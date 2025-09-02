import argparse
import time
import torch
import numpy as np
from transformers import AutoTokenizer
import os
import psutil
import gc

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
    from ..model.multi_gpu_inference import load_model_multi_gpu
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *
    from eagle.model.multi_gpu_inference import load_model_multi_gpu


def get_memory_usage():
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'cpu_percent': cpu_percent
    }

def print_memory_usage(phase=""):
    """Print current memory usage"""
    usage = get_memory_usage()
    print(f"  Memory usage {phase}: RSS={usage['rss_mb']:.1f}MB, VMS={usage['vms_mb']:.1f}MB, CPU={usage['cpu_percent']:.1f}%")


class OptimizedEaModel(EaModel):
    """Single-GPU EaModel with multi-GPU optimizations applied"""
    
    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
            
        # Avoid modifying the input_ids in-place
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # Prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        
        print(f"Starting optimized generation loop (max {max_length} iterations)...")
        for idx in range(max_length):
            # Progress indicator every 10 iterations (same as multi-GPU)
            if idx % 10 == 0:
                print(f"  Generation step {idx}/{max_length} - Generated {new_token} tokens so far")
                # Force garbage collection every 10 steps to reduce memory pressure
                gc.collect()
                torch.cuda.empty_cache()  # Clear CUDA cache
                
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            
            # Target model forward, get logits
            if idx % 50 == 0:  # Reduce logging frequency to reduce CPU load
                print(f"    Step {idx}: Running tree decoding...")
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            
            # Prepare candidates
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            
            # Verification
            if idx % 50 == 0:
                print(f"    Step {idx}: Evaluating posterior probabilities...")
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
        
            # Adjusting the input sequence, draft model forward
            if idx % 50 == 0:
                print(f"    Step {idx}: Updating inference inputs (accepted {accept_length} tokens)...")
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    print(f"  Generation stopped at step {idx}: Found stop token")
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                print(f"  Generation stopped at step {idx}: Found EOS token")
                break
            if new_token > max_new_tokens:
                print(f"  Generation stopped at step {idx}: Reached max tokens ({new_token}/{max_new_tokens})")
                break
            if input_ids.shape[1] > max_length:
                print(f"  Generation stopped at step {idx}: Reached max length")
                break
        
        print(f"Optimized generation loop completed after {idx+1} steps, generated {new_token} tokens")
                
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx


def test_optimized_single_gpu():
    """Test the optimized single-GPU model"""
    base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    ea_model_path = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    device = "cuda:0"
    prompt = "Write a short story about a robot who learns to feel emotions."
    max_new_tokens = 128
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Prepare input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    print(f"Testing optimized single-GPU model...")
    print(f"Prompt: {prompt}")
    print(f"Input length: {input_ids.shape[1]} tokens")
    
    # Load model with optimizations
    print("\nLoading optimized single-GPU model...")
    model = load_model_multi_gpu(
        base_model_path,
        ea_model_path,
        use_eagle3=True,
        use_multi_gpu=False,
        base_device=device,
        depth=7,
        threshold=1.0
    )
    
    # Replace the model's eagenerate method with optimized version
    model.__class__ = OptimizedEaModel
    
    print("\nRunning optimized generation...")
    print_memory_usage("before generation")
    
    start_time = time.time()
    output_ids, new_tokens, steps = model.eagenerate(
        input_ids=input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=max_new_tokens,
        is_llama3=True,
        log=True
    )
    end_time = time.time()
    
    print_memory_usage("after generation")
    
    total_time = end_time - start_time
    tokens_per_second = new_tokens / total_time if total_time > 0 else 0
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\n===== OPTIMIZED SINGLE-GPU RESULTS =====")
    print(f"Device: {device}")
    print(f"Generation time: {total_time:.4f}s")
    print(f"Tokens generated: {new_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Steps taken: {steps}")
    print(f"Output length: {len(output_text)} characters")
    
    print(f"\n===== OUTPUT =====")
    print(output_text)


if __name__ == "__main__":
    test_optimized_single_gpu()