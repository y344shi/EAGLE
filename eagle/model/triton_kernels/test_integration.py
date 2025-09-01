import torch
import pytest
import numpy as np
from ..utils import prepare_logits_processor
from ..kv_cache import initialize_past_key_values
from .fused_transformer import draft_token_generation, tree_decoding, evaluate_posterior, update_inference_inputs
from .fused_draft_kernel import fused_draft_token_generation
from .gpu_assert import GPUAssertionError


class TestIntegration:
    """Integration tests for the fused transformer kernel."""
    
    @pytest.fixture
    def setup_model(self):
        """Set up a model for testing."""
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Import here to avoid import errors when CUDA is not available
        from ..ea_model import EaModel
        
        # Load a small model for testing
        model = EaModel.from_pretrained(
            use_eagle3=True,
            base_model_path="meta-llama/Llama-2-7b-hf",
            ea_model_path="eagle-llama2-7b",
            total_token=10,
            depth=3,
            top_k=4,
            threshold=1.0,
            device_map="auto",
        )
        
        # Move model to CUDA
        model = model.cuda()
        
        # Create a simple input
        tokenizer = model.get_tokenizer()
        input_text = "Hello, world!"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
        
        return model, input_ids, tokenizer
    
    def test_end_to_end_generation(self, setup_model):
        """Test end-to-end token generation."""
        model, input_ids, tokenizer = setup_model
        
        # Generate tokens using both implementations
        with torch.no_grad():
            # Original implementation
            original_output = model.eagenerate(
                input_ids=input_ids,
                temperature=0.0,
                top_p=0.0,
                top_k=0.0,
                max_new_tokens=20,
                max_length=2048,
            )
            
            # Reset model state
            model.ea_layer.reset_kv()
            if hasattr(model, "past_key_values"):
                model.current_length_data.zero_()
            
            # Fused kernel implementation
            fused_output = fused_draft_token_generation(
                model=model,
                input_ids=input_ids,
                temperature=0.0,
                top_p=0.0,
                top_k=0.0,
                max_new_tokens=20,
                max_length=2048,
            )
            
            # Decode outputs
            original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
            fused_text = tokenizer.decode(fused_output[0], skip_special_tokens=True)
            
            # Print outputs
            print(f"Original output: {original_text}")
            print(f"Fused output: {fused_text}")
            
            # Check that the outputs are reasonable
            assert len(original_text) > len(input_ids[0]), "Original implementation didn't generate new tokens"
            assert len(fused_text) > len(input_ids[0]), "Fused implementation didn't generate new tokens"
    
    def test_gpu_assertion(self):
        """Test that GPU assertions work correctly."""
        # Skip test if CUDA is available
        if torch.cuda.is_available():
            pytest.skip("CUDA available, skipping GPU assertion test")
        
        # Import functions
        from .fused_transformer import draft_token_generation
        from .fused_draft_kernel import fused_draft_token_generation
        
        # Create dummy input
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Check that functions raise GPUAssertionError
        with pytest.raises(GPUAssertionError):
            draft_token_generation(None, input_ids)
        
        with pytest.raises(GPUAssertionError):
            fused_draft_token_generation(None, input_ids)
    
    def test_critical_checkpoints(self, setup_model):
        """Test critical checkpoints in the generation process."""
        model, input_ids, _ = setup_model
        
        # Generate draft tokens
        with torch.no_grad():
            # Get hidden states from the model
            outputs, hidden_state = model.forward(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=None,
                output_orig=False,
            )
            
            # Generate draft tokens using both implementations
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
                hidden_state, input_ids, model.base_model.lm_head, None
            )
            
            fused_draft_tokens, fused_retrieve_indices, fused_tree_mask, fused_tree_position_ids, _, _, _ = draft_token_generation(
                model, input_ids, None
            )
            
            # Compare draft tokens
            assert torch.allclose(draft_tokens, fused_draft_tokens), "Draft tokens do not match"
            
            # Initialize past key values
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
                model.base_model, max_length=2048
            )
            
            # Set tree mask
            model.base_model.model.tree_mask = tree_mask
            
            # Perform tree decoding using both implementations
            # Original implementation
            draft_tokens = draft_tokens.to(input_ids.device)
            outputs = model.base_model.model(
                input_ids=draft_tokens,
                past_key_values=past_key_values,
                position_ids=tree_position_ids,
                use_cache=True,
            )
            hidden_state_new = outputs[0]
            logits = model.base_model.lm_head(hidden_state_new)
            
            # Fused kernel implementation
            fused_logits, fused_hidden_state_new, _ = tree_decoding(
                model, draft_tokens, past_key_values, tree_position_ids, input_ids, retrieve_indices
            )
            
            # Compare logits
            assert torch.allclose(logits, fused_logits, atol=1e-5), "Logits do not match"
            
            # Get candidates
            candidates = draft_tokens[0, retrieve_indices]
            
            # Evaluate posterior using both implementations
            best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates, None)
            
            # Update inference inputs using both implementations
            new_token = 0
            updated_input_ids, _, _, _, _, updated_new_token, _, _ = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length, retrieve_indices,
                None, new_token, past_key_values_data, current_length_data,
                model, hidden_state_new, sample_p
            )
            
            # Check that tokens were added
            assert updated_input_ids.shape[1] > input_ids.shape[1], "No tokens were added"
            assert updated_new_token > new_token, "New token count was not incremented"


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])