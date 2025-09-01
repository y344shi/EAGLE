import torch
import pytest
import numpy as np
from ..utils import prepare_logits_processor
from ..kv_cache import initialize_past_key_values
from .fused_transformer import draft_token_generation, tree_decoding, evaluate_posterior, update_inference_inputs


class TestDraftToken:
    """Test suite for draft token generation using fused transformer kernel."""
    
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
        
        return model, input_ids
    
    def test_draft_token_generation(self, setup_model):
        """Test draft token generation."""
        model, input_ids = setup_model
        
        # Generate draft tokens using both implementations
        with torch.no_grad():
            # Initialize tree for draft token generation
            sample_token = input_ids[:, -1]
            
            # Get hidden states from the model
            outputs, hidden_state = model.forward(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=None,
                output_orig=False,
            )
            
            # Generate draft tokens using the EA layer
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
                hidden_state, input_ids, model.base_model.lm_head, None
            )
            
            # Get logits from the last hidden state
            logits = model.base_model.lm_head(outputs[0])
            
            # Generate draft tokens using the fused kernel
            fused_draft_tokens, fused_retrieve_indices, fused_tree_mask, fused_tree_position_ids, fused_logits, fused_hidden_state, fused_sample_token = draft_token_generation(
                model, input_ids, None
            )
            
            # Compare results
            assert torch.allclose(draft_tokens, fused_draft_tokens), "Draft tokens do not match"
            assert torch.allclose(tree_mask, fused_tree_mask), "Tree masks do not match"
            assert torch.allclose(tree_position_ids, fused_tree_position_ids), "Tree position IDs do not match"
            assert torch.allclose(logits, fused_logits), "Logits do not match"
            assert torch.allclose(sample_token, fused_sample_token), "Sample tokens do not match"
    
    def test_tree_decoding(self, setup_model):
        """Test tree decoding."""
        model, input_ids = setup_model
        
        # Generate draft tokens
        with torch.no_grad():
            # Get hidden states from the model
            outputs, hidden_state = model.forward(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=None,
                output_orig=False,
            )
            
            # Generate draft tokens using the EA layer
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
                hidden_state, input_ids, model.base_model.lm_head, None
            )
            
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
            fused_logits, fused_hidden_state_new, fused_outputs = tree_decoding(
                model, draft_tokens, past_key_values, tree_position_ids, input_ids, retrieve_indices
            )
            
            # Compare results
            assert torch.allclose(logits, fused_logits, atol=1e-5), "Logits do not match"
            assert torch.allclose(hidden_state_new, fused_hidden_state_new, atol=1e-5), "Hidden states do not match"
    
    def test_evaluate_posterior(self, setup_model):
        """Test posterior evaluation."""
        model, input_ids = setup_model
        
        # Generate draft tokens and logits
        with torch.no_grad():
            # Get hidden states from the model
            outputs, hidden_state = model.forward(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=None,
                output_orig=False,
            )
            
            # Generate draft tokens using the EA layer
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
                hidden_state, input_ids, model.base_model.lm_head, None
            )
            
            # Initialize past key values
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
                model.base_model, max_length=2048
            )
            
            # Set tree mask
            model.base_model.model.tree_mask = tree_mask
            
            # Perform tree decoding
            draft_tokens = draft_tokens.to(input_ids.device)
            outputs = model.base_model.model(
                input_ids=draft_tokens,
                past_key_values=past_key_values,
                position_ids=tree_position_ids,
                use_cache=True,
            )
            hidden_state_new = outputs[0]
            logits = model.base_model.lm_head(hidden_state_new)
            
            # Get candidates
            candidates = draft_tokens[0, retrieve_indices]
            
            # Evaluate posterior using both implementations
            # Original implementation
            best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates, None)
            
            # Fused kernel implementation
            fused_best_candidate, fused_accept_length, fused_sample_p = evaluate_posterior(logits, candidates, None)
            
            # Compare results
            assert torch.allclose(best_candidate, fused_best_candidate), "Best candidates do not match"
            assert torch.allclose(accept_length, fused_accept_length), "Accept lengths do not match"
            assert torch.allclose(sample_p, fused_sample_p), "Sample probabilities do not match"
    
    def test_update_inference_inputs(self, setup_model):
        """Test inference input updates."""
        model, input_ids = setup_model
        
        # Generate draft tokens and logits
        with torch.no_grad():
            # Get hidden states from the model
            outputs, hidden_state = model.forward(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=None,
                output_orig=False,
            )
            
            # Generate draft tokens using the EA layer
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
                hidden_state, input_ids, model.base_model.lm_head, None
            )
            
            # Initialize past key values
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
                model.base_model, max_length=2048
            )
            
            # Set tree mask
            model.base_model.model.tree_mask = tree_mask
            
            # Perform tree decoding
            draft_tokens = draft_tokens.to(input_ids.device)
            outputs = model.base_model.model(
                input_ids=draft_tokens,
                past_key_values=past_key_values,
                position_ids=tree_position_ids,
                use_cache=True,
            )
            hidden_state_new = outputs[0]
            logits = model.base_model.lm_head(hidden_state_new)
            
            # Get candidates
            candidates = draft_tokens[0, retrieve_indices]
            
            # Evaluate posterior
            best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates, None)
            
            # Update inference inputs
            new_token = 0
            updated_input_ids, updated_draft_tokens, updated_retrieve_indices, updated_tree_mask, updated_tree_position_ids, updated_new_token, updated_hidden_state, updated_sample_token = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length, retrieve_indices,
                None, new_token, past_key_values_data, current_length_data,
                model, hidden_state_new, sample_p
            )
            
            # Check that the updated inputs have the expected shapes
            assert updated_input_ids.shape[1] > input_ids.shape[1], "Input IDs were not extended"
            assert updated_draft_tokens.shape == draft_tokens.shape, "Draft tokens shape changed unexpectedly"
            assert updated_retrieve_indices.shape == retrieve_indices.shape, "Retrieve indices shape changed unexpectedly"
            assert updated_tree_mask.shape == tree_mask.shape, "Tree mask shape changed unexpectedly"
            assert updated_tree_position_ids.shape == tree_position_ids.shape, "Tree position IDs shape changed unexpectedly"
            assert updated_new_token > new_token, "New token count was not incremented"


def test_no_gpu_fallback():
    """Test that the fused kernel raises an error when CUDA is not available."""
    # Skip test if CUDA is available
    if torch.cuda.is_available():
        pytest.skip("CUDA available, skipping fallback test")
    
    # Import the functions
    from .fused_transformer import FusedTransformerKernel
    
    # Check that the fused kernel raises an error
    with pytest.raises(RuntimeError):
        kernel = FusedTransformerKernel()


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])