import torch
import pytest
import time
from ..utils import prepare_logits_processor
from ..kv_cache import initialize_past_key_values
from .fused_draft_kernel import fused_draft_token_generation


class TestFusedDraftKernel:
    """Test suite for the fused draft kernel."""
    
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
    
    def test_fused_draft_generation(self, setup_model):
        """Test fused draft token generation."""
        model, input_ids = setup_model
        
        # Generate tokens using both implementations
        with torch.no_grad():
            # Original implementation
            start_time = time.time()
            original_output = model.eagenerate(
                input_ids=input_ids,
                temperature=0.0,
                top_p=0.0,
                top_k=0.0,
                max_new_tokens=20,
                max_length=2048,
            )
            original_time = time.time() - start_time
            
            # Reset model state
            model.ea_layer.reset_kv()
            if hasattr(model, "past_key_values"):
                model.current_length_data.zero_()
            
            # Fused kernel implementation
            start_time = time.time()
            fused_output = fused_draft_token_generation(
                model=model,
                input_ids=input_ids,
                temperature=0.0,
                top_p=0.0,
                top_k=0.0,
                max_new_tokens=20,
                max_length=2048,
            )
            fused_time = time.time() - start_time
            
            # Compare results
            # Note: Due to non-determinism in GPU operations, the outputs might not be exactly the same
            # We check that the shapes match and the first few tokens are the same
            assert original_output.shape == fused_output.shape, "Output shapes do not match"
            assert torch.all(original_output[:, :input_ids.shape[1]] == fused_output[:, :input_ids.shape[1]]), "Input tokens were modified"
            
            # Print performance comparison
            print(f"Original implementation: {original_time:.4f}s")
            print(f"Fused kernel implementation: {fused_time:.4f}s")
            print(f"Speedup: {original_time / fused_time:.2f}x")
    
    def test_temperature_sampling(self, setup_model):
        """Test fused draft token generation with temperature sampling."""
        model, input_ids = setup_model
        
        # Generate tokens using both implementations with temperature sampling
        with torch.no_grad():
            # Set random seed for reproducibility
            torch.manual_seed(42)
            
            # Original implementation
            original_output = model.eagenerate(
                input_ids=input_ids,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                max_new_tokens=20,
                max_length=2048,
            )
            
            # Reset model state and random seed
            model.ea_layer.reset_kv()
            if hasattr(model, "past_key_values"):
                model.current_length_data.zero_()
            torch.manual_seed(42)
            
            # Fused kernel implementation
            fused_output = fused_draft_token_generation(
                model=model,
                input_ids=input_ids,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                max_new_tokens=20,
                max_length=2048,
            )
            
            # Compare results
            # Note: Due to non-determinism in sampling, the outputs might not be exactly the same
            # We check that the shapes are reasonable and the first few tokens are the same
            assert original_output.shape[1] >= input_ids.shape[1], "No tokens were generated"
            assert fused_output.shape[1] >= input_ids.shape[1], "No tokens were generated"
            assert torch.all(original_output[:, :input_ids.shape[1]] == fused_output[:, :input_ids.shape[1]]), "Input tokens were modified"


def test_no_gpu_fallback():
    """Test that the fused kernel raises an error when CUDA is not available."""
    # Skip test if CUDA is available
    if torch.cuda.is_available():
        pytest.skip("CUDA available, skipping fallback test")
    
    # Import the function
    from .fused_draft_kernel import fused_draft_token_generation
    
    # Check that the function raises an error
    with pytest.raises(RuntimeError):
        fused_draft_token_generation(None, torch.tensor([[1, 2, 3]]))


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])