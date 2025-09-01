
"""
frontier_example.py — Example usage of the single-file frontier API.

Two ways to run:
1) Stub backend (no weights): quick end-to-end plumbing test.
2) EA layer backend: call existing ea_layer.topK_genrate(...) with your features and input_ids.
"""

import torch
from eagle.model.frontier_api import FrontierConfig, frontier_generate

def demo_stub():
    cfg = FrontierConfig(total_token=60, depth=5, top_k=10, vocab_size=128256, hidden_size=12288,
                         use_concat_taps=True, device=torch.device("cpu"))
    # Fake features (concatenated or already aligned — the stub ignores content)
    Xcat = torch.randn(1, 36, cfg.hidden_size)
    out = frontier_generate(cfg, features_concat=Xcat, backend="stub")
    draft_tokens, retrieve_indices, tree_mask, tree_pos = out
    print("stub ->", draft_tokens.shape, retrieve_indices.shape, tree_mask.shape, tree_pos.shape)

def demo_ea_layer(model, input_ids, features_concat):
    cfg = FrontierConfig(total_token=60, depth=5, top_k=10, vocab_size=model.lm_head.weight.shape[0],
                         hidden_size=features_concat.shape[-1], use_concat_taps=True, device=features_concat.device)
    out = frontier_generate(cfg, features_concat=features_concat, backend="ea_layer",
                            ea_layer=model.ea_layer, input_ids=input_ids, logits_processor=None)
    print("ea_layer ->", out.draft_tokens.shape, out.retrieve_indices.shape)

if __name__ == "__main__":
    demo_stub()
    # For ea_layer demo, uncomment and provide a real model + features:
    # model = ...  # EaModel with model.ea_layer
    # input_ids = ...
    # features_concat = ...
    # demo_ea_layer(model, input_ids, features_concat)
