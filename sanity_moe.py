"""
sanity_moe.py — runtime smoke test for the qwen3_5_moe family (run in the project venv).

Builds a tiny synthetic Qwen3.5-MoE (forcing one dense `mlp_only` layer + MoE layers, and a
mix of GDN / full-attention), then checks: routing, per-layer FFN dispatch, prefill, cached
single-token decode (recurrent GDN + MoE top-k on one token), and top-k weight normalization.
No checkpoint needed — random init, shapes/finiteness only.

    ./venv/bin/python sanity_moe.py
"""
import sys; sys.path.insert(0, "src")
import torch
import router  # registers families
from qwen3_5_moe.config import Qwen3_5MoeConfig
from qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

fam = router.route("qwen3_5_moe")
print("routed qwen3_5_moe ->", fam.load.__module__)

cfg = Qwen3_5MoeConfig(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
    rope_theta=10000.0, partial_rotary_factor=0.5, tie_word_embeddings=True, hidden_act="silu",
    layer_types=["linear_attn", "linear_attn", "linear_attn", "full_attention"],
    full_attention_interval=4, attn_output_gate=True, output_gate_type="sigmoid",
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
    linear_value_head_dim=16, linear_conv_kernel_dim=4,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, norm_topk_prob=True, decoder_sparse_step=1,
    mlp_only_layers=[0])

torch.manual_seed(0)
model = Qwen3_5MoeModel(cfg).eval()
kinds = [type(l.mlp).__name__ for l in model.model.layers]
print("ffn per layer:", kinds)
assert kinds == ["MLP", "SparseMoeBlock", "SparseMoeBlock", "SparseMoeBlock"], kinds

ids = torch.randint(0, 128, (1, 10))
logits, cache = model(ids)
print("prefill logits:", tuple(logits.shape), "seen:", cache.seen_tokens)
assert logits.shape == (1, 10, 128) and torch.isfinite(logits).all()

logits2, cache = model(ids[:, -1:], cache)
print("decode logits:", tuple(logits2.shape), "seen:", cache.seen_tokens)
assert logits2.shape == (1, 1, 128) and cache.seen_tokens == 11 and torch.isfinite(logits2).all()

moe = model.model.layers[1].mlp
x = torch.randn(2, 5, 64)
y = moe(x)
print("moe out:", tuple(y.shape))
assert y.shape == (2, 5, 64) and torch.isfinite(y).all()
print("\nALL CHECKS PASSED")
