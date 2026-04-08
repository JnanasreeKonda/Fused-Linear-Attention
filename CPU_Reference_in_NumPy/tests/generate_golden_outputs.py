# tests/generate_golden_outputs.py
import numpy as np
import os
from reference import fused_qkv_attention_reference

os.makedirs("tests/golden", exist_ok=True)
np.random.seed(42)

configs = [
    (1,  64,  128, 64),
    (1,  128, 128, 64),
    (4,  256, 128, 64),
    (4,  512, 128, 64),
    (4, 1024, 128, 64),
]

for (B, S, d_model, d_head) in configs:
    X   = np.random.randn(B, S, d_model).astype(np.float32)
    W_q = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
    W_k = np.random.randn(d_model, d_head).astype(np.float32) * 0.02
    W_v = np.random.randn(d_model, d_head).astype(np.float32) * 0.02

    O, _, _, _, _ = fused_qkv_attention_reference(X, W_q, W_k, W_v)

    tag = f"B{B}_S{S}_dm{d_model}_dh{d_head}"
    np.save(f"tests/golden/{tag}_X.npy",   X)
    np.save(f"tests/golden/{tag}_Wq.npy",  W_q)
    np.save(f"tests/golden/{tag}_Wk.npy",  W_k)
    np.save(f"tests/golden/{tag}_Wv.npy",  W_v)
    np.save(f"tests/golden/{tag}_O.npy",   O)
    print(f"Saved {tag}")

print("\nGolden outputs saved to tests/golden/")