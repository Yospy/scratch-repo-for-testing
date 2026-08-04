import torch
import torch.nn.functional as F

batch = 4
seq = 16
num_q_heads = 4
num_kv_heads = 2
head_dim = 4
d_model = num_q_heads * head_dim

q = torch.randn(batch, num_q_heads, seq, head_dim)
k = torch.randn(batch, num_kv_heads, seq, head_dim)
v = torch.randn(batch, num_kv_heads, seq, head_dim)

print("Q:", q.shape)  # (4, 4, 16, 4)
print("K:", k.shape)  # (4, 2, 16, 4)
print("V:", v.shape)  # (4, 2, 16, 4)

# Replaces:
# scores = q @ k.transpose(-2, -1)
# scores /= sqrt(head_dim)
# weights = softmax(scores)
# attn = weights @ v

attn = F.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True,
    enable_gqa=True,
)

print("Per-head output:", attn.shape)
# (4, 4, 16, 4)

# Move heads next to head_dim:
attn = attn.transpose(1, 2)
# (4, 16, 4, 4)

# Concatenate 4 heads × 4 dimensions:
attn = attn.contiguous().reshape(batch, seq, d_model)

print("Combined output:", attn.shape)
# (4, 16, 16)