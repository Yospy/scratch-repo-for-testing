# scratch-repo-for-testing

A small PyTorch scratch project demonstrating **Group Query Attention (GQA)** with
`torch.nn.functional.scaled_dot_product_attention`.

## What it does

`simple.py` builds random query, key, and value tensors for a multi-head attention
setup where the number of query heads exceeds the number of KV heads (4 query heads,
2 KV heads, head dim 4). Each KV head is shared across `num_q_heads // num_kv_heads`
query heads via `repeat_interleave`, then attention is computed with a causal mask.

The script prints the per-head and combined output shapes:

```
Q: torch.Size([4, 4, 16, 4])
K: torch.Size([4, 4, 16, 4])
V: torch.Size([4, 4, 16, 4])
Per-head output: torch.Size([4, 4, 16, 4])
Combined output: torch.Size([4, 16, 16])
```

## Requirements

- Python 3
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python simple.py
```
