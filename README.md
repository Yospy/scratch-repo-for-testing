# scratch-repo-for-testing

A small PyTorch scratch project demonstrating **Group Query Attention (GQA)** with
`torch.nn.functional.scaled_dot_product_attention`.

## What it does

`simple.py` builds random query, key, and value tensors for a multi-head attention
setup where the number of query heads exceeds the number of KV heads (4 query heads,
2 KV heads, head dim 4). Instead of manually expanding the K/V heads, it passes them
straight to `torch.nn.functional.scaled_dot_product_attention` with `enable_gqa=True`,
which broadcasts the KV heads across query heads natively, then attention is computed
with a causal mask.

The script prints the per-head and combined output shapes:

```
Q: torch.Size([4, 4, 16, 4])
K: torch.Size([4, 2, 16, 4])
V: torch.Size([4, 2, 16, 4])
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
