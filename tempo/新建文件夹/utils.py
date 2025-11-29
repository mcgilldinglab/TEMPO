# tempo_sc/utils.py
from __future__ import annotations
import math, random
import numpy as np
import torch

def _fmt_secs(sec):
    if sec < 1e-3: return f"{sec*1e6:.1f}µs"
    if sec < 1.0: return f"{sec*1e3:.1f}ms"
    return f"{sec:.3f}s"

def set_seed(s, device_str="cpu"):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if "cuda" in device_str:
        torch.cuda.manual_seed_all(s)

def sinusoidal_pe(t_idx: np.ndarray, d_model: int):
    pe = np.zeros((len(t_idx), d_model), dtype=np.float32)
    pos = t_idx[:, None].astype(np.float32)
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe

def _cosine_knn_graph(X_GT, k, device):
    Xn = X_GT / (np.linalg.norm(X_GT, axis=1, keepdims=True) + 1e-12)
    S_mat = Xn @ Xn.T
    np.fill_diagonal(S_mat, 0)
    rows, cols, vals = [], [], []
    G_ = X_GT.shape[0]
    for g in range(G_):
        idx = np.argpartition(-S_mat[g], k-1)[:k]
        for j in idx:
            rows.append(g); cols.append(j); vals.append(S_mat[g, j])
    rows.extend(range(G_))
    cols.extend(range(G_))
    vals.extend([1.0]*G_)
    edge_i = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_v = torch.tensor(vals, dtype=torch.float32, device=device)
    deg = torch.zeros(G_, device=device).scatter_add_(0, edge_i[0], edge_v)
    dinv = deg.pow(-0.5); dinv[deg==0] = 0
    edge_v = dinv[edge_i[0]] * edge_v * dinv[edge_i[1]]
    return torch.sparse_coo_tensor(edge_i, edge_v, (G_, G_)).coalesce()
