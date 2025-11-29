# tempo_sc/models.py
from __future__ import annotations
import torch
import torch.nn as nn

class TinyAE(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, d_latent: int = 64):
        super().__init__()
        self.d_latent = d_latent
        self.enc = nn.Sequential(
            nn.Linear(d_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_latent), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(d_latent, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_in)
        )

    def forward(self, x):
        z = self.enc(x)
        rec = self.dec(z)
        return rec, z

class Regressor(nn.Module):
    def __init__(self, d_in, hidden_dim, dropout):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x))

class GraphConv(nn.Module):
    def __init__(self, din, dout, adj_matrix):
        super().__init__()
        self.lin = nn.Linear(din, dout, bias=False)
        self.adj = adj_matrix 

    def forward(self, x):
        return torch.relu(torch.sparse.mm(self.adj, self.lin(x)))

class GeneGCN(nn.Module):
    def __init__(self, t_in, gcn_dim1, gcn_dim2, adj_matrix):
        super().__init__()
        self.gc1 = GraphConv(t_in, gcn_dim1, adj_matrix)
        self.gc2 = GraphConv(gcn_dim1, gcn_dim2, adj_matrix)

    def forward(self, X_mask):
        return self.gc2(self.gc1(X_mask))
