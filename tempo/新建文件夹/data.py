# tempo_sc/data.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import torch

from torch.utils.data import Dataset, IterableDataset

from .utils import sinusoidal_pe

class ReconDatasetMaskedFullT(Dataset):
    def __init__(self, X_log, S, gene_idx, T_total):
        super().__init__()
        self.S = np.array(S)
        self.gene_idx = np.array(gene_idx)
        X_full = X_log.T.astype(np.float32)
        mask = np.zeros(T_total, dtype=np.float32); mask[self.S] = 1
        self.X_masked = X_full * mask[None,:]
        self.X_target = X_full

    def __len__(self):
        return len(self.gene_idx)

    def __getitem__(self, idx):
        g = self.gene_idx[idx]
        return (torch.from_numpy(self.X_masked[g]), torch.from_numpy(self.X_target[g]))

class PairSampler(IterableDataset):
    def __init__(self, genes_all, target_times_all, S, X_log, ae, gcn,
                 z_gcn_full, batch_g, require_grad, device, pe_dim, T_total):
        super().__init__()
        self.genes_all = genes_all
        self.target_times_all = target_times_all
        self.S = list(S)
        self.X_log = X_log
        self.ae = ae
        self.gcn = gcn
        self.z_gcn_full = z_gcn_full
        self.batch_g = batch_g
        self.require_grad = require_grad
        self.N = len(genes_all)
        self.device = device
        self.pe_dim = pe_dim
        self.T_total = T_total

    def __iter__(self):
        order = np.arange(self.N); np.random.shuffle(order)
        for i in range(0, self.N, self.batch_g):
            idx = order[i:i+self.batch_g]
            g_batch = self.genes_all[idx]
            t_batch = self.target_times_all[idx]

            S_arr = np.array(self.S, dtype=np.int64)
            X_full = self.X_log.T.astype(np.float32)
            mask = np.zeros(self.T_total, dtype=np.float32); mask[S_arr] = 1
            X_mask = X_full * mask[None, :]

            gv = X_mask[g_batch].astype(np.float32)
            gv_t = torch.from_numpy(gv).to(self.device)

            if self.require_grad:
                z_ae = self.ae.enc(gv_t)
            else:
                with torch.no_grad():
                    z_ae = self.ae.enc(gv_t)

            if self.z_gcn_full is not None:
                z_gcn = self.z_gcn_full[torch.from_numpy(g_batch).to(self.device)]
            else:
                X_G_T_masked = torch.from_numpy(X_mask).to(self.device)
                with torch.set_grad_enabled(self.require_grad):
                    z_all = self.gcn(X_G_T_masked)
                z_gcn = z_all[torch.from_numpy(g_batch).to(self.device)]

            pe_t_np = sinusoidal_pe(t_batch, d_model=self.pe_dim)
            pe_t = torch.from_numpy(pe_t_np).to(self.device)

            Xb = torch.cat([z_ae, z_gcn, pe_t], dim=1)
            yb = torch.from_numpy(
                self.X_log[t_batch, g_batch].astype(np.float32)[:,None]
            ).to(self.device)
            yield Xb, yb
