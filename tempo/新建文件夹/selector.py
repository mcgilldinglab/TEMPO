# tempo_sc/selector.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from time import perf_counter

import numpy as np
import anndata as ad
import scanpy as sc

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import (
    _fmt_secs,
    set_seed,
    sinusoidal_pe,
    _cosine_knn_graph,
)
from .models import TinyAE, Regressor, GeneGCN
from .data import ReconDatasetMaskedFullT, PairSampler


# ----------------------------------------------------------------------
#                         TEMPO Selector Main Class
# ----------------------------------------------------------------------
class TEMPO_Selector:
    def __init__(
        self,
        seed=1234,
        device=None,
        max_ref=15,
        train_frac=0.8,
        shuffle_split=True,
        ae_epochs=10,
        reg_epochs=10,
        ft_epochs=10,
        batch_g=128,
        lr=1e-3,
        ft_lr_factor=0.2,
        lambda_r=1.0,
        hidden=128,
        dropout=0.0,
        pe_dim=16,
        k_neighb=30,
        gcn_dim1=128,
        gcn_dim2=64,
        target_lib=1e6,
        beam_width=32,
        precompute_gcn=False,
    ):
        self.SEED = seed
        self.DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # selection parameters
        self.MAX_REF = max_ref
        self.TRAIN_FRAC = train_frac
        self.SHUFFLE_SPLIT = shuffle_split

        # training hyperparameters
        self.AE_EPOCHS = ae_epochs
        self.REG_EPOCHS = reg_epochs
        self.FT_EPOCHS = ft_epochs
        self.BATCH_G = batch_g
        self.LR = lr
        self.FT_LR_FACTOR = ft_lr_factor
        self.LAMBDA_R = lambda_r

        # model dimensions
        self.HIDDEN = hidden
        self.DROPOUT = dropout
        self.PE_DIM = pe_dim

        self.K_NEIGHB = k_neighb
        self.GCN_DIM1 = gcn_dim1
        self.GCN_DIM2 = gcn_dim2

        self.TARGET_LIB = target_lib
        self.BEAM_WIDTH = beam_width
        self.PRECOMPUTE_GCN = precompute_gcn

        set_seed(self.SEED, self.DEVICE)

    # ------------------------------------------------------------------
    #                           Normalization
    # ------------------------------------------------------------------
    def _normalize(self, adata):
        if self.TARGET_LIB is not None:
            sc.pp.normalize_total(adata, target_sum=self.TARGET_LIB)
        sc.pp.log1p(adata)
        return adata

    # ------------------------------------------------------------------
    def build_gene_masked_fullT_tensor(self, X_log, S):
        T_total, G_total = X_log.shape
        X_full = X_log.T.astype(np.float32)
        mask = np.zeros(T_total, dtype=np.float32)
        mask[np.array(S, dtype=np.int64)] = 1.0
        X_mask = X_full * mask[None, :]
        return torch.from_numpy(X_mask).to(self.DEVICE)

    # ------------------------------------------------------------------
    #                           Fit / Split
    # ------------------------------------------------------------------
    def fit(self, adata_input: ad.AnnData, normalize_data: bool = False, verbose: bool = True):
        set_seed(self.SEED, self.DEVICE)

        # Normalize if requested
        adata = self._normalize(adata_input.copy()) if normalize_data else adata_input

        # Extract data matrix
        X0 = adata.X.A if hasattr(adata.X, "A") else np.array(adata.X, dtype=np.float32)
        T_obs, G_var = adata.n_obs, adata.n_vars

        # guarantee shape = (T, G)
        if X0.shape == (T_obs, G_var):
            X_raw = X0.astype(np.float32)
        elif X0.shape == (G_var, T_obs):
            X_raw = X0.T.astype(np.float32)
        else:
            X_raw = (X0 if X0.shape[0] <= X0.shape[1] else X0.T).astype(np.float32)

        self.X_log = X_raw
        self.T, self.G = self.X_log.shape
        self.times = np.arange(self.T, dtype=np.int64)

        # train/val split across genes
        perm = np.random.permutation(self.G) if self.SHUFFLE_SPLIT else np.arange(self.G)
        n_train = max(1, int(round(self.TRAIN_FRAC * self.G)))
        if n_train >= self.G: 
            n_train = self.G - 1
        genes_train = perm[:n_train]
        genes_val = perm[n_train:]
        if len(genes_val) == 0:
            genes_val = perm[-1:]
            genes_train = perm[:-1]

        self.genes_train = genes_train
        self.genes_val = genes_val

        # Build GCN adjacency
        self.A_NORM = _cosine_knn_graph(self.X_log.T, self.K_NEIGHB, self.DEVICE)

        # Start beam search
        S_sel, pack_sel, hist = self.beam_search_select(
            beam_width=self.BEAM_WIDTH,
            max_ref=self.MAX_REF,
            genes_train_in=genes_train,
            genes_val_in=genes_val,
            verbose=verbose,
        )
        return S_sel, pack_sel, hist

    # ------------------------------------------------------------------
    #                   Pair Sampling for Stage 2 / 3
    # ------------------------------------------------------------------
    def make_pairs(self, S, genes_all, use_all_targets=True, n_targets_per_gene=None):
        remaining = np.setdiff1d(self.times, np.array(S))
        if len(remaining) == 0:
            remaining = self.times.copy()

        if use_all_targets:
            t_list = np.repeat(remaining, len(genes_all))
            g_list = np.tile(genes_all, len(remaining))
        else:
            n = int(n_targets_per_gene or 1)
            choices = np.random.choice(remaining, size=(len(genes_all), n), replace=True)
            g_list = np.repeat(genes_all, n)
            t_list = choices.ravel()

        return g_list.astype(np.int64), t_list.astype(np.int64)

    # ------------------------------------------------------------------
    #                             Stage 1
    # ------------------------------------------------------------------
    def pretrain_ae_stage1(self, S, gene_idx=None, d_latent=64):
        assert len(S) >= 1
        if gene_idx is None:
            gene_idx = self.genes_train

        ds = ReconDatasetMaskedFullT(self.X_log, S, gene_idx, self.T)
        dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)

        ae = TinyAE(self.T, self.HIDDEN, d_latent).to(self.DEVICE)
        opt = optim.Adam(ae.parameters(), lr=self.LR)
        loss_fn = nn.MSELoss()

        S_idx = torch.tensor(S, dtype=torch.long, device=self.DEVICE)

        ae.train()
        for _ in range(self.AE_EPOCHS):
            for xb, yb in dl:
                xb = xb.to(self.DEVICE)
                yb = yb.to(self.DEVICE)
                rec, _ = ae(xb)
                loss = loss_fn(
                    rec.index_select(1, S_idx),
                    yb.index_select(1, S_idx),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

        return ae

    # ------------------------------------------------------------------
    #                             Stage 2
    # ------------------------------------------------------------------
    def stage2_train_regressor(self, S, train_pairs, val_pairs, ae):
        g_tr, t_tr = train_pairs
        g_va, t_va = val_pairs

        # Build GCN
        gcn = GeneGCN(self.T, self.GCN_DIM1, self.GCN_DIM2, self.A_NORM).to(self.DEVICE)

        # optional precompute
        if self.PRECOMPUTE_GCN:
            for p in gcn.parameters():
                p.requires_grad = False
            with torch.no_grad():
                X_mask = self.build_gene_masked_fullT_tensor(self.X_log, S)
                z_gcn_full = gcn(X_mask)
        else:
            z_gcn_full = None

        d_in = ae.d_latent + self.GCN_DIM2 + self.PE_DIM
        reg = Regressor(d_in, self.HIDDEN, self.DROPOUT).to(self.DEVICE)

        # freeze AE
        for p in ae.parameters():
            p.requires_grad = False

        # optimizer
        params = list(reg.parameters())
        if not self.PRECOMPUTE_GCN:
            params += list(gcn.parameters())
        opt = optim.Adam(params, lr=self.LR)
        loss_fn = nn.L1Loss()

        tr_ds = PairSampler(
            g_tr, t_tr, S, self.X_log, ae, gcn, z_gcn_full,
            self.BATCH_G, True, self.DEVICE, self.PE_DIM, self.T,
        )
        va_ds = PairSampler(
            g_va, t_va, S, self.X_log, ae, gcn, z_gcn_full,
            self.BATCH_G, False, self.DEVICE, self.PE_DIM, self.T,
        )
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=None)
        va_dl = torch.utils.data.DataLoader(va_ds, batch_size=None)

        # train
        reg.train()
        gcn.train() if not self.PRECOMPUTE_GCN else gcn.eval()

        for _ in range(self.REG_EPOCHS):
            for xb, yb in tr_dl:
                pred = reg(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # validation
        reg.eval()
        gcn.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                yp = reg(xb)
                y_true.append(yb.cpu().numpy().ravel())
                y_pred.append(yp.cpu().numpy().ravel())

        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        y_bar = float(np.mean(y_true))
        sst = float(np.sum((y_true - y_bar) ** 2))
        R2 = float("nan") if sst <= 0 else 1 - (np.sum((y_pred - y_true) ** 2) / sst)

        return reg, gcn, z_gcn_full, mae, R2

    # ------------------------------------------------------------------
    #                             Stage 3
    # ------------------------------------------------------------------
    def stage3_joint_finetune(self, S, train_pairs, val_pairs, ae, reg, gcn, z_gcn_full):
        lr_ft = self.LR * self.FT_LR_FACTOR

        # allow AE to update
        for p in ae.parameters():
            p.requires_grad = True

        if self.PRECOMPUTE_GCN:
            for p in gcn.parameters():
                p.requires_grad = False
            params = list(ae.parameters()) + list(reg.parameters())
        else:
            params = list(ae.parameters()) + list(reg.parameters()) + list(gcn.parameters())

        opt = optim.Adam(params, lr=lr_ft)
        loss_rec = nn.MSELoss()
        loss_reg = nn.L1Loss()

        # reconstruction dataset
        recon_ds = ReconDatasetMaskedFullT(self.X_log, S, self.genes_train, self.T)
        recon_dl = torch.utils.data.DataLoader(recon_ds, batch_size=2048, shuffle=True)
        S_idx = torch.tensor(S, dtype=torch.long, device=self.DEVICE)

        # stage 2 datasets
        g_tr, t_tr = train_pairs
        g_va, t_va = val_pairs

        tr_ds = PairSampler(
            g_tr, t_tr, S, self.X_log, ae, gcn, z_gcn_full,
            self.BATCH_G, True, self.DEVICE, self.PE_DIM, self.T,
        )
        va_ds = PairSampler(
            g_va, t_va, S, self.X_log, ae, gcn, z_gcn_full,
            self.BATCH_G, False, self.DEVICE, self.PE_DIM, self.T,
        )
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=None)
        va_dl = torch.utils.data.DataLoader(va_ds, batch_size=None)

        # --- fine-tune ---
        for _ in range(self.FT_EPOCHS):
            ae.train()
            reg.train()
            gcn.eval() if self.PRECOMPUTE_GCN else gcn.train()

            recon_iter = iter(recon_dl)

            for xb, yb in tr_dl:
                pred = reg(xb)
                L_R = loss_reg(pred, yb)

                try:
                    xr, yr = next(recon_iter)
                except StopIteration:
                    recon_iter = iter(recon_dl)
                    xr, yr = next(recon_iter)

                xr = xr.to(self.DEVICE)
                yr = yr.to(self.DEVICE)
                rec, _ = ae(xr)

                L_AE = loss_rec(
                    rec.index_select(1, S_idx),
                    yr.index_select(1, S_idx),
                )

                L = L_AE + self.LAMBDA_R * L_R
                opt.zero_grad()
                L.backward()
                opt.step()

        # --- Compute metrics ---
        reg.eval()
        gcn.eval()

        # training-set metrics
        tr_ds2 = PairSampler(
            g_tr, t_tr, S, self.X_log, ae, gcn, z_gcn_full,
            self.BATCH_G, False, self.DEVICE, self.PE_DIM, self.T,
        )
        tr_dl2 = torch.utils.data.DataLoader(tr_ds2, batch_size=None)

        with torch.no_grad():
            y_true_tr, y_pred_tr = [], []
            for xb, yb in tr_dl2:
                yp = reg(xb)
                y_true_tr.append(yb.cpu().numpy().ravel())
                y_pred_tr.append(yp.cpu().numpy().ravel())

        y_true_tr = np.hstack(y_true_tr)
        y_pred_tr = np.hstack(y_pred_tr)
        mae_tr = float(np.mean(np.abs(y_pred_tr - y_true_tr)))
        mse_tr = float(np.mean((y_pred_tr - y_true_tr) ** 2))
        y_bar = float(np.mean(y_true_tr))
        sst = np.sum((y_true_tr - y_bar) ** 2)
        R2_tr = float("nan") if sst <= 0 else 1 - (np.sum((y_pred_tr - y_true_tr) ** 2) / sst)

        # validation-set metrics
        va_ds2 = PairSampler(
            g_va, t_va, S, self.X_log, ae, gcn, z_gcn_full,
            self.BATCH_G, False, self.DEVICE, self.PE_DIM, self.T,
        )
        va_dl2 = torch.utils.data.DataLoader(va_ds2, batch_size=None)

        with torch.no_grad():
            y_true_va, y_pred_va = [], []
            for xb, yb in va_dl2:
                yp = reg(xb)
                y_true_va.append(yb.cpu().numpy().ravel())
                y_pred_va.append(yp.cpu().numpy().ravel())

        y_true_va = np.hstack(y_true_va)
        y_pred_va = np.hstack(y_pred_va)
        mae_va = float(np.mean(np.abs(y_true_va - y_pred_va)))
        mse_va = float(np.mean((y_pred_va - y_true_va) ** 2))
        y_bar = float(np.mean(y_true_va))
        sst = np.sum((y_true_va - y_bar) ** 2)
        R2_va = float("nan") if sst <= 0 else 1 - (np.sum((y_pred_va - y_true_va) ** 2) / sst)

        return (
            ae,
            reg,
            gcn,
            z_gcn_full,
            mae_tr,
            R2_tr,
            mse_tr,
            mae_va,
            R2_va,
            mse_va,
        )

    # ------------------------------------------------------------------
    #                  Evaluate a candidate S during beam search
    # ------------------------------------------------------------------
    def _score_and_train_for_S(self, S_try, genes_train_in, genes_val_in):
        ae = self.pretrain_ae_stage1(S_try, gene_idx=genes_train_in)

        def _targets_per_gene_for_len(s_len):
            return 1 if s_len <= 1 else 2

        ntpg = _targets_per_gene_for_len(len(S_try))
        g_tr, t_tr = self.make_pairs(
            S_try, genes_train_in, use_all_targets=False, n_targets_per_gene=ntpg
        )
        g_va, t_va = self.make_pairs(S_try, genes_val_in, use_all_targets=True)

        reg, gcn, z_gcn_full, mae2, R22 = self.stage2_train_regressor(
            S_try, (g_tr, t_tr), (g_va, t_va), ae
        )

        (
            ae,
            reg,
            gcn,
            z_gcn_full,
            mae_tr,
            R2_tr,
            mse_tr,
            mae_va,
            R2_va,
            mse_va,
        ) = self.stage3_joint_finetune(
            S_try, (g_tr, t_tr), (g_va, t_va), ae, reg, gcn, z_gcn_full
        )

        return (ae, reg, gcn, z_gcn_full), mae_tr, R2_tr, mse_tr, mae_va, R2_va, mse_va

    # ------------------------------------------------------------------
    #                      Beam Search Selection
    # ------------------------------------------------------------------
    def beam_search_select(
        self,
        beam_width,
        max_ref,
        genes_train_in,
        genes_val_in,
        verbose=False,
    ):
        t0 = perf_counter()

        # initial candidates: single S={t}
        init_entries = []
        for t0_idx in range(self.T):
            S0 = [t0_idx]
            (
                pack0,
                mae_tr0,
                R2_tr0,
                mse_tr0,
                mae_va0,
                R2_va0,
                mse_va0,
            ) = self._score_and_train_for_S(S0, genes_train_in, genes_val_in)
            init_entries.append(
                dict(
                    S=S0,
                    pack=pack0,
                    mae_tr=mae_tr0,
                    R2_tr=R2_tr0,
                    mse_tr=mse_tr0,
                    mae_va=mae_va0,
                    R2_va=R2_va0,
                    mse_va=mse_va0,
                    added_t=t0_idx,
                )
            )

        init_entries.sort(key=lambda e: (e["mae_tr"], -np.nan_to_num(e["R2_tr"], nan=-1e9)))
        beam = init_entries[:beam_width]

        step_time = perf_counter() - t0
        history = [
            dict(
                step=1,
                S=beam[0]["S"].copy(),
                added_t=beam[0]["added_t"],
                TRAIN_MAE=round(beam[0]["mae_tr"], 6),
                TRAIN_MSE=round(beam[0]["mse_tr"], 6),
                TRAIN_R2=round(beam[0]["R2_tr"], 6),
                VAL_MAE=round(beam[0]["mae_va"], 6),
                VAL_MSE=round(beam[0]["mse_va"], 6),
                VAL_R2=round(beam[0]["R2_va"], 6),
                time_sec=step_time,
            )
        ]

        if verbose:
            top = history[-1]
            print(
                f"[Step 1] S={top['S']} added_t={top['added_t']} "
                f"MAE_train={top['TRAIN_MAE']:.6f} "
                f"MSE_train={top['TRAIN_MSE']:.6f} "
                f"R2_train={top['TRAIN_R2']:.6f} | "
                f"MAE_val={top['VAL_MAE']:.6f} MSE_val={top['VAL_MSE']:.6f} "
                f"R2_val={top['VAL_R2']:.6f} time={_fmt_secs(top['time_sec'])}"
            )

        # Allocate more
        depth = 1
        L_target = min(max_ref, self.T)
        while depth < L_target:
            depth += 1
            d_start = perf_counter()

            candidates = []
            for entry in beam:
                S_curr = entry["S"]
                remaining = [t for t in range(self.T) if t not in S_curr]

                for t_star in remaining:
                    S_try = S_curr + [t_star]
                    (
                        pack,
                        mae_tr,
                        R2_tr,
                        mse_tr,
                        mae_va,
                        R2_va,
                        mse_va,
                    ) = self._score_and_train_for_S(S_try, genes_train_in, genes_val_in)

                    candidates.append(
                        dict(
                            S=S_try,
                            pack=pack,
                            mae_tr=mae_tr,
                            R2_tr=R2_tr,
                            mse_tr=mse_tr,
                            mae_va=mae_va,
                            R2_va=R2_va,
                            mse_va=mse_va,
                            added_t=t_star,
                        )
                    )

            candidates.sort(
                key=lambda e: (e["mae_tr"], -np.nan_to_num(e["R2_tr"], nan=-1e9))
            )
            beam = candidates[:beam_width]

            d_time = perf_counter() - d_start
            top = beam[0]
            history.append(
                dict(
                    step=depth,
                    S=top["S"].copy(),
                    added_t=top["added_t"],
                    TRAIN_MAE=round(top["mae_tr"], 6),
                    TRAIN_MSE=round(top["mse_tr"], 6),
                    TRAIN_R2=round(top["R2_tr"], 6),
                    VAL_MAE=round(top["mae_va"], 6),
                    VAL_MSE=round(top["mse_va"], 6),
                    VAL_R2=round(top["R2_va"], 6),
                    time_sec=d_time,
                )
            )

            if verbose:
                h = history[-1]
                print(
                    f"[Step {h['step']}] S={h['S']} added_t={h['added_t']} "
                    f"MAE_train={h['TRAIN_MAE']:.6f} MSE_train={h['TRAIN_MSE']:.6f} "
                    f"R2_train={h['TRAIN_R2']:.6f} | "
                    f"MAE_val={h['VAL_MAE']:.6f} MSE_val={h['VAL_MSE']:.6f} "
                    f"R2_val={h['VAL_R2']:.6f} time={_fmt_secs(h['time_sec'])}"
                )

        # final best S
        best = min(
            beam,
            key=lambda e: (e["mae_tr"], -np.nan_to_num(e["R2_tr"], nan=-1e9)),
        )
        return best["S"], best["pack"], history

    # ------------------------------------------------------------------
    #                      Prediction with Trained Pack
    # ------------------------------------------------------------------
    def predict_full_from_pack(self, S, pack):
        ae, reg, gcn, z_gcn_full = pack
        reg.eval()
        ae.eval()
        gcn.eval()

        T_, G_ = self.T, self.G
        allg = np.arange(G_)
        P = np.zeros((G_, T_), dtype=np.float32)

        with torch.no_grad():
            for t in range(T_):
                for i in range(0, G_, self.BATCH_G):
                    g_slice = allg[i : i + self.BATCH_G]
                    t_slice = np.full_like(g_slice, t)

                    # masked full T
                    S_arr = np.array(S, dtype=np.int64)
                    X_full = self.X_log.T.astype(np.float32)
                    mask = np.zeros(self.T, dtype=np.float32)
                    mask[S_arr] = 1
                    X_mask = X_full * mask[None, :]

                    gv = X_mask[g_slice].astype(np.float32)
                    gv_t = torch.from_numpy(gv).to(self.DEVICE)

                    z_ae = ae.enc(gv_t)

                    if z_gcn_full is not None:
                        z_gcn = z_gcn_full[torch.from_numpy(g_slice).to(self.DEVICE)]
                    else:
                        X_G_T_masked = torch.from_numpy(X_mask).to(self.DEVICE)
                        z_all = gcn(X_G_T_masked)
                        z_gcn = z_all[torch.from_numpy(g_slice).to(self.DEVICE)]

                    pe_t_np = sinusoidal_pe(t_slice, d_model=self.PE_DIM)
                    pe_t = torch.from_numpy(pe_t_np).to(self.DEVICE)

                    xb = torch.cat([z_ae, z_gcn, pe_t], dim=1)
                    yp = reg(xb).cpu().numpy().ravel()
                    P[i : i + len(g_slice), t] = yp

        return P
