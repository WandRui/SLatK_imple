# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - SONG@K Optimizer
# Reference:
# Yang W. et al. Breaking the Top-K Barrier. KDD 2025.
# -------------------------------------------------------------------

from typing import List, Tuple
import torch
import torch.nn.functional as F
from .optim_Base import IROptimizer
from ..dataset import IRDataBatch
from ..model import IRModel
import numpy as np

__all__ = ['SONGatKOptimizer']


class SONGatKOptimizer(IROptimizer):
    """
    SONG@K:  Bilevel Compositional Optimization for NDCG@K
    """

    def __init__(
        self,
        model: IRModel,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        neg_num: int = 1000,
        tau: float = 1.0,          # temperature for σ_d
        tau_beta: float = 1.0,     # temperature for σ_w
        K: int = 20,
        epoch_quantile: int = 20,
        gamma_g: float = 0.9,      # moving-average momentum
        train_dict: List[List[int]] = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau = tau
        self.tau_beta = tau_beta
        self.K = K
        self.epoch_quantile = epoch_quantile
        self.gamma_g = gamma_g

        # ---------- quantile ----------
        self.register_buffer('beta', torch.full((model.user_size,), -1e6))
        self.train_dict, self.mask, self.pos_num = self._build_train_dict(train_dict)

        # ---------- moving-average for compositional denominator ----------
        self.register_buffer('g', torch.ones(model.user_size) * 1e-6)

        # ---------- optimizers ----------
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------
    def register_buffer(self, name, tensor):
        """helper: register persistent buffer"""
        setattr(self, name, tensor.to(self.model.device))

    def _build_train_dict(self, train_dict: List[List[int]]):
        """pad -> tensor + mask"""
        pos_num = [len(u) for u in train_dict]
        max_len = int(np.percentile(pos_num, 90)) or 1
        padded = [u[:max_len] + [0] * (max_len - len(u)) for u in train_dict]
        mask = [[1] * min(len(u), max_len) + [0] * (max_len - min(len(u), max_len))
                for u in train_dict]
        return (torch.tensor(padded, device=self.model.device),
                torch.tensor(mask, dtype=torch.bool, device=self.model.device),
                torch.tensor(pos_num, device=self.model.device))

    # ------------------------------------------------------------------
    # quantile update (by sorting)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def cal_quantile(self, batch: IRDataBatch):
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = batch.user                                    # (B,)
        pos_items = self.train_dict[user]                  # (B, L)
        neg_items = batch.neg_items                        # (B, N)

        pos_score = F.cosine_similarity(
            user_emb[user].unsqueeze(1), item_emb[pos_items], dim=2)  # (B, L)
        pos_score = pos_score.masked_fill(~self.mask[user], -1e9)
        neg_score = F.cosine_similarity(
            user_emb[user].unsqueeze(1), item_emb[neg_items], dim=2)  # (B, N)

        topk_score = torch.topk(
            torch.cat([pos_score, neg_score], dim=1), k=self.K, dim=1)[0][:, -1]  # (B,)
        self.beta[user] = topk_score

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        """
        Calculates the K-SONG loss with a crucial fix for numerical stability.
        """
        user_emb, item_emb, *addition= self.model.embed(norm=self.model.norm)
        user = batch.user
        pos_item = batch.pos_item
        neg_items = batch.neg_items

        u = user_emb[user]
        i = item_emb[pos_item]
        j = item_emb[neg_items]

        s_ui = F.cosine_similarity(u, i)
        
        # --- Part 1: Calculate g_hat (the ranking surrogate estimator) ---
        d_uij = F.cosine_similarity(u.unsqueeze(1), j, dim=2) - s_ui.unsqueeze(1)
        exp_neg = torch.exp(d_uij / self.tau)
        g_hat = exp_neg.mean(dim=1)

        # --- Part 2: Calculate the weight `p_qi` without gradients ---
        with torch.no_grad():
            weight_selector = torch.sigmoid((s_ui - self.beta[user].squeeze(-1)) / self.tau_beta)
            
            g = self.g[user]
            # 使用一个更大的 epsilon 来增加分母的稳定性
            g_clamped = torch.clamp(g, min=1e-8) 
            weight_denom = 1.0 / g_clamped
            
            p_qi = weight_selector * weight_denom

            # ======================= 关键修正 =======================
            # 对权重进行归一化，防止因 1/g 导致的梯度爆炸
            # 我们将权重的均值缩放到 1，这样可以保持相对重要性，同时稳定梯度尺度
            p_qi = p_qi / (p_qi.mean() + 1e-8)
            # =========================================================

        # --- Part 3: Construct the final loss ---
        loss = (p_qi * g_hat).mean()

        # Add additional loss
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)

        # --- Part 4: Update the moving average of g for the next iteration ---
        current_g = self.g[user]
        current_g = torch.where(current_g < 1e-6, g_hat.detach(), current_g)
        updated_g = (1 - self.gamma_g) * current_g + self.gamma_g * g_hat.detach()
        self.g[user] = updated_g
        
        return loss


    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, batch: IRDataBatch, epoch: int) -> float:
        self.optimizer.zero_grad()
        loss = self.cal_loss(batch)
        loss.backward()
        self.optimizer.step()

        if (epoch + 1) % self.epoch_quantile == 0:
            self.cal_quantile(batch)
        return loss.item()

    def zero_grad(self):
        self.optimizer.zero_grad()