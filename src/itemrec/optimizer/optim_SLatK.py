# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - SL@K Optimizer
# Description:
#  This module provides the SL@K (Top-K Softmax Loss) Optimizer for ItemRec.
#  SL@K is a NDCG@K oriented loss function for item recommendation.
#  - Yang, W., Chen, J., Zhang, S., Wu, P., Sun, Y., Feng, Y., Chen, C., Wang, C.,
#   Breaking the Top-$K$ Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems.
#   31st SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track.
# -------------------------------------------------------------------

from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from .optim_Base import IROptimizer
from ..dataset import IRDataBatch
from ..model import IRModel

__all__ = ['SLatKOptimizer']


class SLatKOptimizer(IROptimizer):
    r"""
    SL@K Optimizer for ItemRec.
    SL@K is a NDCG@K surrogate loss function for item recommendation.
    """
    def __init__(self, model: IRModel, lr: float = 0.1, weight_decay: float = 0.0,
        neg_num: int = 1000, tau: float = 1.0, tau_beta: float = 1.0, K: int = 20,
        epoch_quantile: int = 20, train_dict: List[List[int]] = None) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau = tau
        self.tau_beta = tau_beta
        self.K = K
        self.epoch_quantile = epoch_quantile
        assert train_dict is not None, 'train_dict is required.'
        self.train_dict, self.mask, self.pos_item_num = self._construct_train_dict(train_dict)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.beta = torch.zeros((model.user_size, 1), dtype=torch.float32, device=model.device)
        self.weight_sigma = lambda x: torch.sigmoid(x / self.tau_beta)

    def _construct_train_dict(self, train_data: List[List[int]], use_cutoff: bool = True) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_item_counts = [len(items) for items in train_data]
        if use_cutoff:
            max_len = int(np.percentile(pos_item_counts, 90))
            train_data = [items[:max_len] for items in train_data]
        max_len = max(len(items) for items in train_data)
        pos_item_counts = torch.tensor(pos_item_counts, dtype=torch.long, device=self.model.device)
        mask = [[1] * len(items) + [0] * (max_len - len(items)) for items in train_data]
        mask = torch.tensor(mask, dtype=torch.bool, device=self.model.device)
        train_data = [items + [0] * (max_len - len(items)) for items in train_data]
        train_data = torch.tensor(train_data, dtype=torch.long, device=self.model.device)
        return train_data, mask, pos_item_counts

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        user_emb, item_emb, *extra = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]
        pos_item = item_emb[batch.pos_item]
        neg_items = item_emb[batch.neg_items]
        pos_scores = F.cosine_similarity(user, pos_item)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)
        diff = neg_scores - pos_scores.unsqueeze(1)
        softmax_loss = torch.logsumexp(diff / self.tau, dim=1)
        user_beta = self.beta[batch.user]
        weights = self.weight_sigma(pos_scores - user_beta.squeeze(1))
        loss = (weights * softmax_loss).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *extra)
        return loss

    def cal_quantile(self, batch: IRDataBatch) -> None:
        with torch.no_grad():
            user_emb, item_emb, *extra = self.model.embed(norm=self.model.norm)
            user = user_emb[batch.user]
            batch_pos_items = self.train_dict[batch.user]
            pos_items = item_emb[batch_pos_items]
            neg_items = item_emb[batch.neg_items]
            pos_scores = F.cosine_similarity(user.unsqueeze(1), pos_items, dim=2)
            batch_mask = self.mask[batch.user]
            pos_scores = torch.masked_fill(pos_scores, ~batch_mask, -1e6)
            neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)
            scores = torch.cat([pos_scores, neg_scores], dim=1)
            beta = torch.topk(scores, self.K, dim=1)[0][:, -1]
            self.beta[batch.user] = beta.unsqueeze(1)

    def step(self, batch: IRDataBatch, epoch: int) -> float:
        self.optimizer.zero_grad()
        loss = self.cal_loss(batch)
        loss.backward()
        self.optimizer.step()
        if (epoch + 1) % self.epoch_quantile == 0:
            self.cal_quantile(batch)
        return loss.cpu().item()

