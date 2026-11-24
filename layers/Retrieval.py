# Adapted from RAFT (https://github.com/archon159/RAFT), MIT License.
import math

import torch
import torch.nn.functional as F


class RetrievalTool():
    """
    Multi-granularity similarity retrieval over a fixed memory bank.

    The memory bank holds every training window (key) together with the horizon that
    follows it (value). Keys and values are compared at ``n_period`` temporal
    granularities (window means over g steps, g in {..., 4, 2, 1}); at each granularity
    the top-m most correlated keys are softmax-weighted (temperature ``temperature``)
    and their values are averaged. Retrieval is non-parametric: nothing here is learned.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)

        self.temperature = temperature
        self.topm = topm

    def build_database(self, train_x, train_y):
        """
        Store the memory bank.

        Args:
            train_x: (T, S, C) training input windows (keys)
            train_y: (T, P, C) the horizons that follow them (values)
        """
        self.n_train = train_x.shape[0]
        # A training query masks up to 2 * (S + P) - 1 entries around itself (see `retrieve`);
        # at least `topm` candidates must remain.
        min_bank = 2 * (self.seq_len + self.pred_len) - 1 + self.topm
        if self.n_train < min_bank:
            raise ValueError(
                f'The training split yields {self.n_train} windows, but retrieval with '
                f'seq_len={self.seq_len}, pred_len={self.pred_len}, topm={self.topm} needs at least {min_bank}.'
            )
        self.train_data_all_mg = self.decompose_mg(train_x)  # G, T, S, C
        self.y_data_all_mg = self.decompose_mg(train_y)  # G, T, P, C

    def decompose_mg(self, data_all):
        """Multi-granularity views of (T, S, C) windows, each shifted by its own last value."""
        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)

            mg.append(cur)

        mg = torch.stack(mg, dim=0)  # G, T, S, C

        for i, data_p in enumerate(mg):
            mg[i] = data_p - data_p[:, -1:, :]

        return mg

    def periodic_batch_corr(self, data_all, key, in_bsz=512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)

        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)

            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)

        return sim

    def retrieve(self, x, index, train=True):
        """
        Args:
            x: (B, S, C) query windows
            index: (B,) position of each query in its own split
            train: if True the queries are training windows; every memory entry whose
                   input or horizon overlaps the query's own input or horizon
                   (|offset| <= S + P - 1) is masked out, so a training query can
                   retrieve neither itself nor any window that reveals its future.

        Returns:
            (G, B, P, C) retrieved horizon per granularity
        """
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_mg = self.decompose_mg(x)  # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2),  # G, T, S * C
            x_mg.flatten(start_dim=2),  # G, B, S * C
        )  # G, B, T

        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)

            sim = sim.masked_fill_(self_mask.bool(), float('-inf'))  # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train)  # G X B, T

        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')

        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train)  # G, B, T

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu()  # G, B, T

        y_data_all = self.y_data_all_mg.flatten(start_dim=2)  # G, T, P * C

        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)

        return pred_from_retrieval
