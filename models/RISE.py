# Parts of this file are adapted from RAFT (https://github.com/archon159/RAFT) and Time-Series-Library
# (https://github.com/thuml/Time-Series-Library), both MIT License.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from layers.Decomposition import extract_trend
from layers.Embed import PatchEmbedding
from layers.Retrieval import RetrievalTool
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class Model(nn.Module):
    """
    RISE: decomposition-guided retrieval-augmented forecasting.

    The input window is shifted by its last value and split by a strictly causal moving
    average into a trend and a detrended component. Three pathways forecast the horizon:

      1. Patch predictor     - patch-based Transformer over the detrended component.
                               With Retrieval-as-Context (RAC) the retrieved horizons are
                               appended to the patch sequence as extra tokens.
      2. Retrieval predictor - multi-granularity similarity retrieval over the detrended
                               component, using a memory bank built from the training
                               split only.
      3. Trend predictor     - channel-independent linear extrapolation of the trend.

    The pathway outputs are fused (`composition_mode`) and the last value is added back.

    Expected attributes of `configs` (see run.py for the command-line equivalents):
        seq_len, pred_len, enc_in                 window sizes and number of channels
        n_period, topm, trend_kernel              retrieval granularities / neighbours, MA kernel
        d_model, n_heads, dropout, patch_len, stride
        composition_mode                          'time_distributed' | 'additive' | 'concat'
        patch_input_mode                          'residual' (detrended) | 'raw'
        causal_ma, trend_cross_channel, use_retrieval_context, aux_loss_weight
        ablate_patch, ablate_trend, ablate_retrieval
        use_gpu, gpu, num_workers
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        use_gpu = getattr(configs, 'use_gpu', True) and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{getattr(configs, 'gpu', 0)}" if use_gpu else 'cpu')
        self.num_workers = getattr(configs, 'num_workers', 0)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.n_period = getattr(configs, 'n_period', 3)
        self.topm = getattr(configs, 'topm', 20)
        dropout = getattr(configs, 'dropout', 0.1)

        # Ablation flags
        self.ablate_patch = getattr(configs, 'ablate_patch', False)
        self.ablate_trend = getattr(configs, 'ablate_trend', False)
        self.ablate_retrieval = getattr(configs, 'ablate_retrieval', False)

        # Count active pathways
        self.n_active = 3 - sum([self.ablate_patch, self.ablate_trend, self.ablate_retrieval])
        assert self.n_active >= 1, "At least one pathway must be active"

        # Fusion of the pathway outputs
        self.composition_mode = getattr(configs, 'composition_mode', 'time_distributed')
        assert self.composition_mode in ('time_distributed', 'additive', 'concat')

        # Weight of the optional auxiliary decomposition loss (additive fusion only; 0 = off)
        self.aux_loss_weight = getattr(configs, 'aux_loss_weight', 0.0)

        # Moving-average kernel of the trend / detrended split
        self.trend_kernel = getattr(configs, 'trend_kernel', 25)
        if self.trend_kernel % 2 == 0:
            self.trend_kernel += 1

        # Causal MA uses past values only. The centered MA looks ahead inside the window
        # and is kept solely to measure the effect of that leakage.
        self.causal_ma = getattr(configs, 'causal_ma', True)

        # Patch predictor input: 'residual' (detrended component) or 'raw' (shifted input)
        self.patch_input_mode = getattr(configs, 'patch_input_mode', 'residual')
        assert self.patch_input_mode in ('residual', 'raw')

        # --- Patch predictor: patch-based Transformer with 50% patch overlap ---
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', self.patch_len // 2)
        self.d_model_patch = getattr(configs, 'd_model', 128)
        self.patch_padding = self.stride

        self.patch_embedding = PatchEmbedding(
            self.d_model_patch, self.patch_len, self.stride,
            self.patch_padding, dropout
        )

        self.patch_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,
                                      attention_dropout=dropout,
                                      output_attention=False),
                        self.d_model_patch, getattr(configs, 'n_heads', 4)),
                    self.d_model_patch,
                    self.d_model_patch * 2,
                    dropout=dropout,
                    activation='gelu'
                ) for _ in range(2)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(self.d_model_patch),
                Transpose(1, 2)
            )
        )

        self.patch_num = int((self.seq_len + self.patch_padding - self.patch_len) / self.stride + 1)
        self.head_nf = self.d_model_patch * self.patch_num
        self.patch_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, self.pred_len),
            nn.Dropout(dropout)
        )

        # --- Trend predictor (parametric, no retrieval) ---
        # Default: one Linear(L, H) shared by all channels and applied to each channel separately.
        # trend_cross_channel=True instead maps the trends of all channels jointly to the
        # target channel (last channel; meant for features='MS').
        self.trend_cross_channel = getattr(configs, 'trend_cross_channel', False)
        if self.trend_cross_channel:
            self.trend_linear_cc = nn.Linear(self.seq_len * self.channels, self.pred_len)
        else:
            self.trend_linear = nn.Linear(self.seq_len, self.pred_len)

        # --- Retrieval predictor: multi-granularity retrieval on the detrended component ---
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
        )

        self.period_num = self.rt.period_num[-1 * self.n_period:]
        max_g = max(self.period_num)
        if self.seq_len % max_g != 0 or self.pred_len % max_g != 0:
            raise ValueError(
                f'seq_len ({self.seq_len}) and pred_len ({self.pred_len}) must be divisible by the '
                f'coarsest retrieval granularity ({max_g}, set by n_period={self.n_period}).'
            )

        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)

        # --- Retrieval-as-Context (RAC): retrieved horizons as extra tokens of the patch encoder ---
        self.use_retrieval_context = getattr(configs, 'use_retrieval_context', True)
        if self.use_retrieval_context and not self.ablate_retrieval and not self.ablate_patch:
            self.n_retrieval_tokens = self.n_period
            self.retrieval_context_proj = nn.ModuleList([
                nn.Linear(self.pred_len, self.d_model_patch)
                for _ in range(self.n_period)
            ])
            self.retrieval_type_embed = nn.Parameter(
                torch.zeros(1, self.n_retrieval_tokens, self.d_model_patch)
            )
            nn.init.normal_(self.retrieval_type_embed, std=0.02)
        else:
            self.use_retrieval_context = False

        # --- Fusion ---
        # 'time_distributed': learnable weighted sum, one scalar weight per pathway that is
        #                     shared by all horizon steps and channels (initialised to 1 / n_active)
        # 'additive':         plain sum, no parameters
        # 'concat':           one Linear(n_active * H, H) over the concatenated pathway outputs
        if self.composition_mode == 'concat':
            self.linear_pred = nn.Linear(self.n_active * self.pred_len, self.pred_len)
            self.fusion_weights = None
        elif self.composition_mode == 'time_distributed':
            self.linear_pred = None
            self.fusion_weights = nn.Linear(self.n_active, 1, bias=False)
            with torch.no_grad():
                self.fusion_weights.weight.fill_(1.0 / self.n_active)
        else:
            self.linear_pred = None
            self.fusion_weights = None

    def prepare_dataset(self, train_data, valid_data, test_data):
        """
        Build the memory bank from the training split and pre-compute the retrieval
        result of every window of every split.

        Keys are the detrended training windows and values the detrended horizons that
        follow them. Validation and test windows only ever query this training bank.
        """
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            y_data_all.append(td[2][-self.pred_len:])

        train_x_raw = torch.tensor(np.stack(train_data_all, axis=0)).float()
        train_y_raw = torch.tensor(np.stack(y_data_all, axis=0)).float()

        with torch.no_grad():
            train_x_detrended = train_x_raw - extract_trend(train_x_raw, self.trend_kernel, causal=self.causal_ma)
            train_y_detrended = train_y_raw - extract_trend(train_y_raw, self.trend_kernel, causal=self.causal_ma)

        self.rt.build_database(train_x_detrended, train_y_detrended)

        self.retrieval_dict = {}

        print('Retrieval: train split')
        train_rt = self._retrieve_detrended(train_data, train=True)

        print('Retrieval: validation split')
        valid_rt = self._retrieve_detrended(valid_data, train=False)

        print('Retrieval: test split')
        test_rt = self._retrieve_detrended(test_data, train=False)

        del self.rt
        torch.cuda.empty_cache()

        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def _retrieve_detrended(self, data, train=False):
        """Query the memory bank with the detrended windows of one split -> (G, N, P, C)."""
        rt_loader = DataLoader(
            data, batch_size=1024, shuffle=False, num_workers=self.num_workers, drop_last=False
        )

        retrievals = []
        with torch.no_grad():
            for index, batch_x, batch_y in rt_loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_detrended = batch_x - extract_trend(batch_x, self.trend_kernel, causal=self.causal_ma)
                pred = self.rt.retrieve(batch_x_detrended, index, train=train)
                retrievals.append(pred.cpu())

        return torch.cat(retrievals, dim=1)

    def forecast(self, x, index, mode, return_components=False):
        """
        Args:
            x: (B, L, C) input windows
            index: (B,) position of each window in its split (selects the retrieval results)
            mode: 'train' | 'valid' | 'test'
            return_components: also return the trend and retrieval pathway outputs
                               (used by the auxiliary decomposition loss)
        """
        index = index.cpu()
        bsz, seq_len, channels = x.shape

        # Shift by the last value
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        # === Decompose ===
        trend = extract_trend(x_norm, self.trend_kernel, causal=self.causal_ma)
        residual = x_norm - trend  # detrended component

        # Order of `components` and of the time_distributed fusion weights: patch, trend, retrieval.
        components = []
        trend_pred = None
        seasonal_pred = None

        # === Retrieval-as-Context tokens (if enabled) ===
        retrieval_context = None
        if self.use_retrieval_context and not self.ablate_retrieval and not self.ablate_patch:
            pred_from_retrieval = self.retrieval_dict[mode][:, index]
            pred_from_retrieval = pred_from_retrieval.to(self.device)
            ctx_tokens = []
            for i, pr in enumerate(pred_from_retrieval):
                pr_flat = pr.permute(0, 2, 1).reshape(bsz * channels, self.pred_len)
                ctx_tok = self.retrieval_context_proj[i](pr_flat)
                ctx_tokens.append(ctx_tok.unsqueeze(1))
            retrieval_context = torch.cat(ctx_tokens, dim=1)
            retrieval_context = retrieval_context + self.retrieval_type_embed

        # === Patch predictor ===
        if not self.ablate_patch:
            if self.patch_input_mode == 'raw':
                x_patch = x_norm.permute(0, 2, 1)  # [B, C, L] shifted input
            else:  # 'residual' (default)
                x_patch = residual.permute(0, 2, 1)  # [B, C, L] detrended input
            enc_out, n_vars = self.patch_embedding(x_patch)

            if retrieval_context is not None:
                enc_out = torch.cat([enc_out, retrieval_context], dim=1)

            enc_out, _ = self.patch_encoder(enc_out)

            if retrieval_context is not None:
                enc_out = enc_out[:, :self.patch_num, :]

            enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
            enc_out = enc_out.permute(0, 1, 3, 2)
            patch_pred = self.patch_head(enc_out)
            patch_pred = patch_pred.permute(0, 2, 1)  # [B, P, C]
            components.append(patch_pred)

        # === Trend predictor ===
        if not self.ablate_trend:
            if self.trend_cross_channel:
                B = trend.shape[0]
                flat = trend.reshape(B, -1)  # (B, L*C)
                trend_target = self.trend_linear_cc(flat)  # (B, H)
                # Only the target (last) channel receives a trend forecast
                trend_pred = torch.zeros(
                    B, self.pred_len, self.channels,
                    device=trend.device, dtype=trend.dtype,
                )
                trend_pred[:, :, -1] = trend_target
            else:
                trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
            components.append(trend_pred)

        # === Retrieval predictor ===
        if not self.ablate_retrieval:
            pred_from_retrieval = self.retrieval_dict[mode][:, index]
            pred_from_retrieval = pred_from_retrieval.to(self.device)

            retrieval_pred_list = []
            for i, pr in enumerate(pred_from_retrieval):
                g = self.period_num[i]
                pr = pr.reshape(bsz, self.pred_len // g, g, channels)
                pr = pr[:, :, 0, :]
                pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
                pr = pr.reshape(bsz, self.pred_len, self.channels)
                retrieval_pred_list.append(pr)

            retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
            seasonal_pred = retrieval_pred_list.sum(dim=1)
            components.append(seasonal_pred)

        # === Fuse the pathways ===
        if self.composition_mode == 'additive':
            pred = sum(components)
        elif self.composition_mode == 'time_distributed':
            # Stack along a pathway dim: (B, H, n_active, C)
            stacked = torch.stack(components, dim=2)
            B, H, K, C = stacked.shape
            flat = stacked.permute(0, 1, 3, 2).reshape(-1, K)  # (B*H*C, K)
            pred = self.fusion_weights(flat).reshape(B, H, C)
        else:  # 'concat'
            pred = torch.cat(components, dim=1)
            pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)

        pred = pred.reshape(bsz, self.pred_len, self.channels)

        # Add the last value back
        pred = pred + x_offset

        if return_components:
            return pred, trend_pred, seasonal_pred
        return pred

    def forward(self, x_enc, index, mode='train', return_components=False):
        out = self.forecast(x_enc, index, mode, return_components=return_components)
        if return_components:
            pred, trend_pred, seasonal_pred = out
            return pred[:, -self.pred_len:, :], trend_pred, seasonal_pred
        return out[:, -self.pred_len:, :]
