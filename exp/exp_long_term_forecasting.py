# Adapted from RAFT (https://github.com/archon159/RAFT) and Time-Series-Library
# (https://github.com/thuml/Time-Series-Library), both MIT License.
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data_provider.data_factory import data_provider
from layers.Decomposition import extract_trend
from models import RISE
from utils.metrics import metric
from utils.tools import adjust_learning_rate


class Exp_Long_Term_Forecast(object):
    def __init__(self, args):
        self.args = args
        self._data = {}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = RISE.Model(self.args).float()

        train_data, _ = self._get_data(flag='train')
        vali_data, _ = self._get_data(flag='val')
        test_data, _ = self._get_data(flag='test')

        # Memory bank from the training split; retrieval results for all three splits.
        model.prepare_dataset(train_data, vali_data, test_data)

        return model

    def _get_data(self, flag):
        if flag not in self._data:
            self._data[flag] = data_provider(self.args, flag)
        return self._data[flag]

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_loader, criterion, mode='valid'):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for index, batch_x, batch_y in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x, index, mode=mode)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = None
        if self.args.lradj == 'cosine_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(model_optim, T_0=5, T_mult=1, eta_min=1e-6)

        # Optional auxiliary decomposition loss (off by default: aux_loss_weight = 0).
        use_aux = (self.model.composition_mode == 'additive'
                   and self.model.aux_loss_weight > 0
                   and not self.model.ablate_trend
                   and not self.model.ablate_retrieval)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (index, batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if use_aux:
                    outputs, trend_pred, seasonal_pred = self.model(
                        batch_x, index, mode='train', return_components=True)
                else:
                    outputs = self.model(batch_x, index, mode='train')

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y_target)

                if use_aux:
                    # Supervise the trend and retrieval pathways with the decomposed ground truth.
                    with torch.no_grad():
                        y_raw = batch_y[:, -self.args.pred_len:, :]
                        x_offset = batch_x[:, -1:, :]
                        y_norm = y_raw - x_offset
                        y_trend = extract_trend(y_norm, self.model.trend_kernel, causal=self.model.causal_ma)
                        y_seasonal = y_norm - y_trend
                    loss_trend = criterion(trend_pred[:, :, f_dim:], y_trend[:, :, f_dim:])
                    loss_seasonal = criterion(seasonal_pred[:, :, f_dim:], y_seasonal[:, :, f_dim:])
                    loss = loss + self.model.aux_loss_weight * (loss_trend + loss_seasonal)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion, mode='valid')
            # The test loss is logged for monitoring only. Training always runs for the full
            # number of epochs: there is no early stopping and no checkpoint selection.
            test_loss = self.vali(test_loader, criterion, mode='test')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            adjust_learning_rate(model_optim, scheduler)

        # Save the model as it is at the end of training (the state that is evaluated).
        torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))

        return self.model

    def test(self, setting, load_checkpoint=False):
        _, test_loader = self._get_data(flag='test')
        if load_checkpoint:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for index, batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, index, mode='test')

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # Metrics are computed in the z-scored space of the data loader.
        mae, mse, rmse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        folder_path = os.path.join(self.args.results, setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        record = {
            'setting': setting,
            'root_path': self.args.root_path,
            'data_path': self.args.data_path,
            'features': self.args.features,
            'seq_len': self.args.seq_len,
            'pred_len': self.args.pred_len,
            'seed': self.args.seed,
            'n_test': int(preds.shape[0]),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
        }
        with open(os.path.join(folder_path, 'metrics.json'), 'w') as f:
            json.dump(record, f, indent=2)

        summary_path = os.path.join(self.args.results, 'summary.csv')
        write_header = not os.path.exists(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        return record
