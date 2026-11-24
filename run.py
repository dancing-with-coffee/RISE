"""
Train and evaluate RISE on one dataset / horizon / seed.

Example (horizon 24, seed 0, main configuration):
    python run.py --root_path ./data/ --data_path sp500.csv --pred_len 24 --seed 0

The defaults are the main configuration of the paper. See README.md for the CSV
format and for the settings of each benchmark.
"""
# Adapted from RAFT (https://github.com/archon159/RAFT) and Time-Series-Library
# (https://github.com/thuml/Time-Series-Library), both MIT License.
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def build_parser():
    parser = argparse.ArgumentParser(description='RISE: decomposition-guided retrieval-augmented forecasting')

    # run
    parser.add_argument('--is_training', type=int, default=1,
                        help='1: train then test; 0: load the saved checkpoint of this setting and test')
    parser.add_argument('--model_id', type=str, default='RISE', help='free-form tag prepended to the run name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # data
    parser.add_argument('--root_path', type=str, default='./data/', help='directory of the CSV file')
    parser.add_argument('--data_path', type=str, required=True, help='CSV file name')
    parser.add_argument('--features', type=str, default='MS', choices=['M', 'MS', 'S'],
                        help='M: multivariate -> multivariate; MS: multivariate -> target only; S: target only')
    parser.add_argument('--target', type=str, default='Close',
                        help='target column; must exist in the CSV in every mode (the forecast variable for '
                             'MS and S; always moved to the last channel)')
    parser.add_argument('--enc_in', type=int, default=None,
                        help='number of input channels (default: inferred from the CSV header)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='where model checkpoints are saved')
    parser.add_argument('--results', type=str, default='./results/', help='where predictions and metrics are saved')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='look-back window L')
    parser.add_argument('--pred_len', type=int, default=24, help='forecast horizon H')

    # decomposition and retrieval
    parser.add_argument('--trend_kernel', type=int, default=25,
                        help='kernel size k of the moving average (odd; an even value is increased by one)')
    parser.add_argument('--centered_ma', dest='causal_ma', action='store_false',
                        help='use a centered instead of the causal moving average. The centered MA looks ahead '
                             'inside each window; it exists only to measure the effect of that leakage')
    parser.add_argument('--n_period', type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help='number of retrieval granularities (3 -> window means over {4, 2, 1} steps)')
    parser.add_argument('--topm', type=int, default=20, help='number of retrieved neighbours m')

    # model
    parser.add_argument('--d_model', type=int, default=128, help='hidden size of the patch encoder')
    parser.add_argument('--n_heads', type=int, default=4, help='attention heads of the patch encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--composition_mode', type=str, default='time_distributed',
                        choices=['time_distributed', 'additive', 'concat'],
                        help='pathway fusion: learnable per-pathway weights, plain sum, or one linear layer')
    parser.add_argument('--patch_input_mode', type=str, default='residual', choices=['residual', 'raw'],
                        help='input of the patch predictor: detrended component or the shifted input')
    parser.add_argument('--use_retrieval_context', type=int, default=1, choices=[0, 1],
                        help='Retrieval-as-Context tokens in the patch encoder (1: on, 0: off)')
    parser.add_argument('--trend_cross_channel', action='store_true',
                        help='trend predictor maps all channels jointly to the target instead of per channel')
    parser.add_argument('--aux_loss_weight', type=float, default=0.0,
                        help='weight of the auxiliary decomposition loss (additive fusion only; 0 disables it)')
    parser.add_argument('--ablate_patch', action='store_true', help='remove the patch predictor')
    parser.add_argument('--ablate_trend', action='store_true', help='remove the trend predictor')
    parser.add_argument('--ablate_retrieval', action='store_true', help='remove the retrieval predictor')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='number of training epochs (no early stopping)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate of Adam')
    parser.add_argument('--lradj', type=str, default='cosine_warm_restarts',
                        choices=['cosine_warm_restarts', 'constant'], help='learning-rate schedule')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')

    # device
    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1], help='1: use CUDA when available; 0: CPU')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')

    return parser


def prepare_args(args):
    """Resolve derived settings (device, number of channels) after parsing."""
    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
    args.use_retrieval_context = bool(args.use_retrieval_context)

    if args.trend_kernel % 2 == 0:
        args.trend_kernel += 1
        print(f'Note: --trend_kernel must be odd; using {args.trend_kernel}.')

    if args.enc_in is None:
        path = os.path.join(args.root_path, args.data_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found. See README.md for the expected CSV format.')
        columns = [c for c in pd.read_csv(path, nrows=0).columns if c != 'date']
        args.enc_in = 1 if args.features == 'S' else len(columns)

    if args.aux_loss_weight > 0 and args.composition_mode != 'additive':
        print('Note: --aux_loss_weight only takes effect with --composition_mode additive.')
    return args


def make_setting(args):
    """Name of the run; used for the checkpoint and result directories."""
    data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    parts = [args.model_id, data_name, f'ft{args.features}', f'sl{args.seq_len}', f'pl{args.pred_len}',
             f'tm{args.topm}', f'k{args.trend_kernel}', args.composition_mode]
    if args.n_period != 3:
        parts.append(f'np{args.n_period}')
    if args.patch_input_mode != 'residual':
        parts.append(args.patch_input_mode)
    if not args.causal_ma:
        parts.append('centeredMA')
    if not args.use_retrieval_context:
        parts.append('noRAC')
    if args.trend_cross_channel:
        parts.append('ccTrend')
    if args.aux_loss_weight > 0:
        parts.append(f'aux{args.aux_loss_weight}')
    for flag, tag in (('ablate_patch', 'noPatch'), ('ablate_trend', 'noTrend'), ('ablate_retrieval', 'noRetrieval')):
        if getattr(args, flag):
            parts.append(tag)
    parts.append(f'seed{args.seed}')
    return '_'.join(parts)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(argv=None):
    args = prepare_args(build_parser().parse_args(argv))
    set_seed(args.seed)

    print('Args in experiment:')
    for key, value in sorted(vars(args).items()):
        print(f'  {key}: {value}')

    setting = make_setting(args)
    exp = Exp_Long_Term_Forecast(args)

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        record = exp.test(setting)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        record = exp.test(setting, load_checkpoint=True)

    torch.cuda.empty_cache()
    return record


if __name__ == '__main__':
    main()
