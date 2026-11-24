# RISE: Retrieve the Right Signal

Official implementation of **RISE** — *Retrieve the Right Signal: Decomposition-Guided Retrieval-Augmented Forecasting for Financial Markets* (CIKM 2026).

Paper: https://doi.org/10.1145/3799682.3841004

## Overview

RISE is a three-pathway forecasting model for financial time series. Each input window is shifted by its last value and split by a strictly causal moving average (kernel k = 25) into a trend and a detrended component. Three pathways then forecast the horizon:

1. **Patch predictor** — a patch-based Transformer over the detrended component. With Retrieval-as-Context (RAC), the retrieved horizons are appended to the patch sequence as extra tokens.
2. **Retrieval predictor** — multi-granularity similarity retrieval over the detrended component (window means over {4, 2, 1} steps, top-m = 20 neighbours, softmax temperature 0.1). The retrieval itself is non-parametric and is pre-computed once before training.
3. **Trend predictor** — a channel-independent linear extrapolation of the causal-MA trend.

The pathway outputs are fused (learnable per-pathway weights, or a plain sum) and the last value is added back.

The pipeline is built so that no validation or test forecast can depend on information from after its own input window:

- the moving average is causal (past values only);
- the standard scaler is fitted on the training split;
- the retrieval memory bank contains training windows only, and validation / test windows only ever query it;
- during training, a query masks every memory entry that overlaps its own input or horizon, so it can retrieve neither itself nor a window that reveals its future.

These properties are covered by the tests in [`tests/`](tests).

## Repository structure

```
.
├── run.py                      # train + evaluate RISE on one dataset / horizon / seed
├── models/RISE.py              # the model: three pathways, RAC, fusion
├── layers/
│   ├── Decomposition.py        # causal moving average (trend / detrended split)
│   ├── Retrieval.py            # multi-granularity retrieval with self-exclusion masking
│   └── ...                     # patch embedding and Transformer encoder blocks
├── exp/                        # training / evaluation loop
├── data_provider/              # CSV dataset: chronological 7:1:2 split, train-fitted scaler
├── utils/                      # metrics, learning-rate schedule
└── tests/                      # leakage and smoke tests
```

## Installation

```bash
git clone https://github.com/dancing-with-coffee/RISE.git
cd RISE
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or newer and PyTorch 2.0 or newer are required. A GPU is used when available but is not required: one run takes on the order of a minute on a CPU (`--use_gpu 0`).

## Data format

No datasets are distributed with this repository. RISE reads one CSV file per dataset, with one row per time step, a `date` column, and one column per variable:

```
date,Open,High,Low,Volume,Close
2015-01-02,...
```

- The `date` column is required but not used by the model.
- `--features MS`: all columns are inputs, and the `--target` column (default `Close`) is the one that is forecast and evaluated.
- `--features M`: every column is forecast and evaluated. `--features S`: the target column only.
- `--target` must name an existing column in every mode.
- Each file is split chronologically 7:1:2 into training, validation and test parts, and all variables are z-scored with the statistics of the training part.

## Usage

Train and evaluate on one dataset, horizon and seed:

```bash
python run.py --root_path ./data/ --data_path sp500.csv \
  --features MS --target Close --seq_len 96 --pred_len 24 --seed 0
```

Each run writes

- `results/<run name>/metrics.json` — MSE, MAE and RMSE on the test split,
- `results/<run name>/pred.npy`, `true.npy` — test predictions and ground truth in z-scored units, with shape (windows, H, variables); with `--features M` the variables are in file order with the target column moved last,
- `results/summary.csv` — one row per run,
- `checkpoints/<run name>/checkpoint.pth` — the trained model.

To evaluate a saved checkpoint again, repeat the command with `--is_training 0`.

The run name is built from the data file name and the main settings. Use `--model_id <tag>` to keep apart runs that differ in other ways, for example the same file name in two data directories.

### Main options

| Option | Default | Meaning |
|--------|---------|---------|
| `--features` | `MS` | `M`: all variables are forecast; `MS`: all variables are inputs, `--target` is forecast; `S`: the target only |
| `--seq_len` / `--pred_len` | 96 / 24 | look-back window L / horizon H (both must be divisible by 4 with the default `--n_period 3`) |
| `--topm` | 20 | number of retrieved neighbours m |
| `--trend_kernel` | 25 | moving-average kernel k (odd; an even value is increased by one) |
| `--n_period` | 3 | number of retrieval granularities (1–5) |
| `--composition_mode` | `time_distributed` | fusion of the pathways: `time_distributed` (one learnable weight per pathway), `additive` (plain sum), `concat` (one linear layer) |
| `--patch_input_mode` | `residual` | patch predictor reads the detrended component (`residual`) or the shifted input (`raw`) |
| `--use_retrieval_context` | 1 | RAC tokens on / off |
| `--ablate_patch`, `--ablate_trend`, `--ablate_retrieval` | off | remove a pathway |
| `--centered_ma` | off | centered instead of causal moving average; it looks ahead inside the window and exists only to measure that effect |
| `--trend_cross_channel` | off | trend predictor maps all channels jointly to the target instead of one channel at a time |
| `--aux_loss_weight` | 0.0 | weight of an auxiliary decomposition loss (additive fusion only); disabled in all reported results |
| `--seed` | 0 | random seed |
| `--use_gpu` / `--gpu` | 1 / 0 | use CUDA when available / device index |

`python run.py --help` lists all options.

## Benchmark settings of the paper

This section gives the settings of the two benchmark evaluations of the paper.

### Training and evaluation protocol

- Chronological 7:1:2 split; z-scoring with training statistics. MSE and MAE are computed in that z-scored space (on the target variable for `MS`, on all variables for `M`).
- Adam, learning rate 1e-4, batch size 32, cosine annealing with warm restarts (period 5 epochs, minimum rate 1e-6), 10 epochs. The training loss is the MSE on the evaluated variables.
- Training runs for the fixed number of epochs. There is no early stopping and no checkpoint selection: the model at the end of training is the one that is evaluated and saved. The validation and test losses printed after each epoch are for monitoring only.
- Patch encoder: hidden size 128, 4 attention heads, 2 layers, dropout 0.1, patch length 16 with stride 8. Moving-average kernel 25; three retrieval granularities; softmax temperature 0.1; RAC on.
- Every configuration of the two benchmarks is run with seeds 0–4; the reported values are the means over the five seeds.
- Runs are seeded (`--seed`) and use deterministic cuDNN settings. The reported numbers were obtained on a CUDA GPU; on other hardware or with other library versions, individual runs differ slightly.

These are the defaults of `run.py`; the two benchmarks differ only in the options listed below.

### Financial datasets

Eight assets, daily bars from Yahoo Finance, 2015-01-01 to 2025-03-20:

| Dataset | Yahoo ticker | Rows | First day – last day |
|---------|--------------|------|----------------------|
| S&P 500 | `^GSPC` | 2,569 | 2015-01-02 – 2025-03-20 |
| NASDAQ Composite | `^IXIC` | 2,569 | 2015-01-02 – 2025-03-20 |
| Nikkei 225 | `^N225` | 2,495 | 2015-01-05 – 2025-03-19 |
| Hang Seng | `^HSI` | 2,513 | 2015-01-02 – 2025-03-20 |
| DAX | `^GDAXI` | 2,593 | 2015-01-02 – 2025-03-20 |
| Gold futures | `GC=F` | 2,567 | 2015-01-02 – 2025-03-20 |
| EUR/USD | `EURUSD=X` | 2,662 | 2015-01-01 – 2025-03-20 |
| Bitcoin | `BTC-USD` | 3,732 | 2015-01-01 – 2025-03-20 |

- Each file has the columns `date, Open, High, Low, Volume, Close` (five input channels) and follows the trading calendar of its own market.
- The bars were obtained with the `yfinance` package (`auto_adjust=True`, start 2015-01-01, end 2025-03-21 exclusive). Missing values were forward-filled and then backward-filled.
- Yahoo Finance reports no volume for the EUR/USD pair, so its `Volume` column is zero throughout.
- Yahoo Finance occasionally revises its history, so data downloaded later can differ slightly from the snapshot used in the paper.

Setting: `--features MS --target Close`, look-back 96, horizons 8, 16, 24, 36, 48 and 60, top-m 20, `--composition_mode time_distributed`:

```bash
for horizon in 8 16 24 36 48 60; do
  for seed in 0 1 2 3 4; do
    python run.py --root_path ./data/ --data_path sp500.csv \
      --features MS --target Close --seq_len 96 --pred_len $horizon --topm 20 \
      --composition_mode time_distributed --seed $seed
  done
done
```

### TFB benchmark

The NASDAQ and NYSE datasets of the [TFB benchmark](https://github.com/decisionintelligence/TFB), five variables each (`Open`, `High`, `Low`, `Close`, `Volume`), all of which are forecast.

Setting: `--features M`, horizons 24, 36, 48 and 60, `--composition_mode additive`, with the following look-back L and top-m per dataset and horizon:

| Dataset | H = 24 | H = 36 | H = 48 | H = 60 |
|---------|--------|--------|--------|--------|
| NASDAQ | L = 104, m = 20 | L = 104, m = 20 | L = 104, m = 20 | L = 104, m = 20 |
| NYSE | L = 36, m = 10 | L = 36, m = 20 | L = 36, m = 10 | L = 36, m = 5 |

```bash
python run.py --root_path ./data/ --data_path NASDAQ.csv \
  --features M --seq_len 104 --pred_len 24 --topm 20 \
  --composition_mode additive --seed 0
```

TFB distributes each dataset as a long table (`date,data,cols`) with one block of rows per variable, which has to be converted to the format above.

- TFB's own loader places the blocks side by side by row position (1,244 rows for NASDAQ, 1,243 for NYSE).
- For the results reported in the paper, the blocks were instead joined on the `date` field and incomplete rows were dropped (1,240 and 1,239 rows; the `date` column of the converted files holds placeholder business days). In these two files the `date` field is an integer counter that starts one position earlier in each successive block, so the join offsets the variables from one another: row t of the joined table holds `Open[t]`, `High[t+1]`, `Low[t+2]`, `Close[t+3]` and `Volume[t+4]` of the positional table.

The reported TFB numbers are reproduced only with the joined table; results on the positional table are not directly comparable with them.

## Tests

```bash
python tests/test_causal_ma.py    # the causal moving average never uses future values
python tests/test_retrieval.py    # self-exclusion mask; no dependence on later splits
python tests/test_smoke.py        # end-to-end run on synthetic data (CPU, a few seconds)
```

The files are also collected by `pytest`.

## Citation

```bibtex
@inproceedings{kim2026rise,
  author    = {Kim, Yongdam and Park, Hyeonji and Lee, Dongwoo and Cho, Jeongseon and Lee, Yein and Yang, Jihoon and Jung, Sungwon},
  title     = {Retrieve the Right Signal: Decomposition-Guided Retrieval-Augmented Forecasting for Financial Markets},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '26},
  year      = {2026},
  pages     = {2268--2277},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Rome, Italy},
  doi       = {10.1145/3799682.3841004},
  url       = {https://doi.org/10.1145/3799682.3841004}
}
```

## Contact

Yongdam Kim (yongdamkim@sogang.ac.kr)

## License and acknowledgments

This code is released under the [MIT License](LICENSE).

The retrieval module builds on [RAFT](https://github.com/archon159/RAFT), and the code structure, data pipeline and Transformer layers follow [Time-Series-Library](https://github.com/thuml/Time-Series-Library); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
