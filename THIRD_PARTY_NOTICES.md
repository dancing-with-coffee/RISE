# Third-party notices

This repository includes source code derived from the open-source projects listed below. The license of each project, including its copyright notice, is reproduced in full. Files that contain derived code carry a header comment naming the project they were adapted from.

## RAFT

- Project: RAFT — Retrieval Augmented Time Series Forecasting
- URL: https://github.com/archon159/RAFT
- License: MIT License

Files in this repository that contain code derived from RAFT:

- `layers/Retrieval.py` (retrieval module)
- `models/RISE.py` (model skeleton and handling of the retrieval results, based on RAFT's `models/RAFT.py`)
- `data_provider/data_loader.py`, `data_provider/data_factory.py`
- `exp/exp_long_term_forecasting.py`
- `run.py`

```text
MIT License

Copyright (c) 2024 Seungeon Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Time-Series-Library

- Project: Time Series Library (TSLib)
- URL: https://github.com/thuml/Time-Series-Library
- License: MIT License

Files in this repository that contain code derived from Time-Series-Library, either directly or through RAFT, which is itself built on Time-Series-Library:

- `layers/Embed.py`, `layers/SelfAttention_Family.py`, `layers/Transformer_EncDec.py`, `utils/masking.py` (patch embedding and Transformer encoder blocks)
- `models/RISE.py` (patch-based encoder, based on the library's `models/PatchTST.py`)
- `layers/Decomposition.py` (moving-average computation, based on the `moving_avg` block of the library's `layers/Autoformer_EncDec.py`)
- `utils/metrics.py`, `utils/tools.py`
- `data_provider/data_loader.py`, `data_provider/data_factory.py`
- `exp/exp_long_term_forecasting.py`
- `run.py`

```text
MIT License

Copyright (c) 2021 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
