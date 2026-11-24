"""
Tests for the retrieval pathway.

1. Self-exclusion: a training query can retrieve neither itself nor any window that
   overlaps its own input or horizon.
2. No look-ahead across splits: everything computed for the training and validation
   splits is independent of later data, i.e. the memory bank and the scaler are built
   from the training split only.
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_provider.data_loader import Dataset_Custom
from layers.Retrieval import RetrievalTool
from models import RISE


def test_self_exclusion_mask():
    torch.manual_seed(0)
    S, P, C = 16, 8, 2
    series = torch.randn(400, C).cumsum(dim=0)
    n = len(series) - S - P + 1
    keys = torch.stack([series[i:i + S] for i in range(n)])
    values = torch.stack([series[i + S:i + S + P] for i in range(n)])

    rt = RetrievalTool(seq_len=S, pred_len=P, channels=C, n_period=3, topm=1)
    rt.build_database(keys, values)
    stored = rt.y_data_all_mg[-1]  # values at the finest granularity, (T, P, C)

    index = torch.arange(100, 164)  # queries that are themselves training windows
    query = keys[index]

    # Without the mask, the nearest neighbour of a training window is the window itself.
    out = rt.retrieve(query, index, train=False)[-1]
    assert torch.equal(out, stored[index])

    # With the mask, the nearest neighbour is at least S + P steps away, so it shares
    # no time step with the query's input or horizon.
    out = rt.retrieve(query, index, train=True)[-1]
    for b, i in enumerate(index.tolist()):
        match = (stored == out[b]).flatten(start_dim=1).all(dim=1).nonzero().flatten()
        assert len(match) == 1, 'the retrieved horizon should identify exactly one memory entry'
        assert abs(match.item() - i) >= S + P, f'query {i} retrieved the overlapping window {match.item()}'

    print('[PASS] Training queries never retrieve themselves or overlapping windows')


def test_mask_radius_is_exact():
    """An exact copy of the query is masked at distance S + P - 1 and retrievable at S + P."""
    torch.manual_seed(1)
    S, P, C = 16, 8, 2
    keys = torch.randn(200, S, C)
    values = torch.randn(200, P, C)
    i = 100
    for sign in (1, -1):
        for distance, masked in ((S + P - 1, True), (S + P, False)):
            j = i + sign * distance
            planted = keys.clone()
            planted[j] = planted[i]  # the best possible match for query i sits at index j
            rt = RetrievalTool(seq_len=S, pred_len=P, channels=C, n_period=3, topm=1)
            rt.build_database(planted, values)
            out = rt.retrieve(planted[i:i + 1], torch.tensor([i]), train=True)[-1, 0]
            retrieved_copy = torch.equal(out, rt.y_data_all_mg[-1][j])
            assert retrieved_copy != masked, f'offset {sign * distance}: expected masked={masked}'

    print('[PASS] The mask covers offsets up to S + P - 1 and nothing beyond')


def _write_csv(path, values):
    df = pd.DataFrame(values, columns=['Open', 'High', 'Low', 'Volume', 'Close'])
    df.insert(0, 'date', pd.date_range('2020-01-01', periods=len(df), freq='D').strftime('%Y-%m-%d'))
    df.to_csv(path, index=False)


def _retrieval_results(root, name, S, P):
    cfg = SimpleNamespace(seq_len=S, pred_len=P, enc_in=5, topm=5, d_model=16, n_heads=2,
                          use_gpu=False, num_workers=0)
    model = RISE.Model(cfg)
    splits = [Dataset_Custom(root, name, flag=flag, size=[S, P], features='MS', target='Close')
              for flag in ('train', 'val', 'test')]
    model.prepare_dataset(*splits)
    return model.retrieval_dict


def test_no_dependence_on_later_splits():
    rng = np.random.default_rng(0)
    n, S, P = 600, 16, 8
    n_train, n_test = int(n * 0.7), int(n * 0.2)

    base = rng.standard_normal((n, 5)).cumsum(axis=0)
    new_test = base.copy()
    new_test[n - n_test:] += 5 * rng.standard_normal((n_test, 5))
    new_val_test = base.copy()
    new_val_test[n_train:] += 5 * rng.standard_normal((n - n_train, 5))

    with tempfile.TemporaryDirectory() as root:
        for name, values in (('base.csv', base), ('new_test.csv', new_test), ('new_val_test.csv', new_val_test)):
            _write_csv(os.path.join(root, name), values)
        r_base = _retrieval_results(root, 'base.csv', S, P)
        r_test = _retrieval_results(root, 'new_test.csv', S, P)
        r_val_test = _retrieval_results(root, 'new_val_test.csv', S, P)

    # Changing the test period changes the test queries and nothing else.
    assert torch.equal(r_base['train'], r_test['train'])
    assert torch.equal(r_base['valid'], r_test['valid'])
    assert not torch.equal(r_base['test'], r_test['test'])

    # Changing the validation and test periods leaves the training retrieval untouched.
    assert torch.equal(r_base['train'], r_val_test['train'])
    assert not torch.equal(r_base['valid'], r_val_test['valid'])

    print('[PASS] Train / validation retrieval is independent of later splits')


if __name__ == '__main__':
    test_self_exclusion_mask()
    test_mask_radius_is_exact()
    test_no_dependence_on_later_splits()
    print('\n=== ALL TESTS PASSED ===')
