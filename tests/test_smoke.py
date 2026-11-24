"""
End-to-end smoke test on synthetic data (CPU, well under a minute).

Trains RISE for one epoch in every fusion mode and in a few other configurations,
checks the saved outputs, and checks that evaluating the saved checkpoint reproduces
the metrics reported at the end of training.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run

N_ROWS, SEQ_LEN, PRED_LEN = 500, 32, 8
N_TEST = int(N_ROWS * 0.2) - PRED_LEN + 1


def _make_csv(path):
    rng = np.random.default_rng(0)
    close = 100 + rng.standard_normal(N_ROWS).cumsum()
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=N_ROWS, freq='D').strftime('%Y-%m-%d'),
        'Open': close + 0.1 * rng.standard_normal(N_ROWS),
        'High': close + 0.5,
        'Low': close - 0.5,
        'Volume': rng.integers(1000, 2000, N_ROWS),
        'Close': close,
    })
    df.to_csv(path, index=False)


def _run(tmp, *extra):
    argv = ['--root_path', tmp, '--data_path', 'toy.csv',
            '--seq_len', str(SEQ_LEN), '--pred_len', str(PRED_LEN), '--topm', '5',
            '--d_model', '16', '--n_heads', '2', '--train_epochs', '1',
            '--num_workers', '0', '--use_gpu', '0',
            '--checkpoints', os.path.join(tmp, 'checkpoints'),
            '--results', os.path.join(tmp, 'results'), *extra]
    record = run.main(argv)
    assert np.isfinite(record['mse']) and np.isfinite(record['mae'])
    pred = np.load(os.path.join(tmp, 'results', record['setting'], 'pred.npy'))
    true = np.load(os.path.join(tmp, 'results', record['setting'], 'true.npy'))
    assert pred.shape == true.shape
    return record, pred


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        _make_csv(os.path.join(tmp, 'toy.csv'))

        # features='MS': five input channels, the target channel is evaluated
        for mode in ('time_distributed', 'additive', 'concat'):
            _, pred = _run(tmp, '--composition_mode', mode)
            assert pred.shape == (N_TEST, PRED_LEN, 1)

        # features='M': every channel is evaluated
        _, pred = _run(tmp, '--features', 'M', '--composition_mode', 'additive')
        assert pred.shape == (N_TEST, PRED_LEN, 5)

        # features='S': the target is the only channel
        _, pred = _run(tmp, '--features', 'S')
        assert pred.shape == (N_TEST, PRED_LEN, 1)

        # pathway ablations
        for flags in (['--ablate_patch'], ['--ablate_trend'], ['--ablate_retrieval'],
                      ['--ablate_trend', '--ablate_retrieval']):
            _run(tmp, *flags)

        # the saved checkpoint is the model that was evaluated at the end of training
        trained, pred_trained = _run(tmp, '--model_id', 'ckpt')
        reloaded, pred_reloaded = _run(tmp, '--model_id', 'ckpt', '--is_training', '0')
        assert trained['mse'] == reloaded['mse'] and trained['mae'] == reloaded['mae']
        assert np.array_equal(pred_trained, pred_reloaded)

    print('[PASS] End-to-end training, evaluation and checkpoint reload')


if __name__ == '__main__':
    test_end_to_end()
    print('\n=== ALL TESTS PASSED ===')
