"""
Unit tests for causal vs centered moving average in Decomposition.py.
Verifies that causal MA only uses past values (no data leakage).
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from layers.Decomposition import extract_trend


def test_causal_ma_no_future_leakage():
    """Verify that causal MA at time t only depends on x[t-k+1:t+1]."""
    torch.manual_seed(42)
    B, T, C = 1, 50, 1
    k = 25
    x = torch.randn(B, T, C)

    trend_causal = extract_trend(x, kernel_size=k, causal=True)

    # For t >= k-1, trend[t] should equal mean(x[t-k+1:t+1])
    for t in range(k - 1, T):
        expected = x[0, t - k + 1:t + 1, 0].mean()
        actual = trend_causal[0, t, 0]
        assert torch.allclose(actual, expected, atol=1e-5), \
            f"t={t}: expected {expected:.6f}, got {actual:.6f}"

    print("[PASS] Causal MA uses only past values for t >= k-1")


def test_causal_ma_edge_padding():
    """Verify edge behavior: for t < k-1, first value is repeated."""
    torch.manual_seed(42)
    B, T, C = 1, 50, 1
    k = 5
    x = torch.randn(B, T, C)

    trend_causal = extract_trend(x, kernel_size=k, causal=True)

    # t=0: window is [x0, x0, x0, x0, x0] = x0
    assert torch.allclose(trend_causal[0, 0, 0], x[0, 0, 0], atol=1e-5), \
        f"t=0 should equal x[0], got {trend_causal[0, 0, 0]:.6f} vs {x[0, 0, 0]:.6f}"

    # t=1: window is [x0, x0, x0, x0, x1] = (4*x0 + x1) / 5
    expected_t1 = (4 * x[0, 0, 0] + x[0, 1, 0]) / 5
    assert torch.allclose(trend_causal[0, 1, 0], expected_t1, atol=1e-5), \
        "t=1 edge padding incorrect"

    print("[PASS] Causal MA edge padding correct")


def test_causal_vs_centered_differ():
    """Causal and centered MA should give different results (except trivial cases)."""
    torch.manual_seed(42)
    B, T, C = 1, 100, 3
    x = torch.randn(B, T, C)

    trend_causal = extract_trend(x, kernel_size=25, causal=True)
    trend_centered = extract_trend(x, kernel_size=25, causal=False)

    # They should NOT be identical
    diff = (trend_causal - trend_centered).abs().mean()
    assert diff > 0.01, f"Causal and centered MA are too similar (diff={diff:.6f})"

    print(f"[PASS] Causal vs Centered differ: mean abs diff = {diff:.4f}")


def test_causal_ma_future_independence():
    """
    Critical test: changing future values should NOT affect causal MA at current time.
    This directly tests for data leakage.
    """
    torch.manual_seed(42)
    B, T, C = 1, 50, 1
    k = 25
    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    # Change future values (t=30 onwards)
    x2[0, 30:, :] = torch.randn(T - 30, C)

    trend1 = extract_trend(x1, kernel_size=k, causal=True)
    trend2 = extract_trend(x2, kernel_size=k, causal=True)

    # For t <= 29, trend should be identical
    diff = (trend1[0, :30, 0] - trend2[0, :30, 0]).abs().max()
    assert diff < 1e-6, f"Causal MA leaks future: max diff at t<30 = {diff:.8f}"

    print("[PASS] Causal MA is independent of future values")


def test_centered_ma_has_future_leakage():
    """Confirm that centered MA DOES depend on future values (expected behavior)."""
    torch.manual_seed(42)
    B, T, C = 1, 50, 1
    k = 25
    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    x2[0, 30:, :] = torch.randn(T - 30, C)

    trend1 = extract_trend(x1, kernel_size=k, causal=False)
    trend2 = extract_trend(x2, kernel_size=k, causal=False)

    # For centered MA, changing t>=30 should affect t<30 (within window)
    diff = (trend1[0, :30, 0] - trend2[0, :30, 0]).abs().max()
    assert diff > 0.01, f"Centered MA should leak future but diff={diff:.8f}"

    print(f"[PASS] Centered MA depends on future values (max diff = {diff:.4f})")


def test_output_shape():
    """Output shape should match input shape for both modes."""
    B, T, C = 4, 96, 7
    x = torch.randn(B, T, C)

    for causal in [True, False]:
        trend = extract_trend(x, kernel_size=25, causal=causal)
        assert trend.shape == (B, T, C), \
            f"Shape mismatch: causal={causal}, expected {(B, T, C)}, got {trend.shape}"

    print("[PASS] Output shapes correct for both modes")


if __name__ == '__main__':
    test_causal_ma_no_future_leakage()
    test_causal_ma_edge_padding()
    test_causal_vs_centered_differ()
    test_causal_ma_future_independence()
    test_centered_ma_has_future_leakage()
    test_output_shape()
    print("\n=== ALL TESTS PASSED ===")
