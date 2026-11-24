# The moving-average computation follows the moving_avg block of Time-Series-Library
# (https://github.com/thuml/Time-Series-Library), MIT License.
import torch
import torch.nn.functional as F


def extract_trend(x, kernel_size=25, causal=True):
    """
    Extract the trend component with a moving average (non-learnable).

    Args:
        x: (B, T, C) - input time series
        kernel_size: int - moving-average window (an even value is increased by one)
        causal: bool - if True, use a causal (backward-only) moving average, which never
                       looks ahead. If False, use a centered moving average, which uses
                       future values inside the window (for analysis only).

    Returns:
        trend: same shape as x
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    B, T, C = x.shape

    if causal:
        # Causal MA: T(t) = (1/k) * sum_{i=0}^{k-1} x(t-i)  (past only)
        # Pad only the front with repeated edge values
        front = x[:, :1, :].repeat(1, kernel_size - 1, 1)
        x_pad = torch.cat([front, x], dim=1)  # (B, T + k-1, C)
    else:
        # Centered MA (has look-ahead bias)
        pad = (kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)  # (B, T + 2*pad, C)

    # Apply moving average via avg_pool1d
    x_pad = x_pad.permute(0, 2, 1)  # (B, C, T + pad_total)
    trend = F.avg_pool1d(x_pad, kernel_size=kernel_size, stride=1, padding=0)
    trend = trend.permute(0, 2, 1)  # (B, T, C)

    return trend

