"""Unemployment-rate history and forward scenarios (synthetic, shaped like the 2021-2026 US path).

Two scenario sets are kept: the one used at the June 30, 2026 quarter end, and the updated one used
at September 30, 2026, after the outlook worsened. The reserve roll-forward uses both.
"""
import numpy as np
import pandas as pd

START, AS_OF, PRIOR = "2021-01", "2026-09", "2026-06"
LONG_RUN = 4.6
REVERSION = 12            # months to revert to the long-run rate after the forecast window
HORIZON = 72


def history():
    months = pd.period_range(START, AS_OF, freq="M")
    knots = {"2021-01": 6.4, "2021-07": 5.4, "2022-01": 4.0, "2022-07": 3.6, "2023-04": 3.5, "2023-12": 3.7,
             "2024-07": 4.2, "2025-01": 4.1, "2025-09": 4.3, "2026-03": 4.3, "2026-06": 4.4, "2026-09": 4.6}
    k = pd.Series(knots)
    k.index = pd.PeriodIndex(k.index, freq="M")
    x = np.array([(p - months[0]).n for p in k.index])
    xi = np.arange(len(months))
    rng = np.random.default_rng(4)
    u = np.interp(xi, x, k.values) + rng.normal(0, 0.04, len(months))
    return pd.Series(np.round(u, 2), index=months, name="unemployment")


def _path(start, peak, peak_month, horizon=HORIZON, forecast=24):
    """Rise or fall to `peak` by `peak_month`, hold to the end of the forecast window, then revert."""
    t = np.arange(1, horizon + 1)
    path = np.where(t <= peak_month, start + (peak - start) * t / peak_month, peak)
    rev = (t > forecast) & (t <= forecast + REVERSION)
    path = np.where(rev, peak + (LONG_RUN - peak) * (t - forecast) / REVERSION, path)
    path = np.where(t > forecast + REVERSION, LONG_RUN, path)
    return np.round(path, 3)


def scenarios(as_of=AS_OF):
    """Scenario paths (monthly unemployment for the next HORIZON months) and weights."""
    hist = history()
    start = hist[pd.Period(as_of, "M")]
    if as_of == PRIOR:
        spec = {"Upside": (3.9, 12), "Base": (4.5, 12), "Downside": (5.6, 12), "Severe": (8.0, 15)}
    else:
        spec = {"Upside": (4.1, 12), "Base": (4.9, 12), "Downside": (6.2, 12), "Severe": (8.8, 15)}
    weights = {"Upside": 0.15, "Base": 0.45, "Downside": 0.30, "Severe": 0.10}
    return {name: _path(start, peak, pm) for name, (peak, pm) in spec.items()}, weights
