from __future__ import annotations
import numpy as np
NEG_INF = -np.inf

def log0(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, NEG_INF)
    pos = x > 0
    out[pos] = np.log(x[pos])
    return out

def logsumexp(x, axis=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max_safe = np.where(np.isfinite(x_max), x_max, 0.0)
    s = np.sum(np.exp(x - x_max_safe), axis=axis, keepdims=True)
    log_s = np.log(s) + x_max_safe
    all_neg_inf = ~np.any(np.isfinite(x), axis=axis, keepdims=True)
    log_s = np.where(all_neg_inf, NEG_INF, log_s)
    if axis is None:
        return log_s.item()
    return np.squeeze(log_s, axis=axis)

def log_normalize(log_vec: np.ndarray, axis: int=-1) -> np.ndarray:
    z = logsumexp(log_vec, axis=axis)
    return log_vec - np.expand_dims(z, axis=axis)