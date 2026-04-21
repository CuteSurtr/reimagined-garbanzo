from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from .logmath import NEG_INF

@dataclass
class GHMMState:
    name: str
    log_duration: Callable[[int], float]
    log_segment_emission: Callable[[np.ndarray, int, int], float]
    max_duration: int
    min_duration: int = 1

@dataclass
class GHMM:
    states: List[GHMMState]
    trans: np.ndarray
    pi: np.ndarray

    @property
    def N(self) -> int:
        return len(self.states)

    def viterbi(self, obs: np.ndarray) -> Tuple[float, List[Tuple[int, int, int]]]:
        obs = np.asarray(obs)
        T = len(obs)
        N = self.N
        log_trans = np.log(self.trans + 1e-300)
        log_pi = np.log(self.pi + 1e-300)
        V = np.full((T + 1, N), NEG_INF)
        BP = np.full((T + 1, N, 3), -1, dtype=int)
        V[0, :] = NEG_INF
        for j in range(N):
            st = self.states[j]
            for d in range(st.min_duration, min(st.max_duration, T) + 1):
                emit = st.log_segment_emission(obs, 0, d)
                cand = log_pi[j] + st.log_duration(d) + emit
                if cand > V[d, j]:
                    V[d, j] = cand
                    BP[d, j] = [-1, 0, d]
        for t in range(1, T + 1):
            for j in range(N):
                st_j = self.states[j]
                for d in range(st_j.min_duration, min(st_j.max_duration, t) + 1):
                    start = t - d
                    if start == 0:
                        continue
                    emit = st_j.log_segment_emission(obs, start, t)
                    dur = st_j.log_duration(d)
                    prev_scores = V[start, :] + log_trans[:, j]
                    best_i = int(np.argmax(prev_scores))
                    cand = prev_scores[best_i] + dur + emit
                    if cand > V[t, j]:
                        V[t, j] = cand
                        BP[t, j] = [best_i, start, d]
        end_state = int(np.argmax(V[T, :]))
        best = float(V[T, end_state])
        segs: List[Tuple[int, int, int]] = []
        t = T
        s = end_state
        while t > 0:
            prev_i, start, d = BP[t, s]
            segs.append((s, start, t))
            if prev_i == -1:
                break
            s = int(prev_i)
            t = int(start)
        segs.reverse()
        return (best, segs)

def geometric_duration(p: float) -> Callable[[int], float]:
    log_fail = np.log(max(1 - p, 1e-300))
    log_succ = np.log(max(p, 1e-300))

    def f(d: int) -> float:
        if d < 1:
            return NEG_INF
        return (d - 1) * log_fail + log_succ
    return f

def empirical_duration(histogram: np.ndarray) -> Callable[[int], float]:
    h = np.asarray(histogram, dtype=float) + 1e-09
    h /= h.sum()
    log_h = np.log(h)

    def f(d: int) -> float:
        if d < 1 or d >= len(log_h):
            return NEG_INF
        return float(log_h[d])
    return f