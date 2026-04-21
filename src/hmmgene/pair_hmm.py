from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .logmath import NEG_INF, logsumexp
M_STATE, X_STATE, Y_STATE = (0, 1, 2)

@dataclass
class PairHMM:
    log_p_xy: np.ndarray
    log_q_x: np.ndarray
    log_q_y: np.ndarray
    delta: float = 0.1
    epsilon: float = 0.4
    tau: float = 0.001

    @property
    def log_trans(self) -> np.ndarray:
        T = np.zeros((3, 3))
        T[M_STATE, M_STATE] = 1 - 2 * self.delta - self.tau
        T[M_STATE, X_STATE] = self.delta
        T[M_STATE, Y_STATE] = self.delta
        T[X_STATE, M_STATE] = 1 - self.epsilon - self.tau
        T[X_STATE, X_STATE] = self.epsilon
        T[Y_STATE, M_STATE] = 1 - self.epsilon - self.tau
        T[Y_STATE, Y_STATE] = self.epsilon
        return np.where(T > 0, np.log(T), NEG_INF)

    def viterbi(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        x = np.asarray(x, dtype=int)
        y = np.asarray(y, dtype=int)
        n, m = (len(x), len(y))
        T = self.log_trans
        V = np.full((3, n + 1, m + 1), NEG_INF)
        BP = np.full((3, n + 1, m + 1), -1, dtype=int)
        V[M_STATE, 0, 0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    emit_m = self.log_p_xy[x[i - 1], y[j - 1]]
                    cands = V[:, i - 1, j - 1] + T[:, M_STATE]
                    best = np.argmax(cands)
                    V[M_STATE, i, j] = cands[best] + emit_m
                    BP[M_STATE, i, j] = best
                if i > 0:
                    emit_x = self.log_q_x[x[i - 1]]
                    cands = V[:, i - 1, j] + T[:, X_STATE]
                    best = np.argmax(cands)
                    V[X_STATE, i, j] = cands[best] + emit_x
                    BP[X_STATE, i, j] = best
                if j > 0:
                    emit_y = self.log_q_y[y[j - 1]]
                    cands = V[:, i, j - 1] + T[:, Y_STATE]
                    best = np.argmax(cands)
                    V[Y_STATE, i, j] = cands[best] + emit_y
                    BP[Y_STATE, i, j] = best
        final_scores = V[:, n, m] + np.log(self.tau)
        end_state = int(np.argmax(final_scores))
        best = float(final_scores[end_state])
        alignment: List[Tuple[int, int]] = []
        i, j, s = (n, m, end_state)
        while i > 0 or j > 0:
            if s == M_STATE:
                alignment.append((int(x[i - 1]), int(y[j - 1])))
                prev = BP[s, i, j]
                i -= 1
                j -= 1
                s = prev
            elif s == X_STATE:
                alignment.append((int(x[i - 1]), -1))
                prev = BP[s, i, j]
                i -= 1
                s = prev
            else:
                alignment.append((-1, int(y[j - 1])))
                prev = BP[s, i, j]
                j -= 1
                s = prev
        alignment.reverse()
        return (best, alignment)

    def forward(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        x = np.asarray(x, dtype=int)
        y = np.asarray(y, dtype=int)
        n, m = (len(x), len(y))
        T = self.log_trans
        F = np.full((3, n + 1, m + 1), NEG_INF)
        F[M_STATE, 0, 0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    emit_m = self.log_p_xy[x[i - 1], y[j - 1]]
                    F[M_STATE, i, j] = logsumexp(F[:, i - 1, j - 1] + T[:, M_STATE]) + emit_m
                if i > 0:
                    emit_x = self.log_q_x[x[i - 1]]
                    F[X_STATE, i, j] = logsumexp(F[:, i - 1, j] + T[:, X_STATE]) + emit_x
                if j > 0:
                    emit_y = self.log_q_y[y[j - 1]]
                    F[Y_STATE, i, j] = logsumexp(F[:, i, j - 1] + T[:, Y_STATE]) + emit_y
        log_p = logsumexp(F[:, n, m] + np.log(self.tau))
        return (F, float(log_p))

    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=int)
        y = np.asarray(y, dtype=int)
        n, m = (len(x), len(y))
        T = self.log_trans
        B = np.full((3, n + 1, m + 1), NEG_INF)
        log_tau = np.log(self.tau)
        for s in range(3):
            B[s, n, m] = log_tau
        for i in range(n, -1, -1):
            for j in range(m, -1, -1):
                if i == n and j == m:
                    continue
                candidates = []
                for s in range(3):
                    ext_m = T[s, M_STATE] + self.log_p_xy[x[i], y[j]] + B[M_STATE, i + 1, j + 1] if i < n and j < m else NEG_INF
                    ext_x = T[s, X_STATE] + self.log_q_x[x[i]] + B[X_STATE, i + 1, j] if i < n else NEG_INF
                    ext_y = T[s, Y_STATE] + self.log_q_y[y[j]] + B[Y_STATE, i, j + 1] if j < m else NEG_INF
                    B[s, i, j] = logsumexp(np.array([ext_m, ext_x, ext_y]))
        return B

    def posterior_match(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        F, logp = self.forward(x, y)
        B = self.backward(x, y)
        n, m = (len(x), len(y))
        post = np.zeros((n, m))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                post[i - 1, j - 1] = float(np.exp(F[M_STATE, i, j] + B[M_STATE, i, j] - logp))
        return post

def simple_pair_hmm(match_prob: float=0.85, background: Optional[np.ndarray]=None) -> PairHMM:
    q = np.ones(4) / 4 if background is None else np.asarray(background)
    p_xy = np.full((4, 4), (1 - match_prob) / 12)
    np.fill_diagonal(p_xy, match_prob / 4)
    with np.errstate(divide='ignore'):
        log_p = np.log(p_xy)
        log_qx = np.log(q)
        log_qy = np.log(q)
    return PairHMM(log_p_xy=log_p, log_q_x=log_qx, log_q_y=log_qy)