from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np
from .logmath import NEG_INF, log0, logsumexp

@dataclass
class DiscreteHMM:
    A: np.ndarray
    B: np.ndarray
    pi: np.ndarray
    state_names: Optional[List[str]] = None
    symbol_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.pi = np.asarray(self.pi, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f'A must be square; got {self.A.shape}')
        N = self.A.shape[0]
        if self.B.shape[0] != N:
            raise ValueError(f'B rows ({self.B.shape[0]}) must equal N ({N})')
        if self.pi.shape != (N,):
            raise ValueError(f'pi must have shape ({N},); got {self.pi.shape}')
        _check_row_stochastic(self.A, 'A')
        _check_row_stochastic(self.B, 'B')
        _check_probability_vector(self.pi, 'pi')

    @property
    def logA(self) -> np.ndarray:
        return log0(self.A)

    @property
    def logB(self) -> np.ndarray:
        return log0(self.B)

    @property
    def logpi(self) -> np.ndarray:
        return log0(self.pi)

    @property
    def N(self) -> int:
        return self.A.shape[0]

    @property
    def M(self) -> int:
        return self.B.shape[1]

    def sample(self, length: int, rng: Optional[np.random.Generator]=None):
        if rng is None:
            rng = np.random.default_rng()
        states = np.empty(length, dtype=int)
        obs = np.empty(length, dtype=int)
        states[0] = rng.choice(self.N, p=self.pi)
        obs[0] = rng.choice(self.M, p=self.B[states[0]])
        for t in range(1, length):
            states[t] = rng.choice(self.N, p=self.A[states[t - 1]])
            obs[t] = rng.choice(self.M, p=self.B[states[t]])
        return (states, obs)

    def forward(self, obs: Sequence[int]) -> Tuple[np.ndarray, float]:
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        logA, logB, logpi = (self.logA, self.logB, self.logpi)
        log_alpha = np.full((T, self.N), NEG_INF)
        log_alpha[0] = logpi + logB[:, obs[0]]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + logA, axis=0) + logB[:, obs[t]]
        log_p = logsumexp(log_alpha[-1])
        return (log_alpha, float(log_p))

    def backward(self, obs: Sequence[int]) -> np.ndarray:
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        logA, logB = (self.logA, self.logB)
        log_beta = np.full((T, self.N), NEG_INF)
        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(logA + (logB[:, obs[t + 1]] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def viterbi(self, obs: Sequence[int]) -> Tuple[float, np.ndarray]:
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        logA, logB, logpi = (self.logA, self.logB, self.logpi)
        delta = np.full((T, self.N), NEG_INF)
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi + logB[:, obs[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.N)] + logB[:, obs[t]]
        log_p_star = float(np.max(delta[-1]))
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return (log_p_star, path)

    def posteriors(self, obs: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, float]:
        log_alpha, log_p = self.forward(obs)
        log_beta = self.backward(obs)
        log_gamma = log_alpha + log_beta - log_p
        gamma = np.exp(log_gamma)
        obs = np.asarray(obs, dtype=int)
        logA, logB = (self.logA, self.logB)
        T = len(obs)
        log_xi = log_alpha[:-1, :, None] + logA[None, :, :] + (log_beta[1:] + logB[:, obs[1:]].T)[:, None, :] - log_p
        xi = np.exp(log_xi)
        return (gamma, xi, log_p)

    def baum_welch(self, sequences: Sequence[Sequence[int]], n_iters: int=50, tol: float=0.0001, pseudo: float=1e-06, verbose: bool=False) -> List[float]:
        seqs = [np.asarray(s, dtype=int) for s in sequences]
        logliks: List[float] = []
        for it in range(n_iters):
            num_pi = np.zeros(self.N)
            num_A = np.full((self.N, self.N), pseudo)
            den_A = np.full(self.N, pseudo * self.N)
            num_B = np.full((self.N, self.M), pseudo)
            den_B = np.full(self.N, pseudo * self.M)
            total_ll = 0.0
            for obs in seqs:
                gamma, xi, log_p = self.posteriors(obs)
                total_ll += log_p
                num_pi += gamma[0]
                num_A += xi.sum(axis=0)
                den_A += gamma[:-1].sum(axis=0)
                for k in range(self.M):
                    num_B[:, k] += gamma[obs == k].sum(axis=0)
                den_B += gamma.sum(axis=0)
            self.pi = num_pi / num_pi.sum()
            self.A = num_A / den_A[:, None]
            self.B = num_B / den_B[:, None]
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.B /= self.B.sum(axis=1, keepdims=True)
            self.pi /= self.pi.sum()
            logliks.append(total_ll)
            if verbose:
                print(f'iter {it:3d}  log-lik = {total_ll:.4f}')
            if len(logliks) > 1 and abs(logliks[-1] - logliks[-2]) < tol:
                break
        return logliks

def _check_row_stochastic(M: np.ndarray, name: str, tol: float=1e-08) -> None:
    sums = M.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=tol):
        raise ValueError(f'{name} rows must sum to 1; got {sums}')
    if (M < -tol).any():
        raise ValueError(f'{name} must be non-negative')

def _check_probability_vector(v: np.ndarray, name: str, tol: float=1e-08) -> None:
    if not np.isclose(v.sum(), 1.0, atol=tol):
        raise ValueError(f'{name} must sum to 1; got {v.sum()}')
    if (v < -tol).any():
        raise ValueError(f'{name} must be non-negative')