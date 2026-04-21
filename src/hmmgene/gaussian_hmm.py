from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
from .logmath import NEG_INF, logsumexp

@dataclass
class GaussianHMM:
    A: np.ndarray
    pi: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    state_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.pi = np.asarray(self.pi, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.covars = np.asarray(self.covars, dtype=float)
        N = self.A.shape[0]
        if self.pi.shape != (N,):
            raise ValueError(f'pi shape mismatch: expected ({N},)')
        if not np.allclose(self.A.sum(axis=1), 1.0, atol=1e-08):
            raise ValueError('A rows must sum to 1')
        if not np.isclose(self.pi.sum(), 1.0, atol=1e-08):
            raise ValueError('pi must sum to 1')
        if self.means.shape[0] != N:
            raise ValueError(f'means shape mismatch: got {self.means.shape}')
        if self.covars.shape[0] != N or self.covars.shape[1] != self.covars.shape[2]:
            raise ValueError(f'covars must be (N,d,d); got {self.covars.shape}')

    @property
    def N(self) -> int:
        return self.A.shape[0]

    @property
    def d(self) -> int:
        return self.means.shape[1]

    def log_emission(self, obs: np.ndarray) -> np.ndarray:
        T = obs.shape[0]
        L = np.zeros((T, self.N))
        for j in range(self.N):
            diff = obs - self.means[j]
            cov = self.covars[j]
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError(f'covariance {j} not positive definite')
            inv = np.linalg.inv(cov)
            quad = np.einsum('ti,ij,tj->t', diff, inv, diff)
            L[:, j] = -0.5 * (self.d * np.log(2 * np.pi) + logdet + quad)
        return L

    def sample(self, length: int, rng: np.random.Generator | None=None):
        if rng is None:
            rng = np.random.default_rng()
        states = np.empty(length, dtype=int)
        obs = np.empty((length, self.d))
        states[0] = rng.choice(self.N, p=self.pi)
        obs[0] = rng.multivariate_normal(self.means[states[0]], self.covars[states[0]])
        for t in range(1, length):
            states[t] = rng.choice(self.N, p=self.A[states[t - 1]])
            obs[t] = rng.multivariate_normal(self.means[states[t]], self.covars[states[t]])
        return (states, obs)

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        T = obs.shape[0]
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        L = self.log_emission(obs)
        log_alpha = np.full((T, self.N), NEG_INF)
        log_alpha[0] = logpi + L[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + logA, axis=0) + L[t]
        return (log_alpha, float(logsumexp(log_alpha[-1])))

    def backward(self, obs: np.ndarray) -> np.ndarray:
        T = obs.shape[0]
        logA = np.log(self.A + 1e-300)
        L = self.log_emission(obs)
        log_beta = np.full((T, self.N), NEG_INF)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(logA + (L[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def viterbi(self, obs: np.ndarray) -> Tuple[float, np.ndarray]:
        T = obs.shape[0]
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        L = self.log_emission(obs)
        delta = np.full((T, self.N), NEG_INF)
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi + L[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.N)] + L[t]
        log_p_star = float(np.max(delta[-1]))
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return (log_p_star, path)

    def baum_welch(self, sequences: Sequence[np.ndarray], n_iters: int=50, tol: float=0.0001, cov_reg: float=0.0001, verbose: bool=False) -> List[float]:
        logliks: List[float] = []
        for it in range(n_iters):
            num_pi = np.zeros(self.N)
            num_A = np.full((self.N, self.N), 1e-09)
            den_A = np.full(self.N, 1e-09 * self.N)
            weighted_sum = np.zeros((self.N, self.d))
            weighted_out = np.zeros((self.N, self.d, self.d))
            state_weight = np.full(self.N, 1e-09)
            total_ll = 0.0
            for obs in sequences:
                obs = np.asarray(obs, dtype=float)
                T = obs.shape[0]
                log_alpha, log_p = self.forward(obs)
                log_beta = self.backward(obs)
                log_gamma = log_alpha + log_beta - log_p
                gamma = np.exp(log_gamma)
                total_ll += log_p
                num_pi += gamma[0]
                L = self.log_emission(obs)
                logA = np.log(self.A + 1e-300)
                for t in range(T - 1):
                    log_xi_t = log_alpha[t][:, None] + logA + (L[t + 1] + log_beta[t + 1])[None, :] - log_p
                    num_A += np.exp(log_xi_t)
                den_A += gamma[:-1].sum(axis=0)
                for j in range(self.N):
                    w = gamma[:, j]
                    weighted_sum[j] += (w[:, None] * obs).sum(axis=0)
                    diff = obs - self.means[j]
                    weighted_out[j] += (w[:, None, None] * np.einsum('ti,tj->tij', diff, diff)).sum(axis=0)
                    state_weight[j] += w.sum()
            self.pi = num_pi / num_pi.sum()
            self.A = num_A / den_A[:, None]
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.means = weighted_sum / state_weight[:, None]
            for j in range(self.N):
                self.covars[j] = weighted_out[j] / state_weight[j] + cov_reg * np.eye(self.d)
            logliks.append(total_ll)
            if verbose:
                print(f'iter {it:3d}  log-lik = {total_ll:.4f}')
            if len(logliks) > 1 and abs(logliks[-1] - logliks[-2]) < tol:
                break
        return logliks