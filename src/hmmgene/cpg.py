from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .hmm import DiscreteHMM
BASES = 'ACGT'
BASE_IDX = {b: i for i, b in enumerate(BASES)}
DURBIN_P_PLUS = np.array([[0.18, 0.274, 0.426, 0.12], [0.171, 0.368, 0.274, 0.188], [0.161, 0.339, 0.375, 0.125], [0.079, 0.355, 0.384, 0.182]])
DURBIN_P_MINUS = np.array([[0.3, 0.205, 0.285, 0.21], [0.322, 0.298, 0.078, 0.302], [0.248, 0.246, 0.298, 0.208], [0.177, 0.239, 0.292, 0.292]])

def build_cpg_hmm(cross_plus_to_minus: float=0.001, cross_minus_to_plus: float=0.0001) -> DiscreteHMM:
    P_plus = DURBIN_P_PLUS / DURBIN_P_PLUS.sum(axis=1, keepdims=True)
    P_minus = DURBIN_P_MINUS / DURBIN_P_MINUS.sum(axis=1, keepdims=True)
    s_p = 1.0 - cross_plus_to_minus
    s_m = 1.0 - cross_minus_to_plus
    A = np.zeros((8, 8))
    A[0:4, 0:4] = P_plus * s_p
    A[0:4, 4:8] = cross_plus_to_minus / 4
    A[4:8, 4:8] = P_minus * s_m
    A[4:8, 0:4] = cross_minus_to_plus / 4
    B = np.zeros((8, 4))
    for i in range(4):
        B[i, i] = 1.0
        B[i + 4, i] = 1.0
    pi = np.ones(8) / 8
    names = [f'{b}+' for b in BASES] + [f'{b}-' for b in BASES]
    return DiscreteHMM(A=A, B=B, pi=pi, state_names=names, symbol_names=list(BASES))

def encode_dna(seq: str) -> np.ndarray:
    mapping = np.full(256, -1, dtype=int)
    for i, b in enumerate('ACGT'):
        mapping[ord(b)] = i
    arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    out = mapping[arr]
    out = np.where(out >= 0, out, 0)
    return out.astype(int)

@dataclass
class Interval:
    start: int
    end: int

    def __repr__(self) -> str:
        return f'[{self.start}:{self.end})'

def predict_islands(seq: str, hmm: DiscreteHMM=None) -> List[Interval]:
    if hmm is None:
        hmm = build_cpg_hmm()
    obs = encode_dna(seq)
    _, path = hmm.viterbi(obs)
    in_plus = path < 4
    return _runs(in_plus)

def posterior_island_probability(seq: str, hmm: DiscreteHMM=None) -> np.ndarray:
    if hmm is None:
        hmm = build_cpg_hmm()
    obs = encode_dna(seq)
    gamma, _, _ = hmm.posteriors(obs)
    return gamma[:, :4].sum(axis=1)

def _runs(mask: np.ndarray) -> List[Interval]:
    diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [Interval(int(s), int(e)) for s, e in zip(starts, ends)]