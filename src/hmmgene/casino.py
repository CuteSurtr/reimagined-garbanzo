from __future__ import annotations
import numpy as np
from .hmm import DiscreteHMM
FAIR = 0
LOADED = 1

def dishonest_casino_hmm() -> DiscreteHMM:
    A = np.array([[0.95, 0.05], [0.1, 0.9]])
    B = np.array([[1 / 6] * 6, [1 / 10] * 5 + [1 / 2]])
    pi = np.array([0.5, 0.5])
    return DiscreteHMM(A=A, B=B, pi=pi, state_names=['Fair', 'Loaded'], symbol_names=[str(i + 1) for i in range(6)])