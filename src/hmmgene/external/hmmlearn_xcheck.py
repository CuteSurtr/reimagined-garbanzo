from __future__ import annotations
from typing import Tuple
import numpy as np
try:
    from hmmlearn import hmm as _hmmlearn
except ImportError as e:
    raise ImportError('hmmlearn required for hmmlearn_xcheck; install with `pip install hmmlearn`') from e
from ..hmm import DiscreteHMM

def build_reference_from_ours(ours: DiscreteHMM) -> '_hmmlearn.CategoricalHMM':
    ref = _hmmlearn.CategoricalHMM(n_components=ours.N, n_features=ours.M, init_params='')
    ref.startprob_ = ours.pi.copy()
    ref.transmat_ = ours.A.copy()
    ref.emissionprob_ = ours.B.copy()
    return ref

def compare_forward_loglik(ours: DiscreteHMM, obs: np.ndarray) -> Tuple[float, float]:
    ref = build_reference_from_ours(ours)
    obs_arr = np.asarray(obs, dtype=int).reshape(-1, 1)
    _, ours_logL = ours.forward(obs)
    ref_logL = float(ref.score(obs_arr))
    return (ours_logL, ref_logL)

def compare_viterbi_path(ours: DiscreteHMM, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = build_reference_from_ours(ours)
    obs_arr = np.asarray(obs, dtype=int).reshape(-1, 1)
    _, ours_path = ours.viterbi(obs)
    _, ref_path = ref.decode(obs_arr, algorithm='viterbi')
    return (ours_path, np.asarray(ref_path, dtype=int))