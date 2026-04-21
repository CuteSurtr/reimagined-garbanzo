import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from hmmgene import GHMM, GHMMState, GaussianHMM, empirical_duration, geometric_duration

def test_gaussian_hmm_sample_and_viterbi():
    hmm = GaussianHMM(A=np.array([[0.9, 0.1], [0.2, 0.8]]), pi=np.array([0.5, 0.5]), means=np.array([[0.0], [5.0]]), covars=np.array([[[1.0]], [[1.0]]]))
    rng = np.random.default_rng(0)
    states, obs = hmm.sample(200, rng=rng)
    _, decoded = hmm.viterbi(obs)
    agree = (decoded == states).mean()
    assert agree >= 0.85

def test_gaussian_hmm_forward_consistent_with_posteriors():
    hmm = GaussianHMM(A=np.array([[0.8, 0.2], [0.3, 0.7]]), pi=np.array([0.6, 0.4]), means=np.array([[0.0], [3.0]]), covars=np.array([[[1.0]], [[1.5]]]))
    rng = np.random.default_rng(1)
    _, obs = hmm.sample(100, rng=rng)
    la, logp = hmm.forward(obs)
    lb = hmm.backward(obs)
    from hmmgene.logmath import logsumexp
    for t in [0, 50, 99]:
        lp_t = logsumexp(la[t] + lb[t])
        assert math.isclose(lp_t, logp, abs_tol=1e-07)

def test_gaussian_hmm_baum_welch_improves_likelihood():
    true_hmm = GaussianHMM(A=np.array([[0.9, 0.1], [0.15, 0.85]]), pi=np.array([0.5, 0.5]), means=np.array([[-2.0], [3.0]]), covars=np.array([[[0.5]], [[0.8]]]))
    rng = np.random.default_rng(2)
    _, obs = true_hmm.sample(1000, rng=rng)
    init = GaussianHMM(A=np.array([[0.6, 0.4], [0.4, 0.6]]), pi=np.array([0.5, 0.5]), means=np.array([[-1.0], [1.0]]), covars=np.array([[[1.0]], [[1.0]]]))
    logliks = init.baum_welch([obs], n_iters=30)
    assert logliks[-1] > logliks[0]

def test_ghmm_viterbi_recovers_segmentation():
    rng = np.random.default_rng(3)

    def make_emission(prob_zero: float):

        def emit(obs, start, end):
            segment = obs[start:end]
            n0 = int((segment == 0).sum())
            n1 = int((segment == 1).sum())
            return n0 * math.log(prob_zero) + n1 * math.log(1 - prob_zero)
        return emit
    stateA = GHMMState(name='A', log_duration=empirical_duration(np.array([0, 0, 0, 1, 1, 1])), log_segment_emission=make_emission(0.9), max_duration=5, min_duration=3)
    stateB = GHMMState(name='B', log_duration=empirical_duration(np.array([0] * 10 + [1] * 6)), log_segment_emission=make_emission(0.1), max_duration=15, min_duration=10)
    model = GHMM(states=[stateA, stateB], trans=np.array([[0.0, 1.0], [1.0, 0.0]]), pi=np.array([1.0, 0.0]))
    obs = np.concatenate([rng.choice(2, size=5, p=[0.9, 0.1]), rng.choice(2, size=12, p=[0.1, 0.9]), rng.choice(2, size=4, p=[0.9, 0.1]), rng.choice(2, size=10, p=[0.1, 0.9])])
    _, segs = model.viterbi(obs)
    assert len(segs) == 4
    state_sequence = [s for s, _, _ in segs]
    assert state_sequence == [0, 1, 0, 1]

def test_geometric_duration_is_normalized_over_large_cutoff():
    f = geometric_duration(0.1)
    total = sum((math.exp(f(d)) for d in range(1, 500)))
    assert abs(total - 1.0) < 0.01