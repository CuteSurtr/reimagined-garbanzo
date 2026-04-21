import itertools
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from hmmgene import DiscreteHMM

def _tiny_hmm():
    return DiscreteHMM(A=np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]), B=np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]]), pi=np.array([0.5, 0.3, 0.2]))

def _path_logprob(hmm, path, obs):
    logp = hmm.logpi[path[0]] + hmm.logB[path[0], obs[0]]
    for t in range(1, len(obs)):
        logp += hmm.logA[path[t - 1], path[t]] + hmm.logB[path[t], obs[t]]
    return logp

def test_viterbi_matches_brute_force():
    hmm = _tiny_hmm()
    obs = np.array([0, 1, 2, 2, 0, 1, 0])
    best_path = None
    best_lp = -np.inf
    for path in itertools.product(range(3), repeat=len(obs)):
        lp = _path_logprob(hmm, path, obs)
        if lp > best_lp:
            best_lp = lp
            best_path = path
    log_p_star, path = hmm.viterbi(obs)
    assert math.isclose(log_p_star, best_lp, abs_tol=1e-09)
    assert tuple(path) == best_path

def test_forward_equals_summed_enumeration():
    hmm = _tiny_hmm()
    obs = np.array([0, 1, 2, 2, 0])
    total = 0.0
    for path in itertools.product(range(3), repeat=len(obs)):
        total += math.exp(_path_logprob(hmm, path, obs))
    _, logp = hmm.forward(obs)
    assert math.isclose(logp, math.log(total), abs_tol=1e-09)

def test_backward_complements_forward():
    hmm = _tiny_hmm()
    obs = np.array([0, 1, 2, 2, 0])
    log_alpha, logp_fwd = hmm.forward(obs)
    log_beta = hmm.backward(obs)
    from hmmgene.logmath import logsumexp
    for t in range(len(obs)):
        lp = logsumexp(log_alpha[t] + log_beta[t])
        assert math.isclose(lp, logp_fwd, abs_tol=1e-09)

def test_gamma_sums_to_one_across_states():
    hmm = _tiny_hmm()
    obs = np.array([0, 1, 2, 2, 0, 1])
    gamma, _, _ = hmm.posteriors(obs)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)

def test_xi_sums_to_gamma_marginal():
    hmm = _tiny_hmm()
    obs = np.array([0, 1, 2, 2, 0, 1])
    gamma, xi, _ = hmm.posteriors(obs)
    for t in range(len(obs) - 1):
        assert np.allclose(xi[t].sum(axis=1), gamma[t], atol=1e-09)

def test_viterbi_on_T1_sequence():
    hmm = _tiny_hmm()
    obs = np.array([2])
    lp, path = hmm.viterbi(obs)
    expected = np.max(hmm.logpi + hmm.logB[:, 2])
    assert math.isclose(lp, expected, abs_tol=1e-12)

def test_baum_welch_monotone_on_synthetic():
    true_hmm = _tiny_hmm()
    rng = np.random.default_rng(0)
    _, obs = true_hmm.sample(2000, rng=rng)
    init = DiscreteHMM(A=np.full((3, 3), 1 / 3), B=np.full((3, 3), 1 / 3), pi=np.full(3, 1 / 3))
    logliks = init.baum_welch([obs], n_iters=30, tol=-1)
    for a, b in zip(logliks[:-1], logliks[1:]):
        assert b >= a - 1e-06