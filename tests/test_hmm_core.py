import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from hmmgene import DiscreteHMM, build_cpg_hmm, dishonest_casino_hmm, encode_dna, logsumexp, read_fasta, read_genbank
from hmmgene.hmm import _check_row_stochastic

def test_logsumexp_scalar_matches_log_sum_exp():
    x = np.array([-1.0, -2.0, -3.0])
    assert math.isclose(logsumexp(x), math.log(math.exp(-1) + math.exp(-2) + math.exp(-3)))

def test_logsumexp_all_neg_inf_returns_neg_inf():
    x = np.array([-np.inf, -np.inf])
    assert logsumexp(x) == -np.inf

def test_logsumexp_axis():
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    out = logsumexp(x, axis=1)
    assert out.shape == (2,)
    assert math.isclose(out[0], math.log(math.exp(-1) + math.exp(-2)))

def test_forward_equals_log_of_summed_gamma():
    hmm = dishonest_casino_hmm()
    rng = np.random.default_rng(0)
    _, obs = hmm.sample(500, rng=rng)
    log_alpha, log_p = hmm.forward(obs)
    gamma, _, log_p2 = hmm.posteriors(obs)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-08)
    assert math.isclose(log_p, log_p2, abs_tol=1e-08)

def test_viterbi_at_least_as_good_as_any_sampled_path():
    hmm = dishonest_casino_hmm()
    rng = np.random.default_rng(1)
    states, obs = hmm.sample(200, rng=rng)
    log_p_star, path = hmm.viterbi(obs)
    p = hmm.logpi[states[0]] + hmm.logB[states[0], obs[0]] + sum((hmm.logA[states[t - 1], states[t]] + hmm.logB[states[t], obs[t]] for t in range(1, 200)))
    assert log_p_star >= p - 1e-09

def test_baum_welch_recovers_parameters_on_synthetic():
    true_hmm = dishonest_casino_hmm()
    rng = np.random.default_rng(2)
    _, obs = true_hmm.sample(5000, rng=rng)
    init = DiscreteHMM(A=np.array([[0.6, 0.4], [0.4, 0.6]]), B=np.array([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]), pi=np.array([0.7, 0.3]))
    logliks = init.baum_welch([obs], n_iters=100, tol=0.001)
    assert logliks[-1] > logliks[0]
    assert int(np.argmax(init.B[1])) == 5

def test_cpg_hmm_valid_and_emits_correctly():
    hmm = build_cpg_hmm()
    assert hmm.N == 8
    assert hmm.M == 4
    for i in range(8):
        assert (hmm.B[i] == 0).sum() == 3
        assert np.isclose(hmm.B[i].sum(), 1.0)

def test_cpg_viterbi_on_simple_input_runs():
    hmm = build_cpg_hmm()
    obs = encode_dna('ACGTACGTACGT')
    _, path = hmm.viterbi(obs)
    assert path.shape == (12,)
    for o, s in zip(obs, path):
        assert s % 4 == int(o)
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

def test_read_fasta_ecoli():
    recs = read_fasta(DATA_DIR / 'ecoli_K12_MG1655_first100kb.fasta')
    assert len(recs) == 1
    assert len(recs[0].sequence) == 100000
    assert set(recs[0].sequence) <= set('ACGTN')

def test_read_genbank_ecoli_has_cds():
    gb = read_genbank(DATA_DIR / 'ecoli_K12_MG1655_first100kb.gb')
    assert gb.length == 100000
    assert len(gb.features) > 50
    forward = [f for f in gb.features if f.strand == '+']
    assert len(forward) > 20
    for f in gb.features:
        assert 1 <= f.start <= gb.length
        assert 1 <= f.end <= gb.length

def test_parameter_validation():
    with pytest.raises(ValueError):
        DiscreteHMM(A=np.array([[0.5, 0.4], [0.3, 0.7]]), B=np.array([[0.5, 0.5], [0.5, 0.5]]), pi=np.array([0.5, 0.5]))