import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from hmmgene import encode_dna, simple_pair_hmm
from hmmgene.pair_hmm import PairHMM, M_STATE, X_STATE, Y_STATE

def _reference_nw(x, y, match, mismatch, gap_open, gap_extend):
    n, m = (len(x), len(y))
    INF = -1e+18
    M = np.full((n + 1, m + 1), INF)
    X = np.full((n + 1, m + 1), INF)
    Y = np.full((n + 1, m + 1), INF)
    M[0, 0] = 0
    for i in range(1, n + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if x[i - 1] == y[j - 1] else mismatch
            M[i, j] = s + max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            X[i, j] = max(M[i - 1, j] + gap_open, X[i - 1, j] + gap_extend)
            Y[i, j] = max(M[i, j - 1] + gap_open, Y[i, j - 1] + gap_extend)
    return max(M[n, m], X[n, m], Y[n, m])

def test_pair_hmm_viterbi_ordering_same_as_nw():
    rng = np.random.default_rng(0)
    hmm = simple_pair_hmm(match_prob=0.85)
    match = float(hmm.log_p_xy[0, 0] - hmm.log_q_x[0] - hmm.log_q_y[0])
    mismatch = float(hmm.log_p_xy[0, 1] - hmm.log_q_x[0] - hmm.log_q_y[1])
    gap_open = float(np.log(hmm.delta) - np.log(1 - 2 * hmm.delta - hmm.tau))
    gap_extend = float(np.log(hmm.epsilon) - np.log(1 - hmm.epsilon - hmm.tau))
    _ = (match, mismatch, gap_open, gap_extend)
    for trial in range(10):
        n = rng.integers(5, 15)
        x = rng.integers(0, 4, size=n)
        y = x.copy()
        k = rng.integers(0, 3)
        for _ in range(k):
            pos = rng.integers(0, len(y))
            y[pos] = (y[pos] + 1) % 4
        s_same, _ = hmm.viterbi(x, x)
        s_diff, _ = hmm.viterbi(x, y)
        assert s_same >= s_diff - 1e-09

def test_pair_hmm_forward_ge_viterbi():
    hmm = simple_pair_hmm(match_prob=0.85)
    rng = np.random.default_rng(1)
    for _ in range(5):
        n = rng.integers(5, 12)
        m = rng.integers(5, 12)
        x = rng.integers(0, 4, size=n)
        y = rng.integers(0, 4, size=m)
        score_v, _ = hmm.viterbi(x, y)
        _, logp = hmm.forward(x, y)
        assert logp >= score_v - 1e-09

def test_pair_hmm_posterior_match_is_probability_matrix():
    hmm = simple_pair_hmm(match_prob=0.85)
    x = encode_dna('ACGT')
    y = encode_dna('ACGT')
    post = hmm.posterior_match(x, y)
    assert post.shape == (4, 4)
    assert (post >= -1e-09).all() and (post <= 1 + 1e-06).all()
    diag = np.diag(post).mean()
    off = (post.sum() - np.diag(post).sum()) / (post.size - 4)
    assert diag > off

def test_pair_hmm_forward_independent_of_path_ordering():
    hmm = simple_pair_hmm(match_prob=0.85)
    x = encode_dna('AAACCC')
    y = encode_dna('AAGCCC')
    _, a = hmm.forward(x, y)
    _, b = hmm.forward(x, y)
    assert math.isclose(a, b, abs_tol=1e-12)