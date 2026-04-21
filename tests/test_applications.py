import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from hmmgene import BacterialGeneFinder, PairHMM, ProfileHMM, build_cpg_hmm, encode_dna, predict_islands, posterior_island_probability, read_fasta, read_genbank, simple_pair_hmm
DATA = Path(__file__).resolve().parents[1] / 'data'

def test_cpg_detects_implanted_island():
    rng = np.random.default_rng(3)
    hmm = build_cpg_hmm()
    from hmmgene.cpg import DURBIN_P_PLUS, DURBIN_P_MINUS, BASES

    def sample_markov(P, length, first):
        P = P / P.sum(axis=1, keepdims=True)
        seq = [first]
        cur = first
        for _ in range(length - 1):
            nxt = rng.choice(4, p=P[cur])
            seq.append(nxt)
            cur = nxt
        return np.asarray(seq, dtype=int)
    background1 = sample_markov(DURBIN_P_MINUS, 3000, 0)
    island = sample_markov(DURBIN_P_PLUS, 1500, 1)
    background2 = sample_markov(DURBIN_P_MINUS, 3000, 0)
    seq_int = np.concatenate([background1, island, background2])
    seq_str = ''.join((BASES[i] for i in seq_int))
    islands = predict_islands(seq_str, hmm)
    mask = np.zeros(len(seq_int), dtype=bool)
    for iv in islands:
        mask[iv.start:iv.end] = True
    truth = np.zeros(len(seq_int), dtype=bool)
    truth[3000:3000 + 1500] = True
    overlap = (mask & truth).sum()
    assert overlap / truth.sum() >= 0.6

def test_posterior_island_probability_shape():
    seq = 'ACGTACGT' * 10
    p = posterior_island_probability(seq)
    assert p.shape == (80,)
    assert (p >= 0).all() and (p <= 1 + 1e-06).all()

def test_gene_finder_benchmark_structure():
    from hmmgene import benchmark
    preds = [(1, 100, '+'), (200, 300, '+')]
    truth = [(1, 100, '+'), (210, 310, '+'), (400, 500, '+')]
    r = benchmark(preds, truth, stop_tol=5)
    assert r['exact']['TP'] == 1
    assert r['stop_tol']['TP'] == 1
    assert r['bp']['sensitivity'] > 0

def test_bacterial_gene_finder_trains_without_errors():
    recs = read_fasta(DATA / 'ecoli_K12_MG1655_first100kb.fasta')
    gb = read_genbank(DATA / 'ecoli_K12_MG1655_first100kb.gb')
    seq = recs[0].sequence[:50000]
    train = [(f.start, f.end, f.strand) for f in gb.features if f.end <= 50000]
    finder = BacterialGeneFinder.default()
    finder.fit(seq, train)
    assert finder.coding is not None
    assert finder.coding.cond.shape == (3, 4 ** 5, 4)
    assert np.allclose(finder.coding.cond.sum(axis=2), 1.0, atol=1e-06)

def test_pair_hmm_viterbi_identical_sequences():
    hmm = simple_pair_hmm(match_prob=0.9)
    x = encode_dna('ACGTACGT')
    y = encode_dna('ACGTACGT')
    score, alignment = hmm.viterbi(x, y)
    gaps = sum((1 for a, b in alignment if a == -1 or b == -1))
    assert gaps == 0
    assert len(alignment) == 8

def test_pair_hmm_viterbi_one_gap():
    hmm = simple_pair_hmm(match_prob=0.9)
    x = encode_dna('ACGTACGT')
    y = encode_dna('ACGTACG')
    score, alignment = hmm.viterbi(x, y)
    gaps = sum((1 for a, b in alignment if a == -1 or b == -1))
    assert gaps == 1

def test_profile_hmm_builds_from_msa():
    msa = ['ACGTACGT', 'ACGTACGT', 'ACGTACAT', 'ACGT-CGT', 'ACGTACGT']
    p = ProfileHMM.from_msa(msa)
    assert p.L == 8
    good = encode_dna('ACGTACGT')
    bad = encode_dna('TTTTTTTT')
    score_good = p.viterbi(good)
    score_bad = p.viterbi(bad)
    assert score_good > score_bad