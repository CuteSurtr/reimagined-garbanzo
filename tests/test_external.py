import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from hmmgene import BacterialGeneFinder, DiscreteHMM, dishonest_casino_hmm, read_fasta
DATA = Path(__file__).resolve().parents[1] / 'data'

def test_hmmlearn_matches_our_forward_and_viterbi():
    pytest.importorskip('hmmlearn')
    from hmmgene.external.hmmlearn_xcheck import compare_forward_loglik, compare_viterbi_path
    hmm = dishonest_casino_hmm()
    rng = np.random.default_rng(0)
    _, obs = hmm.sample(500, rng=rng)
    ours_logL, ref_logL = compare_forward_loglik(hmm, obs)
    assert math.isclose(ours_logL, ref_logL, abs_tol=1e-06), (ours_logL, ref_logL)
    ours_path, ref_path = compare_viterbi_path(hmm, obs)
    agree = float((ours_path == ref_path).mean())
    assert agree >= 0.99

def test_pyrodigal_finds_genes_in_ecoli_fragment():
    pytest.importorskip('pyrodigal')
    from hmmgene.external.prodigal_bench import run_prodigal
    rec = read_fasta(DATA / 'ecoli_K12_MG1655_first100kb.fasta')[0]
    seq = rec.sequence[:20000]
    preds = run_prodigal(seq, meta=True)
    assert len(preds) >= 5
    for s, e, strand in preds:
        assert 1 <= s <= e <= len(seq)
        assert strand in ('+', '-')

def test_prodigal_head_to_head_benchmark_structure():
    pytest.importorskip('pyrodigal')
    from hmmgene import read_genbank
    from hmmgene.external.prodigal_bench import head_to_head
    rec = read_fasta(DATA / 'ecoli_K12_MG1655_first100kb.fasta')[0]
    gb = read_genbank(DATA / 'ecoli_K12_MG1655_first100kb.gb')
    seq = rec.sequence[:20000]
    truth = [(f.start, f.end, f.strand) for f in gb.features if f.end <= 20000]
    result = head_to_head(seq, truth, our_preds=[])
    assert 'prodigal' in result and 'hmmgene' in result
    assert result['prodigal']['n_predictions'] > 0
    assert result['prodigal']['bp']['sensitivity'] > 0.5

def test_pyhmmer_loads_real_pfam_hmm():
    pytest.importorskip('pyhmmer')
    from hmmgene.external.hmmer_bridge import HMMERProfile
    hmm_path = DATA / 'Pfam_PF13560.hmm'
    if not hmm_path.exists():
        pytest.skip('Pfam HMM file not present')
    prof = HMMERProfile.from_hmm_file(hmm_path)
    assert prof.n_match_states == 64

def test_pyhmmer_search_returns_hits():
    pytest.importorskip('pyhmmer')
    from hmmgene.external.hmmer_bridge import HMMERProfile
    hmm_path = DATA / 'Pfam_PF13560.hmm'
    fa_path = DATA / 'hth_test_proteins.fasta'
    if not hmm_path.exists() or not fa_path.exists():
        pytest.skip('test data not present')
    prof = HMMERProfile.from_hmm_file(hmm_path)
    recs = read_fasta(fa_path)
    queries = [(r.accession, r.sequence) for r in recs]
    hits = prof.search(queries, e_cutoff=0.1)
    assert isinstance(hits, list)
    for h in hits:
        assert h.e_value >= 0
        assert h.score is not None

def test_biopython_translate_cds():
    pytest.importorskip('Bio')
    from hmmgene.external.biopython_io import translate_cds
    protein = translate_cds('ATGGACCCATAA')
    assert protein.startswith('MDP')
    assert '*' in protein