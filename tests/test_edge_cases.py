import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from hmmgene import DiscreteHMM, dishonest_casino_hmm, encode_dna
from hmmgene.io_fasta import read_fasta

def _tiny():
    return DiscreteHMM(A=np.array([[0.8, 0.2], [0.3, 0.7]]), B=np.array([[0.6, 0.4], [0.3, 0.7]]), pi=np.array([0.5, 0.5]))

def test_viterbi_on_empty_fails_cleanly():
    hmm = _tiny()
    with pytest.raises((IndexError, ValueError)):
        hmm.viterbi([])

def test_viterbi_on_single_observation():
    hmm = _tiny()
    lp, path = hmm.viterbi([0])
    assert path.shape == (1,)
    expected = int(np.argmax(hmm.logpi + hmm.logB[:, 0]))
    assert path[0] == expected

def test_forward_handles_all_same_observation():
    hmm = _tiny()
    _, logp = hmm.forward([0] * 100)
    assert np.isfinite(logp)

def test_encode_dna_handles_ambiguous_nucleotides():
    enc = encode_dna('ACGTNX')
    assert len(enc) == 6
    assert enc[0] == 0
    assert enc[3] == 3

def test_hmm_validates_non_stochastic_input():
    with pytest.raises(ValueError):
        DiscreteHMM(A=np.array([[0.5, 0.4], [0.3, 0.7]]), B=np.array([[0.5, 0.5], [0.5, 0.5]]), pi=np.array([0.5, 0.5]))

def test_hmm_validates_shape_mismatch():
    with pytest.raises(ValueError):
        DiscreteHMM(A=np.array([[1.0, 0.0], [0.0, 1.0]]), B=np.array([[0.5, 0.5], [0.5, 0.5]]), pi=np.array([0.5, 0.3, 0.2]))

def test_read_fasta_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_fasta('/nonexistent/file.fasta')

def test_baum_welch_converges_on_flat_sequence():
    hmm = _tiny()
    obs = [0] * 50
    logliks = hmm.baum_welch([obs], n_iters=5, tol=1e-12)
    assert len(logliks) >= 1