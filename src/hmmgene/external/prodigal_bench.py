from __future__ import annotations
from typing import Dict, List, Tuple
try:
    import pyrodigal
except ImportError as e:
    raise ImportError('pyrodigal is required for prodigal_bench; install with `pip install pyrodigal`') from e

def run_prodigal(sequence: str, meta: bool=True) -> List[Tuple[int, int, str]]:
    gene_finder = pyrodigal.GeneFinder(meta=meta)
    genes = gene_finder.find_genes(sequence.encode('ascii'))
    out = []
    for g in genes:
        strand = '+' if g.strand == 1 else '-'
        out.append((int(g.begin), int(g.end), strand))
    return out

def head_to_head(test_sequence: str, truth_intervals: List[Tuple[int, int, str]], our_preds: List[Tuple[int, int, str]], stop_tol: int=9) -> Dict:
    from ..gene_finder import benchmark
    prodigal_preds = run_prodigal(test_sequence)
    prodigal_scores = benchmark(prodigal_preds, truth_intervals, stop_tol=stop_tol)
    our_scores = benchmark(our_preds, truth_intervals, stop_tol=stop_tol)
    return {'prodigal': {'n_predictions': len(prodigal_preds), **prodigal_scores}, 'hmmgene': {'n_predictions': len(our_preds), **our_scores}}