from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple
try:
    from Bio import AlignIO, SeqIO
    from Bio.Seq import Seq as _BioSeq
except ImportError as e:
    raise ImportError('biopython is required for biopython_io; install with `pip install biopython`') from e

def read_stockholm_msa(path: str | Path) -> Tuple[List[str], List[str]]:
    aln = AlignIO.read(str(path), 'stockholm')
    names = [rec.id for rec in aln]
    seqs = [str(rec.seq) for rec in aln]
    return (names, seqs)

def translate_cds(dna: str, table: int=11) -> str:
    s = _BioSeq(dna)
    return str(s.translate(table=table, to_stop=False))