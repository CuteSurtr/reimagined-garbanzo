from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
try:
    import pyhmmer
    from pyhmmer.easel import Alphabet, DigitalSequence, TextSequence
    from pyhmmer.plan7 import Builder, Background, HMMFile, Pipeline
except ImportError as e:
    raise ImportError('pyhmmer is required for hmmer_bridge; install with `pip install pyhmmer`') from e

@dataclass
class HMMERHit:
    query: str
    target: str
    score: float
    e_value: float
    start: int
    end: int

class HMMERProfile:

    def __init__(self, hmm: 'pyhmmer.plan7.HMM', alphabet: 'Alphabet'):
        self.hmm = hmm
        self.alphabet = alphabet

    @classmethod
    def build_from_msa(cls, msa: Sequence[str], name: str, alphabet: str='DNA', seq_names: Optional[Sequence[str]]=None) -> 'HMMERProfile':
        alph = Alphabet.dna() if alphabet.upper() == 'DNA' else Alphabet.rna() if alphabet.upper() == 'RNA' else Alphabet.amino()
        if seq_names is None:
            seq_names = [f'seq{i}' for i in range(len(msa))]
        text_seqs = [TextSequence(sequence=s, name=n.encode('ascii')) for s, n in zip(msa, seq_names)]
        from pyhmmer.easel import DigitalMSA
        digital_seqs = [t.digitize(alph) for t in text_seqs]
        msa_obj = DigitalMSA(name=name.encode('ascii'), sequences=digital_seqs)
        builder = Builder(alphabet=alph)
        background = Background(alph)
        hmm, _, _ = builder.build_msa(msa_obj, background)
        return cls(hmm, alph)

    @classmethod
    def from_hmm_file(cls, path: str | Path) -> 'HMMERProfile':
        with HMMFile(str(path)) as f:
            hmm = f.read()
        return cls(hmm, hmm.alphabet)

    def search(self, queries: Sequence[tuple[str, str]], e_cutoff: float=0.001) -> List[HMMERHit]:
        from pyhmmer.easel import DigitalSequenceBlock
        digital_seqs = []
        for name, seq in queries:
            ts = TextSequence(name=name.encode('ascii'), sequence=seq.upper())
            digital_seqs.append(ts.digitize(self.alphabet))
        block = DigitalSequenceBlock(self.alphabet, digital_seqs)
        pipeline = Pipeline(alphabet=self.alphabet, background=Background(self.alphabet))
        hits_out: List[HMMERHit] = []
        tophits = pipeline.search_hmm(query=self.hmm, sequences=block)

        def _to_str(x):
            if x is None:
                return ''
            return x.decode('ascii') if isinstance(x, (bytes, bytearray)) else str(x)
        for hit in tophits:
            if hit.evalue >= e_cutoff:
                continue
            for domain in hit.domains:
                hits_out.append(HMMERHit(query=_to_str(hit.name), target=_to_str(self.hmm.name), score=float(hit.score), e_value=float(hit.evalue), start=int(domain.alignment.target_from), end=int(domain.alignment.target_to)))
        return hits_out

    @property
    def n_match_states(self) -> int:
        return int(self.hmm.M)

    def __repr__(self) -> str:
        return f'HMMERProfile(name={self.hmm.name}, M={self.n_match_states})'