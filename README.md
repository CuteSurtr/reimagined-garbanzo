# hmmgene — HMMs for gene prediction

A from-scratch library building HMMs progressively from the classical
three-problem framework (Rabiner 1989, Durbin et al. 1998) up to a
working generalized-HMM gene finder for bacterial genomes.

See [docs/lit_review.md](docs/lit_review.md) for the literature
synthesis and [docs/PLAN.md](docs/PLAN.md) for the layered roadmap.

## Layout
```
literature/   foundational papers (Rabiner, Durbin book, GENSCAN, …)
data/         real genomic sequences (E. coli 100 kb, human BRCA1)
docs/         design docs, literature review, plan
notes/        per-paper notes during reading
src/hmmgene/  the library
tests/        pytest suite
results/      output of demos (plots, benchmarks)
```

## Quick start
```bash
cd src && python3 -m hmmgene.demo
```

## Planned modules

| Module | Role |
|--------|------|
| `io_fasta.py`  | minimal FASTA / GenBank parsers |
| `hmm.py`       | DiscreteHMM + fwd/bwd/Viterbi/Baum-Welch |
| `logmath.py`   | numerically-stable log-space utilities |
| `casino.py`    | occasionally-dishonest-casino demo |
| `cpg.py`       | 8-state CpG-island HMM (Durbin Fig 3.3) |
| `gene_finder.py` | bacterial gene finder with codon-phase states |
| `profile.py`   | profile HMM (M/I/D architecture) |
| `pair_hmm.py`  | pair HMM for pairwise alignment |
| `ghmm.py`      | generalized HMM with state durations |
| `viz.py`       | matplotlib figures |
```
