# Project Plan — HMMs for Gene Prediction

## Goal
Build a research-quality library (`hmmgene`) that exercises
**probability + dynamic programming + linear algebra + EM theory** on
real genomic data, progressively from the classical CpG-island toy
model all the way to a working generalized-HMM gene finder that
predicts genes on E. coli's first 100 kb and is scored against the
GenBank annotation.

## Layers (increasing math depth)

### Layer 0 — Foundations (probability + DP)
- `hmm.DiscreteHMM` dataclass with (A, B, π) validation, log-space
  parameter storage.
- FASTA / GenBank I/O (minimal parsers — no BioPython in core path).
- `log_sum_exp`, numerically-stable log-space utilities.

### Layer 1 — The three classical algorithms
- `forward(obs)`   → log P(O|λ), full α lattice
- `backward(obs)`  → β lattice
- `viterbi(obs)`   → (log P*, path)
- `posteriors(obs)` → γ_t(i), ξ_t(i,j)

Numerical stability baked in: both scaled-forward and log-space
variants, validated to agree to 1e-9.

### Layer 2 — Learning
- `baum_welch(sequences, n_iters)` — EM on one or many sequences,
  supports tied-parameter constraints.
- `sample(length)` — simulate from an HMM for end-to-end validation.
- Pseudocount smoothing + Dirichlet priors.

### Layer 3 — Applications
**(a) Occasionally-dishonest casino** (Durbin Fig 3.5): sanity check,
recovers fair/loaded switch on simulated rolls.

**(b) CpG island detector** (Durbin Fig 3.3):
- 8-state HMM; emissions deterministic.
- Train transition matrix on real CpG-annotated sequence segments
  (use BRCA1 promoter-region positive set, non-promoter negative set).
- Viterbi decoding predicts island intervals.
- Posterior decoding gives per-base probabilities.

**(c) Bacterial gene finder** (minimalist GeneMark-like):
- States: intergenic, start_ATG, coding_phase_{0,1,2}, stop_codon.
- Emissions: 5th-order Markov in coding, 1st-order in intergenic
  (GeneMark observation: hexamer in-phase frequencies are the signal).
- Train on first 50 kb of E. coli (forward strand ORFs from GenBank
  annotation), test on remaining 50 kb.
- Report sensitivity / specificity on exact start/stop matches.

### Layer 4 — Profile HMMs (Eddy 1998)
- `ProfileHMM` with M / I / D architecture.
- Build from a multiple sequence alignment via Dirichlet pseudocounts.
- Viterbi alignment + forward scoring.
- Apply to: align a query DNA sequence to a small RNA family profile
  (use a handful of tRNA sequences as the training alignment).

### Layer 5 — Pair HMMs (Durbin ch 4)
- 3-state pair HMM (M, X, Y).
- Viterbi on the pair HMM recovers Needleman–Wunsch global alignment
  with affine gaps (verified against a reference NW impl).
- Forward gives P(x, y) integrated over all alignments.
- Posterior decoding highlights alignment uncertainty.

### Layer 6 — Generalized HMM (GHMM / semi-Markov)
- `GeneralizedHMM` with state duration distributions f_S(d).
- Viterbi over GHMMs (O(T² × N²) naive, O(T × N²) with known max
  duration per state).
- Minimal GENSCAN-style gene model for bacteria: intergenic + full
  gene (start, triplet-coding, stop) with duration = multiple of 3
  constraint.
- Measure accuracy on E. coli test set.

## Milestones

| # | Deliverable | Layer | Budget |
|---|-------------|-------|--------|
| M1 | DiscreteHMM + log-space fwd/bwd/Viterbi + tests | 0–1 | 1 session |
| M2 | Baum-Welch + occasionally-dishonest casino demo | 2–3a | 1 session |
| M3 | CpG islands + BRCA1 real-data demo + viz | 3b | 1 session |
| M4 | Bacterial gene finder + E. coli benchmark | 3c | 1 session |
| M5 | Profile HMM from MSA | 4 | 1 session |
| M6 | Pair HMM == NW alignment demo | 5 | 1 session |
| M7 | Generalized HMM for bacterial genes | 6 | 2 sessions |

## Data
- `data/ecoli_K12_MG1655_first100kb.fasta` + `.gb` annotation  (95 real
  CDS features; downloaded from NCBI U00096.3).
- `data/BRCA1_human.fasta` (NM_007294.4 transcript).
- Synthetic sequences generated from a trained HMM for learning tests.

## Stack
- Python 3.10+ (dataclasses, typing).
- NumPy for lattice DP; SciPy.special.logsumexp.
- matplotlib for visualization.
- pytest for tests.
- **No** BioPython in the core path — we parse FASTA / GenBank
  ourselves to keep the math transparent.

## Success criteria
- Synthetic-data test: Baum-Welch on sequences sampled from a known
  HMM recovers parameters within 0.05 (discrete states, T ≥ 1000).
- CpG-islands: Viterbi correctly re-discovers implanted CpG-rich
  regions in a synthetic test at ≥ 95% per-base accuracy.
- Bacterial gene finder: achieves ≥ 80% sensitivity on the E. coli
  test set (exact stop-codon match), using parameters trained on the
  train set.
- Pair HMM alignment agrees with a reference NW implementation on 20
  random sequence pairs.
- Profile HMM scores homologous queries above the background null at
  E-value ≤ 1e-3.
