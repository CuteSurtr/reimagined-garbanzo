# Literature Synthesis — Project 1: HMMs for Gene Prediction

## 1. HMM formalism (Rabiner 1989; Durbin ch 3)

A **hidden Markov model** is a tuple λ = (A, B, π):

* **N** hidden states S₁,…,S_N; **M** observation symbols v₁,…,v_M.
* **A** = (aᵢⱼ), aᵢⱼ = P(q_{t+1}=Sⱼ | q_t=Sᵢ).  Rows sum to 1.
* **B** = (bⱼ(k)), bⱼ(k) = P(v_k | q_t=Sⱼ).  Rows sum to 1.
* **π** = (πᵢ), πᵢ = P(q₁=Sᵢ).  Sums to 1.

Durbin's notation unifies this by adding a silent **Begin** state 0 so
that πᵢ = a_{0,i} and likewise End = 0 for termination.

## 2. The three fundamental problems

### 2.1 Problem 1 — Evaluation: P(O|λ)
**Forward algorithm** (Rabiner §III-A, Durbin §3.2):

- α₁(i) = πᵢ bᵢ(O₁)
- α_{t+1}(j) = [ Σᵢ α_t(i) aᵢⱼ ] · bⱼ(O_{t+1})
- P(O|λ) = Σᵢ α_T(i)

Runs in O(N²T).

### 2.2 Problem 2 — Decoding: argmax_Q P(Q,O|λ)
**Viterbi algorithm** (Rabiner eq 30–35; Durbin §3.2):

- δ₁(i) = πᵢ bᵢ(O₁),   ψ₁(i) = 0
- δ_t(j) = max_i [δ_{t-1}(i) aᵢⱼ] · bⱼ(O_t),   ψ_t(j) = argmax_i [δ_{t-1}(i) aᵢⱼ]
- q*_T = argmax_i δ_T(i);  q*_{t-1} = ψ_t(q*_t)

Runs in O(N²T), must be done in **log-space** to avoid underflow.

### 2.3 Problem 3 — Learning: adjust λ to maximize P(O|λ)
**Baum–Welch (EM)** (Rabiner §III-C; derivation in Stephen Tu's writeup):

- **Backward algorithm**: β_T(i)=1; β_t(i) = Σⱼ aᵢⱼ bⱼ(O_{t+1}) β_{t+1}(j).
- **Posteriors**:
  - γ_t(i) = α_t(i) β_t(i) / P(O|λ)       (state occupancy)
  - ξ_t(i,j) = α_t(i) aᵢⱼ bⱼ(O_{t+1}) β_{t+1}(j) / P(O|λ)   (pair occupancy)
- **Re-estimation**:
  - π̄ᵢ = γ₁(i)
  - āᵢⱼ = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
  - b̄ⱼ(k) = Σ_{t : O_t=v_k} γ_t(j) / Σ_t γ_t(j)

Converges to a local maximum of the likelihood; initialization matters.

### 2.4 Numerical stability
- **Log-space Viterbi**: replace `max_i δ a` with `max_i (log δ + log a)`.
- **Scaled forward–backward**: at each t, rescale α_t(i) by c_t = 1/Σᵢ α_t(i);
  bookkeeping gives log P(O|λ) = −Σ_t log c_t (Rabiner §V-A).
- **Log-sum-exp trick** for forward in log space:
  log(Σ exp(x_i)) = x_max + log Σ exp(x_i − x_max).

## 3. CpG islands — the canonical bioinformatics HMM (Durbin Fig 3.3)

**Problem:** locate **CpG-enriched regions** ("islands") in genomic DNA.
Biology: cytosine methylation over evolutionary time depletes CpG
dinucleotides, *except* near gene promoters where islands persist.

**Model:**
- 8 emitting states: A⁺, C⁺, G⁺, T⁺ (island) and A⁻, C⁻, G⁻, T⁻ (non-island).
- Deterministic emissions: state X⁺ or X⁻ emits base X with probability 1.
- Transition matrix a_{k,l} has four "blocks":
  - within-⁺: a Markov chain reflecting CpG-island dinucleotide stats
    (relatively high G|C, C|G).
  - within-⁻: the background genome stats (G|C depleted).
  - ⁺ → ⁻  and ⁻ → ⁺ with small cross probabilities.

**Outputs:**
- Viterbi path → CpG-island intervals (runs of ⁺ states).
- Posterior γ_t(⁺) → soft probability of being inside an island at each
  base, useful for downstream scoring.

## 4. Bacterial / prokaryotic gene finding

### 4.1 Simple ORF + HMM (Durbin-style teaching model)
States:
- Intergenic (background 0-th order / uniform-ish composition).
- Start codon (ATG / alternates).
- Coding triplet in phase 0, 1, 2 — a **3-periodic** sub-chain.
- Stop codon (TAA/TAG/TGA).
- Reverse-strand mirror states (optional).

Emissions in coding states reflect **codon bias** (in-phase hexamer
frequencies; Borodovsky's observation, implemented as 5th-order Markov
in GeneMark).

### 4.2 Glimmer — Interpolated Markov Models (IMMs)
Salzberg et al. 1998. Rather than fixing a Markov order, Glimmer uses
an **interpolated** order that adapts to local data: given a k-mer
context, choose the largest order k for which the k-mer is "well
supported" in training data; blend lower orders with weights χ_k to
smooth sparse counts. IMM is not an HMM per se — it is an emission
model that *plugs into* a Markov state-space.

### 4.3 GeneMark.hmm — Lukashin & Borodovsky 1998
Formally embeds inhomogeneous Markov chains (one per codon position)
as emission models inside a state HMM whose states are coding (+/−
strands, 3 phases) and non-coding. Viterbi on the full model gives
predicted gene boundaries.

## 5. Eukaryotic gene finding

### 5.1 Generalized HMMs (GHMM / HSMM)
Kulp, Haussler, Reese, Eeckman 1996 (Genie). Burge & Karlin 1997
(GENSCAN). Stanke 2003/2006 (AUGUSTUS).

Key generalization: **state durations** are drawn from arbitrary
length distributions, not implicitly geometric as in a plain HMM. For
each state S we specify
- a duration pdf f_S(d) (e.g. empirical exon-length histogram),
- a sub-emission model p_S(x_{1:d}) (e.g. 5th-order Markov, position-
  specific scoring matrix for signal models like splice sites).

The joint generative process: pick state S, draw duration d ~ f_S,
emit d symbols from p_S(·), transition to next state per A.

### 5.2 GENSCAN architecture (Burge–Karlin 1997, Fig 3)
States include:
- Intergenic regions (N)
- Promoter (P), polyA signal (A)
- 5'UTR, 3'UTR
- Initial / internal / terminal / single exons (E_init, E_k, E_term, E_sgl)
- Introns in phases 0/1/2
- Signal sub-models: splice sites (donor GT / acceptor AG) via weight
  matrices + MDD (Maximal Dependence Decomposition).

Separate parameter sets per GC-content compositional region of the
genome.

### 5.3 AUGUSTUS intron submodel (Stanke 2003)
Replaces the single intron state with a three-part model:
- 5' splice-site region (fixed-length PWM)
- variable-length interior
- 3' splice-site region
Improves accuracy on long introns.

## 6. Profile HMMs — protein/DNA families (Eddy 1998)

A **profile HMM** has a left-to-right architecture with three kinds of
states per consensus column j:
- Match M_j — emits with position-specific emission probabilities.
- Insert I_j — emits (extra residues relative to consensus).
- Delete D_j — silent, allows skipping consensus columns.

Transitions: M_j → {M_{j+1}, I_j, D_{j+1}}, I_j → {M_{j+1}, I_j}, D_j
→ {M_{j+1}, D_{j+1}} (plus begin/end tiers). Log-odds scoring against
a null background gives a bit score that's used in HMMER for homology
search.

Training: collect a multiple sequence alignment, assign columns as
match/insert by a heuristic (fraction non-gap > threshold), count
transitions/emissions with pseudocounts (Dirichlet priors).

## 7. Pair HMMs — pairwise alignment (Durbin ch 4)

Three emitting states:
- M — emits (x_i, y_j) — a match / mismatch
- X — emits (x_i, -) — gap in y
- Y — emits (-, y_j) — gap in x

Transition structure: M ↔ X, M ↔ Y, X ↔ X, Y ↔ Y.
- **Viterbi** on pair HMM ≡ Needleman–Wunsch with affine gaps (for
  appropriate emission / transition log-probs).
- **Forward** gives total alignment probability P(x, y).
- **Posterior decoding** → alignment uncertainty at each column.

## 8. Practical considerations across all applications

- **Pseudocounts**: add Laplace (+1) or Dirichlet priors to avoid
  zero-probability parameters during training.
- **Multiple training sequences**: sum the numerators and denominators
  of Baum-Welch re-estimation across sequences, dividing at the end.
- **Parallelism**: forward/backward are easily parallel per time step
  *within* an HMM; Viterbi is too but with an argmax reduction.

---

## Reading list (files in `literature/`)

| File | Role |
|------|------|
| Rabiner_1989_HMM_tutorial.pdf | canonical algorithmic reference |
| Durbin_biological_sequence_analysis.pdf | bioinformatics textbook |
| Tu_BaumWelch_derivation.pdf | clean EM derivation |
| Eddy_profile_HMMs.pdf | profile HMMs for families |
| Glimmer_1998.pdf | interpolated Markov models |
| Kulp_Haussler_1996_GHMM_gene_recognition.pdf | GHMM framework |
| Burge_Karlin_1997_GENSCAN.pdf | canonical eukaryotic gene finder |
| Stanke_2003_new_intron_submodel.pdf | AUGUSTUS intron model |
| Stanke_2006_AUGUSTUS_GHMM_hints.pdf | AUGUSTUS with hints |
