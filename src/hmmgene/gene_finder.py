from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .cpg import BASES, BASE_IDX, encode_dna
from .hmm import DiscreteHMM
from .logmath import NEG_INF
STATE_NAMES = ['INTERGENIC', 'START_A', 'START_T', 'START_G', 'CP0', 'CP1', 'CP2', 'STOP_T', 'STOP_2', 'STOP_3']
N_STATES = len(STATE_NAMES)
INTERGENIC, START_A, START_T, START_G, CP0, CP1, CP2, STOP_T, STOP_2, STOP_3 = range(N_STATES)
_COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')

def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]

@dataclass
class CodingMarkov5:
    cond: np.ndarray
    fallback: List[np.ndarray] = field(default_factory=list)

    @classmethod
    def train(cls, seq_enc: np.ndarray, phase_of: np.ndarray, pseudo: float=1.0) -> 'CodingMarkov5':
        fallback = []
        for k in range(6):
            dim = 4 ** k
            counts = np.full((3, dim, 4), pseudo)
            for t in range(k, len(seq_enc)):
                p = phase_of[t]
                if p < 0:
                    continue
                if k == 0:
                    ctx = 0
                else:
                    ctx = 0
                    for i in range(k):
                        b = seq_enc[t - k + i]
                        if b < 0 or b > 3:
                            ctx = -1
                            break
                        ctx = ctx * 4 + b
                    if ctx < 0:
                        continue
                target = seq_enc[t]
                if target < 0 or target > 3:
                    continue
                counts[p, ctx, target] += 1
            denom = counts.sum(axis=2, keepdims=True)
            fallback.append(counts / denom)
        return cls(cond=fallback[5], fallback=fallback)

    def log_cond(self, seq_enc: np.ndarray, t: int, phase: int) -> float:
        target = seq_enc[t]
        if target < 0 or target > 3:
            return np.log(0.25)
        k = min(5, t)
        ctx = 0
        for i in range(k):
            b = seq_enc[t - k + i]
            if b < 0 or b > 3:
                k = i
                break
            ctx = ctx * 4 + b
        cond = self.fallback[k]
        return float(np.log(cond[phase, ctx, target] + 1e-300))
STOP_CODONS = ('TAA', 'TAG', 'TGA')

@dataclass
class BacterialGeneFinder:
    A: np.ndarray
    pi: np.ndarray
    intergenic_log: np.ndarray
    stop_codon_logprior: np.ndarray
    coding: Optional[CodingMarkov5] = None
    min_gene_length: int = 90

    @classmethod
    def default(cls, p_start: float=0.001, p_stop_per_codon: float=1 / 333) -> 'BacterialGeneFinder':
        A = np.zeros((N_STATES, N_STATES))
        A[INTERGENIC, INTERGENIC] = 1 - p_start
        A[INTERGENIC, START_A] = p_start
        A[START_A, START_T] = 1.0
        A[START_T, START_G] = 1.0
        A[START_G, CP0] = 1.0
        A[CP0, CP1] = 1.0
        A[CP1, CP2] = 1.0
        A[CP2, CP0] = 1 - p_stop_per_codon
        A[CP2, STOP_T] = p_stop_per_codon
        A[STOP_T, STOP_2] = 1.0
        A[STOP_2, STOP_3] = 1.0
        A[STOP_3, INTERGENIC] = 1.0
        pi = np.zeros(N_STATES)
        pi[INTERGENIC] = 1.0
        intergenic_log = np.log(np.array([0.25, 0.25, 0.25, 0.25]))
        stop_prior = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
        return cls(A=A, pi=pi, intergenic_log=intergenic_log, stop_codon_logprior=stop_prior, coding=None)

    def fit(self, sequence: str, gene_intervals: List[Tuple[int, int, str]]) -> None:
        enc = encode_dna(sequence)
        rc_enc = encode_dna(revcomp(sequence))
        L = len(enc)
        phase_fwd = np.full(L, -1, dtype=int)
        phase_rc = np.full(L, -1, dtype=int)
        for start, end, strand in gene_intervals:
            s = start - 1
            e = end
            if e - s < 9:
                continue
            if strand == '+':
                for i in range(s + 3, e - 3):
                    phase_fwd[i] = (i - s) % 3
            else:
                for i in range(s + 3, e - 3):
                    rc_pos = L - 1 - i
                    phase_rc[rc_pos] = (e - 1 - i) % 3
        concat_enc = np.concatenate([enc, rc_enc])
        concat_phase = np.concatenate([phase_fwd, phase_rc])
        self.coding = CodingMarkov5.train(concat_enc, concat_phase)
        inter_mask = (phase_fwd < 0) & np.full(L, True)
        counts = np.bincount(enc[inter_mask], minlength=4).astype(float) + 1.0
        self.intergenic_log = np.log(counts / counts.sum())
        stop_counts = np.array([1.0, 1.0, 1.0])
        for start, end, strand in gene_intervals:
            if strand != '+':
                continue
            if end - start + 1 < 3:
                continue
            codon = sequence[end - 3:end]
            if codon in STOP_CODONS:
                stop_counts[STOP_CODONS.index(codon)] += 1
        self.stop_codon_logprior = np.log(stop_counts / stop_counts.sum())

    def _log_emit(self, enc: np.ndarray, t: int, state: int) -> float:
        o = enc[t]
        if o < 0 or o > 3:
            return np.log(0.25)
        if state == INTERGENIC:
            return float(self.intergenic_log[o])
        if state == START_A:
            return 0.0 if o == 0 else NEG_INF
        if state == START_T:
            return 0.0 if o == 3 else NEG_INF
        if state == START_G:
            return 0.0 if o == 2 else NEG_INF
        if state in (CP0, CP1, CP2):
            phase = state - CP0
            return self.coding.log_cond(enc, t, phase) if self.coding else np.log(0.25)
        if state == STOP_T:
            return 0.0 if o == 3 else NEG_INF
        if state == STOP_2:
            p_a = float(np.exp(self.stop_codon_logprior[0]) + np.exp(self.stop_codon_logprior[1]))
            p_g = float(np.exp(self.stop_codon_logprior[2]))
            if o == 0:
                return float(np.log(max(p_a, 1e-300)))
            if o == 2:
                return float(np.log(max(p_g, 1e-300)))
            return NEG_INF
        if state == STOP_3:
            p_a = float(np.exp(self.stop_codon_logprior[0]) + np.exp(self.stop_codon_logprior[2]))
            p_g = float(np.exp(self.stop_codon_logprior[1]))
            if o == 0:
                return float(np.log(max(p_a, 1e-300)))
            if o == 2:
                return float(np.log(max(p_g, 1e-300)))
            return NEG_INF
        raise ValueError(f'unknown state {state}')

    def _viterbi(self, enc: np.ndarray) -> Tuple[float, np.ndarray]:
        T = len(enc)
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        delta = np.full((T, N_STATES), NEG_INF)
        psi = np.zeros((T, N_STATES), dtype=int)
        for s in range(N_STATES):
            delta[0, s] = logpi[s] + self._log_emit(enc, 0, s)
        for t in range(1, T):
            emit_t = np.array([self._log_emit(enc, t, s) for s in range(N_STATES)])
            scores = delta[t - 1][:, None] + logA
            best_from = np.argmax(scores, axis=0)
            delta[t] = scores[best_from, np.arange(N_STATES)] + emit_t
            psi[t] = best_from
        log_p = float(np.max(delta[-1]))
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return (log_p, path)

    def predict(self, sequence: str) -> List[Tuple[int, int, str]]:
        enc_fwd = encode_dna(sequence)
        enc_rc = encode_dna(revcomp(sequence))
        L = len(sequence)
        preds: List[Tuple[int, int, str]] = []
        for enc, strand in [(enc_fwd, '+'), (enc_rc, '-')]:
            _, path = self._viterbi(enc)
            in_gene = path > 0
            i = 0
            while i < L:
                if in_gene[i]:
                    start = i
                    while i < L and in_gene[i]:
                        i += 1
                    end = i
                    if end - start >= self.min_gene_length:
                        if strand == '+':
                            preds.append((start + 1, end, '+'))
                        else:
                            fwd_start = L - end + 1
                            fwd_end = L - start
                            preds.append((fwd_start, fwd_end, '-'))
                else:
                    i += 1
        return preds

def benchmark(predicted: List[Tuple[int, int, str]], truth: List[Tuple[int, int, str]], stop_tol: int=0) -> Dict:

    def _norm(items):
        out = []
        for it in items:
            if len(it) == 2:
                out.append((it[0], it[1], '+'))
            else:
                out.append(tuple(it))
        return out
    pred = _norm(predicted)
    truth = _norm(truth)

    def stop_of(x):
        s, e, strand = x
        return e if strand == '+' else s
    truth_stops = {stop_of(t): t[2] for t in truth}
    pred_stops = {stop_of(p): p[2] for p in pred}
    tp_exact = sum((1 for s, st in pred_stops.items() if s in truth_stops and truth_stops[s] == st))
    fp_exact = len(pred_stops) - tp_exact
    fn_exact = len(truth_stops) - tp_exact
    truth_list = [(stop_of(t), t[2]) for t in truth]
    matched = set()
    tp_tol = 0
    for ps, st in pred_stops.items():
        for k, (ts, tst) in enumerate(truth_list):
            if k in matched:
                continue
            if abs(ps - ts) <= stop_tol and st == tst:
                matched.add(k)
                tp_tol += 1
                break
    fp_tol = len(pred_stops) - tp_tol
    fn_tol = len(truth_stops) - tp_tol
    if truth or pred:
        L = max(max((t[1] for t in truth), default=0), max((p[1] for p in pred), default=0)) + 1
    else:
        L = 0
    pm = [0] * L
    tm = [0] * L
    for s, e, _ in pred:
        for i in range(s - 1, min(e, L)):
            pm[i] = 1
    for s, e, _ in truth:
        for i in range(s - 1, min(e, L)):
            tm[i] = 1
    tp_bp = sum((a & b for a, b in zip(pm, tm)))
    fp_bp = sum((a & 1 - b for a, b in zip(pm, tm)))
    fn_bp = sum((1 - a & b for a, b in zip(pm, tm)))
    bp_sens = tp_bp / (tp_bp + fn_bp) if tp_bp + fn_bp else 0.0
    bp_spec = tp_bp / (tp_bp + fp_bp) if tp_bp + fp_bp else 0.0
    return {'exact': {'TP': tp_exact, 'FP': fp_exact, 'FN': fn_exact, 'sensitivity': tp_exact / max(tp_exact + fn_exact, 1)}, 'stop_tol': {'tol': stop_tol, 'TP': tp_tol, 'FP': fp_tol, 'FN': fn_tol, 'sensitivity': tp_tol / max(tp_tol + fn_tol, 1)}, 'bp': {'TP': tp_bp, 'FP': fp_bp, 'FN': fn_bp, 'sensitivity': bp_sens, 'specificity': bp_spec}}