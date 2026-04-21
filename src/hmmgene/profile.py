from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from .logmath import NEG_INF, log0

@dataclass
class ProfileHMM:
    emit_M: np.ndarray
    emit_I: np.ndarray
    trans_mm: np.ndarray
    trans_mi: np.ndarray
    trans_md: np.ndarray
    trans_ii: np.ndarray
    trans_im: np.ndarray
    trans_dm: np.ndarray
    trans_dd: np.ndarray
    background: np.ndarray

    @property
    def L(self) -> int:
        return self.emit_M.shape[0]

    @classmethod
    def from_msa(cls, msa: Sequence[str], alphabet: str='ACGT', match_threshold: float=0.5, pseudo: float=1.0) -> 'ProfileHMM':
        sym_idx = {s: i for i, s in enumerate(alphabet)}
        A = len(alphabet)
        ncol = len(msa[0])
        nrow = len(msa)
        is_match = np.zeros(ncol, dtype=bool)
        for c in range(ncol):
            column = [row[c] for row in msa]
            nongap = sum((1 for ch in column if ch != '-'))
            is_match[c] = nongap / nrow >= match_threshold
        match_cols = np.where(is_match)[0]
        L = len(match_cols)
        if L == 0:
            raise ValueError('no match columns in alignment')
        background = np.ones(A) / A
        emit_M = np.full((L, A), pseudo)
        emit_I = np.full((L, A), pseudo)
        for mc_idx, c in enumerate(match_cols):
            for row in msa:
                ch = row[c]
                if ch in sym_idx:
                    emit_M[mc_idx, sym_idx[ch]] += 1
        insert_of = -np.ones(ncol, dtype=int)
        j = 0
        for c in range(ncol):
            if is_match[c]:
                j = match_cols.tolist().index(c)
            else:
                insert_of[c] = j
        for c in range(ncol):
            if not is_match[c] and insert_of[c] >= 0:
                for row in msa:
                    ch = row[c]
                    if ch in sym_idx:
                        emit_I[insert_of[c], sym_idx[ch]] += 1
        emit_M /= emit_M.sum(axis=1, keepdims=True)
        emit_I /= emit_I.sum(axis=1, keepdims=True)
        tcount = {k: np.full(L, pseudo) for k in ['mm', 'mi', 'md', 'ii', 'im', 'dm', 'dd']}
        for row in msa:
            for mc_idx in range(L - 1):
                c_j = match_cols[mc_idx]
                c_next = match_cols[mc_idx + 1]
                prev_ch = row[c_j]
                between = row[c_j + 1:c_next]
                next_ch = row[c_next]
                inserted = sum((1 for ch in between if ch != '-'))
                cur_match = prev_ch != '-'
                nxt_match = next_ch != '-'
                if cur_match and inserted == 0 and nxt_match:
                    tcount['mm'][mc_idx] += 1
                elif cur_match and inserted > 0:
                    tcount['mi'][mc_idx] += 1
                    if inserted > 1:
                        tcount['ii'][mc_idx] += inserted - 1
                    tcount['im'][mc_idx] += 1 if nxt_match else 0
                elif cur_match and (not nxt_match):
                    tcount['md'][mc_idx] += 1
                elif not cur_match and nxt_match:
                    tcount['dm'][mc_idx] += 1
                elif not cur_match and (not nxt_match):
                    tcount['dd'][mc_idx] += 1
        m_total = tcount['mm'] + tcount['mi'] + tcount['md']
        i_total = tcount['ii'] + tcount['im']
        d_total = tcount['dm'] + tcount['dd']
        trans_mm = tcount['mm'] / np.maximum(m_total, 1e-09)
        trans_mi = tcount['mi'] / np.maximum(m_total, 1e-09)
        trans_md = tcount['md'] / np.maximum(m_total, 1e-09)
        trans_ii = tcount['ii'] / np.maximum(i_total, 1e-09)
        trans_im = tcount['im'] / np.maximum(i_total, 1e-09)
        trans_dm = tcount['dm'] / np.maximum(d_total, 1e-09)
        trans_dd = tcount['dd'] / np.maximum(d_total, 1e-09)
        return cls(emit_M=emit_M, emit_I=emit_I, trans_mm=trans_mm, trans_mi=trans_mi, trans_md=trans_md, trans_ii=trans_ii, trans_im=trans_im, trans_dm=trans_dm, trans_dd=trans_dd, background=background)

    def viterbi(self, obs: np.ndarray, return_alignment: bool=False):
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        L = self.L
        bg = log0(self.background)
        logM_emit = log0(self.emit_M)
        logI_emit = log0(self.emit_I)
        score_M = logM_emit - bg
        score_I = logI_emit - bg
        with np.errstate(divide='ignore'):
            l_mm = np.log(self.trans_mm + 1e-12)
            l_mi = np.log(self.trans_mi + 1e-12)
            l_md = np.log(self.trans_md + 1e-12)
            l_ii = np.log(self.trans_ii + 1e-12)
            l_im = np.log(self.trans_im + 1e-12)
            l_dm = np.log(self.trans_dm + 1e-12)
            l_dd = np.log(self.trans_dd + 1e-12)
        V_M = np.full((T + 1, L), NEG_INF)
        V_I = np.full((T + 1, L), NEG_INF)
        V_D = np.full((T + 1, L), NEG_INF)
        BP_M = np.full((T + 1, L, 3), -1, dtype=int)
        BP_I = np.full((T + 1, L, 3), -1, dtype=int)
        BP_D = np.full((T + 1, L, 3), -1, dtype=int)
        V_M[0, 0] = 0.0
        for t in range(1, T + 1):
            o = obs[t - 1]
            for j in range(L):
                m_from = V_M[t - 1, j - 1] + l_mm[j - 1] if j > 0 else NEG_INF
                i_from = V_I[t - 1, j - 1] + l_im[j - 1] if j > 0 else NEG_INF
                d_from = V_D[t - 1, j - 1] + l_dm[j - 1] if j > 0 else NEG_INF
                if j > 0:
                    choices = [m_from, i_from, d_from]
                    best = int(np.argmax(choices))
                    V_M[t, j] = choices[best] + score_M[j, o]
                    BP_M[t, j] = [best, t - 1, j - 1]
                elif t == 1:
                    V_M[t, j] = 0.0 + score_M[j, o]
                    BP_M[t, j] = [-1, 0, 0]
                m_to_i = V_M[t - 1, j] + l_mi[j]
                i_to_i = V_I[t - 1, j] + l_ii[j]
                choices = [m_to_i, i_to_i]
                best = int(np.argmax(choices))
                V_I[t, j] = choices[best] + score_I[j, o]
                BP_I[t, j] = [best, t - 1, j]
            for j in range(1, L):
                m_to_d = V_M[t, j - 1] + l_md[j - 1]
                d_to_d = V_D[t, j - 1] + l_dd[j - 1]
                choices = [m_to_d, d_to_d]
                best = int(np.argmax(choices))
                V_D[t, j] = choices[best] + l_md[0] * 0
                V_D[t, j] = choices[best]
                BP_D[t, j] = [best * 2, t, j - 1]
        end_scores = [V_M[T, L - 1], V_I[T, L - 1], V_D[T, L - 1]]
        end_state = int(np.argmax(end_scores))
        score = float(end_scores[end_state])
        if not return_alignment:
            return score
        path = []
        state = end_state
        t, j = (T, L - 1)
        while not (t == 0 and j == 0 and (state == 0)):
            if state == 0:
                prev_code, pt, pj = BP_M[t, j]
                path.append(('M', j, t - 1))
                state = prev_code if prev_code >= 0 else 0
                t, j = (pt, pj)
                if state == -1:
                    break
            elif state == 1:
                prev_code, pt, pj = BP_I[t, j]
                path.append(('I', j, t - 1))
                state = prev_code
                t, j = (pt, pj)
            else:
                prev_code, pt, pj = BP_D[t, j]
                path.append(('D', j, -1))
                state = prev_code // 2 * 2
                if state == 2:
                    state = 2
                else:
                    state = 0
                t, j = (pt, pj)
            if t < 0 or j < 0:
                break
        path.reverse()
        return (score, path)