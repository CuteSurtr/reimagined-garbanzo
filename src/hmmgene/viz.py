from __future__ import annotations
from typing import List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np

def plot_state_trajectory(observations: Sequence[int], true_states: Optional[Sequence[int]], decoded_states: Sequence[int], symbol_names: List[str], state_names: List[str], ax: Optional[plt.Axes]=None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    T = len(observations)
    x = np.arange(T)
    ax.step(x, np.array(observations, dtype=float) + 2.5, color='black', linewidth=0.7)
    if true_states is not None:
        ax.plot(x, np.array(true_states, dtype=float) + 1.0, color='tab:blue', linewidth=1.0, label='true state')
    ax.plot(x, np.array(decoded_states, dtype=float), color='tab:red', linewidth=1.0, label='Viterbi')
    ax.set_xlabel('t')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax

def plot_posteriors(posterior: np.ndarray, ax: Optional[plt.Axes]=None, title: str='') -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.fill_between(np.arange(len(posterior)), posterior, color='tab:purple', alpha=0.5)
    ax.set_ylabel('P(island)')
    ax.set_xlabel('position (bp)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    return ax

def plot_transition_matrix(A: np.ndarray, state_names: Sequence[str], ax: Optional[plt.Axes]=None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(state_names)))
    ax.set_yticks(range(len(state_names)))
    ax.set_xticklabels(state_names, rotation=60, ha='right', fontsize=8)
    ax.set_yticklabels(state_names, fontsize=8)
    ax.set_title('Transition matrix')
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax

def plot_log_likelihood_trace(logliks: Sequence[float], ax: Optional[plt.Axes]=None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(logliks, color='tab:green', linewidth=1.5)
    ax.set_xlabel('EM iteration')
    ax.set_ylabel('log-likelihood')
    ax.set_title('Baum-Welch convergence')
    ax.grid(alpha=0.3)
    return ax

def plot_sequence_logo(emit: np.ndarray, alphabet: str='ACGT', ax=None, title: str='Sequence logo'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, emit.shape[0] * 0.3), 2.5))
    L, A = emit.shape
    colors = {'A': 'tab:green', 'C': 'tab:blue', 'G': 'tab:orange', 'T': 'tab:red', 'U': 'tab:red'}
    for j in range(L):
        p = emit[j]
        H = -sum((pi * np.log2(pi) for pi in p if pi > 0))
        info = max(0.0, 2.0 - H)
        order = np.argsort(p)
        y = 0.0
        for b_idx in order:
            letter = alphabet[b_idx]
            h = float(p[b_idx] * info)
            if h > 0.01:
                ax.text(j, y + h / 2, letter, ha='center', va='center', fontsize=max(6, h * 12), color=colors.get(letter, 'black'), fontweight='bold')
            y += h
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(0, 2)
    ax.set_ylabel('bits')
    ax.set_xlabel('column')
    ax.set_title(title)
    ax.set_xticks(range(L))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    return ax

def plot_dotplot(posterior_match: np.ndarray, x_label: str='x', y_label: str='y', ax=None, title: str='Pair HMM posterior match'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(posterior_match, cmap='magma_r', origin='lower', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel(f'position in {y_label}')
    ax.set_ylabel(f'position in {x_label}')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, label='P(match)')
    return ax

def plot_state_diagram(A: np.ndarray, state_names, ax=None, title: str='HMM state diagram', threshold: float=0.01):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    N = A.shape[0]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    for i in range(N):
        ax.plot(*pos[i], 'o', color='tab:red', markersize=14, zorder=3)
        ax.text(pos[i, 0] * 1.18, pos[i, 1] * 1.18, state_names[i], ha='center', va='center', fontsize=9, zorder=4)
    for i in range(N):
        for j in range(N):
            if A[i, j] < threshold:
                continue
            alpha = min(1.0, A[i, j] * 2)
            if i == j:
                ax.annotate('', xy=(pos[i, 0] * 1.1, pos[i, 1] * 1.1), xytext=(pos[i, 0] * 1.25, pos[i, 1] * 1.25), arrowprops=dict(arrowstyle='->', color='tab:blue', alpha=alpha))
            else:
                ax.annotate('', xy=pos[j] * 0.92, xytext=pos[i] * 0.92, arrowprops=dict(arrowstyle='->', color='gray', alpha=alpha, lw=1 + A[i, j] * 3))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title)
    return ax

def plot_cpg_prediction(sequence: str, island_intervals, posterior: np.ndarray, ax: Optional[plt.Axes]=None, title: str='CpG islands') -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))
    xs = np.arange(len(sequence))
    ax.fill_between(xs, posterior, color='tab:purple', alpha=0.6, label='P(island)')
    for iv in island_intervals:
        ax.axvspan(iv.start, iv.end, color='tab:orange', alpha=0.2)
    ax.set_ylabel('P(island)')
    ax.set_xlabel('position (bp)')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    return ax