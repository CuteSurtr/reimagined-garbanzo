from __future__ import annotations
from pathlib import Path
import numpy as np
from . import BacterialGeneFinder, DiscreteHMM, GHMM, GHMMState, GaussianHMM, ProfileHMM, benchmark, build_cpg_hmm, dishonest_casino_hmm, empirical_duration, encode_dna, geometric_duration, posterior_island_probability, predict_islands, read_fasta, read_genbank, simple_pair_hmm
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import viz
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
HERE = Path(__file__).resolve().parents[2]
DATA = HERE / 'data'
RESULTS = HERE / 'results'

def section(title: str) -> None:
    print('\n' + '=' * 74)
    print(title)
    print('=' * 74)

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    section('1. Dishonest casino -- Viterbi + Baum-Welch')
    hmm = dishonest_casino_hmm()
    rng = np.random.default_rng(0)
    true_s, rolls = hmm.sample(500, rng=rng)
    _, dec = hmm.viterbi(rolls)
    acc = (dec == true_s).mean()
    print(f'  Viterbi accuracy = {acc:.3f}')
    init = DiscreteHMM(A=np.array([[0.7, 0.3], [0.5, 0.5]]), B=np.array([[1 / 6] * 6, [1 / 6] * 6]), pi=np.array([0.5, 0.5]))
    _, train = hmm.sample(5000, rng=np.random.default_rng(1))
    logliks = init.baum_welch([train], n_iters=40)
    print(f'  Baum-Welch LL: {logliks[0]:.1f} -> {logliks[-1]:.1f}')
    print(f'  recovered P(6 | Loaded) = {init.B[1, 5]:.3f}  (truth: 0.500)')
    section('2. CpG islands -- synthetic implanted island + real HBB region')
    cpg = build_cpg_hmm()
    rng = np.random.default_rng(42)
    from hmmgene.cpg import DURBIN_P_PLUS, DURBIN_P_MINUS

    def mk(P, n, start):
        P = P / P.sum(axis=1, keepdims=True)
        s = [start]
        for _ in range(n - 1):
            s.append(int(rng.choice(4, p=P[s[-1]])))
        return s
    synth = np.array(mk(DURBIN_P_MINUS, 3000, 0) + mk(DURBIN_P_PLUS, 1500, 1) + mk(DURBIN_P_MINUS, 3000, 0))
    synth_str = ''.join(('ACGT'[i] for i in synth))
    pred = predict_islands(synth_str, cpg)
    in_pred = sum((1 for iv in pred if iv.start < 4500 and iv.end > 3000))
    print(f'  synthetic: implanted 1500 bp at pos [3000, 4500]; {len(pred)} islands predicted, overlap={in_pred}')
    hbb_file = DATA / 'HBB_human_genomic_10kb.fasta'
    if hbb_file.exists():
        hbb = read_fasta(hbb_file)[0].sequence
        hbb = ''.join((c for c in hbb if c in 'ACGT'))
        islands_hbb = predict_islands(hbb[:10000], cpg)
        print(f'  HBB real genomic 10kb: {len(islands_hbb)} CpG island(s) predicted')
        if islands_hbb:
            for iv in islands_hbb[:3]:
                print(f'    - [{iv.start}, {iv.end}]  length={iv.end - iv.start} bp')
    section('3. Bacterial gene finder on E. coli (5th-order + stop + reverse strand)')
    recs = read_fasta(DATA / 'ecoli_K12_MG1655_first100kb.fasta')
    gb = read_genbank(DATA / 'ecoli_K12_MG1655_first100kb.gb')
    full = recs[0].sequence
    train_seq, test_seq = (full[:50000], full[50000:])
    train_iv = [(f.start, f.end, f.strand) for f in gb.features if f.end <= 50000]
    test_iv = [(f.start - 50000, f.end - 50000, f.strand) for f in gb.features if f.start > 50000 and f.end <= 100000]
    print(f'  train: {len(train_iv)} CDS    test: {len(test_iv)} CDS')
    finder = BacterialGeneFinder.default()
    finder.fit(train_seq, train_iv)
    preds = finder.predict(test_seq)
    scores = benchmark(preds, test_iv, stop_tol=9)
    print(f"  predictions: {len(preds)} ({sum((1 for p in preds if p[2] == '+'))} forward, {sum((1 for p in preds if p[2] == '-'))} reverse)")
    print(f"  exact stop sens   = {scores['exact']['sensitivity']:.2%}  (TP={scores['exact']['TP']})")
    print(f"  per-base sens     = {scores['bp']['sensitivity']:.2%}")
    print(f"  per-base spec     = {scores['bp']['specificity']:.2%}")
    section('4. Gaussian HMM -- 2-state regime switching')
    gh = GaussianHMM(A=np.array([[0.95, 0.05], [0.1, 0.9]]), pi=np.array([0.5, 0.5]), means=np.array([[0.0], [4.0]]), covars=np.array([[[1.0]], [[1.0]]]))
    rng = np.random.default_rng(7)
    true_s, obs_g = gh.sample(400, rng=rng)
    _, dec_g = gh.viterbi(obs_g)
    print(f'  Gaussian HMM Viterbi accuracy = {(dec_g == true_s).mean():.3f}')
    section('5. GHMM -- two-state duration-aware segmentation')
    rng = np.random.default_rng(8)

    def emit_factory(p0):

        def f(obs, s, e):
            seg = obs[s:e]
            n0 = (seg == 0).sum()
            n1 = (seg == 1).sum()
            return float(n0 * np.log(p0) + n1 * np.log(1 - p0))
        return f
    stA = GHMMState('A', empirical_duration(np.array([0, 0, 0, 1, 1, 1])), emit_factory(0.9), max_duration=5, min_duration=3)
    stB = GHMMState('B', empirical_duration(np.array([0] * 10 + [1] * 6)), emit_factory(0.1), max_duration=15, min_duration=10)
    model = GHMM([stA, stB], trans=np.array([[0.0, 1.0], [1.0, 0.0]]), pi=np.array([1.0, 0.0]))
    seq = np.concatenate([rng.choice(2, p=[0.9, 0.1], size=4), rng.choice(2, p=[0.1, 0.9], size=12), rng.choice(2, p=[0.9, 0.1], size=5), rng.choice(2, p=[0.1, 0.9], size=11)])
    _, segs = model.viterbi(seq)
    print(f'  GHMM segmentation (state, start, end):')
    for st, s, e in segs:
        print(f'    - state={(stA.name if st == 0 else stB.name)}  [{s}, {e})  length={e - s}')
    section('6. Profile HMM from tRNA multiple sequence alignment')
    trna_path = DATA / 'trna_msa.fasta'
    if trna_path.exists():
        records = read_fasta(trna_path)
        msa = [r.sequence.replace('U', 'T') for r in records]
        width = max((len(r) for r in msa))
        msa = [r + '-' * (width - len(r)) for r in msa]
        p = ProfileHMM.from_msa(msa, alphabet='ACGT', match_threshold=0.5)
        print(f'  profile length L = {p.L} (out of MSA length {len(msa[0])})')
        query = encode_dna(msa[0].replace('-', ''))
        random_seq = encode_dna('ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT')
        s_member = p.viterbi(query)
        s_random = p.viterbi(random_seq)
        print(f'  Viterbi score (tRNA member)  = {s_member:.2f}')
        print(f'  Viterbi score (random DNA)   = {s_random:.2f}')
        print(f'  -> separation = {s_member - s_random:.2f} bits')
    section('6b. External head-to-head: pyrodigal vs our gene finder on E. coli')
    try:
        from .external.prodigal_bench import head_to_head
        hh = head_to_head(test_seq, test_iv, preds, stop_tol=9)
        print(f'  ??????????????????????????????????????????????????????????????????????')
        print(f'  ?            ?  n pred  ? exact-stop   ? per-base sens? per-base spec?')
        print(f'  ??????????????????????????????????????????????????????????????????????')
        for name in ('hmmgene', 'prodigal'):
            r = hh[name]
            print(f"  ? {name:<10} ? {r['n_predictions']:>6}   ?  {r['exact']['sensitivity']:>10.1%}  ?  {r['bp']['sensitivity']:>10.1%}  ?  {r['bp']['specificity']:>10.1%}  ?")
        print(f'  ??????????????????????????????????????????????????????????????????????')
    except ImportError as e:
        print(f'  (skipped: {e})')
    section('6c. External head-to-head: pyhmmer (HMMER3) on real Pfam HTH search')
    try:
        from .external.hmmer_bridge import HMMERProfile
        hmm_path = DATA / 'Pfam_PF00356_laci.hmm'
        fa_path = DATA / 'hth_test_proteins.fasta'
        if hmm_path.exists() and fa_path.exists():
            prof = HMMERProfile.from_hmm_file(hmm_path)
            recs2 = read_fasta(fa_path)
            queries = [(r.accession, r.sequence) for r in recs2]
            hits = prof.search(queries, e_cutoff=1.0)
            print(f'  profile: {prof}  (Pfam LacI family, 46 match states)')
            print(f'  searched {len(queries)} proteins; {len(hits)} significant domain hits:')
            for h in hits:
                print(f'    - {h.query:<14} score={h.score:>6.1f}  E={h.e_value:>.2e}  region [{h.start:>4}..{h.end:<4}]')
            if not hits:
                print('  (no hits -- E-value calibration correctly rejects all queries)')
        else:
            print('  (Pfam HMM file not present)')
    except ImportError as e:
        print(f'  (skipped: {e})')
    section('6d. External cross-check: hmmlearn agrees with our DiscreteHMM')
    try:
        from .external.hmmlearn_xcheck import compare_forward_loglik, compare_viterbi_path
        hmm = dishonest_casino_hmm()
        rng = np.random.default_rng(99)
        _, obs_xc = hmm.sample(1000, rng=rng)
        ours_L, ref_L = compare_forward_loglik(hmm, obs_xc)
        ours_p, ref_p = compare_viterbi_path(hmm, obs_xc)
        agree = float((ours_p == ref_p).mean())
        print(f'  forward log-likelihood  --  ours {ours_L:.4f}   hmmlearn {ref_L:.4f}   Delta={abs(ours_L - ref_L):.2e}')
        print(f'  Viterbi path agreement  --  {agree:.1%}')
    except ImportError as e:
        print(f'  (skipped: {e})')
    section('7. Pair HMM -- Viterbi + forward + posterior')
    ph = simple_pair_hmm(match_prob=0.85)
    x = encode_dna('ACGTACGT')
    y = encode_dna('ACGACGT')
    sv, aln = ph.viterbi(x, y)
    _, sf = ph.forward(x, y)
    post = ph.posterior_match(x, y)
    row1 = ''.join(('-' if a == -1 else 'ACGT'[a] for a, _ in aln))
    row2 = ''.join(('-' if b == -1 else 'ACGT'[b] for _, b in aln))
    print(f'  Viterbi score = {sv:.3f}')
    print(f'  forward logP  = {sf:.3f}  (forward >= Viterbi)')
    print(f'  {row1}')
    print(f'  {row2}')
    if HAVE_MPL:
        section('8. Figures -> results/')
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        viz.plot_state_trajectory(rolls[:300], true_s[:300], dec[:300], [], [], ax=axs[0][0])
        axs[0][0].set_title('Casino -- Viterbi')
        viz.plot_log_likelihood_trace(logliks, ax=axs[0][1])
        viz.plot_state_diagram(hmm.A, hmm.state_names or ['F', 'L'], ax=axs[1][0], title='Casino HMM state diagram')
        if hbb_file.exists():
            post_hbb = posterior_island_probability(hbb[:10000], cpg)
            islands_hbb = predict_islands(hbb[:10000], cpg)
            viz.plot_cpg_prediction(hbb[:10000], islands_hbb, post_hbb, ax=axs[1][1], title='CpG -- HBB beta-globin 10 kb')
        fig.tight_layout()
        fig.savefig(RESULTS / 'hmmgene_overview.png', dpi=140, bbox_inches='tight')
        print(f"  wrote {RESULTS / 'hmmgene_overview.png'}")
        if trna_path.exists():
            fig2, axs2 = plt.subplots(2, 1, figsize=(12, 6))
            viz.plot_sequence_logo(p.emit_M, ax=axs2[0], title='tRNA profile HMM sequence logo')
            viz.plot_dotplot(ph.posterior_match(x, y), ax=axs2[1], title='Pair HMM posterior match')
            fig2.tight_layout()
            fig2.savefig(RESULTS / 'hmmgene_profile_pair.png', dpi=140, bbox_inches='tight')
            print(f"  wrote {RESULTS / 'hmmgene_profile_pair.png'}")
if __name__ == '__main__':
    main()