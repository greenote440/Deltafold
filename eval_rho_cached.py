"""
Informative rho evaluation against the pre-computed TM cache (tm_score_analysis.md §4.3).

evaluate_correlation.py samples 30 random proteins -> ~90% of pairs have TM<0.2, so its
rho is dominated by the unrelated-pair noise floor and can't see whether the model ordered
the pairs that matter. This script instead evaluates rho on the cached pairs (which live in
the meaningful TM 0.3-1.0 range) and stratifies by TM band, so a step-function embedding
(high ARI, zero rho) is distinguishable from a true continuous metric.

Good result = strongly NEGATIVE rho (high TM -> low cosine distance) and monotonically
increasing mean distance as the TM band drops.

    python eval_rho_cached.py
    python eval_rho_cached.py --val-only   # honest: only pairs with BOTH proteins in the phylo-val split
"""
import os
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

EMB_FILE = './data/virome_embeddings.pt'
CACHE_FILE = './checkpoints/tm_score_cache.pt'


def load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', default=EMB_FILE)
    ap.add_argument('--cache', default=CACHE_FILE)
    ap.add_argument('--val-only', action='store_true',
                    help="Restrict to pairs whose BOTH proteins are in the phylo-val split (honest generalization rho).")
    args = ap.parse_args()

    emb = load(args.emb)
    cache = load(args.cache)
    emb = {k: np.asarray(v, dtype=np.float64) for k, v in emb.items()}
    print(f"{len(emb)} embeddings | {len(cache)} cached TM pairs")

    val_set = None
    if args.val_only:
        from train import get_phylogenetic_split, PROC_DIR
        _, val_files = get_phylogenetic_split(PROC_DIR, split_ratio=0.8, seed=42, level='taxid')
        val_set = {os.path.basename(f).replace('.pt', '') for f in val_files}
        print(f"val split: {len(val_set)} proteins")

    dists, tms = [], []
    for (a, b), tm in cache.items():
        ka, kb = a.replace('.pt', ''), b.replace('.pt', '')
        if ka not in emb or kb not in emb:
            continue
        if val_set is not None and (ka not in val_set or kb not in val_set):
            continue
        za, zb = emb[ka], emb[kb]
        cos = float(za @ zb / (np.linalg.norm(za) * np.linalg.norm(zb) + 1e-12))
        dists.append(1.0 - cos)   # cosine distance
        tms.append(float(tm))

    dists, tms = np.array(dists), np.array(tms)
    if len(dists) < 5:
        print(f"Only {len(dists)} usable pairs — run extract_embeddings.py first / widen the cache.")
        return

    rho, p = spearmanr(dists, tms)
    print("\n" + "=" * 56)
    print(f"Pairs evaluated: {len(dists)}")
    print(f"Overall Spearman rho(cosine-distance, TM): {rho:+.4f}  (p={p:.2e})")
    print(f"  [want strongly NEGATIVE: high TM -> low distance]")
    print("=" * 56)

    bands = [("TM > 0.7  (close)", tms > 0.7),
             ("0.4-0.7   (distant)", (tms > 0.4) & (tms <= 0.7)),
             ("0.2-0.4   (superficial)", (tms > 0.2) & (tms <= 0.4)),
             ("TM <= 0.2 (unrelated)", tms <= 0.2)]
    print(f"{'band':<26}{'n':>6}{'mean_dist':>11}{'within-band rho':>18}")
    for name, m in bands:
        n = int(m.sum())
        if n == 0:
            print(f"{name:<26}{n:>6}{'-':>11}{'-':>18}")
            continue
        md = dists[m].mean()
        br = spearmanr(dists[m], tms[m])[0] if n >= 5 else float('nan')
        print(f"{name:<26}{n:>6}{md:>11.4f}{br:>18.4f}")
    print("\nMonotonic mean_dist across bands (close<distant<superficial<unrelated) = the model "
          "encodes a continuous metric; flat/!monotonic = step-function (ARI but no rho).")


if __name__ == '__main__':
    main()
