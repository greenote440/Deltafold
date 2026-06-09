"""
Phase 4 - Biological validation ("easter eggs").

4.0 Inventory: keyword-search the dataset for the paper's cross-homology test cases
    (cGAMP/Acb1, ENT4 mimicry, poxvirus I3L / T7 SSB OB-fold). filter_dataset.py
    stratified on some of these keywords, so several should be present.
4.0 Substitutes: when a host/virus pair can't be formed inside a virus-only set,
    build substitute cases from the TM cache: pairs in DIFFERENT Foldseek clusters
    with a high TM-score (genuine structural homology the binary clustering missed).
4.1 Pairwise lookup: cosine distance, same/different Foldseek cluster, cached TM,
    and whether the model puts the pair in the "close" band (distance < 0.45).
4.2 Nearest-neighbour retrieval: 10 closest proteins for each test protein, with
    accession, Foldseek cluster, genome type and annotation.

    python validate_test_cases.py
    python validate_test_cases.py --pair <idA> <idB> --query <id>
"""
import os
import argparse
from collections import defaultdict

import numpy as np

import cluster_common as cc

TARGET_CASES = {
    'cGAMP / Acb1 (anti-immunity)': ['acb1', 'cgamp', 'poxin'],
    'ENT4 mimicry (nucleoside transporter)': ['ent4', 'equilibrative', 'nucleoside transporter'],
    'I3L / T7 SSB (OB-fold ssDNA-binding)': ['i3l', 'ssb', 'single-stranded dna',
                                             'single stranded dna', 'ob-fold', 'ob fold'],
}


def short(pid, width=46):
    name = cc.parse_protein_id(pid)[0]
    return (name[:width - 1] + '…') if len(name) > width else name


def inventory(ids, meta):
    print("=" * 70)
    print("4.0  INVENTORY of target test cases (keyword search on protein name)")
    print("=" * 70)
    hits = {}
    for case, kws in TARGET_CASES.items():
        found = [p for p in ids if any(k in meta[p]['name'].lower() for k in kws)]
        hits[case] = found
        print(f"\n{case}: {len(found)} match(es)")
        for p in found[:12]:
            print(f"   {short(p):<48} {meta[p]['family']:<18} {meta[p]['genome_type']}")
        if len(found) > 12:
            print(f"   ... and {len(found) - 12} more")
    return hits


def target_pairs(hits):
    """From the inventory, any test case with >=2 annotated hits yields putative
    homolog pairs (e.g. poxvirus I3L vs phage/baculovirus SSB - both OB-fold)."""
    out = []
    for case, found in hits.items():
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                out.append((case, found[i], found[j]))
    return out


def substitute_pairs(ids, fs_rep, tm_cache, meta, idset, k=5, min_tm=0.5,
                     same_genome=True):
    """Substitute validation cases: pairs in DIFFERENT Foldseek clusters with a high
    cached TM = genuine structural homology the binary clustering split apart.

    Defaults to SAME genome type and dedupes proteins across pairs. Same-genome is the
    credible-homology filter: TM-score is length-normalised, so the very highest
    cross-genome cross-cluster TMs tend to be short-protein alignment artifacts
    (e.g. a 240aa poxvirus protein scoring "TM 0.9" against a 70aa arterivirus ORF)."""
    cands = []
    for pair, tm in tm_cache.items():
        a, b = tuple(pair)
        if a not in idset or b not in idset:
            continue
        if fs_rep.get(a) == fs_rep.get(b):
            continue  # want DIFFERENT foldseek clusters
        if same_genome and meta[a]['genome_type'] != meta[b]['genome_type']:
            continue
        if tm >= min_tm:
            cands.append((tm, a, b))
    cands.sort(reverse=True)
    picked, used = [], set()
    for tm, a, b in cands:
        if a in used or b in used:
            continue  # one homology case per protein -> k distinct cases
        picked.append((tm, a, b))
        used.update((a, b))
        if len(picked) >= k:
            break
    return picked


def report_pair(a, b, idx, X, fs_rep, tm_cache, meta):
    za, zb = X[idx[a]], X[idx[b]]
    dist = 1.0 - float(za @ zb)
    same = fs_rep.get(a) == fs_rep.get(b)
    tm = cc.tm_lookup(tm_cache, a, b)
    close = dist < cc.CLOSE_THRESHOLD
    print(f"\n  A: {short(a)}  [{meta[a]['family']}, {meta[a]['genome_type']}]")
    print(f"  B: {short(b)}  [{meta[b]['family']}, {meta[b]['genome_type']}]")
    print(f"     cosine distance     : {dist:.4f}")
    print(f"     Foldseek clusters   : {'SAME' if same else 'DIFFERENT'}")
    print(f"     cached TM-score      : {tm:.4f}" if tm is not None else
          "     cached TM-score      : (not cached - alignment needed)")
    verdict = "CLOSE (< 0.45)" if close else "far (>= 0.45)"
    flag = "  <-- homology recovered across Foldseek clusters" if (close and not same) else ""
    print(f"     model band          : {verdict}{flag}")
    return close, same


def nn_retrieval(query, idx, ids, X, fs_rep, meta, k=10):
    qi = idx[query]
    # Compute only the query's distance ROW on the fly: 1 - X @ X[qi]. This avoids
    # materialising the dense (N,N) matrix, which is infeasible at full-dataset scale
    # (67k -> ~36GB). One matvec is O(N*dim) and trivially cheap per query.
    d_row = 1.0 - (X @ X[qi])
    order = np.argsort(d_row)
    print(f"\n  Query: {short(query)}  [{meta[query]['family']}, {meta[query]['genome_type']}, "
          f"foldseek={cc.parse_protein_id(fs_rep.get(query, '?'))[0][:24]}]")
    print(f"    {'rank':<5}{'dist':<8}{'accession':<15}{'foldseek':<26}{'genome':<10}annotation")
    shown = 0
    for j in order:
        nid = ids[j]
        if nid == query:
            continue
        shown += 1
        fs = cc.parse_protein_id(fs_rep.get(nid, '?'))[0][:24]
        print(f"    {shown:<5}{d_row[j]:<8.4f}{meta[nid]['accession']:<15}{fs:<26}"
              f"{meta[nid]['genome_type']:<10}{short(nid, 40)}")
        if shown >= k:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb', default=cc.EMB_FILE)
    ap.add_argument('--pair', nargs=2, metavar=('A', 'B'), action='append',
                    help="Explicit test pair (repeatable). Overrides auto substitutes.")
    ap.add_argument('--query', action='append', help="Explicit NN query id (repeatable).")
    ap.add_argument('--n-substitutes', type=int, default=5)
    ap.add_argument('--min-tm', type=float, default=0.5)
    ap.add_argument('--cross-genome', action='store_true',
                    help="Allow cross-genome-type substitute pairs (off by default; the "
                         "highest cross-genome TMs are often short-protein artifacts).")
    ap.add_argument('--k', type=int, default=10, help="Neighbours per query.")
    args = ap.parse_args()

    ids, X = cc.load_embeddings(args.emb)
    idset = set(ids)
    idx = {p: i for i, p in enumerate(ids)}
    meta = cc.build_metadata(ids)
    fs_rep = cc.load_foldseek_clusters()
    tm_cache = cc.load_tm_cache()

    hits = inventory(ids, meta)

    # Choose test pairs.
    print("\n" + "=" * 70)
    print("4.1  PAIRWISE DISTANCE LOOKUP")
    print("=" * 70)
    pairs = []
    if args.pair:
        for a, b in args.pair:
            if a in idset and b in idset:
                report_pair(a, b, idx, X, fs_rep, tm_cache, meta)
                pairs.append((a, b))
            else:
                print(f"  skip (id not found): {a} | {b}")
    else:
        # (a) genuine target-case pairs found by the inventory
        tpairs = target_pairs(hits)
        if tpairs:
            print(f"\nTarget-case pairs from inventory ({len(tpairs)}): putative homologs by annotation.")
            for case, a, b in tpairs:
                print(f"\n[{case}]")
                report_pair(a, b, idx, X, fs_rep, tm_cache, meta)
            pairs += [(a, b) for _, a, b in tpairs]
        # (b) substitute cases from the TM cache
        subs = substitute_pairs(ids, fs_rep, tm_cache, meta, idset,
                                k=args.n_substitutes, min_tm=args.min_tm,
                                same_genome=not args.cross_genome)
        scope = "cross-genome allowed" if args.cross_genome else "same genome type"
        print(f"\nSubstitute cases ({len(subs)}): different Foldseek clusters, TM >= "
              f"{args.min_tm}, {scope}.")
        for _, a, b in subs:
            report_pair(a, b, idx, X, fs_rep, tm_cache, meta)
        pairs += [(a, b) for _, a, b in subs]

    recovered = sum(1 for a, b in pairs
                    if (1.0 - float(X[idx[a]] @ X[idx[b]])) < cc.CLOSE_THRESHOLD
                    and fs_rep.get(a) != fs_rep.get(b))
    cross = sum(1 for a, b in pairs if fs_rep.get(a) != fs_rep.get(b))
    if pairs:
        print(f"\n  => cross-Foldseek-cluster homologies placed in the close band "
              f"(< {cc.CLOSE_THRESHOLD}): {recovered}/{cross}")

    # NN retrieval.
    print("\n" + "=" * 70)
    print("4.2  NEAREST-NEIGHBOUR RETRIEVAL")
    print("=" * 70)
    if args.query:
        queries = [q for q in args.query if q in idset]
    else:
        # query the members of the chosen pairs plus any keyword hits
        queries = []
        for a, b in pairs:
            queries += [a, b]
        for found in hits.values():
            queries += found[:2]
        # de-dup preserving order, cap
        seen, q2 = set(), []
        for q in queries:
            if q not in seen:
                seen.add(q)
                q2.append(q)
        queries = q2[:6]
    for q in queries:
        nn_retrieval(q, idx, ids, X, fs_rep, meta, k=args.k)


if __name__ == '__main__':
    main()
