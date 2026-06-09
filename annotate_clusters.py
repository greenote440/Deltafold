"""
Phase 5.2 / 5.4 / 5.5 - Annotate the embedding space and export the master table.

Joins, one row per protein:
  - model cluster      (clusters/model_clusters.tsv, Phase 3)
  - Foldseek cluster   (data/cluster.tsv)
  - genome type + viral family + taxid (cluster_common metadata)
  - UMAP coordinates   (clusters/umap_coords.tsv, Phase 5.1)

5.5 writes clusters/cluster_annotations.tsv (the master table for the report).
5.4 finds cross-genome-type bridge clusters: model clusters whose members span >=2
    genome types - structural conservation across viral lineages - and logs each to
    stdout and clusters/bridge_clusters.txt.

    python annotate_clusters.py
"""
import os
import argparse
from collections import defaultdict

import cluster_common as cc


def integer_codes(rep_of, ids):
    """Map Foldseek representative ids to compact integer codes:
    multi-member clusters -> 0..M, singletons -> unique negatives. Returns
    (code_of_protein, rep_of_protein) restricted to `ids`."""
    sizes = defaultdict(int)
    for p in ids:
        if p in rep_of:
            sizes[rep_of[p]] += 1
    multi = sorted(r for r, n in sizes.items() if n >= 2)
    code = {r: i for i, r in enumerate(multi)}
    out, rep = {}, {}
    nxt = -1
    for p in ids:
        r = rep_of.get(p)
        rep[p] = r if r is not None else p
        if r is not None and sizes[r] >= 2:
            out[p] = code[r]
        else:
            out[p] = nxt
            nxt -= 1
    return out, rep


def load_umap(path):
    coords = {}
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found - run project_umap.py first; umap_x/umap_y will be NA")
        return coords
    with open(path) as f:
        next(f, None)  # header
        for line in f:
            cols = line.rstrip('\n').split('\t')
            if len(cols) >= 3:
                coords[cols[0]] = (cols[1], cols[2])
    return coords


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clusters', default=os.path.join(cc.OUT_DIR, 'model_clusters.tsv'))
    ap.add_argument('--umap', default=os.path.join(cc.OUT_DIR, 'umap_coords.tsv'))
    ap.add_argument('--out', default=os.path.join(cc.OUT_DIR, 'cluster_annotations.tsv'))
    ap.add_argument('--bridge-log', default=os.path.join(cc.OUT_DIR, 'bridge_clusters.txt'))
    args = ap.parse_args()

    cc.ensure_out_dir(os.path.dirname(args.out) or '.')
    model = cc.load_cluster_tsv(args.clusters)
    ids = sorted(model.keys())
    meta = cc.build_metadata(ids)
    rep_of = cc.load_foldseek_clusters()
    fs_code, fs_rep = integer_codes(rep_of, ids)
    umap = load_umap(args.umap)

    # --- 5.5 master annotation table ------------------------------------------
    header = ['protein_id', 'model_cluster_id', 'foldseek_cluster_id', 'genome_type',
              'viral_family', 'taxid', 'umap_x', 'umap_y', 'accession', 'foldseek_rep']
    with open(args.out, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for p in ids:
            ux, uy = umap.get(p, ('NA', 'NA'))
            row = [p, str(model[p]), str(fs_code[p]), meta[p]['genome_type'],
                   meta[p]['family'], meta[p]['taxid'], ux, uy,
                   meta[p]['accession'], fs_rep[p]]
            f.write('\t'.join(row) + '\n')
    print(f"5.5  wrote master table -> {args.out}  ({len(ids)} proteins, {len(header)} cols)")

    # --- 5.4 cross-genome-type bridge clusters --------------------------------
    members = defaultdict(list)
    for p in ids:
        if model[p] >= 0:  # multi-member model clusters only
            members[model[p]].append(p)

    bridges = []
    for cid, mem in members.items():
        gtypes = sorted({meta[p]['genome_type'] for p in mem})
        if len(gtypes) >= 2:
            bridges.append((len(mem), len(gtypes), cid, gtypes, mem))
    bridges.sort(reverse=True)

    lines = []
    lines.append("=" * 70)
    lines.append("5.4  CROSS-GENOME-TYPE BRIDGE CLUSTERS")
    lines.append(f"     {len(bridges)} model clusters span >=2 genome types "
                 f"(of {len(members)} multi-member clusters)")
    lines.append("=" * 70)
    for size, ng, cid, gtypes, mem in bridges:
        fams = sorted({meta[p]['family'] for p in mem})
        lines.append(f"\nmodel cluster #{cid}  | members={size}  | genome types={', '.join(gtypes)}")
        lines.append(f"  families: {', '.join(fams)}")
        for p in mem:
            lines.append(f"    {meta[p]['accession']:<15} {meta[p]['genome_type']:<9} "
                         f"{meta[p]['family']:<18} {cc.parse_protein_id(p)[0][:46]}")

    report = '\n'.join(lines)
    print('\n' + report)
    with open(args.bridge_log, 'w') as f:
        f.write(report + '\n')
    print(f"\n5.4  wrote bridge-cluster log -> {args.bridge_log}")


if __name__ == '__main__':
    main()
