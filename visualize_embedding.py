"""
Phase 5.3 - Primary visualisation of the embedding space.

Reads the master annotation table (clusters/cluster_annotations.tsv) and renders one
self-contained HTML file (clusters/embedding_viz.html, plotly.js from CDN) holding two
interactive UMAP scatter plots:

  (1) coloured by GENOME TYPE  - the primary figure: does a structural family cross
      genome-type boundaries? Hover shows protein_id, viral family, cluster id, genome.
  (2) coloured by FOLDSEEK CLUSTER (multi-member only; singletons grey) - does the
      learned metric space respect the Foldseek boundaries?

    python visualize_embedding.py
    python visualize_embedding.py --annotations clusters/cluster_annotations.tsv
"""
import os
import argparse

import cluster_common as cc

GENOME_COLORS = {
    'dsDNA': '#1f77b4', 'ssDNA': '#ff7f0e', 'dsRNA': '#2ca02c',
    'ssRNA(+)': '#d62728', 'ssRNA(-)': '#9467bd', 'ssRNA-RT': '#8c564b',
    'dsDNA-RT': '#e377c2', 'unknown': '#7f7f7f',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', default=os.path.join(cc.OUT_DIR, 'cluster_annotations.tsv'))
    ap.add_argument('--out', default=os.path.join(cc.OUT_DIR, 'embedding_viz.html'))
    args = ap.parse_args()

    import pandas as pd
    import plotly.graph_objects as go
    import plotly.colors as pcolors

    if not os.path.exists(args.annotations):
        raise SystemExit(f"{args.annotations} not found - run annotate_clusters.py "
                         f"(and project_umap.py) first.")
    df = pd.read_csv(args.annotations, sep='\t', dtype=str)
    # numeric coords; drop rows without a UMAP position
    for c in ('umap_x', 'umap_y'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['umap_x', 'umap_y']).reset_index(drop=True)
    df['model_cluster_id'] = df['model_cluster_id'].astype(int)
    df['foldseek_cluster_id'] = df['foldseek_cluster_id'].astype(int)
    df['protein_name'] = df['protein_id'].map(lambda p: cc.parse_protein_id(p)[0])
    print(f"{len(df)} proteins with UMAP coordinates")

    def hover(row):
        return (f"<b>{row.protein_name[:50]}</b><br>"
                f"family: {row.viral_family}<br>"
                f"genome: {row.genome_type}<br>"
                f"model cluster: {row.model_cluster_id}<br>"
                f"foldseek cluster: {row.foldseek_cluster_id}<br>"
                f"accession: {row.accession}")
    df['hover'] = df.apply(hover, axis=1)

    # --- Figure 1: colour by genome type --------------------------------------
    fig1 = go.Figure()
    present = [g for g in cc.GENOME_ORDER if g in set(df['genome_type'])]
    for g in present:
        sub = df[df['genome_type'] == g]
        fig1.add_trace(go.Scattergl(
            x=sub['umap_x'], y=sub['umap_y'], mode='markers', name=f"{g} ({len(sub)})",
            marker=dict(size=4, color=GENOME_COLORS.get(g, '#7f7f7f'), opacity=0.75),
            text=sub['hover'], hovertemplate='%{text}<extra></extra>',
        ))
    fig1.update_layout(
        title="UMAP of learned embeddings - coloured by genome type",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2", legend_title="genome type",
        width=1100, height=750, template='plotly_white',
    )

    # --- Figure 2: colour by Foldseek cluster (multi-member only) --------------
    fig2 = go.Figure()
    singles = df[df['foldseek_cluster_id'] < 0]
    multi = df[df['foldseek_cluster_id'] >= 0]
    fig2.add_trace(go.Scattergl(
        x=singles['umap_x'], y=singles['umap_y'], mode='markers',
        name=f"singletons ({len(singles)})",
        marker=dict(size=3, color='#cccccc', opacity=0.5),
        text=singles['hover'], hovertemplate='%{text}<extra></extra>',
    ))
    palette = pcolors.qualitative.Alphabet + pcolors.qualitative.Dark24
    multi_colors = [palette[cid % len(palette)] for cid in multi['foldseek_cluster_id']]
    fig2.add_trace(go.Scattergl(
        x=multi['umap_x'], y=multi['umap_y'], mode='markers',
        name=f"multi-member ({len(multi)})",
        marker=dict(size=5, color=multi_colors, opacity=0.85),
        text=multi['hover'], hovertemplate='%{text}<extra></extra>',
    ))
    fig2.update_layout(
        title=(f"UMAP - coloured by Foldseek cluster "
               f"({multi['foldseek_cluster_id'].nunique()} multi-member clusters, "
               f"recurring palette; singletons grey)"),
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        width=1100, height=750, template='plotly_white',
    )

    # --- Write one self-contained HTML ----------------------------------------
    cc.ensure_out_dir(os.path.dirname(args.out) or '.')
    html1 = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    html2 = fig2.to_html(full_html=False, include_plotlyjs=False)
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Deltafold embedding space - Phase 5 visualisation</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:32px}} p{{color:#555;max-width:1100px}}</style>
</head><body>
<h1>Deltafold embedding space (UMAP, cosine metric)</h1>
<p>{len(df)} viral proteins. Plot 1 tests whether structural families cross genome-type
boundaries; plot 2 tests whether the learned metric respects Foldseek cluster boundaries.</p>
<h2>1. Coloured by genome type</h2>
{html1}
<h2>2. Coloured by Foldseek cluster</h2>
{html2}
</body></html>"""
    with open(args.out, 'w') as f:
        f.write(doc)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
