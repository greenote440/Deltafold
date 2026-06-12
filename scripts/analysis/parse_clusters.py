#!/usr/bin/env python
"""
Parse and summarize cluster structure without loading all data at once.

Efficiently reads clusters/model_clusters.tsv and generates concise summaries:
- Cluster size distribution
- Top N largest clusters with family composition
- Bridges (clusters uniting >=2 Foldseek clusters)
- Splits (Foldseek clusters split across >=2 model clusters)
- Singleton statistics
- Cross-family cluster composition

Usage:
    python parse_clusters.py                    # Print summary to stdout
    python parse_clusters.py --json clusters.json  # Save detailed report as JSON
    python parse_clusters.py --top 20          # Show top 20 clusters
    python parse_clusters.py --bridges         # Only show bridge clusters
    python parse_clusters.py --query "Poxviridae"  # Filter by family
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import gzip

sys.path.insert(0, str(Path(__file__).parent))
import cluster_common as cc


def parse_cluster_file(cluster_file):
    """
    Parse clusters/model_clusters.tsv into a dictionary:
    cluster_id -> [list of protein IDs]

    Returns: (clusters_dict, total_proteins, n_clusters)
    """
    clusters = defaultdict(list)
    total_proteins = 0

    with open(cluster_file) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                protein_id = parts[0]
                cluster_id = parts[1]
                clusters[cluster_id].append(protein_id)
                total_proteins += 1

    return dict(clusters), total_proteins, len(clusters)


def load_foldseek_clusters(cluster_file='data/cluster.tsv'):
    """Load Foldseek clustering for bridge/split analysis."""
    foldseek = defaultdict(list)
    foldseek_rep = {}

    if not os.path.exists(cluster_file):
        return foldseek, foldseek_rep

    with open(cluster_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep = parts[0]
                member = parts[1]
                foldseek[rep].append(member)
                foldseek_rep[member] = rep

    return foldseek, foldseek_rep


def load_protein_metadata():
    """Load protein metadata (family, genome type)."""
    metadata = {}

    # Try to load from cluster_common's metadata
    try:
        # This would require cluster_common to export metadata
        # For now, we'll parse from the annotation file if it exists
        if os.path.exists(cc.OUT_DIR + '/cluster_annotations.tsv'):
            with open(cc.OUT_DIR + '/cluster_annotations.tsv') as f:
                header = f.readline().strip().split('\t')
                for line in f:
                    parts = dict(zip(header, line.strip().split('\t')))
                    protein_id = parts.get('protein_id')
                    if protein_id:
                        metadata[protein_id] = {
                            'family': parts.get('viral_family', 'unknown'),
                            'genome': parts.get('genome_type', 'unknown'),
                        }
    except:
        pass

    return metadata


def analyze_cluster_composition(clusters, foldseek_rep, metadata):
    """
    Analyze clusters for:
    - Size distribution
    - Bridges (merge >=2 Foldseek clusters)
    - Splits (split >=1 Foldseek cluster)
    - Family composition
    """
    size_dist = defaultdict(int)
    bridges = {}
    splits = defaultdict(set)
    cluster_families = {}

    for cluster_id, proteins in clusters.items():
        size = len(proteins)
        size_dist[size] += 1

        # Find unique Foldseek clusters in this model cluster
        foldseek_clusters = set()
        families = set()
        for protein in proteins:
            if protein in foldseek_rep:
                foldseek_clusters.add(foldseek_rep[protein])
            if protein in metadata:
                families.add(metadata[protein]['family'])

        # Check for bridges (merges)
        if len(foldseek_clusters) >= 2:
            bridges[cluster_id] = {
                'size': size,
                'n_foldseek': len(foldseek_clusters),
                'families': sorted(families),
            }

        # Track splits
        for fs_cluster in foldseek_clusters:
            splits[fs_cluster].add(cluster_id)

        cluster_families[cluster_id] = sorted(families)

    # Filter splits to only those with >=2 model clusters
    splits = {k: v for k, v in splits.items() if len(v) >= 2}

    return size_dist, bridges, splits, cluster_families


def print_summary(clusters, foldseek, foldseek_rep, metadata, top_n=10):
    """Print a human-readable summary of clusters."""
    total_proteins = sum(len(p) for p in clusters.values())
    n_clusters = len(clusters)
    singletons = sum(1 for p in clusters.values() if len(p) == 1)
    multi = n_clusters - singletons

    size_dist, bridges, splits, cluster_families = analyze_cluster_composition(
        clusters, foldseek_rep, metadata
    )

    print("\n" + "=" * 100)
    print("CLUSTER SUMMARY")
    print("=" * 100)
    print(f"\nOverall Statistics:")
    print(f"  Total proteins:           {total_proteins:,}")
    print(f"  Total clusters:           {n_clusters:,}")
    print(f"  Multi-member clusters:    {multi:,}")
    print(f"  Singletons:               {singletons:,} ({100*singletons/n_clusters:.1f}%)")
    print(f"  Avg cluster size:         {total_proteins/n_clusters:.2f}")

    # Size distribution
    sorted_sizes = sorted(size_dist.keys())
    print(f"\nCluster Size Distribution:")
    for size_range in [(1, 1), (2, 2), (3, 5), (6, 10), (11, 25), (26, 100), (101, float('inf'))]:
        low, high = size_range
        count = sum(size_dist.get(s, 0) for s in range(low, min(high+1, max(size_dist.keys())+1)))
        if high == float('inf'):
            count = sum(size_dist.get(s, 0) for s in range(low, max(size_dist.keys())+1))
            print(f"  size {low}+:      {count:>6} clusters")
        else:
            print(f"  size {low:>3}-{high:<3}: {count:>6} clusters")

    # Top clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
    print(f"\nTop {top_n} Largest Clusters:")
    print(f"{'Cluster ID':<15} {'Size':<8} {'Families':<60}")
    print("-" * 100)
    for cluster_id, proteins in sorted_clusters[:top_n]:
        families = cluster_families.get(cluster_id, [])
        families_str = ', '.join(families)[:57] + ('...' if len(', '.join(families)) > 57 else '')
        print(f"{cluster_id:<15} {len(proteins):<8} {families_str:<60}")

    # Bridges
    if bridges:
        print(f"\nBridges (model clusters merging >=2 Foldseek clusters): {len(bridges)}")
        sorted_bridges = sorted(bridges.items(), key=lambda x: -x[1]['size'])
        print(f"{'Cluster ID':<15} {'Size':<8} {'Foldseek':<12} {'Families':<60}")
        print("-" * 100)
        for cluster_id, info in sorted_bridges[:top_n]:
            families_str = ', '.join(info['families'])[:57] + ('...' if len(', '.join(info['families'])) > 57 else '')
            print(f"{cluster_id:<15} {info['size']:<8} {info['n_foldseek']:<12} {families_str:<60}")

    # Splits
    if splits:
        print(f"\nSplits (Foldseek clusters split across >=2 model clusters): {len(splits)}")
        sorted_splits = sorted(splits.items(), key=lambda x: -len(x[1]))
        print(f"{'Foldseek ID':<30} {'Split into':<12} {'Model Clusters':<60}")
        print("-" * 100)
        for foldseek_id, model_clusters in sorted_splits[:top_n]:
            print(f"{foldseek_id:<30} {len(model_clusters):<12} {str(sorted(model_clusters))[:57]}")

    print("\n" + "=" * 100)


def save_json_report(clusters, foldseek, foldseek_rep, metadata, output_file):
    """Save detailed cluster report as JSON."""
    size_dist, bridges, splits, cluster_families = analyze_cluster_composition(
        clusters, foldseek_rep, metadata
    )

    report = {
        'metadata': {
            'total_proteins': sum(len(p) for p in clusters.values()),
            'total_clusters': len(clusters),
            'singletons': sum(1 for p in clusters.values() if len(p) == 1),
            'multi_member': sum(1 for p in clusters.values() if len(p) > 1),
        },
        'size_distribution': {int(k): v for k, v in size_dist.items()},
        'bridges': {
            str(k): {
                'size': v['size'],
                'n_foldseek': v['n_foldseek'],
                'families': v['families'],
            }
            for k, v in bridges.items()
        },
        'splits': {
            str(k): sorted(v)
            for k, v in splits.items()
        },
        'largest_clusters': [
            {
                'cluster_id': cid,
                'size': len(proteins),
                'families': cluster_families.get(cid, []),
            }
            for cid, proteins in sorted(clusters.items(), key=lambda x: -len(x[1]))[:100]
        ],
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved to {output_file}")


def filter_by_query(clusters, foldseek_rep, metadata, query):
    """Filter clusters containing proteins matching query (family, genome, protein name)."""
    query_lower = query.lower()
    matching_proteins = set()

    for protein_id, info in metadata.items():
        if (query_lower in info['family'].lower() or
            query_lower in info['genome'].lower() or
            query_lower in protein_id.lower()):
            matching_proteins.add(protein_id)

    # Filter clusters
    filtered = {}
    for cluster_id, proteins in clusters.items():
        matching = [p for p in proteins if p in matching_proteins]
        if matching:
            filtered[cluster_id] = matching

    return filtered


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--clusters', default='clusters/model_clusters.tsv',
                    help='Path to model_clusters.tsv')
    ap.add_argument('--foldseek', default='data/cluster.tsv',
                    help='Path to Foldseek cluster.tsv')
    ap.add_argument('--json', dest='json_out', default=None,
                    help='Save JSON report to this file')
    ap.add_argument('--top', type=int, default=10,
                    help='Show top N clusters (default 10)')
    ap.add_argument('--bridges', action='store_true',
                    help='Only show bridge clusters')
    ap.add_argument('--query', default=None,
                    help='Filter by family/genome/protein name')
    args = ap.parse_args()

    # Load data
    print("Loading clusters...")
    clusters, total_proteins, n_clusters = parse_cluster_file(args.clusters)
    print(f"  {total_proteins:,} proteins in {n_clusters:,} clusters")

    print("Loading Foldseek clustering...")
    foldseek, foldseek_rep = load_foldseek_clusters(args.foldseek)
    print(f"  {len(foldseek):,} Foldseek clusters")

    print("Loading protein metadata...")
    metadata = load_protein_metadata()
    print(f"  {len(metadata):,} proteins with metadata")

    if args.query:
        print(f"Filtering by query: {args.query}")
        clusters = filter_by_query(clusters, foldseek_rep, metadata, args.query)
        print(f"  Matches: {sum(len(p) for p in clusters.values())} proteins in {len(clusters)} clusters")

    # Generate output
    if args.json_out:
        save_json_report(clusters, foldseek, foldseek_rep, metadata, args.json_out)
    else:
        print_summary(clusters, foldseek, foldseek_rep, metadata, args.top)


if __name__ == '__main__':
    main()
