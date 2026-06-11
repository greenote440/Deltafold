#!/usr/bin/env bash
# Action plan Phases 3-5: end-to-end downstream analysis on virome_embeddings.pt.
# Decision-gate criterion 4 wants the whole thing under 5 minutes on the downsampled set.
#
#   bash run_phases_3_5.sh
#
# Run inside conda env ml_env (umap-learn, hdbscan, scikit-learn, plotly, pandas).
set -euo pipefail
cd "$(dirname "$0")"

echo "### Phase 3.1  cluster_embeddings.py"
python scripts/analysis/cluster_embeddings.py
echo; echo "### Phase 3.2/3.3  compare_clusters.py"
python scripts/analysis/compare_clusters.py
echo; echo "### Phase 4  validate_test_cases.py"
python validate_test_cases.py
echo; echo "### Phase 5.1  project_umap.py"
python project_umap.py
echo; echo "### Phase 5.2/5.4/5.5  annotate_clusters.py"
python scripts/utilities/annotate_clusters.py
echo; echo "### Phase 5.3  visualize_embedding.py"
python visualize_embedding.py
echo; echo "### Done. Outputs in ./clusters/"
ls -la clusters/
