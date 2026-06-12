# run_phases_3_5.sh - Import Status ✅

**Date:** 2026-06-12  
**Status:** All imports fixed and verified  
**Bash syntax:** ✓ Valid

---

## Script Execution Path

```bash
bash run_phases_3_5.sh
```

This runs the following phases:

| Phase | Script | Location | Status |
|-------|--------|----------|--------|
| **3.1** | cluster_embeddings.py | scripts/analysis/ | ✅ Fixed |
| **3.2/3.3** | compare_clusters.py | scripts/analysis/ | ✅ Fixed |
| **4** | validate_test_cases.py | root | ✅ Fixed |
| **5.1** | project_umap.py | root | ✅ Fixed |
| **5.2/5.4/5.5** | annotate_clusters.py | scripts/utilities/ | ✅ Fixed |
| **5.3** | visualize_embedding.py | root | ✅ Fixed |

---

## Import Verification

All scripts have been tested and verified to import correctly:

### Root-Level Scripts (project_umap.py, visualize_embedding.py, validate_test_cases.py)

These scripts add `scripts/analysis/` to `sys.path` dynamically:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'analysis'))
import cluster_common as cc
```

**Status:** ✅ All working

### Scripts in scripts/analysis/

- cluster_embeddings.py
- compare_clusters.py

**Dependencies:** Import from cluster_common.py (in same directory)

**Status:** ✅ All working

### Scripts in scripts/utilities/

- annotate_clusters.py

**Dependencies:** Import from cluster_common.py via dynamic path

**Fix Applied:** Added path injection to find cluster_common in scripts/analysis/
```python
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))
import cluster_common as cc
```

**Status:** ✅ Fixed (commit ecbc5f7)

---

## What cluster_common Provides

Central module (`scripts/analysis/cluster_common.py`) exports:
- `EMB_FILE` — Path to embeddings
- `OUT_DIR` — Output directory (./clusters/)
- `GENOME_ORDER` — Canonical genome type ordering
- `load_embeddings()` — Load embeddings from file
- `ensure_out_dir()` — Create output directory
- `parse_protein_id()` — Parse protein identifiers

Used by all downstream scripts.

---

## Tested Imports

```
✓ cluster_common (scripts/analysis/)
✓ pandas (data manipulation)
✓ plotly (visualization)
✓ numpy (numerical ops)
✓ umap (dimensionality reduction)
✓ hdbscan (clustering)
✓ sklearn (metrics)
```

---

## How to Run

```bash
cd /Users/macbook/Documents/Deltafold

# Run full pipeline
bash run_phases_3_5.sh

# Or run individual phases:
python scripts/analysis/cluster_embeddings.py
python scripts/analysis/compare_clusters.py
python validate_test_cases.py
python project_umap.py
python scripts/utilities/annotate_clusters.py
python visualize_embedding.py
```

---

## Output Files Generated

```
clusters/
├── model_clusters.tsv          # Phase 3.1 output
├── (Phase 3.2 prints comparison)
├── (Phase 4 prints validation)
├── umap_coords.tsv             # Phase 5.1 output
├── cluster_annotations.tsv     # Phase 5.2 output
└── embedding_viz.html          # Phase 5.3 output
```

---

## Summary

✅ **All imports working**  
✅ **Bash script validated**  
✅ **All dependencies available**  
✅ **Ready to run pipeline**

No further import fixes needed. The script is ready for production use.

