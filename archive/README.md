# Research source archive

This directory preserves readable snapshots of the author's recovered research code. Use the root `agcgcnn` package and `notebooks/` for the tested release.

| Folder | Contents |
|---|---|
| `research/trainer/` | Primary training and prediction scripts; model/data variants; SHAP analysis notebooks |
| `research/colab/` | Original Colab SHAP workflows, interpretability plots, learning curves and data preparation |
| `research/root/` | Top-level project scripts and notebooks |
| `research/legacy/` | Earlier CGCNN, multiclass and comparison experiments, with contributor folder context retained |

Python snapshots use `.py.txt`; notebook snapshots use `.ipynb.json`. These are reference formats, not supported execution paths. Notebook outputs and execution metadata were removed. Historical bugs, old dependencies and environment-specific assumptions remain visible for provenance, so these snapshots should not be renamed and executed without a separate review.

Automatic cloud uploads, broad output-packaging utilities, credentials, Git internals, data archives and serialized experiment objects are excluded. See [the source manifest](../docs/source-manifest.json), [release changes](../docs/PROVENANCE.md), and [component attribution](../NOTICE.md).
