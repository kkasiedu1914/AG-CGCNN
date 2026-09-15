# Provenance and release changes

## Recovered sources

The author supplied the Google Drive `work_cgcnn` project and Colab notebooks. Discovery covered the primary `cgcnn_Trainer` code, its model/data variants, earlier multiclass and CGCNN experiments, comparison code, and relevant training, data-preparation, plotting and SHAP notebooks.

The authoritative architecture for this release comes from `work_cgcnn/cgcnn_Trainer/cgcnn/model.py`. The explanation walkthrough derives from the Colab `SHAP_ANALYSIS_v1.6-class.ipynb` workflow and related plotting notebooks. The repository title uses the published name **AG-CGCNN**.

[source-manifest.json](source-manifest.json) records source-relative paths, SHA-256 hashes of recovered originals, publication paths and omissions. Source Drive links, machine credentials, notebook execution outputs and cloud account details are not needed to use the release.

## Archive organization

`archive/research/` contains reference snapshots. Python files have `.py.txt` suffixes and notebooks have `.ipynb.json` suffixes so they are not mistaken for supported entry points. Notebook outputs and execution metadata are cleared. Original algorithm code and comments remain available for comparison. These snapshots retain historical bugs and assumptions; do not rename and run them as the release.

Obsolete automatic cloud-upload, broad archive-collection and desktop-clicking utilities are omitted. Their source hashes and omission reasons remain in the manifest. Credentials, Git internals, caches, temporary output bundles, large CIF databases, training logs and serialized experiment objects are not part of the public code tree.

## What changed in the runnable release

| Area | Recovered behavior / issue | Release behavior |
|---|---|---|
| Dataset creation | Latest trainer did not assign its constructed dataset | Explicit validated dataset construction |
| CLI | List parsing and default indexing could fail | Typed flags with explicit defaults and validation |
| Zero geometry | Returned embedding could be undefined | Embedding is always returned |
| Sample ranges | Feature dictionary and shuffled row selection could disagree | One dataset row representation and recorded split IDs |
| Normalization | Original sampling could include non-training examples | Scalers fitted only to training rows; constant columns handled |
| Checkpoints | General pickle deserialization, including experiment objects | Tensor/primitive release checkpoints; `weights_only=True`; no unsafe fallback |
| ZIP access | Historical notebooks included extraction routines | Direct bounded reads of named CIF members; no extraction |
| Files | Cloud collection could include credential JSON | No automatic cloud upload; no broad collection utility |
| Explainability | Hardcoded 69-dimensional/Keras transfer assumed one dense layer | Exact PyTorch head supports the checkpoint's layer count and widths |
| Reproducibility | Mixed notebook paths and hidden session state | Seed, split IDs, new output directories and documented notebook sequence |

The new training loop uses Adam with documented defaults and validation-loss selection. It is a reproducible release interface, not a claim of bitwise identity with historical training runs. Existing raw research checkpoints are not automatically migrated.

## Maintenance status

Active research development is paused. The repository remains readable and reusable; the included checks document the release's tested scope. See [VALIDATION.md](VALIDATION.md) before treating the software as a numerical reproduction of the publication.
