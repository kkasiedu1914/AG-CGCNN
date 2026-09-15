# Usage

## Installation

Use an isolated Python environment. From the repository root:

```bash
python -m pip install -e ".[notebooks,explain,test]"
python -m agcgcnn.train --help
python -m agcgcnn.predict --help
python -m agcgcnn.explain --help
```

The `agcgcnn-train`, `agcgcnn-predict` and `agcgcnn-explain` commands are equivalent to the module commands after installation. `mainz.py` and `predict.py` are convenience wrappers for the release CLI. The original positional-argument CLI is preserved only in source snapshots; it is not the release interface.

## Regression

Prepare `data/my-mofs/atom_init.json`, `structures.zip` and a headerless `id_prop.csv` following [DATA.md](DATA.md).

```bash
python -m agcgcnn.train --data data/my-mofs --features 5 --targets 2 --task regression --epochs 100 --batch-size 64 --output runs/h2-regression
```

The last two CSV columns are targets. Loss is mean squared error on standardized targets; the reported MAE is in the original target units, averaged across targets. For outputs with different units, inspect per-target predictions rather than treating the aggregate MAE as a single physical quantity.

## Classification

```bash
python -m agcgcnn.train --data data/my-mofs --features 5 --targets 1 --task classification --classes 4 --epochs 100 --output runs/h2-classification
```

The final CSV column must contain integer labels from 0 through 3. Labels and threshold definitions must be prepared consistently before training. The output layer returns log probabilities during training with negative log-likelihood loss. Predictions and SHAP explanations use probabilities.

## Key settings

| Setting | Default | Meaning |
|---|---:|---|
| `--features` | 5 | Number of geometric descriptors before the target columns |
| `--targets` | 1 | Number of regression outputs; always 1 for classification |
| `--atom-fea-len` | 64 | Hidden atom / pooled crystal representation width |
| `--convolutions` | 3 | Number of CGCNN message-passing layers |
| `--hidden` | 128 | Dense hidden-layer width |
| `--hidden-layers` | 1 | Number of hidden dense layers |
| `--train-ratio`, `--val-ratio` | 0.8, 0.1 | Remaining examples form the test split |
| `--seed` | 123 | Controls splitting and random initialization |
| `--lr` | 0.001 | Adam learning rate for this release |
| `--device` | cpu | Use `cuda` only with a compatible GPU/PyTorch installation |

These release defaults are documented software defaults, not a complete reconstruction of every paper experiment. To omit geometric descriptors, use `--features 0` and supply only identifiers and targets in the CSV.

Each run saves `best.pt`, `splits.json`, `history.csv`, and `metrics.json`. Scalers are fitted only on training data. The best validation-loss checkpoint is evaluated on the held-out test set. Output directories must be empty to avoid overwriting earlier results. Seeded runs improve repeatability; exact GPU results may still vary across libraries and hardware.

## Prediction

```bash
python -m agcgcnn.predict --checkpoint runs/h2-regression/best.pt --data data/my-mofs --output runs/h2-predictions
```

Outputs include a readable `predictions.csv`, the model inputs to the prediction head in `encoded.npy`, full outputs in `predictions.npy`, and ordered structure identifiers in `ids.json`. The target columns are retained for evaluation; for genuinely unlabeled inputs supply finite placeholder target values and do not interpret those values as observations.

## Explainability

Use notebook 02 to select a training/reference background and a separate held-out explanation set. The equivalent CLI is:

```bash
python -m agcgcnn.explain --checkpoint runs/h2-regression/best.pt --encoded runs/heldout.npy --background runs/background.npy --reference runs/heldout-predictions.npy --output-index 0 --samples 5 --background-size 30 --nsamples 512 --output runs/h2-explanations
```

`--reference` checks that the prediction head reproduces full-model outputs for the supplied embeddings. Preserve the row order when subsetting arrays. Feature labels default to `VF GSA PLD LCD VSA` for five descriptors; pass `--feature-names` if your CSV uses another order. Never relabel a saved embedding without checking its training schema.

Results are `waterfall.png`, `explanations.npz` and `summary.json`. NPZ files contain numeric values and Unicode feature labels and load with `allow_pickle=False`. SHAP is sampling-based: increase the sample budget and check stability before drawing research conclusions. The latent crystal-feature attributions can be summed for display, but are not attributions to specific atoms and are not equivalent to rerunning a grouped-feature SHAP game.

## Common errors

- **Dataset columns do not match:** use no header; put the identifier first, descriptors next, targets last.
- **Missing CIF:** ZIP members must be named exactly `<identifier>.cif` at the archive root.
- **Unsupported checkpoint:** the release intentionally refuses unrestricted pickle checkpoints. Old research checkpoints may contain pandas objects or custom classes and require a separately reviewed migration; do not disable the safety check.
- **Output directory is not empty:** choose a new directory or inspect and move previous outputs yourself.
- **Too few data points:** train, validation and test splits must each contain at least one sample.
- **CUDA unavailable:** use the CPU default or install an appropriate official PyTorch build.
