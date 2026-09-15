# Data conventions

## Required files

```text
my-mofs/
  atom_init.json
  structures.zip
  id_prop.csv
```

The headerless CSV contains:

```text
structure_id,VF,GSA,PLD,LCD,VSA,target_1[,target_2,...]
```

The line above documents column meanings; **do not include it as a header**. The five-descriptor order is taken from the original SHAP notebook: void fraction, gravimetric surface area, pore limiting diameter, largest cavity diameter, volumetric surface area. Typical units are dimensionless, m²/g, Å, Å and m²/cm³, respectively. Preserve the actual units and order used to train a particular checkpoint.

`structure_id` must uniquely identify `<structure_id>.cif` at the root of `structures.zip`. Identifiers start with an ASCII letter or digit and contain only letters, digits, underscores, periods or hyphens (maximum 200 characters). This also prevents formula-like identifiers from being exported into spreadsheet-readable CSVs. Descriptors and targets must be finite numbers. Classification uses one final integer class label; its thresholds and meaning are experiment-specific.

`atom_init.json` maps atomic numbers, as strings, to equal-length numeric vectors. Research CGCNN models normally use the original 92-dimensional [CGCNN atom embeddings](https://github.com/txie-93/cgcnn/blob/master/data/sample-regression/atom_init.json). Their MIT attribution must be retained when redistributing them. The demo uses artificial four-dimensional embeddings and cannot load research-model weights.

## Original research data

The recovered work references hMOF/MOFX-DB, ToBaCCo, and CoRE MOF datasets. Consult the paper and supporting information for the exact adsorption conditions, source citations, unit conversions and label construction. The source snapshots retain data-preparation and characterization routines.

The Drive workspace includes CIF archives of approximately 1.09 GB (`hMof_cifs.zip`), 356 MB (`Tobacco_data.zip`) and 118 MB (`CoreMOF2019.zip`). These databases are not embedded in this code repository. Dataset licenses are separate from the repository's MIT code license; obtain the appropriate source data and preserve its attribution.

Original training checkpoints, prediction logs and serialized SHAP objects are also outside the code release. The release does not silently deserialize old pickle files or claim that newly generated demo checkpoints reproduce the published models.

## Validation and preprocessing

- Split before fitting feature or target scalers. The release fits scalers only on training rows.
- Keep duplicate crystal identifiers out of a dataset. If related structures can cause leakage, prepare group-aware research splits externally; the supplied splitter is a seeded random split, not a chemical-family split.
- Match gas, temperature, pressure and adsorption units across all rows of a target.
- Keep classification thresholds fixed across train, validation and test data.
- Save descriptor order and units alongside any scientific dataset.
- The loader requires ordered structures, bounds CIF member size and reads ZIP members without filesystem extraction.
