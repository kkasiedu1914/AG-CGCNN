"""Validated CIF graph loading, adapted from CGCNN's MIT-licensed data pipeline."""
from pathlib import Path, PurePosixPath
import csv
import json
import zipfile
import re
import numpy as np
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset

MAX_CIF_BYTES = 10 * 1024 * 1024


class CrystalDataset(Dataset):
    """CSV rows: identifier, geometric descriptors, then target(s); no header."""

    def __init__(self, directory, csv_name='id_prop.csv', archive='structures.zip',
                 features=5, targets=1, classification=False, radius=8.0, neighbors=12):
        self.directory = Path(directory).resolve()
        if features < 0 or targets < 1 or radius <= 0 or neighbors < 1:
            raise ValueError('Invalid feature, target, radius, or neighbor count')
        self.features, self.targets = features, targets
        self.radius, self.neighbors = radius, neighbors
        self.archive = self._inside(archive)
        rows = []
        with self._inside(csv_name).open(newline='', encoding='utf-8-sig') as stream:
            for line, row in enumerate(csv.reader(stream), 1):
                if len(row) != 1 + features + targets:
                    raise ValueError(f'CSV row {line}: expected {1 + features + targets} columns')
                name = row[0]
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', name):
                    raise ValueError(f'CSV row {line}: unsafe structure identifier')
                try:
                    values = np.array(row[1:], dtype=np.float32)
                except ValueError as exc:
                    raise ValueError(f'CSV row {line}: descriptors and targets must be numeric') from exc
                if not np.isfinite(values).all():
                    raise ValueError(f'CSV row {line}: non-finite values')
                if classification and (targets != 1 or values[-1] < 0 or values[-1] != int(values[-1])):
                    raise ValueError('Classification needs one nonnegative integer label per row')
                rows.append((name, values))
        if not rows or len({r[0] for r in rows}) != len(rows):
            raise ValueError('Dataset must be nonempty and structure identifiers unique')
        self.rows = rows
        embedding_path = self._inside('atom_init.json')
        if embedding_path.stat().st_size > MAX_CIF_BYTES:
            raise ValueError('Atom embedding file is too large')
        embedding = json.loads(embedding_path.read_text())
        self.embedding = {int(k): np.asarray(v, dtype=np.float32) for k, v in embedding.items()}
        shapes = {v.shape for v in self.embedding.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1 or next(iter(shapes))[0] == 0:
            raise ValueError('Atom embeddings must be equal-length nonempty vectors')
        if not all(np.isfinite(v).all() for v in self.embedding.values()):
            raise ValueError('Atom embeddings must be finite')
        self.atom_dimension = next(iter(shapes))[0]
        self.centers = np.arange(0, radius + 0.2, 0.2)
        with zipfile.ZipFile(self.archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP members are not allowed')
            for name, _ in rows:
                info = z.getinfo(name + '.cif')
                if info.file_size > MAX_CIF_BYTES or info.flag_bits & 1:
                    raise ValueError('CIF is too large or encrypted')

    def _inside(self, name):
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Dataset file paths must stay within the dataset directory')
        return path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, values = self.rows[index]
        # Never extract archives to disk. Read only the validated, bounded CIF member.
        with zipfile.ZipFile(self.archive) as z, z.open(name + '.cif') as stream:
            text = stream.read(MAX_CIF_BYTES + 1)
        if len(text) > MAX_CIF_BYTES:
            raise ValueError('CIF exceeds size limit')
        crystal = Structure.from_str(text.decode('utf8'), fmt='cif')
        if not crystal.is_ordered or len(crystal) == 0 or len(crystal)>10000:
            raise ValueError(f'{name}: expected an ordered structure with 1 to 10000 atoms')
        atoms = np.stack([self.embedding[site.specie.Z] for site in crystal])
        indexes, distances = [], []
        for neighbors in crystal.get_all_neighbors(self.radius, include_index=True):
            near = sorted(neighbors, key=lambda n: n.nn_distance)[:self.neighbors]
            indexes.append([int(n.index) for n in near] + [0] * (self.neighbors - len(near)))
            distances.append([float(n.nn_distance) for n in near] + [self.radius + 1] * (self.neighbors - len(near)))
        bonds = np.exp(-((np.asarray(distances)[..., None] - self.centers) ** 2) / 0.2 ** 2)
        return (torch.tensor(atoms), torch.tensor(bonds, dtype=torch.float32),
                torch.tensor(indexes, dtype=torch.long), torch.tensor(values[:self.features]),
                torch.tensor(values[self.features:]), name)


def collate(samples):
    atoms, bonds, indexes, maps, features, targets, names = [], [], [], [], [], [], []
    offset = 0
    for atom, bond, idx, feature, target, name in samples:
        atoms.append(atom); bonds.append(bond); indexes.append(idx + offset)
        maps.append(torch.arange(offset, offset + len(atom)))
        offset += len(atom)
        features.append(feature); targets.append(target); names.append(name)
    return (torch.cat(atoms), torch.cat(bonds), torch.cat(indexes), maps,
            torch.stack(features), torch.stack(targets), names)


def split_indices(size, train_ratio=0.8, val_ratio=0.1, seed=123):
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError('Train/validation ratios must be positive and leave a test set')
    a, b = int(size * train_ratio), int(size * val_ratio)
    if min(a, b, size-a-b) < 1:
        raise ValueError('Dataset is too small for three nonempty splits')
    indices = np.random.default_rng(seed).permutation(size)
    return indices[:a].tolist(), indices[a:a+b].tolist(), indices[a+b:].tolist()
