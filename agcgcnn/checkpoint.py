"""Tensor-only checkpoint IO. No unrestricted pickle fallback."""
from pathlib import Path
import zipfile
import torch
from cgcnn.model import CrystalGraphConvNet_1


def load_checkpoint(path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError('Checkpoint exceeds the 512 MiB limit')
    if not zipfile.is_zipfile(path):
        raise ValueError('Checkpoint is not a supported safe tensor checkpoint; unrestricted pickle loading is disabled')
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 1024**3:
            raise ValueError('Uncompressed checkpoint exceeds the 1 GiB limit')
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Checkpoint is not a supported safe tensor checkpoint; unrestricted pickle loading is disabled') from exc
    if not isinstance(state, dict) or state.get('format_version') != 1:
        raise ValueError('Expected an AG-CGCNN release checkpoint (format_version=1)')
    architecture=state.get('architecture')
    limits={'orig_atom_fea_len':1024,'nbr_fea_len':1024,'added_fea_len':1024,
            'classnum':1024,'atom_fea_len':1024,'n_conv':32,'h_fea_len':2048,'n_h':32}
    if not isinstance(architecture,dict) or set(architecture)!=set(limits)|{'classification'}:
        raise ValueError('Invalid checkpoint architecture fields')
    for name,limit in limits.items():
        value=architecture[name]
        if type(value) is not int or not (0 if name=='added_fea_len' else 1)<=value<=limit:
            raise ValueError('Checkpoint architecture dimensions exceed supported limits')
    if type(architecture['classification']) is not bool or state.get('task') not in ['classification','regression']:
        raise ValueError('Invalid checkpoint task')
    if architecture['classification'] != (state['task']=='classification'):
        raise ValueError('Checkpoint task and architecture disagree')
    for key,width in [('feature_scaler',architecture['added_fea_len']),('target_scaler',1 if architecture['classification'] else architecture['classnum'])]:
        scaler=state.get(key,{})
        for field in ['mean','std']:
            value=scaler.get(field)
            if not isinstance(value,torch.Tensor) or value.shape!=(width,) or not torch.isfinite(value).all():
                raise ValueError('Invalid checkpoint scaler')
        if not (scaler['std']>0).all(): raise ValueError('Checkpoint standard deviations must be positive')
    model = CrystalGraphConvNet_1(**architecture)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, state


def fit_scaler(values):
    if values.ndim != 2 or len(values)==0 or not torch.isfinite(values).all():
        raise ValueError('Scaler input must be a nonempty finite matrix')
    if values.shape[1]==0:
        return {'mean':torch.empty(0,dtype=values.dtype),'std':torch.empty(0,dtype=values.dtype)}
    mean = values.mean(0)
    std = values.std(0, unbiased=False)
    return {'mean': mean, 'std': torch.where(std > 1e-12, std, torch.ones_like(std))}


def scale(values, scaler):
    return (values - scaler['mean']) / scaler['std']


def unscale(values, scaler):
    return values * scaler['std'] + scaler['mean']
