import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from agcgcnn.data import CrystalDataset, collate, split_indices, MAX_CIF_BYTES
from agcgcnn.checkpoint import fit_scaler, scale, load_checkpoint
from agcgcnn.explain import PredictionHead, run as explain
from agcgcnn.train import parser, run as train
from agcgcnn.predict import run as predict
from cgcnn.model import CrystalGraphConvNet_1
from scripts.make_demo_data import make_demo


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def dataset(tmp_path, **kwargs):
    directory=tmp_path/'data'
    make_demo(directory,**kwargs)
    return directory


@pytest.mark.parametrize('features',[0,5])
@pytest.mark.parametrize('classification',[False,True])
@pytest.mark.parametrize('hidden_layers',[1,3])
def test_graph_and_head_equivalence(tmp_path,features,classification,hidden_layers):
    directory=dataset(tmp_path,features=features,classification=classification)
    data=CrystalDataset(directory,features=features,classification=classification)
    a,b,i,m,f,y,n=collate([data[0],data[1]])
    assert torch.all(i[2:]>=2)
    model=CrystalGraphConvNet_1(4,41,features,2,atom_fea_len=8,n_conv=2,h_fea_len=12,n_h=hidden_layers,classification=classification).eval()
    with torch.no_grad():
        output,encoded=model(a,b,i,m,f)
        head=PredictionHead(model).eval()(encoded)
    assert encoded.shape==(2,8+features)
    assert torch.allclose(head,output.exp() if classification else output,atol=1e-6)
    if classification: assert torch.allclose(head.sum(1),torch.ones(2))


def test_splits_and_scaling():
    groups=split_indices(100,seed=1)
    assert groups==split_indices(100,seed=1)
    assert sorted(sum(groups,[]))==list(range(100))
    assert all(not (set(a)&set(b)) for j,a in enumerate(groups) for b in groups[j+1:])
    values=torch.tensor([[2.,4.],[2.,8.]])
    assert torch.isfinite(scale(values,fit_scaler(values))).all()
    with pytest.raises(ValueError): split_indices(2)
    with pytest.raises(ValueError): split_indices(10,.8,.3)


@pytest.mark.parametrize('change', ['traversal','nan','width','duplicate','label','formula'])
def test_bad_csv_rejected(tmp_path,change):
    directory=dataset(tmp_path,classification=True)
    path=directory/'id_prop.csv';lines=path.read_text().splitlines()
    if change=='traversal': lines[0]=lines[0].replace('demo-000','../outside')
    if change=='nan': lines[0]=lines[0].replace(',100.0,',',nan,')
    if change=='width': lines[0]+= ',4'
    if change=='duplicate': lines[1]=lines[0]
    if change=='label': lines[0]=lines[0].rsplit(',',1)[0]+',0.5'
    if change=='formula': lines[0]=lines[0].replace('demo-000','=1+1')
    path.write_text('\n'.join(lines))
    with pytest.raises(ValueError): CrystalDataset(directory,classification=True)


def test_dataset_paths_and_zip_bomb(tmp_path):
    directory=dataset(tmp_path)
    with pytest.raises(ValueError): CrystalDataset(directory,csv_name='../outside.csv')
    with zipfile.ZipFile(directory/'structures.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('demo-000.cif','a'*(MAX_CIF_BYTES+1))
    with pytest.raises(ValueError): CrystalDataset(directory)
    assert not (tmp_path/'outside.cif').exists()


def test_unrestricted_pickle_is_rejected(tmp_path):
    class Unsupported:
        pass
    # A global unsupported by the restricted loader; no malicious payload is executed.
    path=tmp_path/'unsafe.pt';torch.save({'object':Path('not-a-tensor')},path)
    with pytest.raises(ValueError,match='unrestricted pickle'): load_checkpoint(path)
    path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError): load_checkpoint(path)


@pytest.mark.parametrize('classification,features,targets',[(False,5,1),(False,0,2),(True,5,1)])
def test_train_predict_explain(tmp_path,classification,features,targets):
    directory=dataset(tmp_path,classification=classification,features=features,targets=targets)
    run_dir=tmp_path/'training'
    command=['--data',str(directory),'--output',str(run_dir),'--epochs','2','--batch-size','8',
             '--atom-fea-len','8','--hidden','12','--convolutions','1','--features',str(features),'--targets',str(targets)]
    if classification: command+=['--task','classification','--classes','2']
    args=parser().parse_args(command);result=train(args)
    assert np.isfinite(result['test_metric'])
    model,state=load_checkpoint(run_dir/'best.pt')
    split=json.loads((run_dir/'splits.json').read_text())
    data=CrystalDataset(directory,features=features,targets=targets,classification=classification)
    by_id=dict(data.rows)
    expected=np.stack([by_id[i][:features] for i in split['train']]).mean(0)
    assert np.allclose(state['feature_scaler']['mean'].numpy(),expected)
    pred_dir=tmp_path/'prediction'; predict(run_dir/'best.pt',directory,pred_dir)
    assert np.load(pred_dir/'encoded.npy').shape==(30,8+features)
    report=explain(run_dir/'best.pt',pred_dir/'encoded.npy',pred_dir/'encoded.npy',tmp_path/'shap',
                   samples=2,background_size=3,nsamples=32,reference=pred_dir/'predictions.npy')
    assert report['max_additivity_residual']<1e-4
    assert (tmp_path/'shap/waterfall.png').stat().st_size>1000
    with pytest.raises(ValueError): train(args)
    # Tampered architecture/scalers fail before model use.
    bad=dict(state);bad['architecture']=dict(state['architecture'],n_conv=100000)
    torch.save(bad,tmp_path/'bad-architecture.pt')
    with pytest.raises(ValueError,match='dimensions'): load_checkpoint(tmp_path/'bad-architecture.pt')
    bad=dict(state);bad['target_scaler']={'mean':torch.zeros(targets),'std':-torch.ones(targets)}
    torch.save(bad,tmp_path/'bad-scaler.pt')
    with pytest.raises(ValueError,match='positive'): load_checkpoint(tmp_path/'bad-scaler.pt')


def test_notebooks_have_clean_outputs():
    import nbformat
    for path in (Path(__file__).parents[1]/'notebooks').glob('*.ipynb'):
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code': assert cell.outputs==[] and cell.execution_count is None
