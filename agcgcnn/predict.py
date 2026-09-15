"""Predict labels/adsorption and export the exact pooled model representation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoint import load_checkpoint, scale, unscale
from .data import CrystalDataset, collate


def run(checkpoint, data_dir, output, csv_name='id_prop.csv', archive='structures.zip', batch_size=64):
    if batch_size < 1: raise ValueError('Batch size must be positive')
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Output directory must be empty')
    model,state=load_checkpoint(checkpoint)
    a=state['architecture']; classification=state['task']=='classification'
    data=CrystalDataset(data_dir,csv_name,archive,a['added_fea_len'],
                        1 if classification else a['classnum'],classification)
    if data.atom_dimension != a['orig_atom_fea_len']:
        raise ValueError('Atom embedding width does not match the checkpoint')
    predictions=[]; embeddings=[]; rows=[]
    with torch.no_grad():
        for atoms,bonds,indices,maps,features,targets,names in DataLoader(data,batch_size=batch_size,collate_fn=collate):
            pred,encoded=model(atoms,bonds,indices,maps,scale(features,state['feature_scaler']))
            pred=pred.exp() if classification else unscale(pred,state['target_scaler'])
            predictions.append(pred.numpy());embeddings.append(encoded.numpy())
            for name,target,prediction in zip(names,targets.tolist(),pred.tolist()):
                row=[name,*target,*prediction]
                if classification: row.append(int(np.argmax(prediction)))
                rows.append(row)
    out.mkdir(parents=True,exist_ok=True)
    labels=['id']+[f'target_{i}' for i in range(data.targets)]+[('probability_' if classification else 'prediction_')+str(i) for i in range(a['classnum'])]
    if classification: labels.append('predicted_class')
    with (out/'predictions.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(labels);w.writerows(rows)
    np.save(out/'encoded.npy',np.concatenate(embeddings),allow_pickle=False)
    np.save(out/'predictions.npy',np.concatenate(predictions),allow_pickle=False)
    (out/'ids.json').write_text(json.dumps([row[0] for row in data.rows],indent=2)+'\n')
    return {'samples':len(rows),'output':str(out)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',default='runs/predictions');p.add_argument('--csv',default='id_prop.csv')
    p.add_argument('--archive',default='structures.zip');p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    try: print(json.dumps(run(args.checkpoint,args.data,args.output,args.csv,args.archive,args.batch_size)))
    except (ValueError,FileNotFoundError,KeyError) as exc: p.error(str(exc))


if __name__=='__main__': main()
