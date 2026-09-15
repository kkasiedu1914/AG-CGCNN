"""Reproducible training entry point using the original AG-CGCNN architecture."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from cgcnn.model import CrystalGraphConvNet_1
from .data import CrystalDataset, collate, split_indices
from .checkpoint import fit_scaler, scale, unscale


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--csv', default='id_prop.csv')
    p.add_argument('--archive', default='structures.zip')
    p.add_argument('--features', type=int, default=5)
    p.add_argument('--targets', type=int, default=1)
    p.add_argument('--task', choices=['regression','classification'], default='regression')
    p.add_argument('--classes', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--atom-fea-len', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--convolutions', type=int, default=3)
    p.add_argument('--hidden-layers', type=int, default=1)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--output', default='runs/train')
    return p


def run(args):
    if min(args.epochs,args.batch_size,args.atom_fea_len,args.hidden,args.convolutions,args.hidden_layers) < 1 or args.lr <= 0:
        raise ValueError('Training counts and learning rate must be positive')
    classification = args.task == 'classification'
    if classification and (args.targets != 1 or args.classes < 2):
        raise ValueError('Classification requires one target and at least two classes')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; choose a new run directory')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    data = CrystalDataset(args.data,args.csv,args.archive,args.features,args.targets,classification)
    if classification and max(row[1][-1] for row in data.rows) >= args.classes:
        raise ValueError('A class label exceeds --classes')
    train_ids,val_ids,test_ids = split_indices(len(data),args.train_ratio,args.val_ratio,args.seed)
    train_values = torch.tensor(np.stack([data.rows[i][1] for i in train_ids]))
    feature_scaler = fit_scaler(train_values[:,:args.features])
    target_scaler = fit_scaler(train_values[:,args.features:])
    architecture = dict(orig_atom_fea_len=data.atom_dimension,nbr_fea_len=len(data.centers),
                        added_fea_len=args.features,classnum=args.classes if classification else args.targets,
                        atom_fea_len=args.atom_fea_len,n_conv=args.convolutions,h_fea_len=args.hidden,
                        n_h=args.hidden_layers,classification=classification)
    model = CrystalGraphConvNet_1(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.NLLLoss() if classification else torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(args.seed)
    def loader(ids, shuffle=False):
        return DataLoader(Subset(data,ids),batch_size=args.batch_size,shuffle=shuffle,
                          collate_fn=collate,num_workers=0,generator=generator)
    def epoch_pass(ids, training=False):
        model.train(training); total_loss=0.; count=0; predictions=[]; truths=[]
        for atoms,bonds,indices,maps,features,targets,names in loader(ids,training):
            features=scale(features,feature_scaler).to(device)
            labels=targets[:,0].long().to(device) if classification else scale(targets,target_scaler).to(device)
            with torch.set_grad_enabled(training):
                output,_=model(atoms.to(device),bonds.to(device),indices.to(device),[m.to(device) for m in maps],features)
                loss=criterion(output,labels)
                if not torch.isfinite(loss): raise ValueError('Non-finite loss; inspect input scales')
                if training:
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total_loss+=loss.item()*len(names);count+=len(names)
            pred=output.detach().cpu().argmax(1).view(-1,1) if classification else unscale(output.detach().cpu(),target_scaler)
            predictions.append(pred); truths.append(targets)
        predicted,true=torch.cat(predictions),torch.cat(truths)
        metric=(predicted==true).float().mean().item() if classification else (predicted-true).abs().mean().item()
        return total_loss/count,metric
    out.mkdir(parents=True,exist_ok=True)
    splits={k:[data.rows[i][0] for i in ids] for k,ids in [('train',train_ids),('validation',val_ids),('test',test_ids)]}
    (out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    history=[];best=float('inf')
    for epoch in range(args.epochs):
        train_loss,train_metric=epoch_pass(train_ids,True)
        val_loss,val_metric=epoch_pass(val_ids)
        history.append({'epoch':epoch+1,'train_loss':train_loss,'validation_loss':val_loss,'train_metric':train_metric,'validation_metric':val_metric})
        if val_loss<best:
            best=val_loss
            checkpoint={'format_version':1,'architecture':architecture,
                        'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'feature_scaler':feature_scaler,'target_scaler':target_scaler,
                        'task':args.task,'epoch':epoch+1,'training':vars(args)}
            torch.save(checkpoint,out/'best.pt')
        print(json.dumps(history[-1]),flush=True)
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict'])
    test_loss,test_metric=epoch_pass(test_ids)
    result={'task':args.task,'metric':'accuracy' if classification else 'MAE','test_metric':test_metric,'test_loss':test_loss,'best_epoch':state['epoch'],'seed':args.seed}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'history.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
    return result


def main():
    p=parser();args=p.parse_args()
    try: print(json.dumps(run(args),indent=2))
    except (ValueError,KeyError,FileNotFoundError) as exc: p.error(str(exc))


if __name__=='__main__': main()
