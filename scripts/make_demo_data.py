"""Create explicitly synthetic CIFs and labels for software smoke tests only."""
import argparse
import csv
import json
import zipfile
from pathlib import Path
from pymatgen.core import Lattice, Structure


def make_demo(directory, count=30, classification=False, features=5, targets=1):
    out=Path(directory)
    if out.exists() and any(out.iterdir()): raise ValueError('Demo directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    # These small artificial embeddings are intentionally unrelated to the paper's 92 features.
    (out/'atom_init.json').write_text(json.dumps({'6':[1.,0.,0.,1.],'8':[0.,1.,0.,1.]}))
    rows=[]
    with zipfile.ZipFile(out/'structures.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for i in range(count):
            name=f'demo-{i:03d}'
            lattice=3.5+i*0.035
            structure=Structure(Lattice.cubic(lattice),['C','O'],[[0,0,0],[.5,.5,.5]])
            z.writestr(name+'.cif',structure.to(fmt='cif'))
            descriptors=[(i+1)/(count+1),100.+i*5,2.+i*.02,3.+i*.03,200.+i*7][:features]
            if features>5: descriptors.extend([float(i+j) for j in range(features-5)])
            target=[i%2] if classification else [lattice*(j+1) for j in range(targets)]
            rows.append([name,*descriptors,*target])
    with (out/'id_prop.csv').open('w',newline='') as stream: csv.writer(stream).writerows(rows)
    (out/'README.txt').write_text('Synthetic software-test data. Not MOFs, adsorption simulations, or paper results.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='examples/demo')
    p.add_argument('--classification',action='store_true');a=p.parse_args()
    make_demo(a.output,classification=a.classification)
