"""SHAP explanations of the pooled crystal embedding plus geometric descriptors.

Adapted from the author's SHAP_ANALYSIS_v1.6-class Colab workflow. The PyTorch
head reuses all trained dense layers, avoiding a second Keras/TensorFlow model.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .checkpoint import load_checkpoint


class PredictionHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.projection=copy.deepcopy(model.conv_to_fc)
        self.activation=torch.nn.Softplus()
        self.hidden=copy.deepcopy(getattr(model,'fcs',torch.nn.ModuleList()))
        self.output=copy.deepcopy(model.fc_out)
        self.classification=model.classification

    def forward(self, x):
        x=self.activation(self.projection(self.activation(x)))
        for layer in self.hidden: x=self.activation(layer(x))
        x=self.output(x)
        return torch.softmax(x,dim=1) if self.classification else x


def run(checkpoint, encoded, background, output, output_index=0, samples=5,
        background_size=30, nsamples=256, seed=123, feature_names=None, reference=None):
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if min(samples,background_size,nsamples)<1: raise ValueError('Sample counts must be positive')
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Output directory must be empty')
    model,state=load_checkpoint(checkpoint);head=PredictionHead(model).eval()
    x=np.load(encoded,allow_pickle=False);bg=np.load(background,allow_pickle=False)
    width=model.conv_to_fc.in_features
    for values in [x,bg]:
        if values.ndim!=2 or values.shape[1]!=width or len(values)==0 or not np.isfinite(values).all():
            raise ValueError('Encoded arrays must be nonempty, finite matrices matching the checkpoint')
    if not 0<=output_index<model.fc_out.out_features: raise ValueError('Output index is outside the model outputs')
    def predict_all(values):
        with torch.no_grad():
            result=head(torch.as_tensor(values,dtype=torch.float32)).numpy()
        if state['task']=='regression':
            result=result*state['target_scaler']['std'].numpy()+state['target_scaler']['mean'].numpy()
        return result
    if reference is not None:
        full=np.load(reference,allow_pickle=False)
        if full.shape!=predict_all(x).shape or not np.allclose(predict_all(x),full,rtol=1e-5,atol=1e-5):
            raise ValueError('Head predictions do not match the supplied full-model predictions')
    geometric=model.added_fea_len
    if feature_names is None:
        feature_names=['VF','GSA','PLD','LCD','VSA'] if geometric==5 else [f'Geometry {i+1}' for i in range(geometric)]
    if len(feature_names)!=geometric: raise ValueError('One feature name is required for each geometric descriptor')
    names=[f'Crystal feature {i+1}' for i in range(width-geometric)]+feature_names
    np.random.seed(seed)
    bg=bg[np.random.default_rng(seed).choice(len(bg),min(len(bg),background_size),replace=False)]
    x=x[:samples]
    explainer=shap.KernelExplainer(lambda values:predict_all(values)[:,output_index],bg)
    values=np.asarray(explainer.shap_values(x,nsamples=nsamples,l1_reg=0.0))
    expected=float(explainer.expected_value)
    residual=predict_all(x)[:,output_index]-(expected+values.sum(axis=1))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'explanations.npz',values=values,base_value=expected,data=x,names=np.array(names),residual=residual)
    explanation=shap.Explanation(values=values[0],base_values=expected,data=x[0],feature_names=names)
    shap.plots.waterfall(explanation,max_display=10,show=False)
    plt.gcf().savefig(out/'waterfall.png',dpi=180,bbox_inches='tight');plt.close('all')
    # Sum attributions for display; this is not a separately computed grouped-player SHAP game.
    n_latent=width-geometric
    grouped=np.column_stack([values[:,:n_latent].sum(axis=1),values[:,n_latent:]])
    report={'method':'KernelSHAP on the exact pooled prediction head','output_index':output_index,
            'units':'class probability' if state['task']=='classification' else 'training target units',
            'max_additivity_residual':float(abs(residual).max()),'seed':seed,
            'background_samples':len(bg),'explained_samples':len(x),
            'group_names':['Crystal embedding']+feature_names,'summed_feature_attributions':grouped.tolist()}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','encoded','background']: p.add_argument('--'+name,required=True)
    p.add_argument('--output',default='runs/explanations');p.add_argument('--output-index',type=int,default=0)
    p.add_argument('--samples',type=int,default=5);p.add_argument('--background-size',type=int,default=30)
    p.add_argument('--nsamples',type=int,default=256);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--feature-names',nargs='*');p.add_argument('--reference')
    args=p.parse_args()
    try: print(json.dumps(run(**vars(args)),indent=2))
    except (ValueError,FileNotFoundError,KeyError) as exc: p.error(str(exc))


if __name__=='__main__': main()
