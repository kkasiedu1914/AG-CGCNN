# Attribution and licensing

## CGCNN foundation

The graph convolution, atom/bond feature construction, neighbor aggregation and mean pooling originate from Tian Xie's [CGCNN](https://github.com/txie-93/cgcnn), Copyright (c) 2018 Tian Xie, MIT License. The full copyright and permission notice is retained in `LICENSE` and in the recovered upstream snapshots.

Reference: Tian Xie and Jeffrey C. Grossman, *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*, Physical Review Letters 120, 145301 (2018), DOI: 10.1103/PhysRevLett.120.145301.

## AG-CGCNN contribution

Kwabena Koranteng Asiedu's recovered implementation appends geometric descriptors after pooling, supports multiple targets or classes, and returns the combined representation for SHAP analysis. `cgcnn/model.py` follows that implementation and keeps its parameter names. The public-release tools repair data handling, training orchestration, checkpoint safety and documentation without introducing a different graph architecture.

The accompanying paper is authored by Kwabena Koranteng Asiedu, Luke E. K. Achenie, Tracy Asamoah, Emmanuel Kwesi Arthur and Nana Yaw Asiedu. Authorship of the paper is not a claim that every coauthor wrote every software component. Source folder names and original comments preserve available contributor context.

## SHAP and research snapshots

The explanation workflow adapts the author's SHAP Colab notebooks and uses the SHAP library. Please also cite Scott M. Lundberg and Su-In Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS (2017), when appropriate.

The historical archive includes comparison implementations and upstream-derived experiments. It is provided as reference source with original notices retained; it is not installed, executed by tests or presented as independently authored release code. Any third-party component remains subject to its original license. The source manifest records recovered file names and hashes.

The `legacy/MOF-CGCNN_compare/MOF-CGCNN` comparison snapshots derive from [Ruihan Wang's MOF-CGCNN](https://github.com/ruihwang/MOF-CGCNN), Copyright (c) 2021 Ruihan Wang. Its full MIT notice is retained in [third_party/MOF-CGCNN-LICENSE](third_party/MOF-CGCNN-LICENSE). That comparison architecture is not the AG-CGCNN release architecture.

## Figures and data

Paper figure crops are by Asiedu et al. (2025), licensed **CC BY 4.0**, not MIT. They are adapted only by cropping away page furniture and captions. See `docs/figures/README.md`. External datasets and any future model-weight releases retain their own licensing and attribution requirements.
