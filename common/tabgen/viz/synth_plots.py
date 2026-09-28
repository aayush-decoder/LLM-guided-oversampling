"""
Plots for one finished run (real train vs synthetic), reproducing the paper's figures:
marginals.png           per-feature real vs synthetic histograms
correlation_diff.png    |corr(real) - corr(synthetic)|
tsne.png                Fig. 8: t-SNE of real and synthetic samples coloured by class
dcr.png                 Fig. 6: DCR histograms (real-test -> real-train vs real -> synthetic)
feature_class_importance.png  Fig. 2: mean |SHAP| per (feature, class) from XGBoost pred_contribs
"""

from typing import Any, Optional, List, Dict, Union

def _save(fig, path):
    pass

def marginals(real, fake, cols, path, max_cols):
    pass

def correlation_diff(real, fake, cols, path):
    pass

def tsne(real, fake, feats, path, n, seed):
    pass

def dcr_hist(split, fake, feats, path):
    pass

def feature_class_importance(real, fake, feats, n_classes, path, seed):
    pass

def plot_run(run_dir, split, fake):
    pass
