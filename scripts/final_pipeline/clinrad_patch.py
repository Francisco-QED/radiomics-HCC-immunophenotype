"""Add clinico-radiologic columns (prefix 'clin_') to the 'gpt' radiomics classification pipelines without changing their radiomics logic.

Design of the combined model: the radiomics part is untouched (reader-ICC, QC, median imputation, within-sequence clusters, outcome-ranked representatives).
The clinico-radiologic variables bypass ICC and clustering and are APPENDED to the selected radiomics features:
  ridge   : top-k radiomics clusters (k tuned) + ALL clinical variables (always in the model, shrunk with the rest);
  sparse  : every radiomics cluster representative + all clinical variables form one penalized pool; the usual 10-20 nonzero-coefficient constraint then applies to the total.
QC and median imputation (training medians) apply to clinical columns as well. Everything is fit inside the training fold only.
Use apply(core, sp) once per process, before any fit_one call.
"""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

CLIN = "clin_"


class ForceClinical(BaseEstimator, TransformerMixin):
    def __init__(self, base=None, k=None):
        self.base = base; self.k = k

    def fit(self, X, y):
        self.rad_ = [c for c in X.columns if not c.startswith(CLIN)]; self.clin_ = [c for c in X.columns if c.startswith(CLIN)]
        b = clone(self.base)
        if self.k is not None:
            b.set_params(k=self.k)
        b.fit(X[self.rad_], y); self.base_ = b
        self.selected_records_ = b.selected_records_; self.selected_features_ = list(b.selected_features_) + self.clin_
        self.n_clusters_ = b.n_clusters_; self.min_cluster_abs_rho_ = b.min_cluster_abs_rho_
        return self

    def transform(self, X):
        return pd.concat([self.base_.transform(X[self.rad_]), X[self.clin_]], axis=1)


def apply(core, sp):
    orig_make, orig_sparse, orig_k = core.make_pipeline, sp.sparse_pipeline, core.feasible_k_grid

    def make_pipeline(memory):
        p = orig_make(memory); i = [n for n, _ in p.steps].index("select"); p.steps[i] = ("select", ForceClinical(base=p.steps[i][1])); return p

    def sparse_pipeline(memory=None):
        p = orig_sparse(memory); i = [n for n, _ in p.steps].index("select"); p.steps[i] = ("select", ForceClinical(base=p.steps[i][1])); return p

    def feasible_k_grid(X, y, splits):
        return orig_k(X[[c for c in X.columns if not c.startswith(CLIN)]], y, splits)

    def sparse_audit(estimator, icc_filter):
        model = estimator.named_steps["model"]; names = estimator.named_steps["select"].selected_features_
        coef = np.asarray(model.coef_).ravel(); active = np.flatnonzero(np.abs(coef) > 1e-7)
        return {"n_before_icc": icc_filter.n_before_icc_, "n_after_icc": len(icc_filter.columns_), "n_model_features": len(active),
                "selected_features": [names[i] for i in active], "coefficients": list(coef[active]), "n_clinical_in_model": int(sum(names[i].startswith(CLIN) for i in active))}

    core.make_pipeline, sp.sparse_pipeline, core.feasible_k_grid, sp.sparse_audit = make_pipeline, sparse_pipeline, feasible_k_grid, sparse_audit


def icc_keep_clin(RealICC):
    """ICC filter that estimates reliability on radiomic columns only and passes clinical columns through."""
    class ICCKeepClin(RealICC):
        def fit(self, X, y=None):
            super().fit(X[[c for c in X.columns if not c.startswith(CLIN)]], y)
            self.columns_ = list(self.columns_) + [c for c in X.columns if c.startswith(CLIN)]
            return self
    return ICCKeepClin
