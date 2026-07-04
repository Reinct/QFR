"""models.py — 3 survival models: CoxPH, RSF, GBSA"""
import numpy as np
import pandas as pd
import config
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from config import RSF_PARAM_GRID, RSF_CV_FOLDS
from utils import save_table


def _cindex_scorer():
    return make_scorer(
        lambda y_t, y_p: concordance_index_censored(y_t["retenosis"], y_t["time"], y_p)[0],
        greater_is_better=True)


def train_coxph(X, y, l1_ratio=0.0, alpha=0.0):
    if l1_ratio == 0 and alpha < 1e-6:
        m = CoxPHSurvivalAnalysis()
    else:
        m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha, max_iter=10000)
    m.fit(X, y)
    c = concordance_index_censored(y["retenosis"], y["time"], m.predict(X))[0]
    print(f"[coxph] C={c:.4f}")
    return m, c


def train_rsf(X, y, params=None):
    if params is None:
        params = {"n_estimators": 500, "min_samples_split": 10,
                  "min_samples_leaf": 3, "max_depth": None}
    m = RandomSurvivalForest(n_jobs=-1, random_state=config.SEED, **params)
    m.fit(X, y)
    c = concordance_index_censored(y["retenosis"], y["time"], m.predict(X))[0]
    print(f"[rsf] C={c:.4f}")
    return m, c


def tune_rsf(X, y):
    scorer = _cindex_scorer()
    search = RandomizedSearchCV(
        RandomSurvivalForest(n_jobs=-1, random_state=config.SEED),
        param_distributions=RSF_PARAM_GRID, n_iter=20, cv=RSF_CV_FOLDS,
        scoring=scorer, random_state=config.SEED, n_jobs=1, error_score="raise")
    search.fit(X, y)
    m = search.best_estimator_
    c = concordance_index_censored(y["retenosis"], y["time"], m.predict(X))[0]
    print(f"[rsf-tune] best={search.best_params_}, CV C={search.best_score_:.4f}, Train C={c:.4f}")
    cv_res = pd.DataFrame(search.cv_results_)
    save_table(cv_res[["param_n_estimators", "param_max_depth",
                        "param_min_samples_split", "param_min_samples_leaf",
                        "mean_test_score", "std_test_score"]].sort_values("mean_test_score", ascending=False),
               "table_rsf_tuning")
    return m, search.best_params_, search.best_score_


def train_gbsa(X, y):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.7, 0.8, 1.0],
    }
    scorer = _cindex_scorer()
    search = RandomizedSearchCV(
        GradientBoostingSurvivalAnalysis(random_state=config.SEED),
        param_distributions=param_grid, n_iter=15, cv=5,
        scoring=scorer, random_state=config.SEED, n_jobs=1)
    search.fit(X, y)
    m = search.best_estimator_
    c = concordance_index_censored(y["retenosis"], y["time"], m.predict(X))[0]
    print(f"[gbsa] best={search.best_params_}, CV C={search.best_score_:.4f}, Train C={c:.4f}")
    return m, c


def train_models(X, y, sel_names):
    """全量训练 3 模型"""
    print("Training CoxPH, RSF, GBSA...")
    cox, _ = train_coxph(X, y, l1_ratio=0.0, alpha=0.0)
    rsf, _, _ = tune_rsf(X, y)
    gbsa, _ = train_gbsa(X, y)

    # RSF importance
    try:
        imp = rsf.feature_importances_
    except Exception:
        imp = np.ones(len(sel_names))
    save_table(pd.DataFrame({"variable": sel_names, "importance": imp}).sort_values("importance", ascending=False),
               "table_rsf_importance")

    return {"cox": cox, "rsf": rsf, "gbsa": gbsa, "feature_names": sel_names}
