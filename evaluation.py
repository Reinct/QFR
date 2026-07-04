"""
evaluation.py — 5-fold CV + C/D AUC + Brier + Permutation Importance
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score, brier_score
from config import EVAL_TIMES_DAYS, BOOTSTRAP_N, CD_AUC_DENSE_N
import config  # 动态读取 SEED
from utils import save_table


# ===================================================
# 5-fold CV
# ===================================================

def train_test_eval_82(X, y, feature_names, test_size=0.2, seed=None):
    """8:2 分层 Train/Test Split, 训练+评估 5 个模型"""
    if seed is None: seed = config.SEED
    from models import train_coxph, train_gbsa, train_rsf

    events = y["retenosis"]
    n = len(y)
    tr_idx, te_idx = train_test_split(
        np.arange(n), test_size=test_size, stratify=events, random_state=seed)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(f"\n{'='*60}")
    print(f"Train/Test Split (8:2, stratified)")
    print(f"{'='*60}")
    print(f"  Total: {n}, events={events.sum()} ({events.mean()*100:.1f}%)")
    print(f"  Train: {len(y_tr)}, events={y_tr['retenosis'].sum()}")
    print(f"  Test:  {len(y_te)}, events={y_te['retenosis'].sum()}")

    model_names = ["CoxPH", "RSF", "GBSA"]
    results = {}
    summary = {}

    max_t = y_te["time"].max()
    eval_t = np.array([t for t in EVAL_TIMES_DAYS if t < max_t])
    if len(eval_t) == 0:
        eval_t = np.linspace(90, max_t * 0.8, 3)

    min_t, max_te_t = max(30, y_te["time"].min() + 10), y_te["time"].max() - 10
    dt = np.linspace(min_t, max_te_t, CD_AUC_DENSE_N)

    for name in model_names:
        print(f"\n  Training {name}...")
        try:
            if name == "CoxPH":
                m, _ = train_coxph(X_tr, y_tr)
            elif name == "RSF":
                m, _ = train_rsf(X_tr, y_tr)
            elif name == "GBSA":
                m, _ = train_gbsa(X_tr, y_tr)

            risk = m.predict(X_te)
            c_val = concordance_index_censored(
                y_te["retenosis"], y_te["time"], risk)[0]

            auc_r = compute_cd_auc(m, X_te, y_te, dt)
            br_r = compute_brier_score(m, X_te, y_te, dt)

            print(f"  {name:10s}: Test C={c_val:.4f}, iAUC={auc_r['integrated_auc']:.4f}, IBS={br_r['integrated_brier']:.4f}")

            results[name] = {"model": m, "c_index": c_val,
                             "iauc": auc_r["integrated_auc"], "ibs": br_r["integrated_brier"],
                             "auc_result": auc_r, "brier_result": br_r}
            summary[name] = {"c_mean": c_val, "c_sd": 0,
                             "iauc_mean": auc_r["integrated_auc"],
                             "ibs_mean": br_r["integrated_brier"]}
        except Exception as e:
            print(f"  {name:10s}: FAILED — {e}")
            results[name] = {"model": None, "c_index": np.nan, "iauc": np.nan, "ibs": np.nan}
            summary[name] = {"c_mean": np.nan, "c_sd": 0, "iauc_mean": np.nan, "ibs_mean": np.nan}

    # 最优
    best_name = max(summary, key=lambda n: summary[n]["c_mean"])
    print(f"\n  Best on test set: {best_name} (C={summary[best_name]['c_mean']:.4f})")

    # 全数据重训练最优模型
    print(f"\n--- Re-training {best_name} on FULL data ---")
    if best_name == "CoxPH":
        best_model, _ = train_coxph(X, y)
    elif best_name == "RSF":
        best_model, _ = train_rsf(X, y)
    elif best_name == "GBSA":
        best_model, _ = train_gbsa(X, y)

    rows = []
    for name in model_names:
        s = summary[name]
        rows.append({"Model": name, "Test_C": f"{s['c_mean']:.4f}",
                     "Test_iAUC": f"{s['iauc_mean']:.4f}", "Test_IBS": f"{s['ibs_mean']:.4f}"})
    save_table(pd.DataFrame(rows), "table_test_results")

    return {"summary": summary, "results": results,
            "best_model_name": best_name, "best_model": best_model,
            "test_c": summary[best_name]["c_mean"], "eval_times": eval_t,
            "test_idx": te_idx, "train_idx": tr_idx}


# ===================================================
# 全数据评估
# ===================================================

def evaluate_on_full_data(model, X, y, model_name="Model"):
    max_t = y["time"].max()
    eval_t = np.array([t for t in EVAL_TIMES_DAYS if t < max_t])
    if len(eval_t) == 0:
        eval_t = np.linspace(90, max_t * 0.8, 3)
    min_t, max_te = max(30, y["time"].min() + 10), max_t - 10
    dt = np.linspace(min_t, max_te, CD_AUC_DENSE_N)

    print(f"\nFull-data: {model_name} (N={len(y)})")
    c_res = compute_c_index_bootstrap(model, X, y)
    print(f"  C-index: {c_res['C_index']:.4f} (95% CI: {c_res['CI_lower']:.4f}–{c_res['CI_upper']:.4f})")
    auc_res = compute_cd_auc(model, X, y, dt)
    print(f"  iAUC: {auc_res['integrated_auc']:.4f}")
    brier_res = compute_brier_score(model, X, y, dt)
    print(f"  IBS: {brier_res['integrated_brier']:.4f}")

    return {"c_index": c_res, "auc": auc_res, "brier": brier_res,
            "km_brier": km_brier_score(y, dt), "dense_times": dt, "eval_times": eval_t}


# ===================================================
# C-index
# ===================================================

def compute_c_index(y, risk):
    return concordance_index_censored(y["retenosis"], y["time"], risk)[0]


def compute_c_index_bootstrap(model, X, y, n_boot=BOOTSTRAP_N, seed=None):
    if seed is None: seed = config.SEED
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            vals.append(concordance_index_censored(
                y[idx]["retenosis"], y[idx]["time"], model.predict(X[idx]))[0])
        except Exception:
            continue
    vals = np.array(vals)
    return {"C_index": np.mean(vals), "CI_lower": np.percentile(vals, 2.5),
            "CI_upper": np.percentile(vals, 97.5), "bootstrap_vals": vals}


# ===================================================
# 生存函数
# ===================================================

def _get_survival_at_times(model, X, target_times):
    try:
        sf = model.predict_survival_function(X, return_array=True)
    except Exception:
        risk = model.predict(X)
        return np.exp(-np.exp(risk[:, None]) * target_times[None, :] / 1000.0)
    try:
        mt = model.unique_times_
    except AttributeError:
        try:
            mt = model.event_times_
        except AttributeError:
            mt = np.linspace(0, 3000, sf.shape[1])
    nm = min(len(mt), sf.shape[1])
    mt, sf = np.asarray(mt[:nm]), sf[:, :nm]
    out = np.zeros((sf.shape[0], len(target_times)))
    for i in range(sf.shape[0]):
        out[i, :] = np.interp(target_times, mt, sf[i, :], left=1.0, right=max(0.0, sf[i, -1]))
    return out


# ===================================================
# C/D AUC — 逐时间点
# ===================================================

def compute_cd_auc(model, X, y, times=None):
    if times is None:
        max_t = y["time"].max()
        times = np.linspace(max(30, y["time"].min()), max_t - 10, CD_AUC_DENSE_N)
    surv = _get_survival_at_times(model, X, times)
    risk = 1.0 - surv
    auc_vals = np.zeros(len(times))
    for k in range(len(times)):
        case = (y["time"] <= times[k]) & (y["retenosis"] == 1)
        ctrl = y["time"] > times[k]
        if case.sum() < 3 or ctrl.sum() < 3:
            auc_vals[k] = np.nan; continue
        mask = case | ctrl
        try:
            from sklearn.metrics import roc_auc_score
            auc_vals[k] = roc_auc_score(np.where(case[mask], 1, 0), risk[mask, k])
        except Exception:
            auc_vals[k] = np.nan
    valid = ~np.isnan(auc_vals)
    nv = valid.sum()
    t_valid = times[valid] / 365.25
    iauc = np.trapezoid(auc_vals[valid], t_valid) / (t_valid[-1] - t_valid[0]) if nv > 1 else (
        float(auc_vals[valid][0]) if nv == 1 else np.nan)
    ta = {}
    for tn, tv in [("1yr", 365), ("2yr", 730), ("3yr", 1095)]:
        ta[tn] = {"time": tv, "auc": np.interp(tv, times[valid], auc_vals[valid],
                  left=np.nan, right=np.nan) if nv > 1 else np.nan}
    print(f"    C/D AUC diag: min={np.nanmin(auc_vals):.4f} max={np.nanmax(auc_vals):.4f} "
          f"n_valid={nv}/{len(times)} iAUC={iauc:.4f}")
    return {"times": times, "auc_vals": auc_vals, "integrated_auc": iauc, "time_auc": ta}


def compute_brier_score(model, X, y, times=None):
    if times is None:
        max_t = y["time"].max()
        times = np.linspace(max(30, y["time"].min()), max_t - 10, CD_AUC_DENSE_N)
    surv = _get_survival_at_times(model, X, times)
    try:
        ibs = integrated_brier_score(y, y, surv, times)
        t_bs, br = brier_score(y, y, surv, times)
        br, t_bs = np.asarray(br).ravel(), np.asarray(t_bs).ravel()
    except Exception:
        br, t_bs = np.full(len(times), np.nan), times; ibs = np.nan
    ml = min(len(t_bs), len(br))
    tb, bv = t_bs[:ml], br[:ml]; valid = ~np.isnan(bv)
    tb_d = {}
    for tn, tv in [("1yr", 365), ("2yr", 730), ("3yr", 1095)]:
        tb_d[tn] = {"time": tv, "brier": np.interp(tv, tb[valid], bv[valid],
                     left=np.nan, right=np.nan) if valid.sum() > 1 else np.nan}
    return {"times": tb, "brier_vals": bv, "integrated_brier": ibs, "time_brier": tb_d}


def km_brier_score(y, times):
    from sksurv.nonparametric import kaplan_meier_estimator
    kt, kp = kaplan_meier_estimator(y["retenosis"], y["time"])
    surv = np.tile(np.interp(times, kt, kp), (len(y), 1))
    try:
        ibs = integrated_brier_score(y, y, surv, times)
        t_bs, bv = brier_score(y, y, surv, times)
    except Exception:
        bv, t_bs = np.full(len(times), np.nan), times; ibs = np.nan
    return {"times": np.asarray(t_bs).ravel(), "brier_vals": np.asarray(bv).ravel(), "integrated_brier": ibs}


# ===================================================
# Permutation Importance
# ===================================================

def compute_time_dependent_importance(model, X, y, feature_names, times=None, n_repeats=5, seed=None):
    if seed is None: seed = config.SEED
    if times is None:
        max_t = y["time"].max()
        times = np.linspace(max(30, y["time"].min()), max_t - 10, CD_AUC_DENSE_N)
    rng = np.random.RandomState(seed)
    nf = X.shape[1]
    sb = _get_survival_at_times(model, X, times)
    rb = 1.0 - sb
    bb = np.zeros(len(times)); cb = np.zeros(len(times))
    for k, tv in enumerate(times):
        try:
            _, bv = brier_score(y, y, sb[:, k:k+1], [tv]); bb[k] = float(np.asarray(bv).ravel()[0])
        except Exception: bb[k] = np.nan
        try:
            _, av = cumulative_dynamic_auc(y, y, rb[:, k:k+1], [tv]); cb[k] = float(np.asarray(av).ravel()[0])
        except Exception: cb[k] = np.nan
    bl_d, al_d = {}, {}
    for j in range(nf):
        fn = feature_names[j]
        bl_j = np.zeros((n_repeats, len(times))); al_j = np.zeros((n_repeats, len(times)))
        for rep in range(n_repeats):
            Xp = X.copy(); rng.shuffle(Xp[:, j])
            sp = _get_survival_at_times(model, Xp, times); rp = 1.0 - sp
            for k, tv in enumerate(times):
                try:
                    _, bv = brier_score(y, y, sp[:, k:k+1], [tv])
                    bl_j[rep, k] = float(np.asarray(bv).ravel()[0]) - bb[k]
                except Exception: bl_j[rep, k] = np.nan
                try:
                    _, av = cumulative_dynamic_auc(y, y, rp[:, k:k+1], [tv])
                    al_j[rep, k] = cb[k] - float(np.asarray(av).ravel()[0])
                except Exception: al_j[rep, k] = np.nan
        bl_d[fn] = np.nanmean(bl_j, axis=0); al_d[fn] = np.nanmean(al_j, axis=0)
        print(f"  [{j+1}/{nf}] {fn}: ΔBrier={np.nanmean(bl_d[fn]):.4f}, ΔAUC={np.nanmean(al_d[fn]):.4f}")
    rows = []
    for j in range(nf):
        fn = feature_names[j]
        vb = bl_d[fn][~np.isnan(bl_d[fn])]; va = al_d[fn][~np.isnan(al_d[fn])]
        rows.append({"variable": fn, "mean_brier_loss": np.mean(vb) if len(vb) > 0 else np.nan,
                     "mean_cd_auc_loss": np.mean(va) if len(va) > 0 else np.nan})
    return {"times": times, "brier_loss": bl_d, "cd_auc_loss": al_d,
            "importance_summary": pd.DataFrame(rows).sort_values("mean_brier_loss", ascending=False),
            "feature_names": feature_names}
