"""
main.py — .632+ Bootstrap (100). OOB metrics. Time-point AUC. SHAP importance.
"""
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

from preprocess import preprocess
from feature_selection import feature_selection
from models import train_coxph, train_rsf, train_gbsa
from sksurv.metrics import concordance_index_censored
from visualization import (
    plot_lasso_path, plot_roc_panels,
    plot_shap_importance, plot_shap_beeswarm, plot_shap_waterfall,
    plot_shap_force, plot_calibration_single, plot_cv_panel,
)
from config import OUT_DIR, CD_AUC_DENSE_N
from utils import save_table


def _map(orig_vars, pred_cols):
    idx = []
    for ov in orig_vars:
        for i, pc in enumerate(pred_cols):
            if pc == ov or pc.startswith(ov + "_"):
                if i not in idx:
                    idx.append(i)
    return sorted(idx)


def _get_surv_matrix(model, X, times):
    try:
        sf = model.predict_survival_function(X, return_array=True)
    except Exception:
        return np.full((X.shape[0], len(times)), np.nan)
    try:
        mt = model.unique_times_
    except AttributeError:
        try: mt = model.event_times_
        except AttributeError: mt = np.linspace(0, 3000, sf.shape[1])
    nm = min(len(mt), sf.shape[1])
    mt, sf = np.asarray(mt[:nm]), sf[:, :nm]
    out = np.zeros((sf.shape[0], len(times)))
    for i in range(sf.shape[0]):
        out[i, :] = np.interp(times, mt, sf[i, :], left=1.0, right=max(0.0, sf[i, -1]))
    return out


def _timepoint_auc(surv_full, y, t_target, times_grid):
    """AUC at a single time point — surv_full is (n, n_times) matrix"""
    k = np.argmin(np.abs(times_grid - t_target))
    event = 1.0 - surv_full[:, k]  # single column
    case = (y["time"] <= t_target) & (y["retenosis"] == 1)
    ctrl = y["time"] > t_target
    mask = case | ctrl
    n_case = case.sum(); n_ctrl = ctrl.sum()
    if n_case < 3 or n_ctrl < 3:
        return np.nan
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(np.where(case[mask], 1, 0), event[mask])
    except Exception:
        return np.nan


def _cd_auc_from_surv(surv, y, times):
    risk = 1.0 - surv
    av = np.zeros(len(times))
    for k in range(len(times)):
        case = (y["time"] <= times[k]) & (y["retenosis"] == 1)
        ctrl = y["time"] > times[k]
        if case.sum() < 3 or ctrl.sum() < 3:
            av[k] = np.nan; continue
        mask = case | ctrl
        try:
            from sklearn.metrics import roc_auc_score
            av[k] = roc_auc_score(np.where(case[mask], 1, 0), risk[mask, k])
        except Exception:
            av[k] = np.nan
    v = ~np.isnan(av); nv = v.sum()
    tv = times[v] / 365.25
    iauc = np.trapezoid(av[v], tv) / (tv[-1] - tv[0]) if nv > 1 else np.nan
    return {"times": times, "auc_vals": av, "integrated_auc": iauc, "time_auc": {}}


def _brier_from_surv(surv, y, times, tag=""):
    from sksurv.metrics import integrated_brier_score, brier_score
    # 缩小时间范围到 y 的实际随访范围
    t_max_y = y["time"].max()
    valid_t = times[times < t_max_y]
    if len(valid_t) < 3:
        return {"times": times, "brier_vals": np.full(len(times), np.nan), "integrated_brier": np.nan}
    surv_valid = surv[:, :len(valid_t)] if surv.shape[1] == len(times) else surv
    try:
        ibs = integrated_brier_score(y, y, surv_valid, valid_t)
        t_bs, br = brier_score(y, y, surv_valid, valid_t)
        return {"times": np.asarray(t_bs).ravel(), "brier_vals": np.asarray(br).ravel(), "integrated_brier": ibs}
    except Exception as e:
        if tag:
            print(f"    [diag] Brier {tag}: {e}")
        return {"times": times, "brier_vals": np.full(len(times), np.nan), "integrated_brier": np.nan}


def main():
    try:
        user_seed = input("Enter random seed (default 42): ").strip()
        seed = int(user_seed) if user_seed else 42
    except ValueError:
        seed = 42
    print(f"  Using seed = {seed}\n")
    import config; config.SEED = seed; np.random.seed(seed)
    print("+" + "=" * 58 + "+")
    print("|  .632+ Bootstrap ×100 — OOB time-point AUC + SHAP imp   |")
    print("+" + "=" * 58 + "+")

    # Step 1
    print("\n" + "=" * 60)
    print("=  STEP 1: Preprocessing")
    print("=" * 60)
    data = preprocess()
    X_full, y_full, pred_cols = data["X"], data["y"], data["pred_cols"]
    df_clean = data["df_clean"]
    n = len(y_full)
    print(f"  N={n}, events={y_full['retenosis'].sum()}")

    # ---- Baseline Table ----
    _build_baseline_table(df_clean, y_full)

    # Step 2
    print("\n" + "=" * 60)
    print("=  STEP 2: LASSO (full data)")
    print("=" * 60)
    sel_result = feature_selection(data)
    sel_idx = _map(sel_result["selected_vars"], pred_cols)
    sel_names = [pred_cols[i] for i in sel_idx]
    X = X_full[:, sel_idx]
    print(f"  {len(sel_names)} features")

    tmin = max(30, y_full["time"].min() + 10)
    tmax = y_full["time"].max() - 10
    if tmax <= tmin:
        tmax = y_full["time"].max(); tmin = max(30, y_full["time"].min())
    dt = np.linspace(tmin, tmax, CD_AUC_DENSE_N)

    # Step 3: Bootstrap
    print("\n" + "=" * 60)
    print("=  STEP 3: .632+ Bootstrap ×100")
    print("=" * 60)

    n_boot = 100
    rng = np.random.RandomState(seed)
    mnames = ["CoxPH", "RSF", "GBSA"]

    boot = {nm: {"c_oob": [], "c_app": [],
                 "auc1_oob": [], "auc2_oob": [], "auc3_oob": [],
                 "ibs_oob": []}
            for nm in mnames}
    oob_surv_sum = {nm: np.zeros((n, len(dt))) for nm in mnames}
    oob_surv_cnt = {nm: np.zeros(n, dtype=int) for nm in mnames}

    for b in range(n_boot):
        bidx = rng.choice(n, size=n, replace=True)
        oidx = np.setdiff1d(np.arange(n), bidx)
        if len(oidx) < 10:
            continue
        X_b, y_b = X[bidx], y_full[bidx]
        X_o, y_o = X[oidx], y_full[oidx]

        for nm in mnames:
            try:
                if nm == "CoxPH":
                    m, _ = train_coxph(X_b, y_b)
                elif nm == "RSF":
                    m, _ = train_rsf(X_b, y_b)
                elif nm == "GBSA":
                    m, _ = train_gbsa(X_b, y_b)

                boot[nm]["c_app"].append(
                    concordance_index_censored(y_b["retenosis"], y_b["time"], m.predict(X_b))[0])
                boot[nm]["c_oob"].append(
                    concordance_index_censored(y_o["retenosis"], y_o["time"], m.predict(X_o))[0])

                so = _get_surv_matrix(m, X_o, dt)
                oob_surv_sum[nm][oidx] += np.nan_to_num(so, 0)
                oob_surv_cnt[nm][oidx] += 1

                # Time-point AUC (OOB only)
                for tp, tval in [("auc1", 365), ("auc2", 730), ("auc3", 1095)]:
                    a = _timepoint_auc(so, y_o, tval, dt)
                    if not np.isnan(a):
                        boot[nm][f"{tp}_oob"].append(a)

                # IBS (OOB only)
                try:
                    br = _brier_from_surv(so, y_o, dt)
                    if not np.isnan(br["integrated_brier"]):
                        boot[nm]["ibs_oob"].append(br["integrated_brier"])
                except Exception: pass
            except Exception:
                continue
        if (b + 1) % 25 == 0:
            print(f"  {b+1}/{n_boot}...")

    # OOB averages
    oob_surv_avg = {}
    for nm in mnames:
        cnt = np.maximum(oob_surv_cnt[nm], 1)
        oob_surv_avg[nm] = oob_surv_sum[nm] / cnt[:, None]

    # Full-data training (for Apparent)
    print(f"\n{'='*60}")
    print("Training full-data models...")
    print(f"{'='*60}")
    full_models = {}
    full_app = {}
    for nm in mnames:
        if nm == "CoxPH": m, _ = train_coxph(X, y_full)
        elif nm == "RSF": m, _ = train_rsf(X, y_full)
        else: m, _ = train_gbsa(X, y_full)
        full_models[nm] = m
        s_full = _get_surv_matrix(m, X, dt)
        full_app[nm] = {
            "auc1": _timepoint_auc(s_full, y_full, 365, dt),
            "auc2": _timepoint_auc(s_full, y_full, 730, dt),
            "auc3": _timepoint_auc(s_full, y_full, 1095, dt),
            "ibs": _brier_from_surv(s_full, y_full, dt)["integrated_brier"],
        }

    # .632+ summary
    print(f"\n{'='*60}")
    print(".632+ Bootstrap Summary")
    print(f"{'='*60}")
    gamma = 0.5
    final = {}
    for nm in mnames:
        res = boot[nm]
        nv = len(res["c_oob"])
        if nv < 10:
            continue
        co = np.array(res["c_oob"])
        ca = np.array(res["c_app"])
        ib = np.array([v for v in res["ibs_oob"] if not np.isnan(v)])
        cm_oob, cm_app = np.mean(co), np.mean(ca)
        R = (cm_oob - cm_app) / (gamma - cm_app) if cm_app < gamma else 0
        if R < 0: R = 0
        w = min(1.0, 0.632 / (1 - 0.368 * R))
        c632 = (1 - w) * cm_app + w * cm_oob

        # CI: bootstrap .632+ estimate itself
        bs_c632 = []
        for _ in range(100):
            bs_idx = rng.choice(len(co), size=len(co), replace=True)
            co_bs = co[bs_idx]; ca_bs = ca[bs_idx]
            cm_o = np.mean(co_bs); cm_a = np.mean(ca_bs)
            R = (cm_o - cm_a) / (gamma - cm_a) if cm_a < gamma else 0
            if R < 0: R = 0
            w = min(1.0, 0.632 / (1 - 0.368 * R))
            bs_c632.append((1 - w) * cm_a + w * cm_o)
        clo, chi = np.percentile(bs_c632, [2.5, 97.5])

        im_ibs = np.mean(ib) if len(ib) > 0 else np.nan
        i_oob = np.mean([v for v in res["ibs_oob"] if not np.isnan(v)])
        fa = full_app.get(nm, {})
        i_app = fa.get("ibs", np.nan)
        ibs_632 = (1-w)*i_app + w*i_oob

        final[nm] = {"c_632plus": c632, "c_lo": clo, "c_hi": chi, "w": w,
                     "c_app": cm_app, "c_oob": cm_oob, "optimism": cm_app - cm_oob,
                     "ibs_oob": i_oob, "ibs_632": ibs_632}
        print(f"\n  {nm}:")
        print(f"    .632+ C: {c632:.4f} (95% CI: {clo:.4f}–{chi:.4f})")
        print(f"    OOB C: {cm_oob:.4f} | App C: {cm_app:.4f}")
        print(f"    IBS: OOB={i_oob:.4f} App={i_app:.4f} .632+={ibs_632:.4f}")

    best_name = max(final, key=lambda x: final[x]["c_632plus"])
    print(f"\n  Best: {best_name}")

    rows = []
    for nm in mnames:
        f = final.get(nm, {})
        rows.append({"Model": nm,
                     "C_632plus": f"{f.get('c_632plus', np.nan):.4f}",
                     "CI_95": f"({f.get('c_lo', np.nan):.4f}–{f.get('c_hi', np.nan):.4f})",
                     "OOB_C": f"{f.get('c_oob', np.nan):.4f}",
                     "AUC_1yr": f"{f.get('auc1_632', np.nan):.4f}",
                     "AUC_2yr": f"{f.get('auc2_632', np.nan):.4f}",
                     "AUC_3yr": f"{f.get('auc3_632', np.nan):.4f}",
                     "IBS_OOB": f"{f.get('ibs_oob', np.nan):.4f}",
                     "IBS_App": f"{full_app.get(nm, {}).get('ibs', np.nan):.4f}",
                     "IBS_632": f"{f.get('ibs_632', np.nan):.4f}",
                     "Optimism": f"{f.get('optimism', np.nan):.4f}"})
    save_table(pd.DataFrame(rows), "table_bootstrap_632plus")

    # .632+ corrected curves
    oof_evals = {}
    # .632+ corrected curves
    oof_evals = {}
    for nm in mnames:
        # OOB curves
        surv_oob = oob_surv_avg[nm]
        auc_oob = _cd_auc_from_surv(surv_oob, y_full, dt)
        brier_oob = _brier_from_surv(surv_oob, y_full, dt)

        # Apparent curves (full-data model)
        s_full = _get_surv_matrix(full_models[nm], X, dt)
        auc_app = _cd_auc_from_surv(s_full, y_full, dt)
        brier_app = _brier_from_surv(s_full, y_full, dt)

        # .632+ weighted curves
        cw = final[nm]["w"] if "w" in final.get(nm, {}) else 0.632
        auc_632_vals = (1-cw) * auc_app["auc_vals"] + cw * auc_oob["auc_vals"]
        iauc_632 = np.trapezoid(auc_632_vals[~np.isnan(auc_632_vals)],
                                dt[~np.isnan(auc_632_vals)] / 365.25) / (
            (dt[-1] - dt[0]) / 365.25) if np.any(~np.isnan(auc_632_vals)) else np.nan

        brier_632_vals = (1-cw) * brier_app["brier_vals"] + cw * brier_oob["brier_vals"]

        oof_evals[nm] = {
            "auc": {"times": dt, "auc_vals": auc_632_vals, "integrated_auc": iauc_632, "time_auc": {}},
            "brier": {"times": dt, "brier_vals": brier_632_vals, "integrated_brier": np.nanmean(brier_632_vals)},
        }
        print(f"    {nm}: .632+ iAUC={iauc_632:.4f}")

    from evaluation import km_brier_score
    oof_km = km_brier_score(y_full, dt)
    for nm in oof_evals:
        oof_evals[nm]["km_brier"] = oof_km

    # .632+ AUC: 在 AUC 值层面加权 (先分别算 App/OOB 的 AUC, 再 .632+)
    from sklearn.metrics import roc_auc_score as _ras
    roc_data = {}
    for tp_label, t_target in [("1yr", 365), ("2yr", 730), ("3yr", 1095)]:
        k = np.argmin(np.abs(dt - t_target))
        roc_data[tp_label] = {}
        for nm in mnames:
            event_app = 1.0 - _get_surv_matrix(full_models[nm], X, dt)[:, k]
            event_oob = 1.0 - oob_surv_avg[nm][:, k]
            case_1yr = y_full["time"] <= t_target
            mask = (case_1yr & (y_full["retenosis"] == 1)) | (~case_1yr)
            y_m = y_full["retenosis"][mask]
            cw = final[nm]["w"]
            p_632 = (1-cw)*event_app[mask] + cw*event_oob[mask]
            a_632 = _ras(y_m, p_632) if y_m.sum() > 0 else np.nan
            roc_data[tp_label][nm] = {"pred": p_632, "truth": y_m, "auc_val": a_632}
            if tp_label == "1yr": final[nm]["auc1_632"] = a_632
            elif tp_label == "2yr": final[nm]["auc2_632"] = a_632
            elif tp_label == "3yr": final[nm]["auc3_632"] = a_632

    summary = {nm: {"c_mean": final[nm]["c_632plus"],
                    "iauc_mean": final[nm]["auc2_632"],  # 2yr AUC
                    "ibs_mean": final[nm]["ibs_632"]} for nm in mnames if nm in final}

    # ---- SHAP (best model) ----
    print("\n" + "=" * 60)
    print(f"=  STEP 4: SHAP — {best_name}")
    print("=" * 60)
    best_model = full_models[best_name]

    X_bg = shap.kmeans(X, min(50, n))
    X_bg_arr = np.array(X_bg.data) if hasattr(X_bg, 'data') else np.array(X_bg)
    def pf(x): return best_model.predict(x)
    n_shap = min(50, n)
    print(f"  Computing SHAP on {n_shap} kmeans centroids x 100 nsamples... (~1 min)")
    X_shap = shap.kmeans(X, n_shap)
    X_shap_arr = np.array(X_shap.data) if hasattr(X_shap, 'data') else np.array(X_shap)
    X_bg2 = shap.kmeans(X, min(30, n))
    X_bg2_arr = np.array(X_bg2.data) if hasattr(X_bg2, 'data') else np.array(X_bg2)
    explainer = shap.KernelExplainer(pf, X_bg2_arr)
    shap_vals = explainer.shap_values(X_shap_arr, nsamples=100, silent=True)
    save_table(pd.DataFrame(shap_vals, columns=sel_names), "table_shap_values")
    X_disp = X_shap_arr

    # SHAP importance
    imp = np.abs(shap_vals).mean(axis=0)
    imp_df = pd.DataFrame({"variable": sel_names, "importance": imp}).sort_values("importance", ascending=False)
    save_table(imp_df, "table_shap_importance")

    # Figures
    print("\n" + "=" * 60)
    print("=  Figures (OOB-based)")
    print("=" * 60)
    try: plot_lasso_path(sel_result["lasso_info"]); print("  [OK] Fig 1: LASSO")
    except Exception as e: print(f"  [FAIL] Fig 1: {e}")
    try: plot_roc_panels(roc_data); print("  [OK] Fig 2: 1/2/3yr ROC (.632+)")
    except Exception as e: print(f"  [FAIL] Fig 2: {e}")
    try: plot_shap_importance(imp_df, model_name=best_name); print("  [OK] Fig 3: SHAP importance")
    except Exception as e: print(f"  [FAIL] Fig 3: {e}")
    try: plot_shap_beeswarm(shap_vals, X_disp, sel_names, model_name=best_name); print("  [OK] Fig 4: SHAP beeswarm")
    except Exception as e: print(f"  [FAIL] Fig 4: {e}")
    try: plot_shap_waterfall(shap_vals, X_disp, sel_names, model_name=best_name, patient_idx=0); print("  [OK] Fig 5: SHAP waterfall")
    except Exception as e: print(f"  [FAIL] Fig 5: {e}")
    try: plot_shap_force(shap_vals, X_disp, sel_names, model_name=best_name, patient_idx=1); print("  [OK] Fig 6: SHAP force")
    except Exception as e: print(f"  [FAIL] Fig 6: {e}")
    try:
        event_2yr = 1.0 - oob_surv_avg[best_name][:, np.argmin(np.abs(dt - 730))]
        plot_calibration_single(event_2yr, y_full, model_name=f"{best_name} (OOB, 2yr)")
        print("  [OK] Fig 7: Calibration (OOB)")
    except Exception as e: print(f"  [FAIL] Fig 7: {e}")
    try: plot_cv_panel(summary); print("  [OK] Fig 8: Panel (.632+)")
    except Exception as e: print(f"  [FAIL] Fig 8: {e}")
    try:
        event_2yr = 1.0 - oob_surv_avg[best_name][:, np.argmin(np.abs(dt - 730))]
        plot_calibration_single(event_2yr, y_full, model_name=f"{best_name} (OOB, 2yr)")
        print("  [OK] Fig 10: Calibration (OOB)")
    except Exception as e: print(f"  [FAIL] Fig 10: {e}")
    try: plot_cv_panel(summary); print("  [OK] Fig 11: Panel (.632+)")
    except Exception as e: print(f"  [FAIL] Fig 11: {e}")

    # 保存模型 (与本次分析完全一致)
    import joblib
    from config import scaler
    joblib.dump({"model": best_model,
                 "scaler_means": scaler.means, "scaler_stds": scaler.stds,
                 "sel_vars": sel_result["selected_vars"], "sel_names": sel_names,
                 "indices": sel_idx, "pred_cols": pred_cols,
                 "X_ref": X[:50], "y_ref": y_full},
                str(OUT_DIR / "streamlit_model.joblib"), compress=3)
    print(f"  Model saved: {OUT_DIR / 'streamlit_model.joblib'}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    f = final[best_name]
    print(f"  {best_name}: .632+ C={f['c_632plus']:.4f} (95% CI: {f['c_lo']:.4f}–{f['c_hi']:.4f})")
    print(f"  AUC 1yr: {f['auc1_632']:.4f} | 2yr: {f['auc2_632']:.4f} | 3yr: {f['auc3_632']:.4f} (.632+)")
    print(f"  IBS: OOB={f['ibs_oob']:.4f} App={full_app[best_name]['ibs']:.4f} .632+={f['ibs_632']:.4f}")


def _build_baseline_table(df, y):
    """Table 1: Baseline characteristics by ISR status"""
    from scipy.stats import ttest_ind, chi2_contingency
    event_mask = y["retenosis"] == 1
    rows = []
    for col in df.columns:
        if col in ("retenosis", "time"):
            continue
        vals = df[col]
        vals0 = vals[~event_mask]
        vals1 = vals[event_mask]
        # continuous (>5 unique values and numeric)
        if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 5:
            try:
                _, p = ttest_ind(vals0, vals1)
            except Exception:
                p = np.nan
            p_str = f"{p:.3f}" if not np.isnan(p) else "–"
            rows.append({
                "Characteristic": col,
                "Overall (n=302)": f"{vals.mean():.2f} ± {vals.std():.2f}",
                "No ISR (n={sum(~event_mask)})": f"{vals0.mean():.2f} ± {vals0.std():.2f}",
                "ISR (n={sum(event_mask)})": f"{vals1.mean():.2f} ± {vals1.std():.2f}",
                "P-value": p_str,
            })
        else:
            try:
                tbl_chi = pd.crosstab(vals, event_mask)
                _, p, _, _ = chi2_contingency(tbl_chi)
            except Exception:
                p = np.nan
            p_str = f"{p:.3f}" if not np.isnan(p) else "–"
            n0, n1 = sum(~event_mask), sum(event_mask)
            rows.append({
                "Characteristic": col,
                "Overall (n=302)": f"{int(vals.sum())} ({vals.mean()*100:.1f}%)",
                f"No ISR (n={n0})": f"{int(vals0.sum())} ({vals0.mean()*100:.1f}%)",
                f"ISR (n={n1})": f"{int(vals1.sum())} ({vals1.mean()*100:.1f}%)",
                "P-value": p_str,
            })
    tbl = pd.DataFrame(rows)
    save_table(tbl, "table1_baseline")
    # Also export a clean version without P-values for paper
    tbl_no_p = tbl.drop(columns=["P-value"])
    save_table(tbl_no_p, "table1_baseline_noP")
    print("  [OK] Baseline table (Table 1) saved: table1_baseline.xlsx")


if __name__ == "__main__":
    main()
