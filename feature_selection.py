"""
feature_selection.py — Step 2: 变量筛选
- 单因素Cox回归 (对原始变量, 不做One-hot)
- LASSO Cox (Coxnet) + 交叉验证, 解决哑变量碎片化问题
- VIF 共线性诊断
- 所有系数输出反变换回原单位
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sksurv.metrics import concordance_index_censored

import config
from config import (
    UNIVARIATE_P_THRESHOLD, VIF_THRESHOLD, OUTCOME_EVENT, OUTCOME_TIME,
    LASSO_N_ALPHAS, LASSO_CV_FOLDS, LASSO_L1_RATIO, LASSO_RULE,
    scaler,
)
from utils import save_table, calc_vif, unscale_coef_table, make_surv_array


def univariate_cox_on_raw(df_raw, cont_vars, cat_vars):
    """
    单因素Cox回归 — 使用原始变量 (不做One-hot)
    对于分类变量, lifelines 内部处理为 k-1 个哑变量,
    我们取 Wald 检验的整体 P 值
    """
    time_col = OUTCOME_TIME
    event_col = OUTCOME_EVENT

    # 准备: 确保分类变量是 category dtype
    df = df_raw.copy()
    for v in cat_vars:
        if v in df.columns:
            df[v] = df[v].astype("category")

    all_vars = cont_vars + cat_vars
    results = []
    for var in all_vars:
        if var not in df.columns:
            continue
        try:
            cph = CoxPHFitter(penalizer=0.0)
            df_tmp = df[[var, time_col, event_col]].dropna()
            cph.fit(df_tmp, duration_col=time_col, event_col=event_col)
            summary = cph.summary

            if len(summary) == 1:
                # 连续变量或二分类变量 — 直接取
                p_val = summary["p"].values[0]
                beta  = summary["coef"].values[0]
                se    = summary["se(coef)"].values[0]
            else:
                # 多分类变量 (k>2) — 取整体 Wald p 值
                # lifelines 对 category 变量使用整体检验
                p_vals = summary["p"].values
                # 使用 log-likelihood ratio test
                ll_null = cph.log_likelihood_ratio_test()
                p_val = ll_null.p_value if hasattr(ll_null, 'p_value') else p_vals.min()
                # 取第一个系数和SE作为占位
                beta = summary["coef"].values[0]
                se   = summary["se(coef)"].values[0]

            results.append({
                "variable": var,
                "beta": beta,
                "se": se,
                "p": p_val,
                "HR": np.exp(beta),
                "HR_CI_lower": np.exp(beta - 1.96 * se),
                "HR_CI_upper": np.exp(beta + 1.96 * se),
                "significant": p_val < UNIVARIATE_P_THRESHOLD,
                "n_coefs": len(summary),
            })
        except Exception as e:
            # fallback: 对哑变量逐一测试
            print(f"  [fallback] {var}: {e}")
            try:
                for col in df.columns:
                    if col.startswith(var + "_") or col == var:
                        cph = CoxPHFitter(penalizer=0.0)
                        df_tmp = df[[col, time_col, event_col]].dropna()
                        if df_tmp[col].nunique() < 2:
                            continue
                        cph.fit(df_tmp, duration_col=time_col, event_col=event_col)
                        s = cph.summary
                        results.append({
                            "variable": col,
                            "beta": s["coef"].values[0],
                            "se": s["se(coef)"].values[0],
                            "p": s["p"].values[0],
                            "HR": np.exp(s["coef"].values[0]),
                            "HR_CI_lower": np.exp(s["coef"].values[0] - 1.96 * s["se(coef)"].values[0]),
                            "HR_CI_upper": np.exp(s["coef"].values[0] + 1.96 * s["se(coef)"].values[0]),
                            "significant": s["p"].values[0] < UNIVARIATE_P_THRESHOLD,
                            "n_coefs": 1,
                        })
            except Exception as e2:
                print(f"    [skip] {var}: {e2}")

    df_result = pd.DataFrame(results).sort_values("p")
    n_sig = df_result["significant"].sum() if "significant" in df_result.columns else 0
    print(f"[univariate] {len(results)} variables tested, {n_sig} with P<{UNIVARIATE_P_THRESHOLD}")
    return df_result


def lasso_cox_selection(df_scaled, X_scaled, y, pred_cols, candidate_vars):
    """
    LASSO Cox 变量筛选 (Coxnet, GridSearchCV 选 alpha)

    策略:
    1. 对候选变量构造设计矩阵 (含 One-hot 哑变量)
    2. GridSearchCV 搜索最优 alpha
    3. 非零系数变量入选
    4. 哑变量映射回原始分类变量名

    Returns: selected_orig_vars (list), lasso_model, info_dict
    """
    # 1. 映射: 原始变量名 -> One-hot 列索引
    selected_indices = []
    selected_names = []

    for cv in candidate_vars:
        for i, pc in enumerate(pred_cols):
            if pc == cv or pc.startswith(cv + "_"):
                if i not in selected_indices:
                    selected_indices.append(i)
                    selected_names.append(pc)

    if len(selected_indices) < 3:
        selected_indices = list(range(X_scaled.shape[1]))
        selected_names = list(pred_cols)

    X_cand = X_scaled[:, selected_indices]
    print(f"[lasso] Candidate design matrix: {X_cand.shape[1]} columns from {len(candidate_vars)} original variables")

    # 2. 手动 CV 选 alpha (逐 alpha 拟合, 避免 coef_path_ API 差异)
    n_cv = min(LASSO_CV_FOLDS, X_cand.shape[0] // 10)
    n_cv = max(3, n_cv)

    # 先拟合一个模型获取 alpha 路径
    lasso_init = CoxnetSurvivalAnalysis(
        l1_ratio=LASSO_L1_RATIO, n_alphas=LASSO_N_ALPHAS,
        max_iter=100000, tol=1e-7,
    )
    lasso_init.fit(X_cand, y)
    alphas = lasso_init.alphas_

    # 降采样 alpha (选 representative 点减少拟合次数)
    n_sample = min(50, len(alphas))
    alpha_sample = np.logspace(np.log10(alphas[0]), np.log10(alphas[-1]), n_sample)

    from sklearn.model_selection import StratifiedKFold, KFold
    try:
        kf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=config.SEED)
        splits = list(kf.split(X_cand, y["retenosis"]))
    except ValueError:
        kf = KFold(n_splits=n_cv, shuffle=True, random_state=config.SEED)
        splits = list(kf.split(X_cand))

    # CV: 对每个 alpha, 拟合模型并评分
    cv_scores = np.zeros((n_sample, n_cv))
    for a_idx, alpha in enumerate(alpha_sample):
        for fold_i, (tr_idx, te_idx) in enumerate(splits):
            X_tr, X_te = X_cand[tr_idx], X_cand[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            m = CoxnetSurvivalAnalysis(
                alphas=[alpha], l1_ratio=LASSO_L1_RATIO,
                max_iter=100000, tol=1e-7,
            )
            m.fit(X_tr, y_tr)
            coef = np.asarray(m.coef_).ravel()
            risk = X_te @ coef
            if np.all(np.abs(coef) < 1e-8):
                cv_scores[a_idx, fold_i] = 0.5
            else:
                cv_scores[a_idx, fold_i] = concordance_index_censored(
                    y_te["retenosis"], y_te["time"], risk)[0]

    # 选最优 alpha
    mean_cv = cv_scores.mean(axis=1)
    se_cv = cv_scores.std(axis=1) / np.sqrt(n_cv)

    if LASSO_RULE == "lambda.min":
        best_idx = np.argmax(mean_cv)
    else:
        best_min = np.argmax(mean_cv)
        threshold = mean_cv[best_min] - se_cv[best_min]
        best_idx = best_min
        for i in range(best_min, n_sample):
            if mean_cv[i] >= threshold:
                best_idx = i
            else:
                break

    best_alpha = alpha_sample[best_idx]
    # 用 best_alpha 在全量数据上重拟合
    best_lasso = CoxnetSurvivalAnalysis(
        alphas=[best_alpha], l1_ratio=LASSO_L1_RATIO, max_iter=100000, tol=1e-7,
    )
    best_lasso.fit(X_cand, y)
    best_coef = np.asarray(best_lasso.coef_).ravel()

    # 4. 非零系数变量
    nonzero_mask = np.abs(best_coef) > 1e-6
    selected_final_idx = np.where(nonzero_mask)[0]
    selected_final_names = [selected_names[i] for i in selected_final_idx]

    # 5. 映射回原始变量名 (去重, 去哑变量碎片)
    original_selected = _map_to_original_vars(selected_final_names, candidate_vars)

    print(f"[lasso] Best alpha={best_alpha:.6f}, CV C-index={mean_cv[best_idx]:.4f}")
    print(f"[lasso] Selected {len(original_selected)} original variables:")
    for v in original_selected:
        n_coefs = sum(1 for sn in selected_final_names if sn == v or sn.startswith(v + "_"))
        print(f"         {v} ({n_coefs} coef)")

    # 保存系数表
    lasso_result = pd.DataFrame({
        "variable": selected_names,
        "coefficient": best_coef,
        "selected": nonzero_mask.astype(int),
    }).sort_values("coefficient", key=abs, ascending=False)
    save_table(lasso_result, "table_lasso_coefficients")

    # 构建系数路径 (用于绑图: 对采样 alpha 拟合并记录 top 系数)
    n_top = min(20, X_cand.shape[1])
    top_coef_idx = np.argsort(np.abs(best_coef))[::-1][:n_top]
    coef_path = np.zeros((n_top, n_sample))
    for a_idx, alpha in enumerate(alpha_sample):
        m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=LASSO_L1_RATIO, max_iter=100000)
        m.fit(X_cand, y)
        coef_path[:, a_idx] = np.asarray(m.coef_).ravel()[top_coef_idx]

    return original_selected, best_lasso, {
        "alphas": alpha_sample,
        "coef_path": coef_path,
        "best_alpha": best_alpha,
        "selected_names": selected_final_names,
        "original_selected": original_selected,
        "top_names": [selected_names[i] for i in top_coef_idx],
    }


def _cv_select_alpha(X, y, alphas, coef_path, rule="lambda.1se"):
    """
    交叉验证选择最优 alpha
    使用 concordance index 作为评分
    """
    from sklearn.model_selection import StratifiedKFold

    n = X.shape[0]
    n_alphas = len(alphas)

    # 使用几折 CV
    cv = min(LASSO_CV_FOLDS, n // 10)
    cv = max(3, cv)

    # Stratified by event
    events = y["retenosis"]
    try:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.SEED)
        splits = list(kf.split(X, events))
    except ValueError:
        # 如果某个类别太少, 回退到 KFold
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=config.SEED)
        splits = list(kf.split(X))

    cv_scores = np.zeros((n_alphas, cv))

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 训练模型 (整个路径一次训练, 再对不同 alpha 评估)
        model = CoxnetSurvivalAnalysis(
            l1_ratio=LASSO_L1_RATIO,
            n_alphas=n_alphas,
            max_iter=100000,
            tol=1e-7,
        )
        model.fit(X_tr, y_tr)

        # 在所有 alpha 上评估
        for a_idx in range(len(model.alphas_)):
            coef = model.coef_path_[:, a_idx]
            if np.all(np.abs(coef) < 1e-8):
                cv_scores[a_idx, fold_i] = 0.5
            else:
                risk = X_te @ coef
                cv_scores[a_idx, fold_i] = concordance_index_censored(
                    y_te["retenosis"], y_te["time"], risk
                )[0]

    mean_cv = cv_scores.mean(axis=1)
    se_cv = cv_scores.std(axis=1) / np.sqrt(cv)

    if rule == "lambda.min":
        best_idx = np.argmax(mean_cv)
    else:  # lambda.1se
        best_idx_lambda_min = np.argmax(mean_cv)
        threshold = mean_cv[best_idx_lambda_min] - se_cv[best_idx_lambda_min]
        # 选 largest alpha (most regularization) within 1 SE of max
        for i in range(best_idx_lambda_min, len(alphas)):
            if mean_cv[i] >= threshold:
                best_idx = i
            else:
                break
        else:
            best_idx = best_idx_lambda_min

    best_alpha = alphas[best_idx]
    best_coef = coef_path[:, best_idx] if coef_path.shape[1] > best_idx else coef_path[:, -1]

    print(f"[lasso] CV: lambda.min C={mean_cv[np.argmax(mean_cv)]:.4f}, "
          f"lambda.1se C={mean_cv[best_idx]:.4f} (alpha={best_alpha:.4f})")

    return best_alpha, best_coef


def _map_to_original_vars(selected_dummy_names, original_candidates):
    """
    将 One-hot 列名映射回原始变量名
    e.g., ['ACS_2', 'site_2', 'site_3', 'rws', 'QFR_stent']
    -> ['ACS', 'site', 'rws', 'QFR_stent']
    """
    mapped = []
    seen = set()

    for name in selected_dummy_names:
        # 检查是否属于某个原始分类变量
        matched = False
        for orig in sorted(original_candidates, key=len, reverse=True):
            if name == orig:
                if orig not in seen:
                    mapped.append(orig)
                    seen.add(orig)
                matched = True
                break
            if name.startswith(orig + "_"):
                if orig not in seen:
                    mapped.append(orig)
                    seen.add(orig)
                matched = True
                break
        if not matched:
            # 连续变量或无法映射
            if name not in seen:
                mapped.append(name)
                seen.add(name)

    return mapped


# ═══════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════

def feature_selection(data):
    """Step 2 主入口: 单因素筛选 -> LASSO 选择"""
    print("=" * 60)
    print("STEP 2: Feature Selection (Univariate + LASSO Cox)")
    print("=" * 60)

    df_scaled = data["df_scaled"]
    df_clean = data["df_clean"]
    X = data["X"]
    y = data["y"]
    pred_cols = data["pred_cols"]
    cont_vars = data["cont_vars"]
    cat_vars = data["cat_vars"]

    # 2.1 单因素Cox (仅用于 forest plot, 不用于筛选)
    print("\n--- Univariate Cox (for forest plot only) ---")
    uni_results = univariate_cox_on_raw(df_clean, cont_vars, cat_vars)
    save_table(uni_results, "table_univariate_cox")

    # 直接用全部变量 → LASSO
    candidates = cont_vars + cat_vars
    print(f"\n[LASSO input] {len(candidates)} variables (no univariate pre-filter)")

    # 2.2 VIF 检查 (仅连续变量)
    print("\n--- VIF Collinearity Check ---")
    cont_candidates = [c for c in candidates if c in cont_vars]
    if len(cont_candidates) >= 2:
        X_vif = df_scaled[cont_candidates].dropna()
        vif_df = calc_vif(X_vif)
        save_table(vif_df, "table_vif")
        high_vif = vif_df[vif_df["VIF"] > VIF_THRESHOLD]
        if len(high_vif) > 0:
            for _, row in high_vif.iterrows():
                print(f"  [high VIF] {row['variable']}: VIF={row['VIF']:.1f}")
    else:
        vif_df = pd.DataFrame({"variable": cont_candidates, "VIF": [np.nan] * len(cont_candidates)})
        print("  (not enough continuous variables for VIF)")

    # 2.3 LASSO Cox 选择
    print("\n--- LASSO Cox Regression ---")
    selected_vars, lasso_model, lasso_info = lasso_cox_selection(
        df_scaled, X, y, pred_cols, candidates
    )

    print(f"\nFinal selected variables ({len(selected_vars)}):")
    for i, v in enumerate(selected_vars):
        print(f"  {i+1}. {v}")

    return {
        "uni_results": uni_results,
        "selected_vars": selected_vars,
        "lasso_model": lasso_model,
        "lasso_info": lasso_info,
        "vif_df": vif_df,
    }
