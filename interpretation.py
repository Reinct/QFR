"""
interpretation.py — Step 5: 最优模型解释
- SurvSHAP(t): 时间依赖的SHAP值 (使用 KernelExplainer, 聚合哑变量)
- SurvLIME: 局部代理模型解释
- 时间依赖 Permutation Importance (Brier loss + C/D AUC loss)
"""
import numpy as np
import pandas as pd
import shap
from config import (
    SEED, scaler, EVAL_TIMES_DAYS,
    SHAP_SAMPLE_N, SURVLIME_SAMPLE_N, SURVLIME_TOP_FEATURES,
    CP_PROFILE_N_PATIENTS, CP_PROFILE_N_POINTS, SURVSHAP_NSAMPLES,
)
from utils import save_table, save_figure


# ===================================================
# SurvSHAP(t) — 时间依赖 SHAP
# ===================================================

def survshap(model, X, feature_names, y, times=None, background_n=SHAP_SAMPLE_N):
    """
    时间依赖的 SurvSHAP(t)

    策略:
    1. 对每个时间点, 将模型预测的 survival probability 作为 SHAP 目标
    2. 使用 KernelExplainer 估计 SHAP 值
    3. 聚合 One-hot 哑变量到原始变量名

    Parameters:
        model: 训练好的生存模型 (需有 predict_survival_function)
        X: 预测矩阵 (标准化后)
        feature_names: 变量名列表 (One-hot 编码后的列名)
        y: 结构化生存数组
        times: 评估时间点 (默认 1yr, 2yr)
        background_n: 背景样本数

    Returns:
        dict with keys: shap_by_time, importance (已聚合), times
    """
    if times is None:
        times = np.array(EVAL_TIMES_DAYS)
        max_t = y["time"].max()
        times = times[times < max_t]
        if len(times) == 0:
            times = np.percentile(y["time"], [25, 50, 75])

    # 背景数据: kmeans 选取代表性样本
    n_bg = min(background_n, X.shape[0])
    if n_bg < X.shape[0]:
        X_bg_raw = shap.kmeans(X, n_bg)
        X_bg = np.array(X_bg_raw.data) if hasattr(X_bg_raw, 'data') else np.array(X_bg_raw)
    else:
        X_bg = X.copy()

    print(f"[survshap] Background: {X_bg.shape[0]} samples, "
          f"Evaluating at {len(times)} time points")

    # 定义预测函数: 返回每个样本在 target_times 的 survival probability
    def make_surv_predictor(target_times):
        def surv_predict(X_input):
            try:
                surv_funcs = model.predict_survival_function(X_input, return_array=True)
            except Exception:
                # fallback: approximate from risk scores
                risk = model.predict(X_input).ravel()
                surv_funcs = np.exp(-np.exp(risk[:, None]) * target_times[None, :] / np.mean(y["time"]))
                return surv_funcs

            # 获取模型时间点
            try:
                model_times = model.unique_times_
            except AttributeError:
                try:
                    model_times = model.event_times_
                except AttributeError:
                    model_times = np.linspace(0, 3000, surv_funcs.shape[1])

            n_model = min(len(model_times), surv_funcs.shape[1])
            model_times = np.asarray(model_times[:n_model])
            surv_funcs = surv_funcs[:, :n_model]

            # 插值到 target_times
            surv_interp = np.zeros((surv_funcs.shape[0], len(target_times)))
            for i in range(surv_funcs.shape[0]):
                surv_interp[i, :] = np.interp(
                    target_times, model_times, surv_funcs[i, :],
                    left=1.0, right=max(0.0, surv_funcs[i, -1])
                )
            return surv_interp
        return surv_predict

    # 计算全部时间点的 survival, 然后每个时间点单独做 SHAP
    surv_all = make_surv_predictor(times)(X_bg)

    # 对每个时间点计算 SHAP
    shap_results = {}
    print("[survshap] Computing SHAP for each time point...")

    for i, t in enumerate(times):
        print(f"  t={t:.0f} days ({t/365.25:.1f} yr)...")
        try:
            # 使用该时间点的 survival 作为目标
            def predict_t(X_in):
                return make_surv_predictor(np.array([t]))(X_in)[:, 0]

            explainer = shap.KernelExplainer(predict_t, X_bg)
            shap_vals = explainer.shap_values(
                X_bg, nsamples=SURVSHAP_NSAMPLES, silent=True
            )

            shap_results[t] = {
                "values": shap_vals,
                "base_value": float(explainer.expected_value),
                "feature_names": feature_names,
            }
        except Exception as e:
            print(f"    [skip] t={t}: {e}")
            continue

    # -- 全局变量重要性 (聚合哑变量) --
    if shap_results:
        importance = _group_shap_importance(shap_results, feature_names)
    else:
        importance = pd.DataFrame({"variable": feature_names, "mean_abs_shap": np.nan})

    # 打印 Top 变量
    print("\n[survshap] Top 10 features by mean|SHAP| (aggregated):")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['variable']}: {row['mean_abs_shap']:.4f}")

    print("[survshap] Done")
    return {
        "shap_by_time": shap_results,
        "importance": importance,
        "times": times,
    }


def _group_shap_importance(shap_results, feature_names):
    """
    聚合哑变量 SHAP 重要性到原始变量名
    (site_2 + site_3 + site_4 + site_5 → site)
    """
    # 收集每个时间点的平均 |SHAP|
    records = []
    for t, res in shap_results.items():
        imp = np.abs(res["values"]).mean(axis=0)
        sh_names = res.get("feature_names", feature_names)
        for j, name in enumerate(sh_names):
            if j < len(imp):
                records.append({"time": t, "variable": str(name), "mean_abs_shap": imp[j]})

    df = pd.DataFrame(records)

    # 聚合: 将 One-hot 哑变量合并
    grouped = {}
    for _, row in df.iterrows():
        name = str(row["variable"])
        imp = float(row["mean_abs_shap"])
        # 检查是否是哑变量 (如 site_2, ACS_2)
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            orig = parts[0]
        else:
            orig = name
        if orig not in grouped:
            grouped[orig] = 0.0
        grouped[orig] += imp

    # 按总重要性排序
    result = pd.DataFrame([
        {"variable": k, "mean_abs_shap": v}
        for k, v in sorted(grouped.items(), key=lambda x: -x[1])
    ])

    return result


# ===================================================
# SurvLIME — 局部代理模型
# ===================================================

def survlime(model, X, feature_names, y, instance_idx=None,
             n_samples=SURVLIME_SAMPLE_N, top_k=SURVLIME_TOP_FEATURES):
    """
    SurvLIME: 局部加权线性代理模型

    - 对目标实例 x_i, 生成核加权扰动样本
    - 拟合加权 Ridge 回归 (weights = exp(-dist^2 / sigma^2))
    - 系数 = 局部解释 (正=增加风险, 负=保护)
    """
    from sklearn.linear_model import Ridge

    if instance_idx is None:
        rng = np.random.RandomState(SEED)
        instance_idx = rng.choice(X.shape[0], min(3, X.shape[0]), replace=False)
    if isinstance(instance_idx, (int, np.integer)):
        instance_idx = [instance_idx]

    results = {}
    for idx in instance_idx:
        x_i = X[idx]
        print(f"[survlime] Instance {idx}...")

        # 生成扰动样本
        rng = np.random.RandomState(SEED + idx)
        X_std = np.std(X, axis=0)
        X_std[X_std < 1e-8] = 1.0
        X_pert = x_i[None, :] + 0.2 * X_std[None, :] * rng.randn(n_samples, X.shape[1])

        # 距离加权 (高斯核)
        from scipy.spatial.distance import cdist
        dists = cdist(X_pert, x_i[None, :], metric="euclidean").ravel()
        sigma = np.median(dists) if np.median(dists) > 0 else 1.0
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        weights = weights / weights.sum()

        # 黑箱模型预测
        risk_blackbox = model.predict(X_pert)

        # 拟合加权 Ridge
        local_model = Ridge(alpha=0.5)
        local_model.fit(X_pert, risk_blackbox, sample_weight=weights)

        coef = local_model.coef_
        abs_coef = np.abs(coef)
        top_idx = np.argsort(abs_coef)[::-1][:top_k]

        local_df = pd.DataFrame({
            "variable": [feature_names[j] for j in top_idx],
            "coefficient": [coef[j] for j in top_idx],
            "abs_coefficient": [abs_coef[j] for j in top_idx],
        }).sort_values("abs_coefficient", ascending=False)

        # 原单位转换
        local_df["coef_original"] = [
            scaler.unscale_coef(r["variable"], r["coefficient"])
            for _, r in local_df.iterrows()
        ]

        results[idx] = {
            "instance_idx": idx,
            "local_coefficients": local_df,
        }

    print("[survlime] Done")
    return results


# ===================================================
# 时间依赖 Permutation Importance
# (thin wrapper, 核心实现在 evaluation.py)
# ===================================================

def permutation_importance_time(model, X, y, feature_names, times=None,
                                 n_repeats=5, seed=SEED):
    """
    时间依赖的 Permutation Importance (for best model).
    核心逻辑委托给 evaluation.compute_time_dependent_importance.
    """
    from evaluation import compute_time_dependent_importance
    return compute_time_dependent_importance(
        model, X, y, feature_names, times=times,
        n_repeats=n_repeats, seed=seed,
    )
