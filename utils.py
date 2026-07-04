"""
utils.py — 通用工具函数
"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from config import OUT_DIR, FIGURE_DPI, FIGURE_FMT, SAVE_FIGURES, SAVE_TABLES, scaler

warnings.filterwarnings("ignore")

# ===================================================
# 数据 IO
# ===================================================

def load_data(path):
    """读取 Excel 数据"""
    df = pd.read_excel(path)
    print(f"[load] {df.shape[0]} rows × {df.shape[1]} cols loaded from {path}")
    return df


def save_table(df, name, float_fmt="%.4f"):
    """保存表格到 Excel"""
    if not SAVE_TABLES:
        return
    p = OUT_DIR / f"{name}.xlsx"
    df.to_excel(p, index=False, float_format=float_fmt)
    print(f"[save] {p}")


def save_figure(fig, name):
    """保存图片到 PNG"""
    if not SAVE_FIGURES:
        return
    p = OUT_DIR / f"{name}.{FIGURE_FMT}"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[save] {p}")


# ===================================================
# 生存数据构造
# ===================================================

def make_surv_array(df, event_col, time_col):
    """构造 scikit-survival 需要的结构化数组 y"""
    from sksurv.util import Surv
    y = Surv.from_dataframe(event=event_col, time=time_col, data=df)
    return y


# ===================================================
# VIF 共线性诊断
# ===================================================

def calc_vif(X):
    """计算 Variance Inflation Factor (仅数值列)"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import numpy as np

    # 确保全数值
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] < 2:
        return pd.DataFrame({"variable": X.columns, "VIF": [np.nan] * X.shape[1]})

    vifs = []
    for i in range(X_num.shape[1]):
        try:
            v = variance_inflation_factor(X_num.values, i)
            vifs.append(v)
        except Exception:
            vifs.append(np.nan)
    vif_df = pd.DataFrame({"variable": X_num.columns.tolist(), "VIF": vifs})
    return vif_df.sort_values("VIF", ascending=False)


# ===================================================
# Bootstrap 工具
# ===================================================

def bootstrap_ci(values, alpha=0.05, n_boot=1000, seed=42):
    """Bootstrap 95% CI"""
    rng = np.random.RandomState(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(np.mean(values[idx]))
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return np.mean(boots), lo, hi


# ===================================================
# 比例风险假设检验
# ===================================================

def test_ph_assumption(cph_model):
    """Schoenfeld 残差检验"""
    from lifelines.statistics import proportional_hazard_test
    try:
        results = proportional_hazard_test(cph_model, cph_model.train_log_partial_hazards_,
                                           cph_model.train_residuals_,
                                           cph_model.train_var_names_)
        return results.summary
    except Exception as e:
        print(f"[warn] PH test failed: {e}")
        return None


# ===================================================
# 标准化 ↔ 原单位 工具
# ===================================================

def group_dummy_importance(imp_df, feature_names):
    """
    将 One-hot 哑变量重要性聚合回原始变量
    site_2=0.1, site_3=0.05 -> site=0.15
    """
    if imp_df is None or len(imp_df) == 0:
        return imp_df
    grouped = {}
    for _, row in imp_df.iterrows():
        name = row["variable"]
        imp = float(row.get("importance", 0))
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            orig = parts[0]
        else:
            orig = name
        grouped[orig] = grouped.get(orig, 0) + imp
    import pandas as pd
    return pd.DataFrame(
        [{"variable": k, "importance": v} for k, v in sorted(grouped.items(), key=lambda x: -x[1])]
    )


def unscale_dataframe(df_scaled, vars_=None):
    """将标准化 DataFrame 反变换回原单位"""
    df_original = df_scaled.copy()
    if vars_ is None:
        vars_ = [v for v in df_original.columns if v in scaler.means]
    for v in vars_:
        if v in scaler.means and v in scaler.stds:
            df_original[v] = scaler.unscale_array(v, df_scaled[v].values)
    return df_original


def unscale_coef_table(coef_table):
    """
    将系数表从标准化空间转换到原单位空间。
    coef_table: DataFrame with columns [variable, beta, se, p, ...]
    返回添加了 OR, OR_CI_lower, OR_CI_upper 列的 DataFrame
    """
    table = coef_table.copy()
    ors, ci_lo, ci_hi = [], [], []
    for _, row in table.iterrows():
        var = row["variable"]
        beta = row["beta"]
        se   = row.get("se", None)
        result = scaler.unscale_or_ci(var, beta, se)
        ors.append(result["OR"])
        ci_lo.append(result["CI_lower"])
        ci_hi.append(result["CI_upper"])
    table["OR"]            = ors
    table["OR_CI_lower"]   = ci_lo
    table["OR_CI_upper"]   = ci_hi
    table["beta_original"] = [scaler.unscale_coef(r["variable"], r["beta"]) for _, r in table.iterrows()]
    return table
