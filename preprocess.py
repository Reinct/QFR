"""
preprocess.py — Step 1: 数据预处理
- 读取 Excel -> 处理缺失值(MICE) -> One-hot编码 -> 标准化 -> 构造结构化生存数组
- 所有标准化信息存入 scaler 供后续反变换
"""
import pandas as pd
import numpy as np
from config import (
    DATA_PATH, SEED, CONTINUOUS_VARS, CATEGORICAL_VARS,
    OUTCOME_EVENT, OUTCOME_TIME, MICE_ITERATIONS, MICE_M, scaler,
)
from utils import load_data, make_surv_array, save_table


def identify_variable_types(df):
    """自动识别数据中实际存在的连续变量和分类变量"""
    cont_vars = [v for v in CONTINUOUS_VARS if v in df.columns]
    cat_vars  = [v for v in CATEGORICAL_VARS if v in df.columns]
    print(f"[vars] {len(cont_vars)} continuous, {len(cat_vars)} categorical variables found")
    return cont_vars, cat_vars


def report_missing(df):
    """报告缺失情况"""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("[miss] No missing values")
        return
    print(f"[miss] {len(missing)} variables with missing data:")
    for v, n in missing.items():
        pct = 100 * n / len(df)
        print(f"       {v}: {n} ({pct:.1f}%)")


def mice_impute(df, cont_vars, cat_vars):
    """
    MICE 多重插补 (连续变量用 MICE, 分类变量用众数填充)
    不做 One-hot 编码 — 交给 encode_and_scale()
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: required by sklearn
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.linear_model import BayesianRidge

    df_imputed = df.copy()

    # 1. 连续变量: MICE 多重插补
    cont_existing = [v for v in cont_vars if v in df.columns]
    if cont_existing:
        cont_missing = df[cont_existing].isnull().sum().sum()
        if cont_missing > 0:
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=MICE_ITERATIONS,
                random_state=SEED,
                sample_posterior=True,
            )
            cont_imputed = imputer.fit_transform(df[cont_existing])
            df_imputed[cont_existing] = cont_imputed
            print(f"[impute] MICE: {cont_missing} missing values in {len(cont_existing)} continuous vars filled")

    # 2. 分类变量: 众数填充
    cat_existing = [v for v in cat_vars if v in df.columns]
    if cat_existing:
        cat_missing = df[cat_existing].isnull().sum().sum()
        if cat_missing > 0:
            mode_imputer = SimpleImputer(strategy="most_frequent")
            cat_imputed = mode_imputer.fit_transform(df[cat_existing])
            df_imputed[cat_existing] = cat_imputed
            print(f"[impute] Mode: {cat_missing} missing values in {len(cat_existing)} categorical vars filled")

    print(f"[impute] Complete: {df_imputed.shape[1]} columns")
    return df_imputed


def encode_and_scale(df, cont_vars, cat_vars):
    """
    分类变量 One-hot + 连续变量 Z-score 标准化 (只对连续变量)
    返回: df_final (One-hot后), X_matrix (纯预测变量), encoded_cols
    注意: cat_vars 中仅编码实际存在于 df 中的列 (防止重复编码)
    """
    # Step 1: One-hot 编码分类变量 (仅编码仍存在的列)
    existing_cat = [c for c in cat_vars if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_cat, drop_first=True)

    # Step 2: 收集所有预测变量列
    encoded_cols = list(df_encoded.columns)

    # Step 3: 对连续变量标准化
    df_scaled = scaler.scale(df_encoded, cont_vars)

    # Step 4: 提取预测矩阵 (排除结局列)
    pred_cols = [c for c in encoded_cols if c not in (OUTCOME_EVENT, OUTCOME_TIME)]
    X = df_scaled[pred_cols].values.astype(np.float64)

    print(f"[prep] Final design matrix: {X.shape[0]} rows × {X.shape[1]} cols")
    return df_scaled, X, pred_cols


def preprocess():
    """主入口：执行全部预处理并返回结构化数据"""
    print("=" * 60)
    print("STEP 1: Data Preprocessing")
    print("=" * 60)

    # 1. 读取
    df_raw = load_data(DATA_PATH)

    # 1.5 删除缺失结局的行 (结局不能插补)
    n_before = len(df_raw)
    df_raw = df_raw.dropna(subset=[OUTCOME_EVENT, OUTCOME_TIME]).copy()
    n_dropped = n_before - len(df_raw)
    if n_dropped > 0:
        print(f"[drop] {n_dropped} rows with missing outcome (event/time) removed")

    # 2. 识别变量
    cont_vars, cat_vars = identify_variable_types(df_raw)

    # 3. 缺失值处理
    report_missing(df_raw)
    df_clean = mice_impute(df_raw, cont_vars, cat_vars)

    # 4. 编码 & 标准化
    df_scaled, X, pred_cols = encode_and_scale(df_clean, cont_vars, cat_vars)

    # 5. 构造生存结局
    y = make_surv_array(df_clean, OUTCOME_EVENT, OUTCOME_TIME)

    # 6. 保存预处理总结
    summary = pd.DataFrame({
        "variable": list(scaler.means.keys()),
        "mean": list(scaler.means.values()),
        "std":  list(scaler.stds.values()),
    })
    save_table(summary, "preprocess_scaler_params")

    print(f"[done] Preprocessing complete. X: {X.shape}, y: {len(y)} events: {y['retenosis'].sum()}\n")
    return {
        "X": X,
        "y": y,
        "df_scaled": df_scaled,
        "df_clean": df_clean,
        "pred_cols": pred_cols,
        "cont_vars": cont_vars,
        "cat_vars": cat_vars,
    }
