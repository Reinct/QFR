"""
config.py — 全局配置 & 标准化参数追踪
Scale -> Train -> Unscale: 模型训练用标准化数据, 所有输出反变换回原单位
"""
from pathlib import Path

# -- 路径 ------------------------------------------
ROOT      = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data.xlsx"
OUT_DIR   = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# -- 结果保存 ---------------------------------------
SAVE_TABLES  = True         # 输出 Excel 表格
SAVE_FIGURES = True         # 输出 PNG 图片 (300 dpi)
FIGURE_DPI   = 300
FIGURE_FMT   = "png"

# -- 随机种子 ---------------------------------------
SEED = 42

# -- 变量定义 ---------------------------------------
# (根据实际 data.xlsx 列名调整; 以下为模板)
OUTCOME_EVENT = "retenosis"      # 结局事件 (0/1)
OUTCOME_TIME  = "time"           # 随访时间 (天)

# 主要连续暴露变量 (需标准化)
CONTINUOUS_VARS = [
    "QFR_before", "QFR_stent", "delta_qfr",
    "rws", "rws_percent", "rwsqfr_delta", "rwsqfr_percent",
    "ES_QFR", "RE_qfr",
    "age", "stenosis_percent_before", "stenosis_percent_stent",
    "total cholesterol", "triglycerides", "HDL", "LDL",
]

# 分类变量 (One-hot编码)
CATEGORICAL_VARS = [
    "male", "ACS", "site",
    "hypertension", "hyperlipidemia", "diabetes mellitus",
    "somkers", "previous PCI", "previous  stroke",
]

# 所有候选预测变量
ALL_PREDICTORS = CONTINUOUS_VARS + CATEGORICAL_VARS

# -- 变量筛选阈值 -----------------------------------
UNIVARIATE_P_THRESHOLD  = 0.20   # 单因素Cox入选阈值 (放宽以纳入更多候选)
VIF_THRESHOLD           = 5.0    # 共线性阈值

# -- LASSO 变量筛选 ---------------------------------
LASSO_N_ALPHAS    = 200   # alpha 路径长度
LASSO_CV_FOLDS    = 3     # 3 折 → 每折 ~100 人, 更稳定
LASSO_L1_RATIO    = 1.0   # 1.0 = pure LASSO, <1.0 = elastic net
LASSO_RULE        = "lambda.1se"  # "lambda.min" or "lambda.1se"

# -- 缺失数据处理 -----------------------------------
MICE_ITERATIONS = 10   # 多重插补迭代次数
MICE_M          = 5    # 插补数据集数量

# -- RSF 超参数搜索 --------------------------------
RSF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":          [3, 5, 7, None],
    "min_samples_split":  [5, 10, 20, 30],
    "min_samples_leaf":   [3, 5, 10, 15],
}
RSF_CV_FOLDS = 5

# -- 模型评估时间点 ---------------------------------
EVAL_TIMES_DAYS = [365, 730, 1095]  # 1年, 2年, 3年 (用于时间点报告)
CD_AUC_DENSE_N   = 50              # C/D AUC 曲线的时间点密度
BOOTSTRAP_N     = 1000              # Bootstrap 次数

# -- 解释性分析 -------------------------------------
SHAP_SAMPLE_N        = 50           # SurvSHAP背景样本数 (kmeans质心)
SURVLIME_SAMPLE_N     = 300         # SurvLIME扰动样本数
SURVLIME_TOP_FEATURES = 8           # SurvLIME展示Top特征数
SURVSHAP_NSAMPLES     = 150         # KernelExplainer 采样数 (平衡速度/精度)
CP_PROFILE_N_PATIENTS = 3           # CP Profile展示患者数
CP_PROFILE_N_POINTS   = 50          # CP Profile每个特征的网格点数


# ===================================================
# 标准化参数容器 (在 preprocess.py 中填充)
# ===================================================
class ScalerParams:
    """记录每个连续变量的 mean 和 std，用于反变换"""
    def __init__(self):
        self.means = {}   # {var_name: mean}
        self.stds  = {}   # {var_name: std}

    def scale(self, df, vars_):
        """Z-score 标准化"""
        import pandas as pd
        df_scaled = df.copy()
        for v in vars_:
            if v in df.columns:
                mu = df[v].mean()
                sg = df[v].std()
                if sg == 0:
                    sg = 1e-8
                self.means[v] = mu
                self.stds[v]  = sg
                df_scaled[v] = (df[v] - mu) / sg
        return df_scaled

    def unscale_coef(self, var_name, beta_scaled):
        """将标准化系数反变换回原单位系数: β_original = β_scaled / σ"""
        if var_name in self.stds and self.stds[var_name] > 1e-8:
            return beta_scaled / self.stds[var_name]
        return beta_scaled

    def unscale_or_ci(self, var_name, beta_scaled, se_scaled=None):
        """返回原单位的 OR 和 95% CI"""
        import numpy as np
        beta_original = self.unscale_coef(var_name, beta_scaled)
        OR = np.exp(beta_original)
        if se_scaled is not None:
            se_original = se_scaled / self.stds.get(var_name, 1.0)
            ci_lower = np.exp(beta_original - 1.96 * se_original)
            ci_upper = np.exp(beta_original + 1.96 * se_original)
            return {"OR": OR, "CI_lower": ci_lower, "CI_upper": ci_upper,
                    "beta": beta_original, "se": se_original}
        return {"OR": OR, "beta": beta_original}

    def unscale_value(self, var_name, scaled_value):
        """单个值反变换: x_original = x_scaled * σ + μ"""
        if var_name in self.means and var_name in self.stds:
            return scaled_value * self.stds[var_name] + self.means[var_name]
        return scaled_value

    def unscale_array(self, var_name, scaled_array):
        """批量反变换"""
        import numpy as np
        if var_name in self.means and var_name in self.stds:
            return scaled_array * self.stds[var_name] + self.means[var_name]
        return scaled_array


scaler = ScalerParams()
