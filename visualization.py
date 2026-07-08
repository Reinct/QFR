"""
visualization.py — Publication-quality figures (9 figures)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_curve, roc_auc_score
from config import scaler, OUT_DIR
from utils import save_figure

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
sns.set_style("whitegrid")

PALETTE = {"CoxPH": "#D62728", "RSF": "#1F77B4", "GBSA": "#2CA02C"}
MODEL_ORDER = ["CoxPH", "RSF", "GBSA"]


# ═══════════════════════════════════════════════
# Fig 1: LASSO path
# ═══════════════════════════════════════════════
def plot_lasso_path(lasso_info):
    alphas = lasso_info["alphas"]; coef_path = lasso_info["coef_path"]
    best_alpha = lasso_info["best_alpha"]
    top_names = lasso_info.get("top_names", [])
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, coef_path.shape[0]))
    for j in range(coef_path.shape[0]):
        ax.plot(alphas, coef_path[j, :], color=colors[j], linewidth=1.5, alpha=0.8,
                label=top_names[j] if j < 8 else None)
    ax.axvline(x=best_alpha, color="#D62728", linestyle="--", linewidth=1.5,
               label=f"Best alpha = {best_alpha:.6f}")
    ax.set_xscale("log"); ax.set_xlabel("Alpha (log scale)"); ax.set_ylabel("Coefficient")
    ax.set_title("LASSO Cox Coefficient Path"); ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout(); save_figure(fig, "fig1_lasso"); return fig


# ═══════════════════════════════════════════════
# Fig 2: 1yr/2yr/3yr ROC panels
# ═══════════════════════════════════════════════
def plot_roc_panels(roc_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_i, tp in enumerate(["1yr", "2yr", "3yr"]):
        ax = axes[ax_i]
        for name in MODEL_ORDER:
            if name not in roc_data[tp]: continue
            d = roc_data[tp][name]
            fpr, tpr, _ = roc_curve(d["truth"], d["pred"])
            a = d.get("auc_val", roc_auc_score(d["truth"], d["pred"]))
            order = np.argsort(fpr)
            fpr_u, tpr_u = fpr[order], tpr[order]
            # 滑动平均平滑
            w = max(3, len(fpr_u) // 20)
            tpr_sm = np.convolve(tpr_u, np.ones(w)/w, mode='same')
            tpr_sm[0], tpr_sm[-1] = 0, 1
            ax.plot(fpr_u, tpr_sm, color=PALETTE.get(name, "#333"), linewidth=2, label=f"{name} (AUC={a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=8); ax.set_title(f"{tp} ROC (.632+)")
    fig.suptitle("Time-dependent ROC Curves (.632+)", fontsize=14, y=1.02)
    plt.tight_layout(); save_figure(fig, "fig2_roc"); return fig


# ═══════════════════════════════════════════════
# Fig 3: Time-point AUC bar chart
# ═══════════════════════════════════════════════
def plot_timepoint_auc(full_evals):
    fig, ax = plt.subplots(figsize=(10, 6))
    timepoints = ["1yr", "2yr", "3yr"]
    names = [n for n in MODEL_ORDER if n in full_evals]
    colors = [PALETTE.get(n, "#333") for n in names]
    bar_w = 0.22; x = np.arange(len(timepoints))
    for i, name in enumerate(names):
        vals = [full_evals[name]["auc"]["time_auc"].get(tp, {}).get("auc", np.nan) for tp in timepoints]
        vals = [v if not np.isnan(v) else 0 for v in vals]
        bars = ax.bar(x + i*bar_w, vals, bar_w, color=colors[i], alpha=0.85, edgecolor="white", label=name)
        for bar, v in zip(bars, vals):
            if v > 0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}",
                              ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x + bar_w*(len(names)-1)/2)
    ax.set_xticklabels(["1 Year", "2 Year", "3 Year"])
    ax.set_ylabel("C/D AUC"); ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Time-Point C/D AUC (.632+)")
    plt.tight_layout(); save_figure(fig, "fig3_timepoint_auc"); return fig


# ═══════════════════════════════════════════════
# Fig 4: SHAP importance
# ═══════════════════════════════════════════════
def plot_shap_importance(imp_df, model_name="Best", top_n=15):
    df = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(5, len(df)*0.4)))
    dirs = df["direction"].values if "direction" in df.columns else np.zeros(len(df))
    colors = ["#E64B35" if d > 0 else "#1F77B4" for d in dirs]
    ax.barh(np.arange(len(df)), df["importance"].values, color=colors, edgecolor="white", alpha=0.85)
    ax.set_yticks(np.arange(len(df))); ax.set_yticklabels(df["variable"].values); ax.invert_yaxis()
    ax.set_xlabel("Mean(|SHAP value|)")
    ax.set_title(f"SHAP Importance — {model_name}\n(red=high value→high risk, blue=high value→low risk)", fontsize=11)
    plt.tight_layout(); save_figure(fig, "fig3_shap_importance"); return fig


# ═══════════════════════════════════════════════
# Fig 5: SHAP beeswarm
# ═══════════════════════════════════════════════
def plot_shap_beeswarm(shap_values, X_display, feature_names, model_name="Best"):
    import shap
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_display, feature_names=list(feature_names),
                      show=False, max_display=min(15, len(feature_names)))
    plt.title(f"SHAP Beeswarm — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(); save_figure(fig, "fig4_beeswarm"); return fig


# ═══════════════════════════════════════════════
# Fig 6: SHAP waterfall
# ═══════════════════════════════════════════════
def plot_shap_waterfall(shap_values, X_display, feature_names, model_name="Best", patient_idx=0):
    import shap
    sv = np.array(shap_values); base = float(np.mean(sv))
    exp = shap.Explanation(values=sv[patient_idx], base_values=base,
                           data=np.array(X_display)[patient_idx], feature_names=list(feature_names))
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(exp, show=False)
    plt.title(f"SHAP Waterfall — Patient #{patient_idx} ({model_name})", fontsize=13, fontweight="bold")
    plt.tight_layout(); save_figure(fig, f"fig5_waterfall_p{patient_idx}"); return fig


# ═══════════════════════════════════════════════
# Fig 7: SHAP force
# ═══════════════════════════════════════════════
def plot_shap_force(shap_values, X_display, feature_names, model_name="Best", patient_idx=1):
    import shap
    sv = np.array(shap_values); base = float(np.mean(sv))
    fig = shap.force_plot(base, sv[patient_idx], np.array(X_display)[patient_idx],
                          feature_names=list(feature_names), matplotlib=True, show=False)
    if fig is not None:
        fig.set_size_inches(14, 3); fig.tight_layout(); save_figure(fig, f"fig6_force_p{patient_idx}")
    return fig


# ═══════════════════════════════════════════════
# Fig 8: Calibration
# ═══════════════════════════════════════════════
def plot_calibration_single(pred_risk, y, model_name="Model"):
    fig, ax = plt.subplots(figsize=(7, 6))
    truth = y["retenosis"]
    bins = np.percentile(pred_risk, np.linspace(0, 100, 11))
    bc, bo = [], []
    for k in range(10):
        mask = (pred_risk >= bins[k]) & (pred_risk < bins[k+1])
        if mask.sum() > 5: bc.append(pred_risk[mask].mean()); bo.append(truth[mask].mean())
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.scatter(bc, bo, s=80, color="#E64B35", edgecolors="white", linewidth=1, zorder=5)
    ax.plot(bc, bo, "-", color="#E64B35", linewidth=2, alpha=0.7)
    ax.set_xlabel("Predicted Event Probability (OOB)"); ax.set_ylabel("Observed Event Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Calibration Curve — {model_name}")
    plt.tight_layout(); save_figure(fig, "fig7_calibration"); return fig


# ═══════════════════════════════════════════════
# Fig 9: Performance panel
# ═══════════════════════════════════════════════
def plot_cv_panel(cv_summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = [n for n in MODEL_ORDER if n in cv_summary]
    colors = [PALETTE.get(n, "#333") for n in names]
    for ax, metric, label, ylim in [
        (axes[0], "c_mean", "C-index", (0.5, 1.0)),
        (axes[1], "iauc_mean", "2yr AUC", (0.4, 1.0)),
        (axes[2], "ibs_mean", "IBS", (0, 0.2))]:
        vals = [cv_summary[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01 if ylim else 0,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel(label); ax.set_title(f"{label} (.632+)")
        if ylim: ax.set_ylim(*ylim)
    fig.suptitle("Model Performance (.632+ Bootstrap)", fontsize=14, y=1.02)
    plt.tight_layout(); save_figure(fig, "fig8_panel"); return fig
