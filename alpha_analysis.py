import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import significance_stars

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)

with open("data/models.pkl", "rb") as f:
    models = pickle.load(f)

with open("data/clustered_models.pkl", "rb") as f:
    robust_models = pickle.load(f)

def build_alpha_table():
    """alpha comparison table across all models"""

    rows = []
    for name in models.keys():
        m = models[name]
        rm = robust_models[name]
        robust_sum = rm.summary2().tables[1]

        alpha_coef = m.params['const']
        alpha_se = robust_sum.loc['const', 'Std.Err.'] #robust model standard error
        alpha_pval = robust_sum.loc['const', 'P>|t|'] #robust model p value
        ci_low = robust_sum.loc['const', '[0.025']
        ci_high = robust_sum.loc['const', '0.975]']
        stars = significance_stars(alpha_pval)

        rows.append({
            "Model": name,
            "Alpha": round(alpha_coef, 6),
            "Robust Model SE": round(alpha_se,6),
            "P-value": round(alpha_pval, 4),
            "Significance": stars,
            "CI lower": round(ci_low, 6),
            "CI upper": round(ci_high, 6),
            "Conclusion": "Significant" if alpha_pval < 0.05 else "Not significant"
        })
    table = pd.DataFrame(rows)
    table.to_csv("outputs/tables/alpha_analysis.csv", index=False)
    print("Alpha Across Models saved in outputs")
    return table

def plot_alpha(alpha_table):
    """plot alpha with confidence intervals"""
    model_names = alpha_table['Model']
    alphas = alpha_table['Alpha']
    ci_low = alpha_table['CI lower']
    ci_high = alpha_table['CI upper']

    yerr_low = alphas - ci_low
    yerr_high = ci_high - alphas

    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(
        x=range(len(model_names)),
        y=alphas,
        yerr=[yerr_low, yerr_high],
        fmt='o',
        color='green',
        ecolor='lightgreen',
        elinewidth=2,
        capsize=5,
        markersize=8,
        label='Alpha(intercept)'
    )
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Alpha = 0')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Alpha")
    ax.set_title("Alpha with 95% Confidence Intervals Across Models")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/alpha_analysis.png", dpi=150)
    print("Saved alpha_analysis.png")

def interpret_alpha(alpha_table):
    """interpret the pattern"""
    print("\nAlpha Interpretation")
    for _, row in alpha_table.iterrows():
        direction = "negative" if row['Alpha'] < 0 else "positive"
        print(f"{row['Model']:10s}: alpha={row['Alpha']:.6f} ({direction}), "
              f"p={row['P-value']:.4f}, {row['Conclusion']}")


if __name__ == "__main__":
    alpha_table = build_alpha_table()
    plot_alpha(alpha_table)
    interpret_alpha(alpha_table)