import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

#load models
with open("data/models.pkl", "rb") as f:
    models = pickle.load(f)

final_model = models['FF5_Mom']
fitted = final_model.fittedvalues
residuals = final_model.resid


def plot_resid_vs_fitted():
    """plot linearity: residual vs fitted values"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        axes[i].scatter(model.fittedvalues, model.resid, 
                    alpha=0.1, s=3, color='steelblue')
        axes[i].axhline(0, color='red', linewidth=1)
        axes[i].set_title(f"Residuals vs Fitted — {name}")
        axes[i].set_xlabel("Fitted Values")
        axes[i].set_ylabel("Residuals")

    plt.tight_layout()
    plt.savefig("outputs/figures/residuals_all_models.png", dpi=150)
    plt.show()


def test_heteroskedasticity():
    """
    test heteroskedasticity using Breusch Pagan and White tests
    """
    rows = []
    detected = False
    
    for name, model in models.items():

        X = model.model.exog
        resid = model.resid

        #Breusch Pagan
        bp_stat, bp_pval, _, _ = het_breuschpagan(resid, X)
        #White test
        wh_stat, wh_pval, _, _ = het_white(resid, X)

        bp_detected = bp_pval<0.05
        wh_detected = wh_pval<0.05

        if bp_detected or wh_detected:
            detected = True
        rows.append({
            "Model": name,
            "BP Stat": round(bp_stat, 4),
            "BP p-val": round(bp_pval),
            "BP Result":    "Detected" if bp_detected else "No evidence",
            "White Stat":   round(wh_stat, 4),
            "White p-value": round(wh_pval, 4),
            "White Result": "Detected" if wh_detected else "No evidence",
        })
    table = pd.DataFrame(rows)

    print("\nHeteroskedasticity Tests Across All Models:")
    print(table.to_string(index=False))
    table.to_csv("outputs/tables/heteroskedasticity_tests.csv", index=False)

    # if heteroskedasticity detected, then refit models with robust SEs
    robust_models = None
    if detected:
        print("\nHeteroskedasticity detected — refitting all models with HC3 robust SEs")
        robust_models = {
            name: model.get_robustcov_results(cov_type='HC3')
            for name, model in models.items()
        }
        with open("data/robust_models.pkl", "wb") as f:
            pickle.dump(robust_models, f)
        print("Saved robust_models.pkl")

    return robust_models

def test_normality():
    """testing normality using Q-Q plot and Jarque-Bera"""

    #qq plot
    fig, ax = plt.subplots(figsize=(6,6))
    sm.qqplot(residuals, line='s', ax=ax, alpha=0.3, markersize=2, markerfacecolor='green', markeredgecolor='green')
    ax.set_title("Q-Q Plot of residuals for FF5 + Momentum")
    plt.tight_layout()
    plt.savefig("outputs/figures/qq_plot.png", dpi = 150)

    #jarque-bera
    rows = []
    for name, model in models.items():

        jb_stat, jb_pval, skew, kurtosis = jarque_bera(model.resid)
        rows.append({
            "JB Stat":  round(jb_stat,4),
            "P-value":  round(jb_pval,4),
            "Skewness":  round(skew, 4),
            "Kurtosis":  round(kurtosis, 4),
            "Result":    "Non-normal" if jb_pval < 0.05 else "Normal"
        })
    table = pd.DataFrame(rows)
    print("\nJarque-Bera Normality Tests Across All Models:")
    print(table.to_string(index=False))
    table.to_csv("outputs/tables/normality_tests.csv", index=False)
    print("Saved normality_tests.csv")

    return table

def vif():
    """compute variance inflation factors"""
    X = final_model.model.exog
    cols = final_model.model.exog_names
    vif_data = pd.DataFrame({
        "Variable": cols,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    })

    #dropping the constant row
    vif_data = vif_data[vif_data['Variable'] != 'const'].round(4)

    print("\nVariance Inflation Factors ──")
    print(vif_data.to_string(index=False))
    print("Usually VIF > 5 warrants attention, VIF > 10 is serious")

    vif_data.to_csv("outputs/tables/vif.csv", index=False)
    return vif_data

def plot_cooks_dist():
    """plot cooks dist"""
    influence = final_model.get_influence()
    cooks_d = influence.cooks_distance[0]

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(cooks_d)), cooks_d, alpha=0.5, s=1, color='green')
    plt.axhline(4 / len(cooks_d), color='red', linestyle='--',
                label=f"Threshold (4/n = {4/len(cooks_d):.5f})")
    plt.xlabel("Observation Index")
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance — FF5+MOM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/cooks_distance.png", dpi=150)

    #top 10 most influential observations
    paneldf = pd.read_parquet("data/panel.parquet")
    top_influential = (
        paneldf.copy()
        .assign(cooks_d=cooks_d)
        .nlargest(10, 'cooks_d')
        [['date', 'ticker', 'excess_ret', 'cooks_d']]
    )
    top_influential.to_csv("outputs/tables/influential_obs.csv", index=False)
    print("\nTop 10 Most Influential Observations saved")
    

if __name__ == "__main__":
    plot_resid_vs_fitted()
    robust_models = test_heteroskedasticity()
    if robust_models:
        with open("data/robust_models.pkl", "wb") as f:
            pickle.dump(robust_models, f)
    jb_table = test_normality()
    vif()
    plot_cooks_dist()