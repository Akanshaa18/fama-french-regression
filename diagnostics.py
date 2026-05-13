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
    plt.figure(figsize=(8,5))
    plt.scatter(fitted, residuals, alpha=0.5, s=5, color='green')
    plt.axhline(0, color= 'red', linewidth=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted values for FF5+ Momentum")
    plt.tight_layout()
    plt.savefig("outputs/figures/residuals_vs_fitted.png", dpi=500)
    print("saved residuals_vs_fitted.png")


def test_heteroskedasticity():
    """
    test heteroskedasticity using Breusch Pagan and White tests
    """
    X = final_model.model.exog
    resid = final_model.resid

    #Breusch Pagan
    bp_stat, bp_pval, _, _ = het_breuschpagan(resid, X)
    #White test
    wh_stat, wh_pval, _, _ = het_white(resid, X)

    print("Testing heteroskedasticity")
    print(f"Breusch Pagan: stat={bp_stat:.4f}, p={bp_pval:.4f}")
    print(f"{'Heteroskedasticity detected' if bp_pval < 0.05 else 'No evidence'}")
    print(f"White Test:stat={wh_stat:.4f}, p={wh_pval:.4f} ")
    print(f"{'Heteroskedasticity detected' if wh_pval < 0.05 else '→ No evidence'}")

    # if heteroskedasticity detected, then refit models with robust SEs
    if bp_pval<0.05 or wh_pval<0.05:
        print("Refitting all models with HC3 robust standard errors")
        robust_models = {}
        for name, model in models.items():
            robust_models[name] = model.get_robustcov_results(cov_type='HC3')
        return robust_models
    return None

def test_normality():
    """testing normality using Q-Q plot and Jarque-Bera"""

    #qq plot
    fig, ax = plt.subplots(figsize=(6,6))
    sm.qqplot(residuals, line='s', ax=ax, alpha=0.3, markersize=2, markerfacecolor='green', markeredgecolor='green')
    ax.set_title("Q-Q Plot of residuals for FF5 + Momentum")
    plt.tight_layout()
    plt.savefig("outputs/figures/qq_plot.png", dpi = 150)

    #jarque-bera
    jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)
    print("\nNormality Test (Jarque-Bera)")
    print(f"JB Stat:  {jb_stat:.4f}")
    print(f"P-value:  {jb_pval:.4f}  "
          f"{'Non-normal residuals' if jb_pval < 0.05 else 'Normal residuals'}")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")

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
    test_normality()
    vif()
    plot_cooks_dist()