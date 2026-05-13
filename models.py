"""
OLS Models
1. CAPM
2. FF3
3. FF3 + Momentum
4. FF5 + Momentum
"""

import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.anova import anova_lm
import pickle


#load panel
paneldf = pd.read_parquet("data/panel.parquet")

MODEL_TYPES = {
        "CAPM": ['Mkt-RF'],
        "FF3": ['Mkt-RF', 'SMB', 'HML'],
        "FF3_Mom": ['Mkt-RF', 'SMB', 'HML', 'Mom'],
        "FF5_Mom": ['Mkt-RF', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']
    }
FACTORS = ['const', 'Mkt-RF', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']

def significance_stars(pval):
    if pval < 0.01:  return "***"
    if pval < 0.05:  return "**"
    if pval < 0.1:   return "*"
    return ""

def build_summary(models):
    """build summary table to compare all models"""
    rows = []
    for factor in FACTORS:
        coef_row = {"Variable": factor}
        se_row = {"Variable": ""}

        for name, model in models.items():
            if factor in model.params:
                coef = model.params[factor]
                se = model.bse[factor]
                pval = model.pvalues[factor]
                stars = significance_stars(pval)
                coef_row[name] = f"{coef:.4f}{stars}"
                se_row[name] = f"{se:.4f}"
            else:
                coef_row[name] = "-"
                se_row[name] = ""
        rows.append(coef_row)
        rows.append(se_row)
    
    rows.append({"Variable": "———", **{n: "———" for n in models}})
    rows.append({"Variable": "Adj. R²", **{n: f"{m.rsquared_adj:.4f}" for n, m in models.items()}})
    rows.append({"Variable": "N",       **{n: f"{int(m.nobs):,}" for n, m in models.items()}})

    return pd.DataFrame(rows, columns=["Variable"] + list(models.keys()))

def fit_model():
    """fit all models and return a result dict"""
    models = {}
    for name, factors in MODEL_TYPES.items():
        X = sm.add_constant(paneldf[factors])
        Y = paneldf['excess_ret']
        models[name] = sm.OLS(Y,X).fit()
    return models

def f_test(models):
    """run incremental F-tests for nested model comparisons"""
    tests = [
        ("CAPM", "FF3", "Adding SMB and HML"),
        ("FF3", "FF3_Mom", "Adding Momentum"),
        ("FF3_Mom", "FF5_Mom", "Adding RMW and CMA")
    ]
    
    rows = []

    for m1, m2, desc in tests:
        ftest = anova_lm(models[m1], models[m2])
        fstat1 = ftest['F'].iloc[1]
        pval = ftest['Pr(>F)'].iloc[1]
        conc = "Significant p value" if pval < 0.05 else "P value not significant"

        rows.append({
            "Comparison": f"{m1} -> {m2}",
            "Description": desc,
            "F-statistic": round(fstat1,4),
            "P-value":      round(pval, 4),
            "Conclusion":   conc
        })
    table = pd.DataFrame(rows)
    table.to_csv("outputs/tables/f_tests.csv", index=False)
    print("\nSaved to outputs/tables/f_tests.csv")
    
    return table

if __name__ == "__main__":
    models = fit_model()
    summary = build_summary(models)
    summary.to_csv("outputs/tables/regression_table.csv", index=False)
    print("\nSaved to outputs/tables/regression_table.csv")
    res = f_test(models)
    with open("data/models.pkl", "wb") as f:
        pickle.dump(models, f)
        


