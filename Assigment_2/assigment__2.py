"""
Linear regression: living area -> selling price (villas in Landvetter, Sweden)

Run:    python assignment__2.py [path_to_csv]
        (without an argument, data_assignment2.csv is read from the working directory)
Needs:  pandas, numpy, matplotlib, scikit-learn, scipy
Output:i fgure1..figure4 PNG files (saved, not shown) + a results summary printed at the end
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # save figures to files, no window needed
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data_assignment2.csv"
OUTLIER_THRESHOLD = 3.0                   # |studentized residual| above this = outlier
NEW_AREAS = [100, 150, 200]               # living areas to predict (part 1c), in m2


# ----------------------------------------------------------------------------
# 1. Load the data and check its quality                      [part 1a, cleaning]
#    Reads the CSV, drops rows only if Living_area or Selling_price is missing
#    (the other columns are not used), and checks for duplicates.
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
rows_raw = len(df)

# Only Living_area (x) and Selling_price (y) are used in this model.
# Rooms, Land_size and Biarea contain missing values, but that does not matter
# here, so we do NOT drop rows because of them (that would waste good data).
missing_per_column = df.isna().sum().to_dict()
df = df.dropna(subset=["Living_area", "Selling_price"])

# Exact duplicate rows (all columns except the ID)
n_duplicates = int(df.drop(columns="ID").duplicated().sum())


# ----------------------------------------------------------------------------
# 2. Helper functions: fit a line and compute studentized residuals  [parts 1a, 1b]
#    fit_line does the least-squares fit; studentized puts residuals on a common
#    scale, which is used in the next section to find outliers.
# ----------------------------------------------------------------------------
def fit_line(x, y):
    """Fit y = intercept + slope * x. Returns model, fitted values, residuals."""
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    fitted = model.predict(x.reshape(-1, 1))
    return model, fitted, y - fitted


def studentized(x, resid):
    """Internally studentized residuals: residual / (s * sqrt(1 - leverage)).
    Puts residuals on a common scale so we can say 'more than 3 sd away'."""
    n = len(x)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))              # residual standard error
    lev = 1 / n + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)   # leverage
    return resid / (s * np.sqrt(1 - lev))


# ----------------------------------------------------------------------------
# 3. Data cleaning: first fit on ALL data, then look for outliers   [part 1a, cleaning]
#    Houses with |studentized residual| > 3 are removed, because a few extreme
#    sales would pull the regression line away from the typical house.
# ----------------------------------------------------------------------------
x_all = df["Living_area"].to_numpy(dtype=float)
y_all = df["Selling_price"].to_numpy(dtype=float)

model_raw, fitted_raw, resid_raw = fit_line(x_all, y_all)
df["stud_resid_raw"] = studentized(x_all, resid_raw)
r2_raw = model_raw.score(x_all.reshape(-1, 1), y_all)

# Rows flagged as outliers: |studentized residual| > threshold
outlier_mask = np.abs(df["stud_resid_raw"]) > OUTLIER_THRESHOLD
outliers = df[outlier_mask]
clean = df[~outlier_mask].copy()


# ----------------------------------------------------------------------------
# 4. Final model on the cleaned data                          [parts 1a and 1b]
#    Fits Selling_price = intercept + slope * Living_area (the model of 1a) and
#    reports the slope and intercept (1b), plus R2 and the slope's confidence interval.
# ----------------------------------------------------------------------------
x = clean["Living_area"].to_numpy(dtype=float)
y = clean["Selling_price"].to_numpy(dtype=float)
n = len(x)

model, fitted, resid = fit_line(x, y)
slope, intercept = model.coef_[0], model.intercept_
r2 = model.score(x.reshape(-1, 1), y)
corr = np.corrcoef(x, y)[0, 1]

# Standard error and p-value of the slope, 95% confidence interval
s = np.sqrt(np.sum(resid ** 2) / (n - 2))                  # residual standard error (SEK)
sxx = np.sum((x - x.mean()) ** 2)
se_slope = s / np.sqrt(sxx)
t_crit = stats.t.ppf(0.975, df=n - 2)
p_slope = 2 * (1 - stats.t.cdf(abs(slope / se_slope), df=n - 2))
slope_ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)


# ----------------------------------------------------------------------------
# 5. Predictions for 100, 150 and 200 m2, with 95% prediction intervals   [part 1c]
#    Uses the fitted line to predict the selling price of three living areas.
# ----------------------------------------------------------------------------
pred_rows = []
for a in NEW_AREAS:
    p = model.predict(np.array([[a]]))[0]
    # 95% prediction interval for a NEW single house at this area
    se_pred = s * np.sqrt(1 + 1 / n + (a - x.mean()) ** 2 / sxx)
    pred_rows.append({
        "Living_area_m2": a,
        "Predicted_price_SEK": round(p),
        "PI95_low": round(p - t_crit * se_pred),
        "PI95_high": round(p + t_crit * se_pred),
        "Inside_data_range": bool(x.min() <= a <= x.max()),
        "Houses_within_10m2": int(np.sum(np.abs(x - a) <= 10)),   # how well supported
    })
predictions = pd.DataFrame(pred_rows)


# ----------------------------------------------------------------------------
# 6. Figures
#    Figure 1: outliers and the effect of cleaning                [part 1a]
#    Figure 2: regression line and the three predictions          [parts 1b, 1c]
#    Figure 3: residual plot                                      [part 1d]
#    Figure 4: histogram of the residuals (supports 1d)
# ----------------------------------------------------------------------------
M = 1e6                                                    # show prices in millions SEK
grid = np.linspace(min(x_all.min(), 55), max(x_all.max(), 215), 100)

# --- Figure 1: raw data, outliers flagged, lines with and without outliers ---
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(clean["Living_area"], clean["Selling_price"] / M, color="tab:blue", label="Kept houses")
if len(outliers):
    ax.scatter(outliers["Living_area"], outliers["Selling_price"] / M, color="red", s=90,
               marker="X", label=f"Removed outlier(s) (|stud. resid.| > {OUTLIER_THRESHOLD:g})")
    for _, r in outliers.iterrows():
        ax.annotate(f"ID {int(r['ID'])}", (r["Living_area"], r["Selling_price"] / M),
                    textcoords="offset points", xytext=(8, -4), color="red")
ax.plot(grid, model_raw.predict(grid.reshape(-1, 1)) / M, "--", color="gray",
        label="Line fitted on all data")
ax.plot(grid, model.predict(grid.reshape(-1, 1)) / M, color="tab:orange",
        label="Line fitted on cleaned data")
ax.set(xlabel="Living area (m²)", ylabel="Selling price (million SEK)",
       title="Figure 1: Raw data, outlier(s) and the effect of cleaning")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figure1_raw_data_and_outliers.png", dpi=200); plt.close(fig)

# --- Figure 2: final regression line + predictions ---
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(x, y / M, color="tab:blue", label="Houses (cleaned data)")
ax.plot(grid, model.predict(grid.reshape(-1, 1)) / M, color="tab:orange", lw=2,
        label=f"Price = {intercept:,.0f} + {slope:,.0f} × area")
for _, r in predictions.iterrows():
    ax.scatter(r["Living_area_m2"], r["Predicted_price_SEK"] / M, color="green", s=90,
               zorder=5, edgecolor="black")
    ax.annotate(f"{r['Living_area_m2']} m²: {r['Predicted_price_SEK'] / M:.2f} M",
                (r["Living_area_m2"], r["Predicted_price_SEK"] / M),
                textcoords="offset points", xytext=(-10, 12), ha="center", color="darkgreen")
ax.scatter([], [], color="green", edgecolor="black", s=90, label="Predictions (100, 150, 200 m²)")
ax.set(xlabel="Living area (m²)", ylabel="Selling price (million SEK)",
       title=f"Figure 2: Regression line (R² = {r2:.2f}) and predictions")
ax.grid(alpha=0.3); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("figure2_regression_and_predictions.png", dpi=200); plt.close(fig)

# --- Figure 3: residual plot (part 1d) ---
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(x, resid / M, color="tab:blue")
ax.axhline(0, color="red", lw=1.5)
ax.axhline(2 * s / M, color="gray", ls="--", lw=1, label="± 2 residual std. errors")
ax.axhline(-2 * s / M, color="gray", ls="--", lw=1)
ax.set(xlabel="Living area (m²)", ylabel="Residual = actual − predicted (million SEK)",
       title="Figure 3: Residual plot")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figure3_residual_plot.png", dpi=200); plt.close(fig)

# --- Figure 4: histogram of the residuals (is the error roughly normal?) ---
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.hist(resid / M, bins=12, color="tab:blue", edgecolor="white", density=True)
rr = np.linspace(resid.min(), resid.max(), 200)
ax.plot(rr / M, stats.norm.pdf(rr, 0, resid.std(ddof=1)) * M, color="tab:orange", lw=2,
        label="Normal curve (same std. dev.)")
ax.set(xlabel="Residual (million SEK)", ylabel="Density",
       title="Figure 4: Distribution of the residuals")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figure4_residual_histogram.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------
# 7. Results summary: prints the answers to parts 1a-1d    [parts 1a, 1b, 1c, 1d]
# ----------------------------------------------------------------------------
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n[1a  DATA CLEANING]")
print("Rows in file:", rows_raw)
print("Missing values per column:", missing_per_column)
print("Exact duplicate rows:", n_duplicates)
print(f"Outlier rule: |studentized residual| > {OUTLIER_THRESHOLD:g}")
print("Removed rows:")
print(outliers[["ID", "Living_area", "Age", "Selling_price", "stud_resid_raw"]]
      .round(2).to_string(index=False))
print("Rows after cleaning:", n)
print(f"Model on ALL data:     slope = {model_raw.coef_[0]:,.1f}, "
      f"intercept = {model_raw.intercept_:,.0f}, R2 = {r2_raw:.3f}")

print("\n[1a, 1b  MODEL ON CLEANED DATA]")
print(f"Selling price = {intercept:,.0f} + {slope:,.1f} x Living area")
print(f"Slope (SEK per m2): {slope:,.1f}")
print(f"Intercept (SEK):    {intercept:,.0f}")
print(f"Slope 95% CI: ({slope_ci[0]:,.0f}, {slope_ci[1]:,.0f}) | p-value: {p_slope:.2e}")
print(f"R2 = {r2:.3f} | correlation = {corr:.3f}")
print(f"Residual standard error: {s:,.0f} SEK")

print("\n[1c  PREDICTIONS]")
print(predictions.to_string(index=False))

print("\n[1d  RESIDUAL PLOT]  see figure3_residual_plot.png")
print("\nFigures saved: figure1_raw_data_and_outliers.png, figure2_regression_and_predictions.png,")
print("               figure3_residual_plot.png, figure4_residual_histogram.png")
print("=" * 70)