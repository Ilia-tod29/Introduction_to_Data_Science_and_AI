
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. DOWNLOAD AND PREPARE THE DATA
# ============================================================

# Dataset URLs:
# https://ourworldindata.org/grapher/life-expectancy
# https://ourworldindata.org/grapher/gdp-per-capita-worldbank
# https://ourworldindata.org/grapher/gdp-worldbank

life_url = (
    "https://ourworldindata.org/grapher/life-expectancy.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)

gdp_url = (
    "https://ourworldindata.org/grapher/gdp-per-capita-worldbank.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)

# Total GDP is needed for question d. It comes from the same World Bank
# source and uses the same unit (international-$ at 2021 prices, PPP)
# as GDP per capita, so the two measures can be compared directly.
gdp_total_url = (
    "https://ourworldindata.org/grapher/gdp-worldbank.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)

user_agent = {"User-Agent": "Our World In Data data fetch/1.0"}

life_df = pd.read_csv(life_url, storage_options=user_agent)
gdp_df = pd.read_csv(gdp_url, storage_options=user_agent)
gdp_total_df = pd.read_csv(gdp_total_url, storage_options=user_agent)

# Rename columns to make them easier to understand.
life_df = life_df.rename(columns={
    "entity": "Country",
    "code": "Code",
    "year": "Year",
    "life_expectancy_0": "Life expectancy"
})

gdp_df = gdp_df.rename(columns={
    "entity": "Country",
    "code": "Code",
    "year": "Year",
    "ny_gdp_pcap_pp_kd": "GDP per capita"
})

gdp_total_df = gdp_total_df.rename(columns={
    "code": "Code",
    "year": "Year",
    "ny_gdp_mktp_pp_kd": "GDP"
})

# Keep only the columns needed for the analysis.
life_df = life_df[
    ["Country", "Code", "Year", "Life expectancy"]
]

gdp_df = gdp_df[
    ["Country", "Code", "Year", "GDP per capita"]
]

gdp_total_df = gdp_total_df[
    ["Code", "Year", "GDP"]
]

# Combine GDP and life expectancy using country code and year.
# An inner join keeps only observations present in both datasets.
df = pd.merge(
    life_df,
    gdp_df,
    on=["Code", "Year"],
    how="inner",
    suffixes=("_life", "_gdp")
)

# Add total GDP. A left join keeps every observation used in the
# scatter plots, even where total GDP is missing.
df = pd.merge(
    df,
    gdp_total_df,
    on=["Code", "Year"],
    how="left"
)

# Remove observations without valid GDP or life expectancy.
df = df.dropna(subset=["GDP per capita", "Life expectancy"])

# Keep positive values and exclude aggregate entities without country codes.
df = df[
    (df["GDP per capita"] > 0) &
    (df["Life expectancy"] > 0) &
    df["Code"].notna()
].copy()

# Exclude aggregates that Our World in Data gives an "OWID_" code,
# such as World and the income groups. Kosovo also uses an "OWID_"
# code because it has no ISO code, but it is a country, so keep it.
is_aggregate = (
    df["Code"].str.startswith("OWID_") &
    (df["Code"] != "OWID_KOS")
)
df = df[~is_aggregate].copy()

# Keep one country name column after merging.
if "Country_life" in df.columns:
    df["Country"] = df["Country_life"]


# ============================================================
# 2. GRAPH 1 — ALL COUNTRIES, ALL AVAILABLE YEARS
#    Each point represents a country in a particular year.
#    Historical and modern observations are included together.
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["GDP per capita"],
    df["Life expectancy"],
    alpha=0.3
)

plt.title("GDP per Capita vs Life Expectancy — All Years")
plt.xlabel("GDP per capita (international-$ at 2021 prices, PPP, log scale)")
plt.xscale("log")

# Label the log axis in plain dollars instead of powers of 10.
gdp_ticks = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
plt.xticks(gdp_ticks, [f"${tick:,}" for tick in gdp_ticks])
plt.minorticks_off()
plt.ylabel("Life expectancy (years)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 3. GRAPH 2 — ALL COUNTRIES IN 2023
#    Each point represents one country in 2023.
#    This is a defensible choice because it compares countries
#    at the same point in time, rather than mixing historical
#    and modern life expectancy values.
#
#    The red curve shows a smoothed average trend across
#    GDP-per-capita intervals. It helps reveal the general
#    relationship without displaying every average as a dot.
# ============================================================

selected_year = 2023

year_df = df[df["Year"] == selected_year].copy()

plt.figure(figsize=(10, 6))

# Show the individual countries.
plt.scatter(
    year_df["GDP per capita"],
    year_df["Life expectancy"],
    alpha=0.5,
    label="Countries"
)

# Divide GDP per capita into 20 intervals of equal width on the
# log scale, so each interval covers the same width on the x-axis.
year_df["GDP group"] = pd.cut(
    np.log10(year_df["GDP per capita"]),
    bins=20
)

# Calculate average GDP and life expectancy in each interval.
average_df = year_df.groupby(
    "GDP group",
    observed=True
).agg({
    "GDP per capita": "mean",
    "Life expectancy": "mean"
}).dropna()

# Sort the averages by GDP before smoothing.
average_df = average_df.sort_values("GDP per capita")

# Smooth the trend using a centered rolling average.
average_df["Smoothed life expectancy"] = (
    average_df["Life expectancy"]
    .rolling(window=5, center=True, min_periods=1)
    .mean()
)

# Draw the smoothed trend without individual markers.
plt.plot(
    average_df["GDP per capita"],
    average_df["Smoothed life expectancy"],
    color="red",
    linewidth=2.5,
    label="Smoothed average trend"
)

plt.title(f"GDP per Capita vs Life Expectancy — {selected_year}")
plt.xlabel("GDP per capita (international-$ at 2021 prices, PPP, log scale)")
plt.xscale("log")

# Label the log axis in plain dollars instead of powers of 10.
gdp_ticks = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
plt.xticks(gdp_ticks, [f"${tick:,}" for tick in gdp_ticks])
plt.minorticks_off()
plt.ylabel("Life expectancy (years)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 4. GRAPH 3 — JAPAN OVER TIME
#    Using one country reduces geographical differences,
#    such as climate and location, allowing us to examine
#    how GDP per capita and life expectancy change together
#    within the same country over time.
# ============================================================

selected_country = "Japan"

country_df = df[
    df["Country"] == selected_country
].copy()

# Sort chronologically so the line follows Japan's history.
country_df = country_df.sort_values("Year")

plt.figure(figsize=(10, 6))

# Each point represents Japan in one year.
plt.scatter(
    country_df["GDP per capita"],
    country_df["Life expectancy"],
    alpha=0.6,
    label="Annual observations"
)

# Connect observations chronologically to show historical development.
plt.plot(
    country_df["GDP per capita"],
    country_df["Life expectancy"],
    linewidth=1.5,
    alpha=0.7,
    label="Historical path"
)

plt.title(
    f"GDP per Capita vs Life Expectancy — {selected_country}, All Years"
)
plt.xlabel("GDP per capita (international-$ at 2021 prices, PPP)")
plt.ylabel("Life expectancy (years)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
