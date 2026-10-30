
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Assignment 1
# By Ilia Todorov FN: 2MI3400906 & Georgi Karamishev FN: 2MI3200988
# ============================================================

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

# Exclude territories that are not independent countries. They have
# their own ISO codes, so the filter above keeps them, but they are
# part of another country (e.g. Greenland belongs to Denmark and
# Macao to China). Small, rich territories such as Bermuda and the
# Cayman Islands would otherwise fill the top of the GDP per capita
# ranking. Kosovo and Palestine are kept as partially recognised states.
territory_codes = [
    "ABW",  # Aruba (Netherlands)
    "BMU",  # Bermuda (United Kingdom)
    "CUW",  # Curacao (Netherlands)
    "CYM",  # Cayman Islands (United Kingdom)
    "FRO",  # Faroe Islands (Denmark)
    "GRL",  # Greenland (Denmark)
    "HKG",  # Hong Kong (China)
    "MAC",  # Macao (China)
    "PRI",  # Puerto Rico (United States)
    "SXM",  # Sint Maarten (Netherlands)
    "TCA",  # Turks and Caicos Islands (United Kingdom)
    "VIR",  # United States Virgin Islands (United States)
]
df = df[~df["Code"].isin(territory_codes)].copy()

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


# ============================================================
# 5. GRAPHS 4 AND 5 — QUESTIONS D AND E: STRONG ECONOMIES VS
#    LIFE EXPECTANCY IN 2023
#    The same plot is drawn twice: once with total GDP (question d)
#    and once with GDP per capita (question e) as the indicator of
#    a strong economy. The top countries by that indicator are
#    highlighted so we can see whether all of them are above the
#    "high life expectancy" line.
#
#    We consider "High" life expectancy to be the upper quartile: life
#    expectancy at least as high as 75% of countries in 2023. The same
#    threshold is used in both plots so the results can be compared.
# ============================================================

top_n = 15

high_life_threshold = year_df["Life expectancy"].quantile(0.75)


def plot_strong_economies(column, title_name, xlabel, ticks, tick_labels):
    """Plot an economic indicator against life expectancy for one year.

    column      -- the indicator column, e.g. "GDP" or "GDP per capita"
    title_name  -- how the indicator is named in the title and legend
    xlabel      -- x-axis label, including the unit
    ticks       -- positions of the x-axis ticks
    tick_labels -- text shown at each x-axis tick
    """
    indicator_df = year_df.dropna(subset=[column]).copy()

    top_df = indicator_df.nlargest(top_n, column)

    plt.figure(figsize=(10, 6))

    # Show all countries in grey for context.
    plt.scatter(
        indicator_df[column],
        indicator_df["Life expectancy"],
        color="grey",
        alpha=0.4,
        label="Other countries"
    )

    # Highlight the top countries by the chosen indicator.
    plt.scatter(
        top_df[column],
        top_df["Life expectancy"],
        color="tab:blue",
        label=f"Top {top_n} countries by {title_name}"
    )

    # Mark the threshold for "high" life expectancy.
    plt.axhline(
        high_life_threshold,
        color="red",
        linestyle="--",
        label=f"High life expectancy (upper quartile, {high_life_threshold:.1f} years)"
    )

    # Capitalise only the first letter so "GDP" stays in capitals.
    plt.title(
        f"{title_name[0].upper()}{title_name[1:]} vs Life Expectancy — "
        f"{selected_year}"
    )
    plt.xlabel(xlabel)
    plt.xscale("log")
    plt.xticks(ticks, tick_labels)
    plt.minorticks_off()
    plt.ylabel("Life expectancy (years)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Top {top_n} countries by {title_name} in {selected_year}:")
    print(top_df[["Country", column, "Life expectancy"]].to_string(index=False))
    print()


# Question d: total GDP, with the log axis in billions and trillions.
plot_strong_economies(
    column="GDP",
    title_name="total GDP",
    xlabel="GDP (international-$ at 2021 prices, PPP, log scale)",
    ticks=[1e8, 1e9, 1e10, 1e11, 1e12, 1e13],
    tick_labels=["$100M", "$1B", "$10B", "$100B", "$1T", "$10T"]
)

# Question e: GDP per capita, with the same ticks as graph 2.
plot_strong_economies(
    column="GDP per capita",
    title_name="GDP per capita",
    xlabel="GDP per capita (international-$ at 2021 prices, PPP, log scale)",
    ticks=gdp_ticks,
    tick_labels=[f"${tick:,}" for tick in gdp_ticks]
)
# ============================================================
# 6. QUESTION B - LIFE EXPECTANCY MORE THAN ONE STANDARD
#    DEVIATION ABOVE THE MEAN (2023)
# ============================================================

mean_life = year_df["Life expectancy"].mean()
std_life = year_df["Life expectancy"].std()
cutoff_b = mean_life + std_life

above_df = year_df[year_df["Life expectancy"] > cutoff_b].sort_values(
    "Life expectancy", ascending=False
)

print(f"Question b - {selected_year}")
print(f"Mean: {mean_life:.1f} years, standard deviation: {std_life:.1f} years")
print(f"Cutoff (mean + 1 SD): {cutoff_b:.1f} years")
print(f"{len(above_df)} of {len(year_df)} countries are above the cutoff:")
print(above_df[["Country", "Life expectancy", "GDP per capita"]].round({"Life expectancy": 1, "GDP per capita": 0}).to_string(index=False))
print()

plt.figure(figsize=(10, 6))
plt.scatter(
    year_df["GDP per capita"], year_df["Life expectancy"],
    color="grey", alpha=0.4, label="Other countries"
)
plt.scatter(
    above_df["GDP per capita"], above_df["Life expectancy"],
    color="tab:green", label="Above mean + 1 SD"
)
plt.axhline(
    cutoff_b, color="red", linestyle="--",
    label=f"Mean + 1 SD ({cutoff_b:.1f} years)"
)
plt.title(f"Life Expectancy Above Mean + 1 Standard Deviation - {selected_year}")
plt.xlabel("GDP per capita (international-$ at 2021 prices, PPP, log scale)")
plt.xscale("log")
plt.xticks(gdp_ticks, [f"${tick:,}" for tick in gdp_ticks])
plt.minorticks_off()
plt.ylabel("Life expectancy (years)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. QUESTION C - HIGH LIFE EXPECTANCY BUT LOW GDP PER CAPITA (2023)
#    High: upper quartile of life expectancy (same as d and e).
#    Low (strict): below the median GDP per capita.
#    Low (relaxed): below the mean GDP per capita.
# ============================================================

median_gdp = year_df["GDP per capita"].median()
mean_gdp = year_df["GDP per capita"].mean()
high_df = year_df[year_df["Life expectancy"] >= high_life_threshold]

strict_df = high_df[high_df["GDP per capita"] < median_gdp]
relaxed_df = high_df[high_df["GDP per capita"] < mean_gdp].sort_values(
    "Life expectancy", ascending=False
)

print(f"Question c - {selected_year}")
print(f"High life expectancy: at least {high_life_threshold:.1f} years")
print(f"Strict low GDP (below median ${median_gdp:,.0f}): {len(strict_df)} countries")
print(f"Relaxed low GDP (below mean ${mean_gdp:,.0f}): {len(relaxed_df)} countries")
print(relaxed_df[["Country", "Life expectancy", "GDP per capita"]]
      .round({"Life expectancy": 1, "GDP per capita": 0})
      .to_string(index=False))
print()

plt.figure(figsize=(10, 6))
plt.scatter(year_df["GDP per capita"], year_df["Life expectancy"],
            color="grey", alpha=0.4, label="Other countries")
plt.scatter(relaxed_df["GDP per capita"], relaxed_df["Life expectancy"],
            color="tab:orange", label="High life expectancy, GDP below mean")
plt.axhline(high_life_threshold, color="red", linestyle="--",
            label=f"High life expectancy ({high_life_threshold:.1f} years)")
plt.axvline(median_gdp, color="blue", linestyle="--",
            label=f"Median GDP per capita (${median_gdp:,.0f})")
plt.axvline(mean_gdp, color="green", linestyle="--",
            label=f"Mean GDP per capita (${mean_gdp:,.0f})")
plt.title(f"High Life Expectancy but Low GDP per Capita - {selected_year}")
plt.xlabel("GDP per capita (international-$ at 2021 prices, PPP, log scale)")
plt.xscale("log")
plt.xticks(gdp_ticks, [f"${tick:,}" for tick in gdp_ticks])
plt.minorticks_off()
plt.ylabel("Life expectancy (years)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

