
import pandas as pd
import matplotlib.pyplot as plt
import requests


# ==================================================
# 1. Define URLs
# ==================================================

life_url = (
    "https://ourworldindata.org/grapher/"
    "life-expectancy.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)

life_metadata_url = (
    "https://ourworldindata.org/grapher/"
    "life-expectancy.metadata.json"
    "?v=1&csvType=full&useColumnShortNames=true"
)

gdp_url = (
    "https://ourworldindata.org/grapher/"
    "gdp-per-capita-worldbank.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)

gdp_metadata_url = (
    "https://ourworldindata.org/grapher/"
    "gdp-per-capita-worldbank.metadata.json"
    "?v=1&csvType=full&useColumnShortNames=true"
)

user_agent = {
    "User-Agent": "Our World In Data data fetch/1.0"
}


# ==================================================
# 2. Download datasets
# ==================================================

print("Downloading life expectancy data...")

life_df = pd.read_csv(
    life_url,
    storage_options=user_agent
)

print("Downloading GDP per capita data...")

gdp_df = pd.read_csv(
    gdp_url,
    storage_options=user_agent
)


# ==================================================
# 3. Download metadata
# ==================================================

life_response = requests.get(
    life_metadata_url,
    headers=user_agent,
    timeout=30
)
life_response.raise_for_status()
life_metadata = life_response.json()

gdp_response = requests.get(
    gdp_metadata_url,
    headers=user_agent,
    timeout=30
)
gdp_response.raise_for_status()
gdp_metadata = gdp_response.json()

print("\nGDP metadata:")
print(gdp_metadata)


# ==================================================
# 4. Normalize column names
# ==================================================

# OWID may return lowercase column names when
# useColumnShortNames=true.

life_df.columns = [
    col.strip().lower() for col in life_df.columns
]

gdp_df.columns = [
    col.strip().lower() for col in gdp_df.columns
]

# Rename identifier columns to consistent names.
life_df = life_df.rename(
    columns={
        "entity": "Country",
        "code": "Code",
        "year": "Year",
        "life_expectancy_0": "Life expectancy"
    }
)

gdp_df = gdp_df.rename(
    columns={
        "entity": "Country",
        "code": "Code",
        "year": "Year",
        "ny_gdp_pcap_pp_kd": "GDP per capita"
    }
)

# Check the actual columns before continuing.
print("\nLife expectancy columns:")
print(life_df.columns.tolist())

print("\nGDP columns:")
print(gdp_df.columns.tolist())

if "Life expectancy" not in life_df.columns:
    raise ValueError(
        "Life expectancy column not found. "
        f"Available columns: {life_df.columns.tolist()}"
    )

if "GDP per capita" not in gdp_df.columns:
    raise ValueError(
        "GDP per capita column not found. "
        f"Available columns: {gdp_df.columns.tolist()}"
    )


# ==================================================
# 5. Clean the data
# ==================================================

life_df["Year"] = pd.to_numeric(
    life_df["Year"], errors="coerce"
)
life_df["Life expectancy"] = pd.to_numeric(
    life_df["Life expectancy"], errors="coerce"
)

gdp_df["Year"] = pd.to_numeric(
    gdp_df["Year"], errors="coerce"
)
gdp_df["GDP per capita"] = pd.to_numeric(
    gdp_df["GDP per capita"], errors="coerce"
)

# Remove rows without the required measurements.
life_df = life_df.dropna(
    subset=["Code", "Year", "Life expectancy"]
).copy()

gdp_df = gdp_df.dropna(
    subset=["Code", "Year", "GDP per capita"]
).copy()

# Keep three-letter country codes as an initial filter.
# This excludes many OWID aggregate entries, but the
# resulting entities should still be checked.
life_df = life_df[
    life_df["Code"].str.fullmatch(r"[A-Z]{3}", na=False)
].copy()

gdp_df = gdp_df[
    gdp_df["Code"].str.fullmatch(r"[A-Z]{3}", na=False)
].copy()


# ==================================================
# 6. Plot life expectancy over time
# ==================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    life_df["Year"],
    life_df["Life expectancy"],
    alpha=0.15,
    s=8
)

plt.xlabel("Year")
plt.ylabel("Life expectancy (years)")
plt.title("Life Expectancy Across Countries Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 7. Plot GDP per capita over time
# ==================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    gdp_df["Year"],
    gdp_df["GDP per capita"],
    alpha=0.15,
    s=8
)

plt.xlabel("Year")
plt.ylabel("GDP per capita (international-$, 2021 prices)")
plt.title("GDP per Capita Across Countries Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 8. Combine the datasets
# ==================================================

combined_df = pd.merge(
    gdp_df[["Code", "Year", "GDP per capita"]],
    life_df[["Code", "Year", "Life expectancy"]],
    on=["Code", "Year"],
    how="inner",
    validate="one_to_one"
)

combined_df = combined_df.dropna(
    subset=["GDP per capita", "Life expectancy"]
)

print("\nCombined data:")
print(combined_df.head(10))

print(
    "\nNumber of matched observations:",
    len(combined_df)
)

print(
    "Number of country codes:",
    combined_df["Code"].nunique()
)

print(
    "Years covered:",
    combined_df["Year"].min(),
    "to",
    combined_df["Year"].max()
)


# ==================================================
# 9. Plot GDP per capita vs life expectancy
# ==================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    combined_df["GDP per capita"],
    combined_df["Life expectancy"],
    alpha=0.25,
    s=12
)

plt.xlabel("GDP per capita (international-$, 2021 prices)")
plt.ylabel("Life expectancy (years)")
plt.title("GDP per Capita vs Life Expectancy")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
