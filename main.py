
import pandas as pd
import matplotlib.pyplot as plt


# URLs for the full Our World in Data datasets
life_url = (
    "https://ourworldindata.org/grapher/life-expectancy.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)

gdp_url = (
    "https://ourworldindata.org/grapher/gdp-per-capita-worldbank.csv"
    "?v=1&csvType=full&useColumnShortNames=true"
)


# Download the full datasets
user_agent = {"User-Agent": "Our World In Data data fetch/1.0"}

life_df = pd.read_csv(life_url, storage_options=user_agent)
gdp_df = pd.read_csv(gdp_url, storage_options=user_agent)


# Rename the relevant columns
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


# Keep the columns needed for the scatter plot
life_df = life_df[["Country", "Code", "Year", "Life expectancy"]]
gdp_df = gdp_df[["Country", "Code", "Year", "GDP per capita"]]


# Combine both datasets by country code and year
df = pd.merge(
    life_df,
    gdp_df,
    on=["Code", "Year"],
    how="inner"
)


# Print the combined data
print(df.head())
print("Number of rows:", len(df))


# Create the scatter plot using all combined observations
plt.figure(figsize=(10, 6))

plt.scatter(
    df["GDP per capita"],
    df["Life expectancy"],
    alpha=0.5
)

plt.title("GDP per Capita vs Life Expectancy")
plt.xlabel("GDP per capita (international-$, 2021 prices)")
plt.ylabel("Life expectancy (years)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
