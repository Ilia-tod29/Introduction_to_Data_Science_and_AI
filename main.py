
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. Load the datasets
# ==================================================

life_df = pd.read_csv("life-expectancy_Ireland.csv")
gdp_df = pd.read_csv("gdp-per-capita_Ireland.csv")


# ==================================================
# 2. Inspect the datasets
# ==================================================

print("Life expectancy data:")
print(life_df.head())
print()

print("GDP per capita data:")
print(gdp_df.head())
print()


# ==================================================
# 3. Plot life expectancy over time
# ==================================================

plt.figure(figsize=(10, 5))

plt.scatter(
    life_df["Year"],
    life_df["Life expectancy"]
)

plt.xlabel("Year")
plt.ylabel("Life expectancy (years)")
plt.title("Life Expectancy in Ireland Over Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==================================================
# 4. Plot GDP per capita over time
# ==================================================

plt.figure(figsize=(10, 5))

plt.scatter(
    gdp_df["Year"],
    gdp_df["GDP per capita"]
)

plt.xlabel("Year")
plt.ylabel("GDP per capita")
plt.title("GDP per Capita in Ireland Over Time")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==================================================
# 5. Combine the datasets by year
# ==================================================

combined_df = pd.merge(
    gdp_df,
    life_df,
    on=["Code", "Year"],
    how="inner"
)

# Keep only observations with both measurements
combined_df = combined_df.dropna(
    subset=["GDP per capita", "Life expectancy"]
)

# Sort observations chronologically
combined_df = combined_df.sort_values("Year")


print("Combined data:")
print(
    combined_df[
        ["Entity", "Year", "GDP per capita", "Life expectancy"]
    ]
)
print()

print("Number of matched observations:", len(combined_df))


# ==================================================
# 6. Plot GDP per capita vs. life expectancy
# ==================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    combined_df["GDP per capita"],
    combined_df["Life expectancy"]
)

plt.xlabel("GDP per capita")
plt.ylabel("Life expectancy (years)")
plt.title("GDP per Capita vs Life Expectancy in Ireland")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
