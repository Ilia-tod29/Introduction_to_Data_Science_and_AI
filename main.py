
import pandas as pd
import matplotlib.pyplot as plt

# Load the life expectancy data
life_df = pd.read_csv("life-expectancy_Ireland.csv")

# Load the GDP per capita data
gdp_df = pd.read_csv("gdp-per-capita_Ireland.csv")

# Plot 1: Life expectancy over time
plt.figure(figsize=(10, 5))
plt.scatter(life_df["Year"], life_df["Life expectancy"])

plt.xlabel("Year")
plt.ylabel("Life expectancy (years)")
plt.title("Life Expectancy in Ireland Over Time")
plt.grid(True, alpha=0.3)

plt.show()

# Plot 2: GDP per capita over time
plt.figure(figsize=(10, 5))
plt.scatter(gdp_df["Year"], gdp_df["GDP per capita"])

plt.xlabel("Year")
plt.ylabel("GDP per capita")
plt.title("GDP per Capita in Ireland Over Time")
plt.grid(True, alpha=0.3)

plt.show()
