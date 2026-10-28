
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file from the project root directory
df = pd.read_csv("life-expectancy_Ireland.csv")

# Inspect the data
print(df.head())
print(df.info())

# Create a scatter plot
plt.scatter(df["Year"], df["Life expectancy"])

# Add labels and title
plt.xlabel("Year")
plt.ylabel("Life expectancy (years)")
plt.title("Life Expectancy in Ireland Over Time")

# Display the plot
plt.show()
