import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Step 1: Load and Inspect the Dataset
# Load the CSV
df = pd.read_csv('/Users/sahil/Documents/Meteorite_Landings.csv')

# Show the first few rows
print(df.head())

# Show basic info
print("\nDataset Info:")
print(df.info())
print(df.shape)

# Print summary statistics
print("\nSummary stats:")
print(df.describe()) # count,mean,std,min,max(stats)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum()) # gives missing values with rows 
print(df.isnull().sum().sum()) # gives total missing values

# Step 2: Handle Missing Values
# Fill missing numeric values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing non-numeric values with "Unknown"
df.fillna("Unknown", inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())
print(df.isnull().sum().sum())

# Step 3: Geospatial Visualization
# Set plot style
sns.set(style="whitegrid")

# Scatter plot of latitude vs longitude
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='reclong', y='reclat', hue='mass (g)', size='mass (g)', sizes=(10, 200), palette="viridis", alpha=0.6, legend=False)

plt.title("Meteorite Landings by Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Ensure 'year' column is numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Drop rows with invalid year (if any)
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# Step 4: Trend Analysis Over Time
# Group by year and count meteorites
yearly_counts = df.groupby('year').size().reset_index(name='count')

# Plot: Number of meteorites per year

sns.lineplot(data=yearly_counts, x='year', y='count', marker='o', color='darkblue')
plt.title("Number of Meteorite Landings Over the Years")
plt.xlabel("Year")
plt.ylabel("Count of Meteorites")
plt.grid(True)
plt.show()

# Group by year and calculate average mass
yearly_avg_mass = df.groupby('year')['mass (g)'].mean().reset_index()

# Plot: Average meteorite mass per year

sns.lineplot(data=yearly_avg_mass, x='year', y='mass (g)', marker='o', color='green')
plt.title("Average Meteorite Mass Over the Years")
plt.xlabel("Year")
plt.ylabel("Average Mass (g)")
plt.grid(True)
plt.show()

# Step 5: Classification-Based Insights
top_classes = df['recclass'].value_counts().head(10)
print("Top 10 Meteorite Classes:\n", top_classes)

# Plot: Top 10 Most Common Meteorite Classes

sns.barplot(x=top_classes.index, y=top_classes.values, hue=top_classes.index, palette="mako", dodge=False, legend=False)
plt.title("Top 10 Most Common Meteorite Classes")
plt.xlabel("Meteorite Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

top_class_names = top_classes.index.tolist()
filtered = df[df['recclass'].isin(top_class_names)]

# Plot: Mass Distribution by Meteorite Class

sns.boxplot(x='recclass', y='mass (g)', data=filtered, hue='recclass', palette="coolwarm", dodge=False)
plt.title("Mass Distribution by Meteorite Class")
plt.xlabel("Meteorite Class")
plt.ylabel("Mass (g)")
plt.yscale("log")  # because mass varies widely
plt.xticks(rotation=45)
plt.show()

# Step 6: Fell vs Found Analysis
fall_counts = df['fall'].value_counts()

# Plot: Meteorites Fell vs Found

plt.pie(fall_counts, labels=fall_counts.index, autopct='%1.1f%%', startangle=90, colors=["#66c2a5", "#fc8d62"])
plt.title("Meteorites: Fell vs Found")
plt.axis('equal')
plt.show()

# Step 7: Heaviest Meteorites Identification
heaviest = df.sort_values(by='mass (g)', ascending=False).head(10)
print("Top 10 Heaviest Meteorites:\n", heaviest[['name', 'mass (g)', 'year', 'reclat', 'reclong']])

# Plot: Top 10 Heaviest Meteorites

sns.barplot(x='name', y='mass (g)', data=heaviest, hue='name', palette="rocket", dodge=False, legend=False)
plt.title("Top 10 Heaviest Meteorites")
plt.xlabel("Meteorite Name")
plt.ylabel("Mass (g)")
plt.xticks(rotation=45)
plt.show()

# Step 8: Correlation and Covariance Analysis
# Compute correlation matrix
corr = df[['mass (g)', 'year', 'reclat', 'reclong']].corr()

# Plot heatmap

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 9: Hypothesis Testing
# Test if 'Fell' meteorites have different mass than 'Found'
from scipy.stats import ttest_ind

fell_mass = df[df['fall'] == 'Fell']['mass (g)']
found_mass = df[df['fall'] == 'Found']['mass (g)']

t_stat, p_val = ttest_ind(fell_mass, found_mass, equal_var=False)
print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant difference in mass.")
else:
    print("Fail to reject the null hypothesis: No significant difference in mass.")

# Step 10: Probability Distribution of Mass

sns.histplot(df['mass (g)'], kde=True, bins=50, color='steelblue')
plt.title("Distribution of Meteorite Mass")
plt.xlabel("Mass (g)")
plt.ylabel("Frequency")
plt.xscale('log')  # Because mass is skewed
plt.show()

# Step 11: Outlier Detection using IQR
# Selecting numerical columns for outlier analysis
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns

# Calculating Q1, Q3 and IQR
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Defining lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Counting outliers
outliers = ((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).sum()

print("\nOutlier Count in Each Numeric Column:")
print(outliers)

# Step 12: Z-Score Analysis
# Calculate z-scores for mass
df['mass_zscore'] = (df['mass (g)'] - df['mass (g)'].mean()) / df['mass (g)'].std()

# Basic Histogram for Z-Score Distribution
plt.hist(df['mass_zscore'], bins=50, color='purple', edgecolor='black')
plt.axvline(x=3, color='red', linestyle='--', label='z = 3')
plt.axvline(x=-3, color='red', linestyle='--', label='z = -3')
plt.title("Z-Score Distribution of Meteorite Mass")
plt.xlabel("Z-Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Show extreme z-score meteorites (absolute z > 3)
extreme_z = df[np.abs(df['mass_zscore']) > 3][['name', 'mass (g)', 'mass_zscore']].sort_values(by='mass_zscore', ascending=False)
print("\n first few Z-scores:")
print(df['mass_zscore'].head())

print("\nMeteorites with Extreme Z-Scores (|z| > 3):")
print(extreme_z)