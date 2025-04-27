import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Step 1: Load and Inspect the Dataset
# Load the CSV
df = pd.read_csv('/Users/sahil/Documents/Meteorite_Landings.csv')

print("PROJECT OBJECTIVES:")
print("1. Analyze geospatial patterns in meteorite landings.")
print("2. Identify influential features for meteorite discovery.")
print("3. Predict whether a meteorite was 'Fell' or 'Found' using ML.\n")

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

# Count z-score outliers (|z| > 3)
z_outliers = (np.abs(df['mass_zscore']) > 3).sum()
print("\nZ-Score Outliers (|z| > 3) for mass (g):", z_outliers)

# Plot z-score distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['mass_zscore'], bins=50, kde=True, color='purple')
plt.axvline(x=3, color='r', linestyle='--', label='z=3')
plt.axvline(x=-3, color='r', linestyle='--')
plt.title("Distribution of Z-Scores for Meteorite Mass")
plt.xlabel("Z-Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Show extreme z-score meteorites
extreme_z = df[np.abs(df['mass_zscore']) > 3][['name', 'mass (g)', 'mass_zscore']].sort_values('mass_zscore', ascending=False)
print("\nMeteorites with Extreme Z-Scores (|z| > 3):")
print(extreme_z)


########## Research oriented starts : 

import scipy.stats as stats

# Q-Q Plots to Check Normality Assumption
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
stats.probplot(fell_mass, dist="norm", plot=plt)
plt.title("Q-Q Plot: 'Fell' Meteorites")

plt.subplot(1, 2, 2)
stats.probplot(found_mass, dist="norm", plot=plt)
plt.title("Q-Q Plot: 'Found' Meteorites")

plt.tight_layout()
plt.show()


# Step 3.1: Geospatial Clustering using DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Drop rows with missing coordinates just in case
geo_df = df.dropna(subset=['reclat', 'reclong'])

# Extract coordinates
coords = geo_df[['reclat', 'reclong']].values

# Normalize coordinates
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# Apply DBSCAN clustering
db = DBSCAN(eps=0.3, min_samples=10).fit(coords_scaled)
geo_df['cluster'] = db.labels_

# Plot clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(data=geo_df, x='reclong', y='reclat', hue='cluster', palette='tab10', legend='full', alpha=0.6)
plt.title("Meteorite Landing Clusters (DBSCAN)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

from sklearn.metrics import silhouette_score

# Silhouette Score
if len(set(db.labels_)) > 1 and -1 not in set(db.labels_):
    score = silhouette_score(coords_scaled, db.labels_)
    print("Silhouette Score:", score)
else:
    print("Silhouette Score not applicable (only one cluster or contains noise).")

# Step 6.1: Predicting Fell vs Found using Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Prepare data
ml_df = df.copy()
ml_df = ml_df.dropna(subset=['mass (g)', 'year', 'reclat', 'reclong', 'recclass', 'fall'])

# Encode categorical data
le_recclass = LabelEncoder()
ml_df['recclass_encoded'] = le_recclass.fit_transform(ml_df['recclass'])

le_fall = LabelEncoder()
ml_df['fall_encoded'] = le_fall.fit_transform(ml_df['fall'])

features = ml_df[['mass (g)', 'year', 'reclat', 'reclong', 'recclass_encoded']]
target = ml_df['fall_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\nClassification Report for Predicting 'Fell' vs 'Found':")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance Plot
importances = clf.feature_importances_
feature_names = features.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names, hue=feature_names, palette='viridis', dodge=False, legend=False)
plt.title("Feature Importance for 'Fell' vs 'Found' Classification")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Step 6.2: Novelty Boost - Meteorite Type Clustering via Feature Embeddings

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Prepare embeddings using PCA for visualization
pca = PCA(n_components=2)
embeddings = pca.fit_transform(features)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add cluster labels back to the dataset
ml_df['cluster'] = clusters

# Visualize Clusters from PCA projection
plt.figure(figsize=(10, 6))
sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=clusters, palette="Set2", alpha=0.8)
plt.title("Novelty Step: PCA Projection & KMeans Clustering on Meteorite Features")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Cluster composition
print("\nCluster Distribution based on PCA-KMeans embedding:")
print(ml_df['cluster'].value_counts())

# Step 13: Interactive Map of Meteorite Landings using Folium
import folium
from folium.plugins import MarkerCluster

# Create a map centered around the average coordinates
map_center = [df['reclat'].mean(), df['reclong'].mean()]
meteor_map = folium.Map(location=map_center, zoom_start=2, tiles='CartoDB positron')

# Create marker cluster
marker_cluster = MarkerCluster().add_to(meteor_map)

# Add points to the map
for _, row in df.dropna(subset=['reclat', 'reclong']).iterrows():
    popup_info = f"""
    <b>Name:</b> {row['name']}<br>
    <b>Mass (g):</b> {row['mass (g)']}<br>
    <b>Year:</b> {row['year']}<br>
    <b>Class:</b> {row['recclass']}
    """
    folium.CircleMarker(
        location=[row['reclat'], row['reclong']],
        radius=max(2, np.log1p(row['mass (g)']) / 2),
        color='blue',
        fill=True,
        fill_color='cyan',
        fill_opacity=0.6,
        popup=folium.Popup(popup_info, max_width=300)
    ).add_to(marker_cluster)

# Save the map to an HTML file
meteor_map.save('interactive_meteorite_map.html')
print("Interactive map saved as 'interactive_meteorite_map.html'")


# Notes on limitations and future work
print("\nLimitations & Future Work:")
print("- Dataset may contain inaccuracies or missing values.")
print("- Class imbalance in 'fall' column might affect prediction.")
print("- Could extend to time-series modeling or deep learning.")