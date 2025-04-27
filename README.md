🌎 Meteorite Landings Analysis & Predictive Modeling

A comprehensive data science project exploring global meteorite landings using NASA’s dataset, with predictive modeling, geospatial clustering, and statistical analysis.

📊 Project Overview

This project analyzes meteorite landing patterns, temporal trends, classification insights, and mass distributions. It integrates advanced ML techniques like PCA, DBSCAN, and Random Forest classification to uncover deeper scientific patterns and predictive insights.

🔍 Key Features
	•	Geospatial Analysis: Visualize and cluster landing sites globally using DBSCAN
	•	Temporal Trend Analysis: Track meteorite landings and mass averages over time
	•	Classification Insights: Explore meteorite types and mass variations
	•	Predictive Modeling: Predict “Fell” vs “Found” meteorites using Random Forest (87% accuracy)
	•	Statistical Testing: Perform hypothesis testing, Z-score, and outlier detection
	•	Dimensionality Reduction: Apply PCA for feature embedding and visualization
	•	Interactive Mapping: Create a Folium-based world map with meteorite landings

🧪 Analysis Components
	1.	Data Cleaning & Preprocessing
	•	Handling missing values
	•	Data type conversions
	•	Outlier detection via IQR and Z-Score
	2.	Exploratory Data Analysis (EDA)
	•	Geospatial mapping
	•	Temporal landing trends
	•	Mass and class distribution analysis
	•	“Fell” vs “Found” comparisons
	3.	Machine Learning and Statistical Analysis
	•	Random Forest Classification (87% Accuracy)
	•	Principal Component Analysis (PCA) for clustering
	•	KMeans Clustering on meteorite features
	•	DBSCAN Geospatial Clustering for landing site groupings
	•	Hypothesis Testing (t-tests) for mass differences
	4.	Visualization
	•	Geospatial scatter plots
	•	Heatmaps and time series plots
	•	Mass distributions and boxplots
	•	Folium interactive map with clustered meteorite points

📈 Key Findings
	•	Clear geospatial clustering of meteorite landings across continents.
	•	Observable temporal trends in meteorite discovery rates.
	•	Significant statistical differences between “Fell” and “Found” meteorites.
	•	Dimensionality reduction via PCA revealed hidden structures among meteorite features.
	•	DBSCAN identified dense regions of meteorite impacts without prior assumptions.
	•	Machine Learning model accurately classified meteorites with high predictive performance.

🛠️ Technologies Used
	•	Python: Main programming language
	•	Pandas, NumPy: Data wrangling and computation
	•	Scikit-learn: Machine learning (Random Forest, PCA, KMeans, DBSCAN)
	•	Seaborn, Matplotlib: Data visualization
	•	SciPy: Statistical testing
	•	Folium: Interactive mapping and spatial visualization

🚀 Getting Started

Prerequisites
	•	Python 3.7+
	•	Required libraries: pandas, matplotlib, numpy, seaborn, scipy, scikit-learn, folium

Installation
# Clone the repository
git clone https://github.com/yourusername/meteorite-analysis.git
cd meteorite-analysis

# Create virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Running the Analysis
python CA3.py

📚 Dataset Information
	•	Dataset: NASA Meteorite Landings Dataset
	•	Attributes: Name, Mass, Year, Latitude, Longitude, Classification, Fall Type (“Fell”/“Found”)

📈 Future Improvements
	•	Integrate real-time meteorite detection feeds for live updates
	•	Build a dashboard using Plotly Dash or Streamlit
	•	Expand classification using advanced deep learning (e.g., XGBoost, Neural Networks)
	•	Cross-reference geological data for deeper spatial correlation insights

📄 License

MIT License

👤 Author

Sahil Srivastava - GitHub Profile

🔗 References
	•	NASA Open Data Portal
	•	The Meteoritical Society
