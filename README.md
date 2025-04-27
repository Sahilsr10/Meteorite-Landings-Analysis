# Meteorite Data Analysis Project 🌠

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-latest-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-red)
![Seaborn](https://img.shields.io/badge/Seaborn-latest-purple)

## Overview

This project conducts a comprehensive analysis of meteorite landings data, exploring geospatial patterns, temporal trends, and meteorite characteristics. The analysis incorporates statistical methods, machine learning, and interactive visualizations to derive meaningful insights from the Meteorite Landings dataset.

## Objectives

1. Analyze geospatial patterns in meteorite landings
2. Identify influential features for meteorite discovery
3. Predict whether a meteorite was 'Fell' (observed falling) or 'Found' using machine learning

## Features

### Data Preparation & Cleaning
- Missing value detection and imputation
- Outlier identification using IQR and Z-score methods
- Data integrity verification and preprocessing

### Exploratory Data Analysis
- Geospatial visualization of landing coordinates
- Temporal trend analysis of landing frequency and mass
- Classification-based insights on meteorite types
- Distribution analysis of meteorite characteristics

### Statistical Analysis
- Hypothesis testing (T-test) to compare 'Fell' vs 'Found' meteorites
- Correlation and covariance analysis between features
- Probability distribution modeling of meteorite mass
- Q-Q plots for normality assumption verification

### Advanced Analytics
- Geospatial clustering using DBSCAN algorithm
- Random Forest classification to predict 'Fell' vs 'Found' status
- Feature importance analysis for predictive modeling
- Feature embedding and clustering using PCA and K-Means

### Interactive Visualization
- Dynamic map visualization with Folium
- Cluster analysis with visual representation
- Interactive meteorite landing map with detailed popups

## Key Findings

- Significant patterns in the geographical distribution of meteorite landings
- Temporal trends in meteorite discoveries over years
- Mass distribution varies significantly across meteorite classes
- Year of discovery is a crucial factor in predicting whether a meteorite was observed falling or found later
- Successful 'Fell' vs 'Found' classification with Random Forest model
- Identification of potential landing clusters using DBSCAN

## Visualizations

The project includes various visualizations:

- Scatter plots of geographical landing locations
- Time series analysis of meteorite discoveries
- Distribution plots for meteorite mass
- Bar charts for class comparisons
- Heatmaps for correlation analysis
- Interactive map of landing sites

## Machine Learning Implementation

- **Model**: Random Forest Classifier
- **Features**: Mass, year, latitude, longitude, and meteorite class
- **Target**: Predicting whether a meteorite was observed falling ('Fell') or discovered later ('Found')
- **Performance**: Detailed in classification report with precision, recall, and F1-score metrics

## Technical Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- folium

## Usage

```python
# Clone the repository
git clone https://github.com/yourusername/meteorite-analysis.git

# Install required packages
pip install -r requirements.txt

# Run the analysis
python meteorite_analysis.py
```

## Dataset

The analysis uses the "Meteorite_Landings.csv" dataset, which contains records of meteorites that have fallen to Earth. The dataset includes features such as:

- Name of the meteorite
- Mass in grams
- Year of discovery/fall
- Latitude and longitude coordinates
- Classification type
- Whether it was observed falling ('Fell') or discovered later ('Found')

## Limitations & Future Work

- Dataset may contain inaccuracies or missing values
- Class imbalance in the 'fall' column might affect prediction performance
- Future work could extend to time-series modeling or deep learning approaches
- Additional external data sources could enhance the analysis

## License

[MIT License](LICENSE)

## Contributors

- Sahil Srivastava(https://github.com/Sahilsr10)

## Acknowledgments

- The Meteoritical Society for the dataset
- Contributors to the Python data science ecosystem
