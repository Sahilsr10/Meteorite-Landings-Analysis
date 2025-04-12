# Meteorite Landings Analysis

A comprehensive data analysis project examining global meteorite landing patterns, classifications, and characteristics using the NASA Meteorite Landings dataset.

## 📊 Project Overview

This project conducts an in-depth analysis of meteorite landings across Earth, investigating spatial distributions, temporal patterns, classification insights, and statistical properties of recorded meteorite events.

## 🔍 Key Features

- **Geospatial Analysis**: Visualize global distribution of meteorite landings
- **Temporal Trend Analysis**: Examine changes in frequency and mass over time
- **Classification Insights**: Analyze different meteorite types and their properties
- **Statistical Analysis**: Identify outliers, correlations, and test hypotheses

## 🧪 Analysis Components

1. **Data Cleaning & Preprocessing**
   - Missing value handling
   - Data type conversions
   - Outlier detection

2. **Exploratory Data Analysis**
   - Geographical distribution mapping
   - Temporal trend identification
   - Classification distribution
   - "Fell" vs "Found" comparison

3. **Statistical Analysis**
   - Correlation analysis
   - Hypothesis testing
   - Z-score analysis
   - Extreme value identification

4. **Visualization**
   - Geographic scatter plots
   - Time series analysis
   - Mass distribution charts
   - Classification comparisons

## 📈 Key Findings

- Distribution of meteorite landings shows clear geographical patterns
- Temporal trends in meteorite documentation reflect both natural phenomena and improvements in detection methods
- Significant differences between "Fell" (observed) and "Found" meteorites
- The top meteorite classes exhibit distinct mass distributions
- Several notable outliers identified in the mass distribution

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **NumPy**: Numerical computing
- **SciPy**: Statistical analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required libraries: pandas, matplotlib, numpy, seaborn, scipy

### Installation

```bash
# Clone the repository
git clone https://github.com/username/meteorite-analysis.git
cd meteorite-analysis

# Set up virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
python meteorite_analysis.py
```

## 📁 Project Structure

```
meteorite-analysis/
├── data/
│   └── Meteorite_Landings.csv
├── images/
│   └── [visualization outputs]
├── meteorite_analysis.py
├── requirements.txt
└── README.md
```

## 📝 Dataset Information

The analysis uses the NASA Meteorite Landings dataset, which includes:
- Name and ID of meteorites
- Location data (latitude/longitude)
- Mass information
- Year of landing/discovery
- Classification details
- Whether the meteorite was observed falling ("Fell") or discovered later ("Found")

## 📚 Future Improvements

- Machine learning clustering to identify meteorite landing patterns
- Interactive visualization dashboard
- Additional datasets for cross-referencing with geological features
- Time-based analysis with historical events

## 📄 License

[MIT License](LICENSE)

## 👤 Author

Your Name - [GitHub Profile](https://github.com/)

## 🔗 References

- [NASA Open Data Portal](https://data.nasa.gov/)
- [The Meteoritical Society](https://www.lpi.usra.edu/meteor/)
