# Regime Detection via Unsupervised Learning

This project aims to detect different market regimes using unsupervised learning techniques. It utilizes depth and aggregated trade data to extract features, normalize them, and apply clustering algorithms for analysis.

## Project Structure

- **data/**: Contains the datasets used for analysis.
  - **depth20_1000ms/**: Directory with depth data files.
  - **aggTrade/**: Directory with aggregated trade data files.
  
- **notebooks/**: Contains Jupyter notebooks for data analysis.
  - **new.ipynb**: Main notebook for data loading, preprocessing, feature engineering, and exploration.

- **src/**: Source code for the project.
  - **\_\_init\_\_.py**: Marks the directory as a Python package.
  - **data_preprocessing.py**: Functions for loading and preprocessing data.
  - **feature_engineering.py**: Functions for feature extraction and transformation.
  - **dimensionality_reduction.py**: Implements dimensionality reduction techniques.
  - **clustering.py**: Functions for applying clustering algorithms.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd Regime-Detection-via-Unsupervised-learning-
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter notebook `notebooks/new.ipynb` to start the analysis.
2. Follow the instructions in the notebook to load data, preprocess it, and perform feature engineering.
3. Use the functions in `src/dimensionality_reduction.py` to reduce the feature space.
4. Apply clustering algorithms using the functions in `src/clustering.py` to identify market regimes.

## Overview

This project leverages unsupervised learning techniques to analyze market data, aiming to uncover hidden patterns and regimes that can inform trading strategies. The combination of feature engineering, normalization, dimensionality reduction, and clustering provides a comprehensive approach to market analysis.