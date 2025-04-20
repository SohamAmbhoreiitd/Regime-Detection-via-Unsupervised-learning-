# Regime Detection via Unsupervised Learning

## Steps in the Process

1. **Load Data into DataFrames**  
   Load and inspect the depth and trade data.

2. **Preprocess Depth Data**  
   Clean and preprocess the depth data to extract meaningful features.

3. **Preprocess Trade Data**  
   Clean and preprocess the trade data for further analysis.

4. **Add Price Features**  
   Calculate features like mid-price, log returns, and volatility.

5. **Add Volume Features**  
   Extract volume-related features such as buy/sell volume and VWAP.

6. **Merge Depth and Trade Data**  
   Combine the depth and trade data into a single DataFrame.

7. **Clean Merged Data**  
   Handle missing values and ensure data consistency.

8. **Scale Features**  
   Standardize the features for clustering and dimensionality reduction.

9. **Perform PCA**  
   Reduce dimensionality while retaining most of the variance.

10. **Visualize PCA Results**  
    Plot the first two principal components to understand the data distribution.

11. **KMeans Clustering**  
    Apply KMeans clustering and evaluate using silhouette scores.

12. **HDBSCAN Clustering**  
    Perform HDBSCAN clustering to detect regimes.

13. **Gaussian Mixture Model Clustering**  
    Use GMM for clustering and compare results.

14. **Final HDBSCAN Model**  
    Select the best HDBSCAN parameters and finalize the model.

15. **Analyze Regimes**  
    Group data by regimes and calculate statistics.

16. **Visualize Regimes**  
    Plot regime labels over time and analyze trends.

17. **Dimensionality Reduction with t-SNE**  
    Visualize high-dimensional data in 2D using t-SNE.

18. **Dimensionality Reduction with UMAP**  
    Use UMAP for another perspective on dimensionality reduction.

19. **Regime Transition Matrix**  
    Build and visualize the regime transition probability matrix.

20. **Strategy Ideas Based on Regimes**  
    Suggest trading strategies based on identified regimes