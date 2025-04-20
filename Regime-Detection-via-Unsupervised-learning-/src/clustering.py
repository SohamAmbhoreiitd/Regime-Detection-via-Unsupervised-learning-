import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def normalize_features(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def reduce_dimensions(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def apply_kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

def cluster_data(df, feature_columns, n_clusters=3):
    # Normalize features
    normalized_data = normalize_features(df[feature_columns])
    
    # Reduce dimensions
    reduced_data = reduce_dimensions(normalized_data)
    
    # Apply KMeans clustering
    labels = apply_kmeans_clustering(reduced_data, n_clusters)
    
    # Add cluster labels to the original dataframe
    df['Cluster'] = labels
    return df