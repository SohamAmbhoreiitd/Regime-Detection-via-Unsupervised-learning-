import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def normalize_features(df):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns)

def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df)
    return pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

def apply_tsne(df, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_result = tsne.fit_transform(df)
    return pd.DataFrame(tsne_result, columns=[f'tSNE{i+1}' for i in range(n_components)])