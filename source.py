import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.preprocessing
import sklearn.datasets
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
import io
import base64

# --- Utility Functions (Provided and Refined) ---

def load_synthetic_financial_data(dataset_type: str) -> np.ndarray:
    """
    Generates and returns a synthetic dataset based on the dataset_type string.

    Args:
        dataset_type (str): Type of dataset to generate ('kmeans_portfolio' or 'spectral_assets').

    Returns:
        np.ndarray: Generated synthetic data.

    Raises:
        ValueError: If an invalid dataset_type is provided.
    """
    if dataset_type == 'kmeans_portfolio':
        X, y_true = sklearn.datasets.make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42, n_features=4)
        # Scale features to simulate financial features like growth, volatility, dividend yield
        X[:, 0] = X[:, 0] * 0.01 + 0.05  # Growth rate (0.01-0.1)
        X[:, 1] = np.abs(X[:, 1] * 0.02) + 0.1  # Volatility (0.1-0.5)
        X[:, 2] = X[:, 2] * 0.005 + 0.02  # Dividend yield (0.01-0.05)
        X[:, 3] = np.abs(X[:, 3] * 0.5) + 1  # P/E ratio (1-10)
        return X
    elif dataset_type == 'spectral_assets':
        X, y_true = sklearn.datasets.make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42, n_features=4)
        # Simulate more intertwined relationships
        X[:, 0] = X[:, 0] * 0.01 + 0.05
        X[:, 1] = np.sin(X[:, 1] * 0.5) + X[:, 0]
        X[:, 2] = X[:, 2] * 0.005 + 0.02
        X[:, 3] = np.cos(X[:, 3] * 0.5) + X[:, 1]
        return X
    else:
        raise ValueError("Invalid dataset_type. Choose 'kmeans_portfolio' or 'spectral_assets'.")

def preprocess_data(data: np.ndarray) -> np.ndarray:
    """
    Applies StandardScaler to the input data.

    Args:
        data (np.ndarray): The input data to be scaled.

    Returns:
        np.ndarray: The standardized data.
    """
    scaler = sklearn.preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def get_pca_components(data: np.ndarray, n_components: int = 2, random_state: int = 42) -> tuple[np.ndarray, PCA]:
    """
    Performs PCA to reduce data dimensionality for visualization.

    Args:
        data (np.ndarray): The input data.
        n_components (int): Number of principal components to keep.
        random_state (int): Random state for PCA.

    Returns:
        tuple[np.ndarray, PCA]: A tuple containing the 2D projected data and the fitted PCA model.
    """
    if data.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=random_state)
        data_2d = pca.fit_transform(data)
    else:
        data_2d = data
        pca = None # No PCA performed if data is already <= n_components
    return data_2d, pca

# --- Clustering Algorithm Functions (Provided) ---

def run_kmeans(data: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs K-Means clustering on the provided data.

    Args:
        data (np.ndarray): The input data for clustering.
        n_clusters (int): The number of clusters (k).
        random_state (int): Random state for K-Means initialization.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - cluster_labels (np.ndarray): Array of cluster labels for each data point.
            - cluster_centers (np.ndarray): Coordinates of the cluster centers.
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def run_spectral_clustering(data: np.ndarray, n_clusters: int, affinity_kernel: str = 'rbf', gamma: float = 1.0, random_state: int = 42) -> np.ndarray:
    """
    Runs Spectral Clustering on the provided data.

    Args:
        data (np.ndarray): The input data for clustering.
        n_clusters (int): The number of clusters.
        affinity_kernel (str): The affinity kernel to use (e.g., 'rbf', 'nearest_neighbors').
        gamma (float): Kernel coefficient for 'rbf', 'poly', 'sigmoid', 'laplacian' and 'chi2' kernels.
        random_state (int): Random state for reproducible results.

    Returns:
        np.ndarray: Array of cluster labels for each data point.
    """
    spectral = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity=affinity_kernel, gamma=gamma,
                                                  random_state=random_state, assign_labels='kmeans', n_init=10)
    spectral.fit(data)
    return spectral.labels_

def run_hierarchical_clustering(data: np.ndarray, n_clusters: int, linkage_method: str, affinity_metric: str) -> np.ndarray:
    """
    Runs Agglomerative Hierarchical Clustering.

    Args:
        data (np.ndarray): The input data for clustering.
        n_clusters (int): The number of clusters to form.
        linkage_method (str): Which linkage criterion to use ('ward', 'complete', 'average', 'single').
        affinity_metric (str): The metric used to compute the linkage ('euclidean', 'l1', 'l2', 'manhattan', 'cosine').

    Returns:
        np.ndarray: Array of cluster labels for each data point.
    """
    # AgglomerativeClustering does not support 'cosine' with 'ward' linkage directly through metric.
    # Ward linkage only supports Euclidean distance.
    if linkage_method == 'ward' and affinity_metric != 'euclidean':
        print(f"Warning: Ward linkage only supports euclidean affinity. Using 'euclidean' for affinity_metric.")
        current_affinity_metric = 'euclidean'
    else:
        current_affinity_metric = affinity_metric
    
    # sklearn.cluster.AgglomerativeClustering's 'metric' parameter became 'affinity' in newer versions.
    # It seems the notebook used 'metric'. Let's use 'metric' for compatibility but clarify.
    # The 'metric' parameter applies to the distance between samples, not the linkage itself.
    # For linkage, the 'metric' is part of scipy.cluster.hierarchy.linkage
    # For AgglomerativeClustering, the 'affinity' parameter determines the distance.
    # The `affinity` parameter in AgglomerativeClustering is equivalent to `metric` for linkage functions.
    agg_clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, affinity=current_affinity_metric)
    agg_clustering.fit(data)
    return agg_clustering.labels_

# --- Visualization Functions (Return Plotly/Matplotlib objects) ---

def create_cluster_scatter_plot(
    data_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    feature_columns: list[str],
    centers_2d: np.ndarray = None
) -> go.Figure:
    """
    Generates a Plotly scatter plot for clustering results.

    Args:
        data_2d (np.ndarray): 2D array of data points (e.g., PCA-reduced).
        labels (np.ndarray): Cluster labels for each data point.
        title (str): Title of the plot.
        feature_columns (list[str]): Names for the two feature columns (e.g., ['Feature 1', 'Feature 2 (PCA)']).
        centers_2d (np.ndarray, optional): 2D array of cluster centers, if applicable (e.g., for K-Means). Defaults to None.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    df_plot = pd.DataFrame(data_2d, columns=feature_columns)
    df_plot['Cluster'] = labels.astype(str)

    fig = px.scatter(df_plot, x=df_plot.columns[0], y=df_plot.columns[1],
                     color='Cluster', title=title,
                     hover_data={'Cluster': True, df_plot.columns[0]: ':.2f', df_plot.columns[1]: ':.2f'})

    if centers_2d is not None and len(centers_2d) > 0:
        fig.add_trace(go.Scatter(
            x=centers_2d[:, 0], y=centers_2d[:, 1],
            mode='markers', marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
            name='Centroids',
            hoverinfo='none'
        ))
    
    fig.update_layout(height=600, width=800)
    return fig

def create_dendrogram_plot(
    data: np.ndarray,
    linkage_method: str,
    affinity_metric: str,
    n_clusters_display: int = None,
    title: str = "Hierarchical Clustering Dendrogram"
) -> plt.Figure:
    """
    Generates and returns a Matplotlib dendrogram figure.

    Args:
        data (np.ndarray): The input data for calculating linkage.
        linkage_method (str): The linkage criterion ('ward', 'complete', 'average', 'single').
        affinity_metric (str): The distance metric ('euclidean', 'l1', 'l2', 'manhattan', 'cosine').
        n_clusters_display (int, optional): If provided, draws a horizontal line to indicate a cut-off
                                            for this number of clusters. Defaults to None.
        title (str, optional): Title of the dendrogram. Defaults to "Hierarchical Clustering Dendrogram".

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object.
    """
    # Scipy's linkage function expects a consistent metric, ward only accepts euclidean
    current_affinity_metric = affinity_metric
    if linkage_method == 'ward' and affinity_metric != 'euclidean':
        print(f"Warning: Ward linkage in dendrogram only supports euclidean metric. Using 'euclidean'.")
        current_affinity_metric = 'euclidean'
        
    Z = sch.linkage(data, method=linkage_method, metric=current_affinity_metric)

    fig, ax = plt.subplots(figsize=(15, 7))
    sch.dendrogram(Z, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Data Points')
    ax.set_ylabel(f'{current_affinity_metric.capitalize()} distances')

    if n_clusters_display is not None and n_clusters_display > 1 and n_clusters_display <= len(Z):
        # Determine the height at which the dendrogram would be cut to yield n_clusters
        # This is typically the distance value of the (N - n_clusters + 1)-th merge.
        # Z contains [idx1, idx2, distance, num_points_in_cluster]
        # To get k clusters, you look at the N-k+1 highest merges
        # The threshold is the distance of the (N - n_clusters)th merge from the top, or (len(Z) - n_clusters + 1) index
        # For Z (N-1) rows, to get k clusters, you cut at Z[-(k-1), 2] if k>1
        # Example: N=10, k=3 => Z[-(3-1), 2] = Z[-2, 2]
        # Or more robustly, find the largest distance such that merging below it gives >= n_clusters.
        
        # A simple way to visualize is to find the (N-k+1)th largest distance value.
        # Z is sorted by distance. Z[-k+1, 2] is often used for this.
        if n_clusters_display > 0: # Ensure k is positive
            if n_clusters_display == 1: # Single cluster, cut at max height
                max_d = Z[-1, 2] + 0.1 * Z[-1, 2] # Slightly above the highest merge
            else: # For k > 1 clusters, cut below the (k-1)th highest merge
                max_d = Z[-(n_clusters_display - 1), 2]
            
            ax.axhline(y=max_d, color='r', linestyle='--', label=f'{n_clusters_display} Clusters Threshold')
            ax.legend()
    
    plt.tight_layout()
    return fig

# --- Orchestration Functions (For app.py integration) ---

def perform_kmeans_analysis(
    scaled_data: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> tuple[go.Figure, float, np.ndarray]:
    """
    Performs K-Means clustering, calculates silhouette score, and generates a Plotly scatter plot.

    Args:
        scaled_data (np.ndarray): Pre-processed (scaled) input data.
        n_clusters (int): Number of clusters for K-Means.
        random_state (int): Random state for K-Means and PCA.

    Returns:
        tuple[go.Figure, float, np.ndarray]: A tuple containing:
            - plotly_figure (go.Figure): Plotly scatter plot of clusters and centroids.
            - silhouette_score (float): Silhouette score for the clustering. Returns None if n_clusters <= 1.
            - labels (np.ndarray): Cluster labels.
    """
    kmeans_labels, kmeans_centers = run_kmeans(scaled_data, n_clusters=n_clusters, random_state=random_state)

    silhouette_avg = None
    if n_clusters > 1:
        silhouette_avg = sklearn.metrics.silhouette_score(scaled_data, kmeans_labels)

    # Reduce data and centers to 2D for visualization
    data_2d, pca_model = get_pca_components(scaled_data, n_components=2, random_state=random_state)
    centers_2d = None
    if pca_model is not None:
        centers_2d = pca_model.transform(kmeans_centers)
    elif scaled_data.shape[1] == 2: # If data was already 2D, centers are also 2D
        centers_2d = kmeans_centers
    
    feature_cols = [f'Feature 1 (PCA)' if scaled_data.shape[1] > 2 else 'Feature 1',
                    f'Feature 2 (PCA)' if scaled_data.shape[1] > 2 else 'Feature 2']

    plotly_figure = create_cluster_scatter_plot(
        data_2d, kmeans_labels,
        title=f'K-Means Clustering with k={n_clusters}',
        feature_columns=feature_cols,
        centers_2d=centers_2d
    )

    return plotly_figure, silhouette_avg, kmeans_labels

def perform_spectral_analysis(
    scaled_data: np.ndarray,
    n_clusters: int,
    affinity_kernel: str = 'rbf',
    gamma: float = 1.0,
    random_state: int = 42
) -> tuple[go.Figure, float, np.ndarray]:
    """
    Performs Spectral clustering, calculates silhouette score, and generates a Plotly scatter plot.

    Args:
        scaled_data (np.ndarray): Pre-processed (scaled) input data.
        n_clusters (int): Number of clusters for Spectral Clustering.
        affinity_kernel (str): The affinity kernel for Spectral Clustering.
        gamma (float): Kernel coefficient for 'rbf' affinity.
        random_state (int): Random state for Spectral Clustering and PCA.

    Returns:
        tuple[go.Figure, float, np.ndarray]: A tuple containing:
            - plotly_figure (go.Figure): Plotly scatter plot of clusters.
            - silhouette_score (float): Silhouette score for the clustering. Returns None if n_clusters <= 1.
            - labels (np.ndarray): Cluster labels.
    """
    spectral_labels = run_spectral_clustering(
        scaled_data, n_clusters=n_clusters, affinity_kernel=affinity_kernel, gamma=gamma, random_state=random_state
    )

    silhouette_avg = None
    if n_clusters > 1:
        silhouette_avg = sklearn.metrics.silhouette_score(scaled_data, spectral_labels)

    data_2d, _ = get_pca_components(scaled_data, n_components=2, random_state=random_state)
    
    feature_cols = [f'Feature 1 (PCA)' if scaled_data.shape[1] > 2 else 'Feature 1',
                    f'Feature 2 (PCA)' if scaled_data.shape[1] > 2 else 'Feature 2']

    plotly_figure = create_cluster_scatter_plot(
        data_2d, spectral_labels,
        title=f'Spectral Clustering with k={n_clusters}, affinity={affinity_kernel}',
        feature_columns=feature_cols,
        centers_2d=None # Spectral clustering does not explicitly provide centroids in the original feature space
    )

    return plotly_figure, silhouette_avg, spectral_labels

def perform_hierarchical_analysis(
    scaled_data: np.ndarray,
    n_clusters: int,
    linkage_method: str,
    affinity_metric: str,
    random_state: int = 42 # PCA uses random_state
) -> tuple[go.Figure, str, float, np.ndarray]:
    """
    Performs Agglomerative Hierarchical clustering, calculates silhouette score,
    generates a Plotly scatter plot, and a Matplotlib dendrogram.

    Args:
        scaled_data (np.ndarray): Pre-processed (scaled) input data.
        n_clusters (int): Number of clusters for Hierarchical Clustering.
        linkage_method (str): The linkage criterion.
        affinity_metric (str): The distance metric.
        random_state (int): Random state for PCA.

    Returns:
        tuple[go.Figure, str, float, np.ndarray]: A tuple containing:
            - plotly_figure (go.Figure): Plotly scatter plot of clusters.
            - dendrogram_image_base64 (str): Base64 encoded PNG image of the dendrogram.
            - silhouette_score (float): Silhouette score for the clustering. Returns None if n_clusters <= 1.
            - labels (np.ndarray): Cluster labels.
    """
    # Handle Ward linkage constraint for AgglomerativeClustering (uses 'affinity')
    current_affinity_metric_agg = affinity_metric
    if linkage_method == 'ward' and affinity_metric != 'euclidean':
        current_affinity_metric_agg = 'euclidean'
        print(f"Warning: Ward linkage in AgglomerativeClustering only supports euclidean affinity. Using 'euclidean'.")

    agg_labels = run_hierarchical_clustering(
        scaled_data, n_clusters=n_clusters, linkage_method=linkage_method, affinity_metric=current_affinity_metric_agg
    )

    silhouette_avg = None
    if n_clusters > 1:
        silhouette_avg = sklearn.metrics.silhouette_score(scaled_data, agg_labels)

    data_2d, _ = get_pca_components(scaled_data, n_components=2, random_state=random_state)

    feature_cols = [f'Feature 1 (PCA)' if scaled_data.shape[1] > 2 else 'Feature 1',
                    f'Feature 2 (PCA)' if scaled_data.shape[1] > 2 else 'Feature 2']

    plotly_figure = create_cluster_scatter_plot(
        data_2d, agg_labels,
        title=f'Agglomerative Hierarchical Clustering (k={n_clusters}, linkage={linkage_method})',
        feature_columns=feature_cols,
        centers_2d=None # Hierarchical clustering does not explicitly provide centroids
    )

    # Generate and convert dendrogram to base64 image
    dendrogram_fig = create_dendrogram_plot(
        scaled_data, linkage_method=linkage_method, affinity_metric=affinity_metric,
        n_clusters_display=n_clusters,
        title=f"Hierarchical Clustering Dendrogram (k={n_clusters}, linkage={linkage_method})"
    )
    buf = io.BytesIO()
    dendrogram_fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(dendrogram_fig) # Close the matplotlib figure to free memory
    dendrogram_image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return plotly_figure, dendrogram_image_base64, silhouette_avg, agg_labels

# --- Example Usage (for demonstration, not to be directly imported by app.py) ---
if __name__ == "__main__":
    print("--- Running Demo for app.py functions ---")

    # 1. Load and Preprocess Data
    kmeans_data_raw = load_synthetic_financial_data('kmeans_portfolio')
    scaled_kmeans_data = preprocess_data(kmeans_data_raw)
    print(f"Scaled K-Means Data Shape: {scaled_kmeans_data.shape}")

    spectral_data_raw = load_synthetic_financial_data('spectral_assets')
    scaled_spectral_data = preprocess_data(spectral_data_raw)
    print(f"Scaled Spectral Data Shape: {scaled_spectral_data.shape}")

    # 2. K-Means Analysis
    print("\n--- K-Means Analysis (k=3) ---")
    kmeans_plot, kmeans_silhouette, kmeans_labels = perform_kmeans_analysis(
        scaled_kmeans_data, n_clusters=3
    )
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")
    print(f"K-Means Labels (first 5): {kmeans_labels[:5]}")
    # In an app.py, you would render kmeans_plot using fig.to_json() or similar
    # kmeans_plot.show() # Uncomment to view Plotly figure in a script

    print("\n--- K-Means Analysis (interactive simulation k=5) ---")
    kmeans_plot_int, kmeans_silhouette_int, _ = perform_kmeans_analysis(
        scaled_kmeans_data, n_clusters=5
    )
    print(f"Interactive K-Means Silhouette Score (k=5): {kmeans_silhouette_int:.3f}")
    # kmeans_plot_int.show() # Uncomment to view Plotly figure

    # 3. Spectral Clustering Analysis
    print("\n--- Spectral Clustering Analysis (k=3) ---")
    spectral_plot, spectral_silhouette, spectral_labels = perform_spectral_analysis(
        scaled_spectral_data, n_clusters=3, affinity_kernel='rbf'
    )
    print(f"Spectral Silhouette Score: {spectral_silhouette:.3f}")
    print(f"Spectral Labels (first 5): {spectral_labels[:5]}")
    # spectral_plot.show() # Uncomment to view Plotly figure

    print("\n--- Spectral Clustering Analysis (interactive simulation k=4, nearest_neighbors) ---")
    spectral_plot_int, spectral_silhouette_int, _ = perform_spectral_analysis(
        scaled_spectral_data, n_clusters=4, affinity_kernel='nearest_neighbors'
    )
    print(f"Interactive Spectral Silhouette Score (k=4, nearest_neighbors): {spectral_silhouette_int:.3f}")
    # spectral_plot_int.show() # Uncomment to view Plotly figure

    # 4. Hierarchical Clustering Analysis
    print("\n--- Hierarchical Clustering Analysis (k=3, ward, euclidean) ---")
    agg_plot, agg_dendrogram_b64, agg_silhouette, agg_labels = perform_hierarchical_analysis(
        scaled_kmeans_data, n_clusters=3, linkage_method='ward', affinity_metric='euclidean'
    )
    print(f"Hierarchical Silhouette Score: {agg_silhouette:.3f}")
    print(f"Hierarchical Labels (first 5): {agg_labels[:5]}")
    # agg_plot.show() # Uncomment to view Plotly figure
    # To view dendrogram:
    # from IPython.display import HTML, display
    # display(HTML(f'<img src="data:image/png;base64,{agg_dendrogram_b64}"/>'))

    print("\n--- Hierarchical Clustering Analysis (interactive simulation k=4, complete, manhattan) ---")
    agg_plot_int, agg_dendrogram_b64_int, agg_silhouette_int, _ = perform_hierarchical_analysis(
        scaled_kmeans_data, n_clusters=4, linkage_method='complete', affinity_metric='manhattan'
    )
    print(f"Interactive Hierarchical Silhouette Score (k=4, complete, manhattan): {agg_silhouette_int:.3f}")
    # agg_plot_int.show() # Uncomment to view Plotly figure

    # 5. User data simulation (app.py would handle file upload and pass the dataframe/numpy array)
    print("\n--- User Data Simulation ---")
    user_data_df = pd.DataFrame(np.random.rand(100, 4), columns=[f'User_Feature_{i}' for i in range(4)])
    scaled_user_data = preprocess_data(user_data_df.values)
    print(f"Scaled user data shape: {scaled_user_data.shape}")
    print("First 5 rows of scaled user data:\n", scaled_user_data[:5])

    # Example: Run K-Means on user data
    user_kmeans_plot, user_kmeans_silhouette, _ = perform_kmeans_analysis(
        scaled_user_data, n_clusters=4
    )
    print(f"K-Means on user data Silhouette Score: {user_kmeans_silhouette:.3f}")
    # user_kmeans_plot.show() # Uncomment to view Plotly figure
