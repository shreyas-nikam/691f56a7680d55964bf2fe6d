
# Streamlit Application Specification: Interactive Clustering Workbench

## 1. Application Overview

The **Interactive Clustering Workbench** is designed to provide financial professionals and learners with an engaging platform to explore and apply unsupervised clustering techniques to financial data. The application addresses the challenge of discovering hidden patterns and structures in financial markets, where labeled data is often scarce. It focuses on demonstrating the practical application of k-Means, Spectral Clustering, and Agglomerative Hierarchical Clustering for tasks such as portfolio construction and correlated asset identification.

**High-level Story Flow:**

1.  **Introduction**: The user starts on a home page, gaining an overview of unsupervised learning and the clustering algorithms covered, along with their relevance in finance.
2.  **Data Preparation**: The user navigates to a data selection page where they can choose from pre-loaded synthetic financial datasets (e.g., stock features for portfolio construction, asset returns for correlation analysis) or upload their own CSV data. The chosen data is automatically preprocessed (scaled).
3.  **k-Means Exploration**: The user selects the k-Means clustering page. Here, they learn the algorithm's principles, interactively adjust the number of clusters ($k$), run the algorithm, and visualize the resulting clusters in a 2D plot. The Silhouette Score helps them evaluate clustering quality.
4.  **Spectral Clustering Exploration**: On the Spectral Clustering page, the user is introduced to this graph-based method. They can tune the number of clusters ($k$) and the affinity kernel, run the clustering, and observe how it identifies non-linear relationships in asset returns through 2D visualizations and the Silhouette Score.
5.  **Agglomerative Hierarchical Clustering Exploration**: The Hierarchical Clustering page guides the user through the bottom-up approach. They can adjust $k$, the linkage method, and the distance metric. The results are visualized both as a 2D scatter plot and an interactive dendrogram, allowing for a detailed understanding of cluster formation and evaluation via the Silhouette Score.
6.  **Real-world Application**: Throughout the exploration, the application emphasizes how these techniques can be applied to financial problems, allowing the learner to experience a hands-on workflow of data analysis, model tuning, and interpretation.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# Import all functions from source.py
from source import *
```

### `st.session_state` Design

All `st.session_state` keys will be initialized at the start of `app.py` if they do not already exist, ensuring state persistence across interactions and "page" navigation.

*   `st.session_state.page`: Stores the current active page selected from the sidebar.
    *   **Initialized**: `"Home"`
    *   **Updated**: Via `st.sidebar.selectbox` selection.
    *   **Read**: To conditionally render page content.
*   `st.session_state.raw_data`: Stores the raw, unscaled DataFrame selected or uploaded by the user.
    *   **Initialized**: `None`
    *   **Updated**: When a pre-loaded dataset is selected (`load_synthetic_financial_data`) or a CSV is uploaded.
    *   **Read**: As input for `preprocess_data`.
*   `st.session_state.scaled_data`: Stores the processed (scaled) numpy array ready for clustering.
    *   **Initialized**: `None`
    *   **Updated**: After `st.session_state.raw_data` is set, by calling `preprocess_data`.
    *   **Read**: As primary input for all clustering functions (`run_kmeans`, `run_spectral_clustering`, `run_hierarchical_clustering`).
*   `st.session_state.dataset_name`: Stores a string identifying the current dataset (e.g., "K-Means Portfolio Data", "Uploaded Data").
    *   **Initialized**: `"No data loaded"`
    *   **Updated**: When a pre-loaded dataset is selected or a CSV is uploaded.
    *   **Read**: For display purposes.
*   `st.session_state.kmeans_labels`: Stores the cluster assignments from the latest k-Means run.
    *   **Initialized**: `None`
    *   **Updated**: After `run_kmeans` is called.
    *   **Read**: For k-Means visualization and Silhouette Score calculation.
*   `st.session_state.kmeans_centers`: Stores the cluster centroids from the latest k-Means run.
    *   **Initialized**: `None`
    *   **Updated**: After `run_kmeans` is called.
    *   **Read**: For k-Means visualization.
*   `st.session_state.spectral_labels`: Stores the cluster assignments from the latest Spectral Clustering run.
    *   **Initialized**: `None`
    *   **Updated**: After `run_spectral_clustering` is called.
    *   **Read**: For Spectral Clustering visualization and Silhouette Score calculation.
*   `st.session_state.agg_labels`: Stores the cluster assignments from the latest Agglomerative Hierarchical Clustering run.
    *   **Initialized**: `None`
    *   **Updated**: After `run_hierarchical_clustering` is called.
    *   **Read**: For Hierarchical Clustering visualization and Silhouette Score calculation.
*   `st.session_state.current_pca`: Stores the fitted `PCA` object for consistent dimensionality reduction.
    *   **Initialized**: `None`
    *   **Updated**: When data with `n_features > 2` is processed for visualization, by fitting `PCA`.
    *   **Read**: To transform data for 2D visualization across clustering methods.
*   `st.session_state.current_data_2d`: Stores the 2D projected data for visualization.
    *   **Initialized**: `None`
    *   **Updated**: After `st.session_state.scaled_data` is processed through `PCA` (if needed) or used directly.
    *   **Read**: For all scatter plot visualizations.
*   `st.session_state.current_centers_2d`: Stores the 2D projected centroids for k-Means visualization.
    *   **Initialized**: `None`
    *   **Updated**: After `st.session_state.kmeans_centers` is transformed by `PCA`.
    *   **Read**: For k-Means scatter plot.
*   `st.session_state.linkage_matrix`: Stores the linkage matrix required for dendrogram plotting.
    *   **Initialized**: `None`
    *   **Updated**: After `sch.linkage` is called on the `scaled_data` for Hierarchical Clustering.
    *   **Read**: For dendrogram visualization.

### UI Interactions and Function Invocations

The Streamlit app will be structured into distinct pages, managed by a sidebar `st.selectbox`.

**Sidebar Navigation**:

```python
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Choose a section",
    ["Home", "Select & Preprocess Data", "K-Means Clustering", "Spectral Clustering", "Agglomerative Hierarchical Clustering"]
)
st.session_state.page = page_selection
```

---

#### Page: "Home"

**Layout**: Main content area.

**Markdown**:

```python
st.title("Interactive Clustering Workbench")
st.markdown(f"")
st.markdown(f"This workbench explores various unsupervised clustering techniques. Unsupervised learning discovers hidden patterns in data without predefined labels, making it invaluable for tasks like market segmentation and portfolio construction in finance. We will focus on k-Means, Spectral Clustering, and Agglomerative Hierarchical Clustering.")
st.markdown(f"")
st.header("Learning Outcomes:")
st.markdown(f"- Understand the operational principles of k-Means, Spectral, and Agglomerative Hierarchical clustering algorithms.")
st.markdown(f"- Analyze the impact of different parameters (e.g., number of clusters $k$, linkage methods) on clustering results.")
st.markdown(f"- Visualize cluster assignments in 2D/3D space and interpret dendrograms.")
st.markdown(f"- Gain insights into how clustering can be applied for tasks like market segmentation or portfolio construction.")
st.markdown(f"")
st.markdown(f"Unsupervised learning is a branch of machine learning that encompasses algorithms used to discover hidden patterns and structures in data without labeled examples from which to learn. Unlike supervised learning, there is no 'ground truth' to guide the learning process, which means that the algorithm must discover hidden patterns and relationships in data without any explicit guidance from real-word observations regarding what constitutes the correct answer. Without ground truths, unsupervised learning algorithms must rely on mathematical principles, such as maximizing likelihood or minimizing error, to capture the essence of the data. This makes unsupervised learning both an art and a science, requiring careful consideration of what constitute meaningful patterns versus mere noise.")
st.markdown(f"")
st.markdown(f"In financial contexts, unsupervised learning can be particularly useful because financial markets are often opaque, labeled data are often scarce or expensive to obtain, or such data quickly become obsolete. In other words, the 'correct' answer is often elusive to varying degrees. Financial markets are also dynamic, and as market regimes change, new patterns emerge and traditional relationships often break down. In such cases, unsupervised learning methods can be invaluable in helping practitioners discover structures in financial data that may prove valuable in their portfolio and risk management efforts.")
st.markdown(f"")
st.header("Clustering Overview")
st.markdown(f"Perhaps the most well-known framework for unsupervised learning is clustering. Simpler clustering algorithms, such as k-means clustering (Lloyd 1982), operate according to a criterion of compactness, with observations grouped into different clusters based on their distance from designated centroids. These centroids are the average (mean) positions of all the data points that belong to a particular cluster. The algorithm for k-means clustering is shown in Figure 1.")
st.markdown(f"")
st.markdown(f"A k-means clustering approach makes a good choice when data are numeric, clusters are roughly spherical and similar in size, and a fast, scalable clustering for large datasets is needed. It is mathematically simple, efficient, and easy to interpret. However, k-means also assumes that clusters are spherical and equal sized, which is not always the case. Further, it is sensitive to initialization and outliers and requires a specification of the number of clusters, k. Finally, k-means clustering can detect only clusters that are linearly separable, limiting its usefulness in applications in which nonlinear or otherwise nuanced relationships are present.")
st.markdown(f"")
st.markdown(f"With the foregoing in mind, however, note that k-means has nevertheless been applied to portfolio construction. For example, Wu, Wang, and Wu (2022) used k-means to cluster stocks according to their continuous trend characteristics and then used inverse volatility weighting, risk parity, and mean-variance-type considerations to arrive at final portfolio weights.")
```

---

#### Page: "Select & Preprocess Data"

**Layout**:
*   `st.subheader` for pre-loaded data selection.
*   `st.selectbox` for dataset type.
*   `st.subheader` for user data upload.
*   `st.file_uploader` for CSV upload.
*   `st.button` to process data.
*   `st.write` to display data info and head.

**UI Interactions**:

1.  **Select Pre-loaded Data**:
    *   `dataset_choice = st.selectbox("Choose a pre-loaded dataset:", ["None", "K-Means Portfolio Data", "Spectral Assets Data"])`
    *   `if dataset_choice != "None":`
        *   `if dataset_choice == "K-Means Portfolio Data":`
            *   **Invokes**: `raw_data = load_synthetic_financial_data('kmeans_portfolio')`
            *   `st.session_state.dataset_name = "K-Means Portfolio Data"`
        *   `elif dataset_choice == "Spectral Assets Data":`
            *   **Invokes**: `raw_data = load_synthetic_financial_data('spectral_assets')`
            *   `st.session_state.dataset_name = "Spectral Assets Data"`
        *   `st.session_state.raw_data = raw_data`
        *   **Invokes**: `st.session_state.scaled_data = preprocess_data(st.session_state.raw_data)`
        *   `st.success(f"'{st.session_state.dataset_name}' loaded and scaled.")`

2.  **Upload User Data**:
    *   `uploaded_file = st.file_uploader("Upload your own CSV data", type=["csv"])`
    *   `if uploaded_file is not None:`
        *   `user_df = pd.read_csv(uploaded_file)`
        *   `st.session_state.raw_data = user_df.values` (assuming all columns are features)
        *   `st.session_state.dataset_name = uploaded_file.name`
        *   **Invokes**: `st.session_state.scaled_data = preprocess_data(st.session_state.raw_data)`
        *   `st.success(f"'{st.session_state.dataset_name}' uploaded and scaled.")`

3.  **Display Current Data**:
    *   `if st.session_state.scaled_data is not None:`
        *   `st.markdown(f"**Currently active dataset:** {st.session_state.dataset_name}")`
        *   `st.markdown(f"Shape of scaled data: {st.session_state.scaled_data.shape}")`
        *   `st.dataframe(pd.DataFrame(st.session_state.scaled_data).head())`

**Markdown**:

```python
st.title("Select & Preprocess Data")
st.markdown(f"")
st.markdown(f"This workbench also allows you to upload your own datasets for clustering analysis or choose from pre-loaded financial data. Your dataset should be in a CSV format and contain numerical features. The application will automatically preprocess (scale) your data before applying the clustering algorithms.")
st.markdown(f"")
st.subheader("1. Choose a Pre-loaded Dataset")
# (UI for selecting pre-loaded data)
st.subheader("2. Upload Your Own Dataset")
st.markdown(f"**Instructions:**")
st.markdown(f"1. Ensure your CSV file contains only numerical features relevant for clustering.")
st.markdown(f"2. Upload your file below.")
st.markdown(f"")
# (UI for file uploader)
st.markdown(f"")
st.subheader("Current Data Overview")
# (Display selected/uploaded data info)
```

---

#### Page: "K-Means Clustering"

**Layout**:
*   `st.subheader` for algorithm explanation.
*   Widgets for `n_clusters`.
*   `st.button` to trigger clustering.
*   `st.write` for silhouette score.
*   `st.plotly_chart` for scatter plot.

**UI Interactions**:

1.  **Parameter Selection**:
    *   `n_clusters_kmeans = st.slider("Number of Clusters (k):", min_value=2, max_value=10, value=3, step=1)`

2.  **Run K-Means**:
    *   `if st.button("Run K-Means Clustering") and st.session_state.scaled_data is not None:`
        *   **Invokes**: `kmeans_labels, kmeans_centers = run_kmeans(st.session_state.scaled_data, n_clusters=n_clusters_kmeans)`
        *   `st.session_state.kmeans_labels = kmeans_labels`
        *   `st.session_state.kmeans_centers = kmeans_centers`
        *   **Invokes**: `silhouette_avg = sklearn.metrics.silhouette_score(st.session_state.scaled_data, st.session_state.kmeans_labels)` (if `n_clusters_kmeans > 1`)
        *   `st.session_state.kmeans_silhouette = silhouette_avg`
        *   Perform PCA for 2D visualization:
            *   `if st.session_state.scaled_data.shape[1] > 2:`
                *   `st.session_state.current_pca = PCA(n_components=2, random_state=42)`
                *   `st.session_state.current_data_2d = st.session_state.current_pca.fit_transform(st.session_state.scaled_data)`
                *   `st.session_state.current_centers_2d = st.session_state.current_pca.transform(st.session_state.kmeans_centers)`
            *   `else:`
                *   `st.session_state.current_data_2d = st.session_state.scaled_data`
                *   `st.session_state.current_centers_2d = st.session_state.kmeans_centers`
        *   `st.success("K-Means clustering completed!")`
    *   `elif st.session_state.scaled_data is None:`
        *   `st.warning("Please load a dataset first on the 'Select & Preprocess Data' page.")`

3.  **Display Results**:
    *   `if st.session_state.kmeans_labels is not None:`
        *   `if st.session_state.kmeans_silhouette is not None:`
            *   `st.markdown(f"**Silhouette Score for k={n_clusters_kmeans}:** {st.session_state.kmeans_silhouette:.3f}")`
        *   `else: st.markdown("Silhouette Score not defined for a single cluster.")`

        *   Prepare data for Plotly:
            *   `df_kmeans = pd.DataFrame(st.session_state.current_data_2d, columns=['Feature 1 (PCA)' if st.session_state.scaled_data.shape[1] > 2 else 'Feature 1', 'Feature 2 (PCA)' if st.session_state.scaled_data.shape[1] > 2 else 'Feature 2'])`
            *   `df_kmeans['Cluster'] = st.session_state.kmeans_labels.astype(str)`
        *   Generate Plotly figure:
            *   `fig = px.scatter(df_kmeans, x=df_kmeans.columns[0], y=df_kmeans.columns[1], color='Cluster', title=f'K-Means Clustering with k={n_clusters_kmeans}', hover_data={'Cluster': True, df_kmeans.columns[0]: ':.2f', df_kmeans.columns[1]: ':.2f'})`
            *   `fig.add_trace(go.Scatter(x=st.session_state.current_centers_2d[:, 0], y=st.session_state.current_centers_2d[:, 1], mode='markers', marker=dict(symbol='x', size=15, color='black', line=dict(width=2)), name='Centroids', hoverinfo='none'))`
        *   `st.plotly_chart(fig)`

**Markdown**:

```python
st.title("K-Means Clustering")
st.markdown(f"")
st.markdown(f"For the k-Means demonstration, we'll use a synthetic financial dataset representing 'stock features for portfolio construction'. This dataset will consist of features like simulated growth rates, volatility, and dividend yields for different stocks. We will first load and scale this data.")
st.markdown(f"")
st.markdown(f"The `scaled_kmeans_data` is now ready for clustering. This dataset consists of several data points (representing stocks) each with standardized features, suitable for identifying distinct groups for portfolio construction.")
st.markdown(f"")
st.markdown(f"k-Means clustering is a partition-based algorithm that aims to partition $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (centroid). The algorithm iteratively assigns data points to clusters and updates cluster centroids until convergence.")
st.markdown(r"") # Placeholder for formatting
st.markdown(r"The objective function, often called the inertia, that k-Means aims to minimize is the sum of squared distances of samples to their closest cluster center:")
st.markdown(r"$$ J = \sum_{{i=0}}^{{n}}\min_{{\mu_j \in C}}(||x_i - \mu_j||^2) $$")
st.markdown(r"where $J$ is the objective function, $x_i$ is a data point, $\mu_j$ is the centroid of cluster $j$, and $C$ is the set of all centroids. The algorithm typically proceeds as described in Figure 1:")
st.markdown(f"1. Initialize $k$ centroids randomly.")
st.markdown(f"2. For a fixed number of iterations:")
st.markdown(f"   a. Assign each data point to its closest centroid.")
st.markdown(f"   b. Update each centroid to be the mean of all points assigned to that cluster.")
st.markdown(f"   c. If centroids do not change significantly, break.")
st.markdown(f"3. Output cluster assignments and final centroids.")
st.markdown(f"")
st.markdown(f"The `run_kmeans` function encapsulates the k-Means algorithm, allowing us to easily apply it with different parameters.")
st.markdown(f"")
st.markdown(f"The scatter plot visually represents the distinct clusters identified by the k-Means algorithm. Each color corresponds to a different cluster, and the black 'x' markers indicate the calculated centroids. In a financial context, these clusters could represent different types of stocks (e.g., growth stocks, value stocks, defensive stocks), aiding in diversified portfolio construction.")
st.markdown(f"")
st.markdown(f"The choice of the number of clusters, $k$, is critical for k-Means. We can evaluate the quality of clustering using metrics like the Silhouette Score. A higher Silhouette Score generally indicates better-defined clusters. The Silhouette Coefficient $s(i)$ for a single sample is calculated as:")
st.markdown(r"$$ s(i) = \frac{{b(i) - a(i)}}{{\max(a(i), b(i))}} $$")
st.markdown(r"where $a(i)$ is the mean distance between $i$ and all other data points in the same cluster, and $b(i)$ is the mean distance between $i$ and all other data points in the *next nearest* cluster.")
st.markdown(f"")
st.markdown(f"By adjusting the slider, you can observe how the cluster boundaries and assignments change. The Silhouette Score provides a quantitative measure of clustering quality, helping to identify a suitable number of clusters for the given dataset. Typically, we look for a $k$ that yields a high silhouette score while making sense contextually.")
st.markdown(f"")
```

---

#### Page: "Spectral Clustering"

**Layout**:
*   `st.subheader` for algorithm explanation.
*   Widgets for `n_clusters`, `affinity_kernel`.
*   `st.button` to trigger clustering.
*   `st.write` for silhouette score.
*   `st.plotly_chart` for scatter plot.

**UI Interactions**:

1.  **Parameter Selection**:
    *   `n_clusters_spectral = st.slider("Number of Clusters (k):", min_value=2, max_value=10, value=3, step=1)`
    *   `affinity_kernel_spectral = st.selectbox("Affinity Kernel:", options=['rbf', 'nearest_neighbors'], index=0)`

2.  **Run Spectral Clustering**:
    *   `if st.button("Run Spectral Clustering") and st.session_state.scaled_data is not None:`
        *   **Invokes**: `spectral_labels = run_spectral_clustering(st.session_state.scaled_data, n_clusters=n_clusters_spectral, affinity_kernel=affinity_kernel_spectral)`
        *   `st.session_state.spectral_labels = spectral_labels`
        *   **Invokes**: `silhouette_avg = sklearn.metrics.silhouette_score(st.session_state.scaled_data, st.session_state.spectral_labels)` (if `n_clusters_spectral > 1`)
        *   `st.session_state.spectral_silhouette = silhouette_avg`
        *   Perform PCA for 2D visualization:
            *   `if st.session_state.scaled_data.shape[1] > 2:`
                *   `st.session_state.current_pca = PCA(n_components=2, random_state=42)`
                *   `st.session_state.current_data_2d = st.session_state.current_pca.fit_transform(st.session_state.scaled_data)`
            *   `else:`
                *   `st.session_state.current_data_2d = st.session_state.scaled_data`
        *   `st.success("Spectral clustering completed!")`
    *   `elif st.session_state.scaled_data is None:`
        *   `st.warning("Please load a dataset first on the 'Select & Preprocess Data' page.")`

3.  **Display Results**:
    *   `if st.session_state.spectral_labels is not None:`
        *   `if st.session_state.spectral_silhouette is not None:`
            *   `st.markdown(f"**Silhouette Score for k={n_clusters_spectral}, affinity='{affinity_kernel_spectral}':** {st.session_state.spectral_silhouette:.3f}")`
        *   `else: st.markdown("Silhouette Score not defined for a single cluster.")`

        *   Prepare data for Plotly:
            *   `df_spectral = pd.DataFrame(st.session_state.current_data_2d, columns=['Feature 1 (PCA)' if st.session_state.scaled_data.shape[1] > 2 else 'Feature 1', 'Feature 2 (PCA)' if st.session_state.scaled_data.shape[1] > 2 else 'Feature 2'])`
            *   `df_spectral['Cluster'] = st.session_state.spectral_labels.astype(str)`
        *   Generate Plotly figure:
            *   `fig = px.scatter(df_spectral, x=df_spectral.columns[0], y=df_spectral.columns[1], color='Cluster', title=f'Spectral Clustering with k={n_clusters_spectral}, affinity={affinity_kernel_spectral}', hover_data={'Cluster': True, df_spectral.columns[0]: ':.2f', df_spectral.columns[1]: ':.2f'})`
        *   `st.plotly_chart(fig)`

**Markdown**:

```python
st.title("Spectral Clustering")
st.markdown(f"")
st.markdown(f"For Spectral Clustering, we'll utilize another synthetic financial dataset representing 'asset returns for correlated asset identification'. This dataset will simulate returns of various assets that might exhibit non-linear correlations or lie on manifolds, making it suitable for Spectral Clustering to identify groups of correlated assets.")
st.markdown(f"")
st.markdown(f"The `scaled_spectral_data` is now ready. This dataset represents assets whose relationships might be better captured by graph-based similarity rather than direct Euclidean distance, which is where Spectral Clustering excels.")
st.markdown(f"")
st.markdown(f"Spectral Clustering is a technique that uses the eigenvalues (spectrum) of a similarity matrix to perform dimensionality reduction before clustering in a lower-dimensional space. It is particularly effective for discovering non-globular or intertwined clusters. The process generally follows Figure 2:")
st.markdown(f"1. **Construct similarity matrix $W$**: Measures the similarity between all pairs of data points. A common choice is the Gaussian kernel:")
st.markdown(r"$$ W[i, j] = \exp\left(-\frac{{||x_i - x_j||^2}}{{2\sigma^2}}\right) $$")
st.markdown(r"where $x_i$ and $x_j$ are data points, and $\sigma$ is a scaling parameter.")
st.markdown(f"2. **Compute degree matrix $D$**: A diagonal matrix where $D[i, i]$ is the sum of similarities of data point $i$ with all other data points:")
st.markdown(r"$$ D[i, i] = \sum_j W[i, j] $$")
st.markdown(r"where $D[i, i]$ is the sum of similarities of data point $i$ with all other data points.")
st.markdown(f"3. **Compute normalized Laplacian $L_{{norm}}$**: A matrix often used to reveal the graph structure:")
st.markdown(r"$$ L_{{norm}} = D^{{-1/2}} \times (D - W) \times D^{{-1/2}} $$")
st.markdown(r"where $L_{{norm}}$ is the normalized Laplacian.")
st.markdown(f"4. **Find $k$ smallest eigenvectors of $L_{{norm}}$**: These eigenvectors form a new lower-dimensional representation of the data.")
st.markdown(f"5. **Form matrix $V$**: Consisting of these $k$ eigenvectors as columns.")
st.markdown(f"6. **Normalize rows of $V$ to unit length**.")
st.markdown(f"7. **Apply k-Means clustering to the rows of $V$**: This groups the data points in the transformed space.")
st.markdown(f"8. **Output cluster assignments**.")
st.markdown(f"")
st.markdown(f"The `run_spectral_clustering` function simplifies the application of Spectral Clustering with flexible parameters like the number of clusters and the affinity kernel.")
st.markdown(f"")
st.markdown(f"The visualization shows the clusters identified by Spectral Clustering. This method can uncover groupings that are not linearly separable, which is particularly useful in financial contexts for identifying complex relationships between assets, such as groups of assets that exhibit similar behavior under certain market regimes.")
st.markdown(f"")
st.markdown(f"Similar to k-Means, the number of clusters $k$ is an important parameter. Additionally, the `affinity` kernel (how similarity is defined) significantly impacts Spectral Clustering. We can explore these parameters interactively, along with the Silhouette Score.")
st.markdown(f"")
st.markdown(f"Observe how changes in `n_clusters` and the `affinity_kernel` affect the resulting clusters and the Silhouette Score. Different affinity kernels might be more appropriate for different underlying data structures. For example, 'rbf' is good for capturing dense regions, while 'nearest_neighbors' focuses on local connectivity.")
st.markdown(f"")
```

---

#### Page: "Agglomerative Hierarchical Clustering"

**Layout**:
*   `st.subheader` for algorithm explanation.
*   Widgets for `n_clusters`, `linkage_method`, `affinity_metric`.
*   `st.button` to trigger clustering.
*   `st.write` for silhouette score.
*   `st.plotly_chart` for scatter plot.
*   `st.pyplot` for dendrogram.

**UI Interactions**:

1.  **Parameter Selection**:
    *   `n_clusters_agg = st.slider("Number of Clusters (k):", min_value=2, max_value=10, value=3, step=1)`
    *   `linkage_method_agg = st.selectbox("Linkage Method:", options=['ward', 'complete', 'average', 'single'], index=0)`
    *   `affinity_metric_agg = st.selectbox("Distance Metric:", options=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], index=0)`

2.  **Run Hierarchical Clustering**:
    *   `if st.button("Run Hierarchical Clustering") and st.session_state.scaled_data is not None:`
        *   Handle `ward` linkage constraint:
            *   `if linkage_method_agg == 'ward' and affinity_metric_agg != 'euclidean':`
                *   `st.warning("Warning: Ward linkage only supports euclidean affinity. Using 'euclidean' for affinity_metric.")`
                *   `current_affinity_metric_agg = 'euclidean'`
            *   `else:`
                *   `current_affinity_metric_agg = affinity_metric_agg`
        *   **Invokes**: `agg_labels = run_hierarchical_clustering(st.session_state.scaled_data, n_clusters=n_clusters_agg, linkage_method=linkage_method_agg, affinity_metric=current_affinity_metric_agg)`
        *   `st.session_state.agg_labels = agg_labels`
        *   **Invokes**: `silhouette_avg = sklearn.metrics.silhouette_score(st.session_state.scaled_data, st.session_state.agg_labels)` (if `n_clusters_agg > 1`)
        *   `st.session_state.agg_silhouette = silhouette_avg`
        *   Perform PCA for 2D visualization:
            *   `if st.session_state.scaled_data.shape[1] > 2:`
                *   `st.session_state.current_pca = PCA(n_components=2, random_state=42)`
                *   `st.session_state.current_data_2d = st.session_state.current_pca.fit_transform(st.session_state.scaled_data)`
            *   `else:`
                *   `st.session_state.current_data_2d = st.session_state.scaled_data`
        *   **Invokes**: `st.session_state.linkage_matrix = sch.linkage(st.session_state.scaled_data, method=linkage_method_agg, metric=current_affinity_metric_agg)` (if possible with selected metric)
        *   `st.success("Agglomerative Hierarchical clustering completed!")`
    *   `elif st.session_state.scaled_data is None:`
        *   `st.warning("Please load a dataset first on the 'Select & Preprocess Data' page.")`

3.  **Display Results**:
    *   `if st.session_state.agg_labels is not None:`
        *   `if st.session_state.agg_silhouette is not None:`
            *   `st.markdown(f"**Silhouette Score for k={n_clusters_agg}, linkage='{linkage_method_agg}', affinity='{current_affinity_metric_agg}':** {st.session_state.agg_silhouette:.3f}")`
        *   `else: st.markdown("Silhouette Score not defined for a single cluster.")`

        *   Prepare data for Plotly:
            *   `df_agg = pd.DataFrame(st.session_state.current_data_2d, columns=['Feature 1 (PCA)' if st.session_state.scaled_data.shape[1] > 2 else 'Feature 1', 'Feature 2 (PCA)' if st.session_state.scaled_data.shape[1] > 2 else 'Feature 2'])`
            *   `df_agg['Cluster'] = st.session_state.agg_labels.astype(str)`
        *   Generate Plotly figure:
            *   `fig = px.scatter(df_agg, x=df_agg.columns[0], y=df_agg.columns[1], color='Cluster', title=f'Agglomerative Hierarchical Clustering (k={n_clusters_agg}, linkage={linkage_method_agg})', hover_data={'Cluster': True, df_agg.columns[0]: ':.2f', df_agg.columns[1]: ':.2f'})`
        *   `st.plotly_chart(fig)`

        *   **Dendrogram Display**:
            *   `if st.session_state.linkage_matrix is not None:`
                *   `st.subheader("Dendrogram")`
                *   `fig_dendro = plt.figure(figsize=(15, 7))`
                *   **Invokes**: `sch.dendrogram(st.session_state.linkage_matrix)`
                *   `plt.title(f"Hierarchical Clustering Dendrogram (Linkage: {linkage_method_agg})")`
                *   `plt.xlabel('Data Points')`
                *   `plt.ylabel('Euclidean distances')`
                *   `if n_clusters_agg is not None:`
                    *   `max_d = st.session_state.linkage_matrix[-n_clusters_agg, 2]`
                    *   `plt.axhline(y=max_d, color='r', linestyle='--', label=f'{n_clusters_agg} Clusters Threshold')`
                    *   `plt.legend()`
                *   `st.pyplot(fig_dendro)`

**Markdown**:

```python
st.title("Agglomerative Hierarchical Clustering")
st.markdown(f"")
st.markdown(f"Agglomerative Hierarchical Clustering builds a hierarchy of clusters from individual data points. It is a 'bottom-up' approach, where each data point starts as its own cluster, and then pairs of clusters are iteratively merged based on their proximity until all points belong to a single cluster or a desired number of clusters is reached. This process is detailed in Figure 3.")
st.markdown(f"")
st.markdown(f"Key concepts include:")
st.markdown(f"- **Distance Metric**: How the distance between individual data points is measured (e.g., Euclidean, Manhattan).")
st.markdown(f"- **Linkage Method**: How the distance between two clusters is defined. Common methods include:")
st.markdown(f"    - **Single Linkage**: The shortest distance between any two points in the two clusters.")
st.markdown(r"$$ d(C_i, C_j) = \min_{{x \in C_i, y \in C_j}} d(x, y) $$")
st.markdown(f"    - **Complete Linkage**: The maximum distance between any two points in the two clusters.")
st.markdown(r"$$ d(C_i, C_j) = \max_{{x \in C_i, y \in C_j}} d(x, y) $$")
st.markdown(f"    - **Average Linkage**: The average distance between all pairs of points across the two clusters.")
st.markdown(r"$$ d(C_i, C_j) = \text{{mean}}_{{x \in C_i, y \in C_j}} d(x, y) $$")
st.markdown(f"    - **Ward Linkage**: Minimizes the variance of the clusters being merged. It calculates the increase in the total within-cluster variance after merging.")
st.markdown(f"")
st.markdown(f"The `run_hierarchical_clustering` function allows us to experiment with different linkage methods and distance metrics. The `plot_dendrogram` function provides a visual representation of the hierarchical structure, which is crucial for understanding the merging process.")
st.markdown(f"")
st.markdown(f"The scatter plot shows the clusters identified by hierarchical clustering. The dendrogram provides a tree-like diagram that illustrates the sequence of merges or splits. By cutting the dendrogram at a specific height, one can determine the cluster assignments, allowing for visual inspection of cluster formation at various levels of granularity.")
st.markdown(f"")
st.markdown(f"The power of hierarchical clustering lies in exploring different linkage methods and distance metrics. We can interactively adjust these parameters along with the desired number of clusters ($k$) to understand their impact on the clustering structure and the Silhouette Score.")
st.markdown(f"")
st.markdown(f"Experiment with different linkage methods and distance metrics. You will observe how these choices significantly influence how clusters are formed and how sensitive the algorithm is to different data distributions. For instance, 'single' linkage can find elongated clusters, while 'ward' tends to find more compact, spherical clusters. The Silhouette Score helps quantify the effectiveness of these parameter choices.")
st.markdown(f"")
```

