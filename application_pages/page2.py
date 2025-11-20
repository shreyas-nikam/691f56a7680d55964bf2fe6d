"""application_pages/page2.py"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

def run_page2():
    current_data = st.session_state.current_data
    data_name = st.session_state.data_name

    st.markdown("## 8. Spectral Clustering: Theory")
    st.markdown("""
    Spectral Clustering is a technique that uses the eigenvalues (spectrum) of a similarity matrix to perform dimensionality reduction before clustering in a lower-dimensional space. It is particularly effective for discovering non-globular or intertwined clusters by leveraging the graph structure of the data. The process generally follows these steps:
    1.  **Construct similarity matrix $W$**: Measures the similarity between all pairs of data points. A common choice is the Gaussian (Radial Basis Function) kernel:
        $$ W[i, j] = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right) $$
        where $x_i$ and $x_j$ are data points, and $\sigma$ is a scaling parameter that controls the width of the similarity decay.
    2.  **Compute degree matrix $D$**: A diagonal matrix where $D[i, i]$ is the sum of similarities of data point $i$ with all other data points:
        $$ D[i, i] = \sum_j W[i, j] $$
    3.  **Compute normalized Laplacian $L_{norm}$**: A matrix often used to reveal the graph structure and its connected components:
        $$ L_{norm} = D^{-1/2} \times (D - W) \times D^{-1/2} $$
    4.  **Find $k$ smallest eigenvectors of $L_{norm}$**: These eigenvectors correspond to the lowest frequencies in the graph and form a new lower-dimensional, spectrally transformed representation of the data.
    5.  **Form matrix $V$**: Consisting of these $k$ eigenvectors as columns.
    6.  **Normalize rows of $V$ to unit length**.
    7.  **Apply k-Means clustering to the rows of $V$**: This groups the data points in the transformed, lower-dimensional space.
    8.  **Output cluster assignments**.
    """)

    def run_spectral_clustering(data, n_clusters, affinity_kernel, gamma, random_state=42):
        """Docstring: Executes Spectral Clustering on the given data using sklearn.cluster.SpectralClustering."""
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity_kernel,
            gamma=gamma,
            random_state=random_state,
            n_init=10
        )
        cluster_labels = model.fit_predict(data)
        return cluster_labels

    st.markdown("""
    The `run_spectral_clustering` function simplifies the application of Spectral Clustering with flexible parameters like the number of clusters and the affinity kernel. This allows us to efficiently explore how different similarity measures impact the discovery of intricate data structures.
    """)

    st.markdown("## 10. Spectral Clustering: Interactive Parameter Tuning and Evaluation")
    st.markdown("""
    Similar to k-Means, the number of clusters $k$ is an important parameter for Spectral Clustering. Additionally, the `affinity` kernel (how similarity is defined) significantly impacts the clustering results. We can explore these parameters interactively, along with the Silhouette Score, to understand their influence on cluster formation and quality.
    """)

    if current_data is not None:
        if current_data.shape[1] < 2:
            st.warning("Spectral clustering requires at least 2 features for 2D visualization. Please upload data with more features or select a demo dataset.")
        else:
            col1, col2 = st.columns(2)
            n_clusters_spectral = col1.slider('Number of Clusters (k) for Spectral Clustering:', min_value=2, max_value=10, value=3, key='spectral_n_clusters')
            affinity_kernel_spectral = col2.selectbox('Affinity Kernel for Spectral Clustering:', options=['rbf', 'nearest_neighbors'], value='rbf', key='spectral_affinity_kernel')
            gamma_val_spectral = 1.0 # Default gamma for rbf, not made interactive for brevity

            try:
                spectral_labels = run_spectral_clustering(current_data, n_clusters_spectral, affinity_kernel_spectral, gamma_val_spectral)

                unique_labels_count_spectral = len(np.unique(spectral_labels))
                if unique_labels_count_spectral > 1 and unique_labels_count_spectral <= len(current_data) -1:
                    silhouette_avg_spectral = silhouette_score(current_data, spectral_labels)
                    st.metric(label=f"Silhouette Score for {n_clusters_spectral} clusters:", value=f"{silhouette_avg_spectral:.3f}")
                else:
                    st.info(f"Silhouette Score cannot be computed for {unique_labels_count_spectral} unique clusters (requires >1 and < n_samples).")

                spectral_df_interactive = pd.DataFrame(current_data[:, :2], columns=['Feature 1', 'Feature 2'])
                spectral_df_interactive['Cluster'] = spectral_labels.astype(str)

                title_score_spectral = f'Silhouette Score: {silhouette_avg_spectral:.3f}' if unique_labels_count_spectral > 1 else 'Silhouette Score: N/A'
                fig_spectral = px.scatter(spectral_df_interactive, x='Feature 1', y='Feature 2', color='Cluster',
                                        title=f'Spectral Clustering (k={n_clusters_spectral}, Kernel=\"{affinity_kernel_spectral}\") of {data_name}<br>{title_score_spectral}',
                                        hover_data=['Cluster'])
                fig_spectral.update_layout(legend_title_text='Cluster')
                st.plotly_chart(fig_spectral, use_container_width=True)

                st.markdown("""
                Observe how changes in `n_clusters` and the `affinity_kernel` affect the resulting clusters and the Silhouette Score. Different affinity kernels might be more appropriate for different underlying data structures. For example, 'rbf' is good for capturing dense regions and non-linear boundaries, while 'nearest_neighbors' focuses on local connectivity. This interactive exploration helps in selecting parameters that yield meaningful and stable clusters for financial assets.
                """)
            except Exception as e:
                st.error(f"Spectral Clustering failed with given parameters: {e}")
    else:
        st.info("Please select a data source from the sidebar to run Spectral Clustering.")
