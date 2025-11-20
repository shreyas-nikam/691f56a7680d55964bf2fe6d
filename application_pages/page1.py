"""application_pages/page1.py"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_page1():
    current_data = st.session_state.current_data
    data_name = st.session_state.data_name

    st.markdown("## 4. K-Means Clustering: Theory")
    st.markdown("""
    k-Means clustering is a partition-based algorithm that aims to partition $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (centroid). The algorithm iteratively assigns data points to clusters and updates cluster centroids until convergence.

    The objective function, often called the inertia, that k-Means aims to minimize is the sum of squared distances of samples to their closest cluster center:
    $$ J = \sum_{i=0}^{n}\min_{\mu_j \in C}(||x_i - \mu_j||^2) $$
    where $J$ is the objective function, $x_i$ is a data point, $\mu_j$ is the centroid of cluster $j$, and $C$ is the set of all centroids. The algorithm typically proceeds as follows:
    1. Initialize $k$ centroids randomly.
    2. For a fixed number of iterations:
       a. Assign each data point to its closest centroid.
       b. Update each centroid to be the mean of all points assigned to that cluster.
       c. If centroids do not change significantly, break.
    3. Output cluster assignments and final centroids.
    """)

    def run_kmeans(data, n_clusters, random_state=42):
        """Docstring: Runs K-Means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        kmeans.fit(data)
        return kmeans.labels_, kmeans.cluster_centers_

    st.markdown("""
    The `run_kmeans` function encapsulates the k-Means algorithm, allowing us to easily apply it with different parameters. This abstraction simplifies the process of exploring various cluster configurations and their impact on data segmentation.
    """)

    st.markdown("## 5. K-Means Clustering: Interactive Parameter Tuning and Evaluation")
    st.markdown("""
    The choice of the number of clusters, $k$, is critical for k-Means. We can evaluate the quality of clustering using metrics like the Silhouette Score. A higher Silhouette Score generally indicates better-defined clusters, where data points are well-matched to their own cluster and poorly matched to neighboring clusters. The Silhouette Coefficient $s(i)$ for a single sample is calculated as:
    $$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$
    where $a(i)$ is the mean distance between $i$ and all other data points in the same cluster, and $b(i)$ is the mean distance between $i$ and all other data points in the *next nearest* cluster.

    We can interactively adjust $k$ and observe its impact on the clusters and the Silhouette Score. This helps in finding an optimal $k$ that balances clustering quality with business interpretability.
    """)

    if current_data is not None:
        if current_data.shape[1] < 2:
            st.warning("K-Means clustering requires at least 2 features for 2D visualization. Please upload data with more features or select a demo dataset.")
        else:
            n_clusters_kmeans = st.slider('Number of Clusters (k) for K-Means:', min_value=2, max_value=10, value=3, key='kmeans_n_clusters')

            kmeans_labels, kmeans_centers = run_kmeans(current_data, n_clusters=n_clusters_kmeans)

            if n_clusters_kmeans > 1:
                silhouette_avg_kmeans = silhouette_score(current_data, kmeans_labels)
                st.metric(label=f"Silhouette Score for {n_clusters_kmeans} clusters:", value=f"{silhouette_avg_kmeans:.3f}")
            else:
                st.info("Silhouette Score cannot be calculated for less than 2 clusters.")

            kmeans_df_interactive = pd.DataFrame(current_data[:, :2], columns=['Feature 1', 'Feature 2']) # Limit to 2 features for 2D plot
            kmeans_df_interactive['Cluster'] = kmeans_labels.astype(str)

            fig_kmeans = px.scatter(kmeans_df_interactive, x='Feature 1', y='Feature 2', color='Cluster',
                                    title=f'K-Means Clustering with k={n_clusters_kmeans} of {data_name}',
                                    hover_data=['Cluster'])

            centroids_df_interactive = pd.DataFrame(kmeans_centers[:, :2], columns=['Feature 1', 'Feature 2'])
            centroids_df_interactive['Cluster'] = [f'Centroid {i}' for i in range(len(kmeans_centers))]

            fig_kmeans.add_scatter(x=centroids_df_interactive['Feature 1'], y=centroids_df_interactive['Feature 2'],
                                mode='markers', marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
                                name='Centroids', showlegend=True)

            fig_kmeans.update_layout(legend_title_text='Cluster / Centroid')
            st.plotly_chart(fig_kmeans, use_container_width=True)

            st.markdown("""
            The scatter plot visually represents the distinct clusters identified by the k-Means algorithm. Each color corresponds to a different cluster, and the black 'x' markers indicate the calculated centroids. In a financial context, these clusters could represent different types of stocks (e.g., growth stocks, value stocks, defensive stocks), aiding in diversified portfolio construction by selecting assets from different segments.
            By adjusting the slider, you can observe how the cluster boundaries and assignments change. The Silhouette Score provides a quantitative measure of clustering quality, helping to identify a suitable number of clusters for the given dataset.
            """)
    else:
        st.info("Please select a data source from the sidebar to run K-Means clustering.")
