"""application_pages/page3.py"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def run_page3():
    current_data = st.session_state.current_data
    data_name = st.session_state.data_name

    st.markdown("## 11. Agglomerative Hierarchical Clustering: Theory")
    st.markdown("""
    Agglomerative Hierarchical Clustering builds a hierarchy of clusters from individual data points. It is a \"bottom-up\" approach, where each data point starts as its own cluster, and then pairs of clusters are iteratively merged based on their proximity until all points belong to a single cluster or a desired number of clusters is reached. This process forms a tree-like structure called a dendrogram.

    Key concepts include:
    -   **Distance Metric**: How the distance between individual data points is measured (e.g., Euclidean, Manhattan, Cosine). This defines the proximity between any two data points.
    -   **Linkage Method**: How the distance between two clusters is defined based on the distances between their constituent data points. Common methods include:
        -   **Single Linkage**: The shortest distance between any two points in the two clusters. This method is prone to chaining, where clusters are merged due to single close points.
            $$ d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y) $$
        -   **Complete Linkage**: The maximum distance between any two points in the two clusters. This tends to produce compact, spherical clusters.
            $$ d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y) $$
        -   **Average Linkage**: The average distance between all pairs of points across the two clusters. This offers a compromise between single and complete linkage.
            $$ d(C_i, C_j) = \text{mean}_{x \in C_i, y \in C_j} d(x, y) $$
        -   **Ward Linkage**: Minimizes the variance of the clusters being merged. It calculates the increase in the total within-cluster variance after merging, generally favoring compact, spherical clusters of similar size.

    Understanding these methods is crucial for financial applications, as the choice impacts how asset groups are formed (e.g., tightly correlated groups vs. loosely related ones).
    """)

    def run_hierarchical_clustering(data, n_clusters, linkage_method, affinity_metric):
        """Docstring: Runs Agglomerative Hierarchical Clustering."""
        agg_clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=affinity_metric
        )
        cluster_labels = agg_clustering.fit_predict(data)
        return cluster_labels

    def plot_dendrogram(data, linkage_matrix, n_clusters_display=None, title="Hierarchical Clustering Dendrogram"):
        """Docstring: Generates and displays an interactive dendrogram using Matplotlib."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input 'data' must be a numpy array.")
        
        N = data.shape[0]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(title)

        color_threshold_param = None
        cut_off_height = None
        add_axhline = False

        if n_clusters_display is not None:
            if not isinstance(n_clusters_display, int):
                raise TypeError("n_clusters_display must be an integer or None.")

            if n_clusters_display == 0:
                color_threshold_param = 0
                add_axhline = False
            elif n_clusters_display > 0:
                if linkage_matrix.shape[0] > 0:
                    if n_clusters_display <= N:
                        index_for_threshold = -(n_clusters_display - 1)
                        if index_for_threshold < linkage_matrix.shape[0]: # Ensure index is valid
                            color_threshold_param = linkage_matrix[index_for_threshold, 2]
                            cut_off_height = color_threshold_param
                            add_axhline = (n_clusters_display > 1)
                        else: # Not enough merges to form that many clusters
                            color_threshold_param = None
                            cut_off_height = None
                            add_axhline = False
                    else: # n_clusters > N, not possible
                        color_threshold_param = None
                        cut_off_height = None
                        add_axhline = False
                else: # linkage_matrix is empty, typically for N=1
                    if n_clusters_display == 1 and N == 1:
                        color_threshold_param = 0
                        cut_off_height = 0
                        add_axhline = False
                    else:
                        color_threshold_param = None
                        cut_off_height = None
                        add_axhline = False

        dendrogram(
            linkage_matrix, 
            color_threshold=color_threshold_param, 
            above_threshold_color='k',
            ax=ax
        )

        if add_axhline and cut_off_height is not None:
            ax.axhline(y=cut_off_height, color='r', linestyle='--', label=f'{n_clusters_display} Clusters Cut-off')
            ax.legend()

        ax.set_xlabel("Sample Index or (Cluster Size)")
        ax.set_ylabel("Distance")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)


    st.markdown("""
    The `run_hierarchical_clustering` function allows us to experiment with different linkage methods and distance metrics, providing flexibility in how clusters are formed. The `plot_dendrogram` function provides a visual representation of the hierarchical structure, which is crucial for understanding the merging process and for selecting an appropriate number of clusters by observing the 'cuts' in the tree.
    """)

    st.markdown("## 13. Agglomerative Hierarchical Clustering: Interactive Parameter Tuning and Evaluation")
    st.markdown("""
    The power of hierarchical clustering lies in exploring different linkage methods and distance metrics. We can interactively adjust these parameters along with the desired number of clusters ($k$) to understand their impact on the clustering structure and the Silhouette Score. This interactivity allows for a nuanced understanding of how different assumptions about 'similarity' affect the resulting financial market segments.
    """)

    if current_data is not None:
        if current_data.shape[1] < 2:
            st.warning("Agglomerative Hierarchical clustering requires at least 2 features for 2D visualization. Please upload data with more features or select a demo dataset.")
        else:
            col1, col2, col3 = st.columns(3)
            n_clusters_agg = col1.slider('Number of Clusters (k) for Hierarchical:', min_value=2, max_value=10, value=3, key='agg_n_clusters')
            linkage_method_agg = col2.selectbox('Linkage Method:', options=['ward', 'complete', 'average', 'single'], value='ward', key='agg_linkage_method')
            affinity_metric_agg = col3.selectbox('Distance Metric:', options=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], value='euclidean', key='agg_affinity_metric')

            # Specific validation for 'ward' linkage
            if linkage_method_agg == 'ward' and affinity_metric_agg != 'euclidean':
                st.warning("When linkage_method is 'ward', affinity_metric must be 'euclidean'. Changing metric to 'euclidean'.")
                affinity_metric_agg = 'euclidean' # Force correction for ward

            try:
                agg_labels = run_hierarchical_clustering(current_data, n_clusters=n_clusters_agg, linkage_method=linkage_method_agg, affinity_metric=affinity_metric_agg)

                unique_labels_agg = np.unique(agg_labels)
                if len(unique_labels_agg) < 2 or len(unique_labels_agg) > len(current_data) -1:
                    silhouette_avg_agg = -1.0
                    st.info(f"Silhouette Score cannot be computed as {len(unique_labels_agg)} unique clusters were formed (requires >1 and < n_samples).")
                else:
                    silhouette_avg_agg = silhouette_score(current_data, agg_labels)
                    st.metric(label=f"Silhouette Score for {n_clusters_agg} clusters:", value=f"{silhouette_avg_agg:.4f}")

                agg_df_interactive = pd.DataFrame(current_data[:, :2], columns=['Feature 1', 'Feature 2'])
                agg_df_interactive['Cluster'] = agg_labels.astype(str)

                title_score_agg = f'Silhouette Score: {silhouette_avg_agg:.4f}' if silhouette_avg_agg != -1.0 else 'Silhouette Score: N/A'
                fig_agg = px.scatter(agg_df_interactive, x='Feature 1', y='Feature 2', color='Cluster',
                                    title=f'Agglomerative Hierarchical Clustering (k={n_clusters_agg}, Linkage: {linkage_method_agg}, Metric: {affinity_metric_agg}) of {data_name}<br>{title_score_agg}',
                                    hover_data=['Cluster'])
                fig_agg.update_layout(legend_title_text='Cluster')
                st.plotly_chart(fig_agg, use_container_width=True)

                st.markdown("""
                ---
                ### Dendrogram Visualization
                """)
                # Compute linkage matrix for dendrogram
                Z = linkage(current_data, method=linkage_method_agg, metric=affinity_metric_agg)
                plot_dendrogram(current_data, Z, n_clusters_display=n_clusters_agg, 
                                title=f"Hierarchical Clustering Dendrogram with {linkage_method_agg.title()} Linkage")

                st.markdown("""
                Experiment with different linkage methods and distance metrics. You will observe how these choices significantly influence how clusters are formed and how sensitive the algorithm is to different data distributions. For instance, 'single' linkage can find elongated clusters, while 'ward' tends to find more compact, spherical clusters. The Silhouette Score helps quantify the effectiveness of these parameter choices. In finance, this allows tailoring the clustering approach to specific asset characteristics or market behaviors.
                """)
            except Exception as e:
                st.error(f"Agglomerative Hierarchical Clustering failed with given parameters: {e}")
    else:
        st.info("Please select a data source from the sidebar to run Agglomerative Hierarchical Clustering.")
