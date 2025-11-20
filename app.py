"""Python file for the main Streamlit application (app.py)."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Scikit-learn imports
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import silhouette_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- Utility functions for data generation and preprocessing ---
def load_synthetic_financial_data(dataset_type):
    """
    Generates and returns a synthetic financial dataset based on the specified type.
    
    Arguments:
        dataset_type (str): A string indicating the type of synthetic dataset to generate
                            (e.g., 'kmeans_portfolio', 'spectral_assets').
    
    Output:
        X (np.ndarray): The generated synthetic dataset.
    """
    if not isinstance(dataset_type, str):
        raise TypeError("dataset_type must be a string.")

    if dataset_type == 'kmeans_portfolio':
        X, _ = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=42)
        return X
    elif dataset_type == 'spectral_assets':
        X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
        return X
    elif dataset_type == "":
        raise ValueError("dataset_type cannot be an empty string.")
    else:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'.")

def preprocess_data(data: (np.ndarray | pd.DataFrame)) -> np.ndarray:
    """
    Applies StandardScaler from sklearn.preprocessing to the input data, ensuring all features are
    scaled to a standard range. This preprocessing step is crucial for distance-based clustering
    algorithms to prevent features with larger ranges from dominating the distance calculations.

    Arguments:
        data (np.ndarray or pd.DataFrame): The input data to be preprocessed.

    Output:
        scaled_data (np.ndarray): The scaled data with standardized features.
    """
    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise TypeError("Input data must be a numpy.ndarray or a pandas.DataFrame.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

# Initialize session state for data
if 'scaled_kmeans_data' not in st.session_state:
    st.session_state.kmeans_data_raw = load_synthetic_financial_data('kmeans_portfolio')
    st.session_state.scaled_kmeans_data = preprocess_data(st.session_state.kmeans_data_raw)

if 'scaled_spectral_data' not in st.session_state:
    st.session_state.spectral_data_raw = load_synthetic_financial_data('spectral_assets')
    st.session_state.scaled_spectral_data = preprocess_data(st.session_state.spectral_data_raw)

if 'user_data_df' not in st.session_state:
    st.session_state.user_data_df = None
    st.session_state.scaled_user_data = None

# Initialize current_data and data_name in session state for cross-page access
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_name' not in st.session_state:
    st.session_state.data_name = ""

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we delve into the fascinating world of **Unsupervised Learning**, specifically focusing on clustering techniques. Unsupervised learning is a paradigm in machine learning where algorithms learn patterns from unlabeled data, meaning there are no pre-defined output variables. This is particularly useful in finance for tasks such as identifying distinct market segments, grouping similar assets for portfolio diversification, or detecting anomalous trading behavior.

We will explore three fundamental clustering algorithms:
1.  **k-Means Clustering**: A partition-based algorithm that aims to partition $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (centroid).
2.  **Spectral Clustering**: A technique that uses the eigenvalues (spectrum) of a similarity matrix to perform dimensionality reduction before clustering in a lower-dimensional space, often effective for non-globular clusters.
3.  **Agglomerative Hierarchical Clustering**: A "bottom-up" approach that builds a hierarchy of clusters by progressively merging similar clusters, visualized through a dendrogram.

Through interactive demonstrations and parameter tuning, you will gain hands-on experience in applying these algorithms, understanding their underlying principles, and interpreting their results in a financial context. We will also examine the impact of various parameters, such as the number of clusters ($k$), linkage methods, and affinity kernels, on the final clustering outcomes.
""")

# Data Selection and Upload (Sidebar Component) - Replicated and adapted for app.py main logic
st.sidebar.header("Data Source Selection")
data_source_option = st.sidebar.radio(
    "Choose Data Source:",
    ("K-Means Portfolio Data (Demo)", "Spectral Assets Data (Demo)", "Upload Your Own Data")
)

if data_source_option == "K-Means Portfolio Data (Demo)":
    st.session_state.current_data = st.session_state.scaled_kmeans_data
    st.session_state.data_name = "K-Means Portfolio Data"
    st.sidebar.info(f"Loaded {st.session_state.data_name} with shape: {st.session_state.current_data.shape}")
elif data_source_option == "Spectral Assets Data (Demo)":
    st.session_state.current_data = st.session_state.scaled_spectral_data
    st.session_state.data_name = "Spectral Assets Data"
    st.sidebar.info(f"Loaded {st.session_state.data_name} with shape: {st.session_state.current_data.shape}")
elif data_source_option == "Upload Your Own Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state.user_data_df = pd.read_csv(uploaded_file)
            st.session_state.scaled_user_data = preprocess_data(st.session_state.user_data_df.values)
            st.session_state.current_data = st.session_state.scaled_user_data
            st.session_state.data_name = "Uploaded User Data"
            st.sidebar.success("CSV file successfully uploaded and preprocessed!")
            st.sidebar.info(f"Loaded {st.session_state.data_name} with shape: {st.session_state.current_data.shape}")
        except Exception as e:
            st.sidebar.error(f"Error processing uploaded file: {e}")
            st.session_state.current_data = None
    else:
        st.session_state.user_data_df = None
        st.session_state.scaled_user_data = None
        st.session_state.current_data = None # Ensure no data is selected if no file is uploaded
        st.sidebar.warning("Please upload a CSV file for analysis.")

# Fallback if current_data is None (e.g., initial state of user upload)
if st.session_state.current_data is None:
    st.warning("Please select a data source or upload a valid CSV to proceed with clustering.")

# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Page 1", "Page 2", "Page 3"])
if page == "Page 1":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Page 2":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Page 3":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
