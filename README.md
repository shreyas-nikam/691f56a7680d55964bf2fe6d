# QuLab: Interactive Unsupervised Learning Workbench

![Application Screenshot](https://www.quantuniversity.com/assets/img/logo5.jpg)
*(Placeholder: You might want to replace this with a screenshot of your running application for better visual appeal.)*

## 1. Project Title and Description

**QuLab: Interactive Unsupervised Learning Workbench** is a Streamlit-based educational application designed to provide an interactive environment for exploring and understanding various unsupervised clustering techniques. This workbench allows users to experiment with different algorithms, parameters, and datasets to gain practical insights into how clustering works and its applications, particularly within a financial context.

Unsupervised learning is a powerful branch of machine learning that focuses on discovering hidden patterns and structures in data without relying on predefined labels. This is especially valuable in domains like finance, where labeled data can be scarce, expensive, or rapidly become obsolete. This application focuses on three prominent clustering algorithms: **k-Means Clustering**, **Spectral Clustering**, and **Agglomerative Hierarchical Clustering**.

**Learning Outcomes:**
*   Understand the operational principles of k-Means, Spectral, and Agglomerative Hierarchical clustering algorithms.
*   Analyze the impact of different parameters (e.g., number of clusters `k`, linkage methods, affinity kernels) on clustering results.
*   Visualize cluster assignments in 2D space (using PCA for dimensionality reduction) and interpret dendrograms for hierarchical structures.
*   Gain insights into how clustering can be applied for tasks like market segmentation, portfolio construction, or identifying correlated assets.

The application emphasizes the art and science of unsupervised learning, highlighting how mathematical principles like maximizing likelihood or minimizing error are used to capture the essence of data in the absence of ground truths.

## 2. Features

This application offers a rich set of features for interactive exploration of unsupervised learning:

*   **Interactive Streamlit UI**: A clean, intuitive, and interactive user interface powered by Streamlit.
*   **Sidebar Navigation**: Easy navigation between different sections: Home, Select & Preprocess Data, K-Means Clustering, Spectral Clustering, and Agglomerative Hierarchical Clustering.
*   **Flexible Data Loading**:
    *   **Pre-loaded Datasets**: Choose from synthetic financial datasets like "K-Means Portfolio Data" and "Spectral Assets Data" to jumpstart your analysis.
    *   **Custom CSV Upload**: Upload your own numerical datasets in CSV format for analysis.
*   **Automatic Data Preprocessing**: All loaded data is automatically scaled (standardized) to ensure fair distance calculations across features.
*   **K-Means Clustering Module**:
    *   Adjust the number of clusters (`k`).
    *   Visualize clusters and centroids in a 2D scatter plot (using PCA for higher-dimensional data).
    *   Evaluate clustering quality using the Silhouette Score.
*   **Spectral Clustering Module**:
    *   Adjust the number of clusters (`k`).
    *   Select different affinity kernels (`rbf`, `nearest_neighbors`) to explore graph-based similarities.
    *   Visualize clusters in a 2D scatter plot (using PCA).
    *   Evaluate clustering quality using the Silhouette Score.
*   **Agglomerative Hierarchical Clustering Module**:
    *   Adjust the desired number of clusters (`k`).
    *   Choose from various linkage methods (`ward`, `complete`, `average`, `single`) to define cluster proximity.
    *   Select different distance metrics (`euclidean`, `l1`, `l2`, `manhattan`, `cosine`).
    *   Visualize clusters in a 2D scatter plot (using PCA).
    *   Display an interactive **Dendrogram** to visualize the hierarchical merging process and help determine the optimal `k`.
    *   Evaluate clustering quality using the Silhouette Score.
*   **Session State Management**: Ensures that data and clustering results persist across page navigations, providing a seamless user experience.

## 3. Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quolab-unsupervised-learning.git
    cd quolab-unsupervised-learning
    ```
    *(Note: Replace `https://github.com/your-username/quolab-unsupervised-learning.git` with the actual repository URL if different.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit>=1.0.0
    pandas
    numpy
    plotly
    matplotlib
    scikit-learn
    scipy
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure `source.py` exists:**
    Make sure you have a `source.py` file in the same directory as `app.py`. This file should contain the helper functions: `load_synthetic_financial_data`, `preprocess_data`, `run_kmeans`, `run_spectral_clustering`, and `run_hierarchical_clustering`. A minimal example for `source.py` would look like this:

    ```python
    # source.py
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
    import scipy.cluster.hierarchy as sch

    def load_synthetic_financial_data(dataset_type):
        if dataset_type == 'kmeans_portfolio':
            # Example: stocks with features like growth, volatility, yield
            data = pd.DataFrame(np.random.rand(100, 3) * 100, columns=['GrowthRate', 'Volatility', 'DividendYield'])
            data.iloc[0:30, 0] += 50 # Make a cluster
            data.iloc[30:60, 1] += 50 # Make another cluster
            data.iloc[60:100, 2] += 50 # Make a third cluster
            return data.values
        elif dataset_type == 'spectral_assets':
            # Example: assets with non-linear correlations
            # Generate data in two intertwined half-moons for spectral clustering
            from sklearn.datasets import make_moons
            X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
            # Add some additional noise dimensions to make it 3D
            X = np.hstack((X, np.random.rand(200, 1) * 2))
            return X
        return None

    def preprocess_data(data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def run_kmeans(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_
        return labels, centers

    def run_spectral_clustering(data, n_clusters, affinity_kernel):
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity_kernel, random_state=42, n_init=10)
        labels = spectral.fit_predict(data)
        return labels

    def run_hierarchical_clustering(data, n_clusters, linkage_method, affinity_metric):
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric=affinity_metric)
        labels = agg.fit_predict(data)
        return labels
    ```

## 4. Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if you created one):
    ```bash
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  **Access the application:**
    Your web browser will automatically open a new tab directed to `http://localhost:8501` (or a similar address) where the application is running.

### Basic Workflow:

1.  **Home Page**: Provides an introduction to unsupervised learning and the clustering algorithms covered.
2.  **Select & Preprocess Data**:
    *   Choose one of the pre-loaded synthetic financial datasets (e.g., "K-Means Portfolio Data", "Spectral Assets Data").
    *   Alternatively, upload your own CSV file. Ensure it contains only numerical features.
    *   The data will be automatically scaled and an overview will be displayed.
3.  **Clustering Pages (K-Means, Spectral, Agglomerative)**:
    *   Navigate to your desired clustering algorithm page using the sidebar.
    *   Adjust parameters such as the number of clusters (`k`), linkage method, or affinity kernel using the sliders and select boxes.
    *   Click the "Run Clustering" button (e.g., "Run K-Means Clustering") to execute the algorithm.
    *   Observe the generated scatter plots, dendrograms (for Agglomerative), and Silhouette Scores.
    *   Experiment with different parameters to see their impact on cluster formation and quality.

## 5. Project Structure

The project is organized as follows:

```
quolab-unsupervised-learning/
├── app.py                     # Main Streamlit application script
├── source.py                  # Contains helper functions for data loading, preprocessing, and clustering algorithms
├── requirements.txt           # List of Python dependencies
├── README.md                  # This file
└── .git/                      # Git version control directory
```

## 6. Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For creating the interactive web application interface.
*   **Pandas**: For data manipulation and handling, especially for CSV input and DataFrame operations.
*   **NumPy**: For numerical operations and array manipulations.
*   **Plotly Express / Plotly Graph Objects**: For creating interactive and visually appealing scatter plots.
*   **Matplotlib**: Primarily used for generating and displaying the dendrogram in Hierarchical Clustering.
*   **Scikit-learn (sklearn)**: Provides implementations for:
    *   `StandardScaler` for data preprocessing.
    *   `PCA` for dimensionality reduction (2D visualization).
    *   `KMeans` for k-Means clustering.
    *   `SpectralClustering` for spectral clustering.
    *   `AgglomerativeClustering` for agglomerative hierarchical clustering.
    *   `silhouette_score` for evaluating clustering performance.
*   **SciPy**: Specifically `scipy.cluster.hierarchy` for generating the linkage matrix and plotting dendrograms.

## 7. Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## 8. License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) [Year] [Your Name/QuantUniversity]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
*(Note: Replace `[Year]` and `[Your Name/QuantUniversity]` with the appropriate information.)*

## 9. Contact

For questions or inquiries, please visit the [QuantUniversity website](https://www.quantuniversity.com/) or reach out via [info@quantuniversity.com](mailto:info@quantuniversity.com).


## License

## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
