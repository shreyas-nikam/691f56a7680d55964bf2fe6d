import pytest
import numpy as np
from definition_a9877d1ded894983a3b40078e3d643b4 import run_hierarchical_clustering

# Define test data for various scenarios
# Test Case 1: Standard functionality with two distinct blobs
data_standard = np.array([[1.0, 1.0], [1.5, 1.5], [2.0, 1.0], [2.0, 2.0],
                         [10.0, 10.0], [10.5, 10.5], [11.0, 10.0], [11.0, 11.0]], dtype=float)

# Test Case 2: Single data point
data_single_point = np.array([[1.0, 1.0]], dtype=float)

# Test Case 3: n_clusters > n_samples (e.g., 3 samples, request 4 clusters)
data_too_few_samples = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)

# Test Case 4: Data for invalid linkage_method (using standard data)
data_invalid_linkage = data_standard

# Test Case 5: Data for 'ward' linkage with non-euclidean affinity (using standard data)
data_ward_non_euclidean_affinity = data_standard

@pytest.mark.parametrize(
    "data, n_clusters, linkage_method, affinity_metric, expected",
    [
        # Test Case 1: Standard functionality with two clear clusters
        # Expect two distinct groups of labels. The exact label values (0 or 1) can vary,
        # so we check the grouping structure: first 4 points same label, next 4 points same label,
        # and these two labels are different.
        (data_standard, 2, 'ward', 'euclidean',
         lambda labels: (np.all(labels[:4] == labels[0]) and
                         np.all(labels[4:] == labels[4]) and
                         labels[0] != labels[4])),

        # Test Case 2: Single data point, requesting 1 cluster
        # sklearn.cluster.AgglomerativeClustering with n_clusters=1 will return a single cluster label [0]
        (data_single_point, 1, 'ward', 'euclidean', np.array([0])),

        # Test Case 3: n_clusters is greater than the number of samples
        # sklearn.cluster.AgglomerativeClustering raises ValueError for this.
        (data_too_few_samples, 4, 'ward', 'euclidean', ValueError),

        # Test Case 4: Invalid linkage_method string
        # sklearn.cluster.AgglomerativeClustering raises ValueError for an unknown linkage method.
        (data_invalid_linkage, 2, 'invalid_linkage_method', 'euclidean', ValueError),

        # Test Case 5: 'ward' linkage with a non-euclidean affinity metric
        # sklearn.cluster.AgglomerativeClustering explicitly states 'ward' only supports 'euclidean' affinity.
        (data_ward_non_euclidean_affinity, 2, 'ward', 'manhattan', ValueError),
    ]
)
def test_run_hierarchical_clustering(data, n_clusters, linkage_method, affinity_metric, expected):
    try:
        result_labels = run_hierarchical_clustering(data, n_clusters, linkage_method, affinity_metric)

        # If the expected value is a callable (lambda for structural check)
        if callable(expected):
            assert expected(result_labels)
        # If the expected value is an array (for exact label match)
        elif isinstance(expected, np.ndarray):
            assert np.array_equal(result_labels, expected)
        
        # Additional checks for successful clustering results
        assert isinstance(result_labels, np.ndarray)
        assert result_labels.dtype == np.int_
        assert result_labels.shape == (data.shape[0],)
        if n_clusters > 0: # Labels should be within [0, n_clusters-1] if clustering is successful
            assert np.all(result_labels >= 0) and np.all(result_labels < n_clusters)

    except Exception as e:
        # If an exception is expected, check its type
        assert isinstance(e, expected)
