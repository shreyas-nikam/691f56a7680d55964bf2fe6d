import pytest
import numpy as np
from sklearn.datasets import make_blobs

# Keep the placeholder definition_8780210df8f3426e96a9f0b601429124 as it is. DO NOT REPLACE or REMOVE the block.
from definition_8780210df8f3426e96a9f0b601429124 import run_spectral_clustering


# Helper function to generate data for tests
def generate_test_data(n_samples=100, n_features=2, centers=3, random_state=42):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=0.8, random_state=random_state)
    return X


@pytest.mark.parametrize("input_args, expected", [
    # Test Case 1: Basic functionality with 'rbf' kernel
    (
        (generate_test_data(n_samples=100, centers=3), 3, 'rbf', 1.0, 42),
        {"type": np.ndarray, "shape": (100,), "unique_labels_min": 3}
    ),

    # Test Case 2: Basic functionality with 'nearest_neighbors' kernel
    (
        (generate_test_data(n_samples=50, centers=2), 2, 'nearest_neighbors', 1.0, 42),
        {"type": np.ndarray, "shape": (50,), "unique_labels_min": 2}
    ),

    # Test Case 3: Edge case - n_clusters less than 2 (SpectralClustering expects n_clusters >= 2)
    (
        (generate_test_data(n_samples=30, centers=3), 1, 'rbf', 1.0, 42),
        ValueError
    ),
    
    # Test Case 4: Edge case - n_clusters greater than n_samples (SpectralClustering expects n_clusters <= n_samples)
    (
        (generate_test_data(n_samples=10, centers=3), 11, 'rbf', 1.0, 42),
        ValueError
    ),

    # Test Case 5: Edge case - Invalid data type (e.g., None instead of array-like)
    (
        (None, 3, 'rbf', 1.0, 42),
        TypeError
    ),
])
def test_run_spectral_clustering(input_args, expected):
    data, n_clusters, affinity_kernel, gamma, random_state = input_args

    try:
        result = run_spectral_clustering(data, n_clusters, affinity_kernel, gamma, random_state)
        
        # If an exception was expected, this point should not be reached
        if isinstance(expected, type) and issubclass(expected, Exception):
            pytest.fail(f"Expected {expected.__name__} but no exception was raised.")

        assert isinstance(result, expected["type"])
        assert result.shape == expected["shape"]
        # Cluster labels should be non-negative and less than n_clusters
        assert np.all(result >= 0)
        assert np.all(result < n_clusters)
        
        unique_labels = np.unique(result)
        # Check that a reasonable number of unique clusters were found (at least the minimum expected)
        assert len(unique_labels) >= expected["unique_labels_min"]
        assert len(unique_labels) <= n_clusters

    except Exception as e:
        # Check if the raised exception matches the expected exception type
        if isinstance(expected, type) and issubclass(expected, Exception):
            assert isinstance(e, expected)
        else:
            # If an unexpected exception was raised
            pytest.fail(f"Unexpected exception raised: {type(e).__name__}: {e}")