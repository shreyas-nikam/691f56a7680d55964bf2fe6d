import pytest
import numpy as np

# This block should remain as is:
from definition_56ebed368fee4f53b86c45180f21abef import run_kmeans

@pytest.mark.parametrize(
    "data, n_clusters, random_state, expected",
    [
        # Test Case 1: Basic functionality with two distinct clusters
        (
            np.array([[1,1],[1,2],[4,4],[4,5]]),
            2,
            42,
            (np.array([0,0,1,1]), np.array([[1. , 1.5], [4. , 4.5]]))
        ),
        # Test Case 2: Edge case - n_clusters=1 (all points in one cluster)
        (
            np.array([[1,1],[1,2],[4,4],[4,5]]),
            1,
            42,
            (np.array([0,0,0,0]), np.array([[2.5, 3.]]))
        ),
        # Test Case 3: Edge case - n_clusters > n_samples (should raise ValueError)
        (
            np.array([[1,1],[1,2]]), # 2 samples
            3, # Requesting 3 clusters
            42,
            ValueError # sklearn.cluster.KMeans raises ValueError if n_samples < n_clusters
        ),
        # Test Case 4: Edge case - Empty data array (should raise ValueError)
        (
            np.empty((0, 2)), # 0 samples, 2 features
            2,
            42,
            ValueError # sklearn.cluster.KMeans raises ValueError if n_samples=0
        ),
        # Test Case 5: Invalid n_clusters type (should raise TypeError)
        (
            np.array([[1,1],[1,2],[4,4],[4,5]]),
            "2", # Invalid type for n_clusters
            42,
            TypeError # sklearn.cluster.KMeans constructor/fit expects int for n_clusters
        ),
    ]
)
def test_run_kmeans(data, n_clusters, random_state, expected):
    try:
        labels, centers = run_kmeans(data, n_clusters, random_state)
        # If no exception, then expected should be a tuple (labels, centers)
        assert isinstance(expected, tuple)
        expected_labels, expected_centers = expected

        assert isinstance(labels, np.ndarray)
        assert isinstance(centers, np.ndarray)

        assert np.array_equal(labels, expected_labels)
        assert np.allclose(centers, expected_centers) # Use allclose for float comparisons
    except Exception as e:
        # If an exception occurred, then expected should be the exception type
        assert isinstance(e, expected)