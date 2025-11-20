import pytest
from definition_7463defa992b4ae8a42556b6289cbb8f import interactive_hierarchical

@pytest.mark.parametrize("n_clusters, linkage_method, affinity_metric, expected_exception", [
    # Test Case 1: Valid inputs, 'ward' linkage with 'euclidean' affinity
    (3, 'ward', 'euclidean', None),
    # Test Case 2: Valid inputs, a different linkage and affinity combination
    (4, 'complete', 'manhattan', None),
    # Test Case 3: Invalid n_clusters (less than 2), expected ValueError
    (1, 'ward', 'euclidean', ValueError),
    # Test Case 4: Incompatible linkage method and affinity metric ('ward' only with 'euclidean'), expected ValueError
    (3, 'ward', 'manhattan', ValueError),
    # Test Case 5: Invalid type for linkage_method (int instead of str), expected TypeError
    (3, 123, 'euclidean', TypeError),
])
def test_interactive_hierarchical(n_clusters, linkage_method, affinity_metric, expected_exception):
    if expected_exception is None:
        # Expect no exception to be raised for valid inputs
        try:
            interactive_hierarchical(n_clusters, linkage_method, affinity_metric)
            assert True  # Test passes if no exception is raised
        except Exception as e:
            pytest.fail(f"Unexpected exception raised for valid input: {e}")
    else:
        # Expect a specific exception type to be raised
        with pytest.raises(expected_exception):
            interactive_hierarchical(n_clusters, linkage_method, affinity_metric)