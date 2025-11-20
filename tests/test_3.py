import pytest
from definition_8a50852ff379498780fc8227cee1f421 import interactive_kmeans

@pytest.mark.parametrize("n_clusters_input, expected_outcome", [
    # Test 1: Happy Path - Valid n_clusters within the interactive range (2-10)
    (3, None), 
    # Test 2: Edge Case - Minimum valid n_clusters as per the interactive slider (min=2)
    (2, None), 
    # Test 3: Invalid Type - n_clusters is not an integer
    ("invalid_type", TypeError), 
    # Test 4: Invalid Value - n_clusters is less than the minimum allowed (2)
    # A K-Means interactive tool typically requires at least 2 clusters for meaningful analysis and Silhouette Score calculation.
    (1, ValueError), 
    # Test 5: Invalid Value - n_clusters is greater than the maximum allowed by the interactive slider (max=10)
    # A robust implementation would validate n_clusters against the slider's defined range.
    (11, ValueError), 
])
def test_interactive_kmeans(n_clusters_input, expected_outcome):
    if expected_outcome is None:
        # For valid inputs, the function is expected to execute without raising an exception.
        # It should internally run K-Means, calculate score, and display plots.
        # Since the function returns None, we assert that.
        assert interactive_kmeans(n_clusters_input) is None
    else:
        # For invalid inputs (type or value), we expect a specific exception to be raised.
        with pytest.raises(expected_outcome):
            interactive_kmeans(n_clusters_input)