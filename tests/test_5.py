import pytest
from definition_07e5f7dc25e848ffa27ff730f5e9c777 import interactive_spectral

@pytest.mark.parametrize("n_clusters, affinity_kernel, expected_exception", [
    (3, 'rbf', None),  # Valid: typical use case
    (2, 'nearest_neighbors', None),  # Edge case: minimum n_clusters (as per slider spec min=2)
    (1, 'rbf', ValueError),  # Invalid: n_clusters too low (clustering requires k >= 2)
    ('3', 'rbf', TypeError),  # Invalid: n_clusters of wrong type
    (3, 'unsupported_kernel', ValueError),  # Invalid: affinity_kernel not supported
])
def test_interactive_spectral(n_clusters, affinity_kernel, expected_exception):
    if expected_exception is None:
        # For valid inputs, expect no exception and the function to complete (return None)
        result = interactive_spectral(n_clusters, affinity_kernel)
        assert result is None
    else:
        # For invalid inputs, expect a specific exception
        with pytest.raises(expected_exception):
            interactive_spectral(n_clusters, affinity_kernel)
