import pytest
import numpy as np

# definition_9a4826f60579440e8a975536c971273d block
from definition_9a4826f60579440e8a975536c971273d import load_synthetic_financial_data
# End definition_9a4826f60579440e8a975536c971273d block

@pytest.mark.parametrize("dataset_type, expected_output_type, expected_shape, expected_exception", [
    # Test Case 1: Valid dataset_type 'kmeans_portfolio'
    # Expects a numpy array of shape (300, 2) as make_blobs defaults to 2 features.
    ('kmeans_portfolio', np.ndarray, (300, 2), None),
    # Test Case 2: Valid dataset_type 'spectral_assets'
    # Expects a numpy array of shape (300, 2), similar to kmeans_portfolio per spec.
    ('spectral_assets', np.ndarray, (300, 2), None),
    # Test Case 3: Invalid dataset_type (unrecognized string)
    # Should raise a ValueError as the type is not defined.
    ('unknown_dataset', None, None, ValueError),
    # Test Case 4: Invalid dataset_type (non-string input)
    # Should raise a TypeError for incorrect argument type.
    (123, None, None, TypeError),
    # Test Case 5: Edge case - empty string for dataset_type
    # Should raise a ValueError as it's an unsupported type.
    ("", None, None, ValueError),
])
def test_load_synthetic_financial_data(dataset_type, expected_output_type, expected_shape, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            load_synthetic_financial_data(dataset_type)
    else:
        result = load_synthetic_financial_data(dataset_type)
        assert isinstance(result, expected_output_type)
        assert result.shape == expected_shape
        # Basic check to ensure the array is not just empty or filled with zeros (i.e., data was generated)
        assert not np.all(result == 0)