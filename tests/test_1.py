import pytest
import numpy as np
import pandas as pd

# Keep the definition_9dd832c6f3ac4a25b3d8d78fedc38758 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_9dd832c6f3ac4a25b3d8d78fedc38758 import preprocess_data

@pytest.mark.parametrize("input_data, expected_output, expected_exception", [
    # Test Case 1: Standard numpy array with multiple features
    # Input: A 2D numpy array with distinct numerical values.
    # Expected: The standardized version where each column's mean is approximately 0 and standard deviation is approximately 1.
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
     np.array([[-1.22474487, -1.22474487],
               [ 0.        ,  0.        ],
               [ 1.22474487,  1.22474487]]),
     None),

    # Test Case 2: Standard pandas DataFrame with multiple features
    # Input: A pandas DataFrame, which should be internally handled and scaled.
    # Expected: Same scaled numpy array as in Test Case 1, as the underlying data is identical.
    (pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=['Feature_A', 'Feature_B']),
     np.array([[-1.22474487, -1.22474487],
               [ 0.        ,  0.        ],
               [ 1.22474487,  1.22474487]]),
     None),

    # Test Case 3: Empty numpy array (0 samples, N features)
    # Edge Case: Input is an empty array but with a defined number of features (e.g., 2).
    # Expected: StandardScaler typically returns an empty array of the same shape without error.
    (np.empty((0, 2)),
     np.empty((0, 2)),
     None),

    # Test Case 4: Single feature numpy array
    # Edge Case: Input is a 2D numpy array with only one feature.
    # Expected: The standardized version of that single feature.
    (np.array([[10.0], [20.0], [30.0]]),
     np.array([[-1.22474487],
               [ 0.        ],
               [ 1.22474487]]),
     None),

    # Test Case 5: Invalid input type (e.g., a scalar integer)
    # Edge Case: Input is neither a numpy array nor a pandas DataFrame.
    # Expected: A TypeError, as the StandardScaler or an initial validation step should fail.
    (123, # A scalar integer
     None, # No expected output for an exception case
     TypeError),
])
def test_preprocess_data(input_data, expected_output, expected_exception):
    try:
        result = preprocess_data(input_data)
        # If we reached here, no exception was raised. Assert that no exception was expected.
        assert expected_exception is None, f"An exception {expected_exception.__name__} was expected but none was raised."

        # Verify the type of the output
        assert isinstance(result, np.ndarray), "Output must be a numpy.ndarray"

        # For non-empty inputs, compare numerical results precisely
        if input_data.shape[0] > 0:
            np.testing.assert_allclose(result, expected_output, rtol=1e-5, atol=1e-8,
                                       err_msg="Scaled data values do not match expected values.")
        else: # For empty data, just check shape
            assert result.shape == expected_output.shape, "Shape of empty scaled data does not match expected."

        # Additionally, verify statistical properties for meaningful scaling
        # This check is only relevant for data with more than one sample and at least one feature.
        if result.shape[0] > 1 and result.shape[1] > 0:
            assert np.allclose(np.mean(result, axis=0), 0.0, atol=1e-7), "Mean of scaled data not close to 0."
            assert np.allclose(np.std(result, axis=0), 1.0, atol=1e-7), "Standard deviation of scaled data not close to 1."

    except Exception as e:
        # If an exception was raised, assert that an exception was expected and verify its type.
        assert expected_exception is not None, f"Unexpected exception raised: {type(e).__name__} - {e}"
        assert isinstance(e, expected_exception), \
            f"Expected {expected_exception.__name__} but caught {type(e).__name__}."