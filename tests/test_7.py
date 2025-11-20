import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt

# Placeholder for the module import
from definition_f513525fe54e4ddbb4b1338105cf1c01 import plot_dendrogram

# Helper to generate a valid linkage matrix for testing
def _generate_test_linkage_matrix(num_points=5):
    if num_points < 2:
        return np.array([])
    # Create distinct clusters for a more meaningful linkage
    data = np.random.rand(num_points, 2)
    # Ensure some variance to create meaningful clusters
    data[0:num_points//2] += 5
    Z = linkage(data, method='ward')
    return Z

# Mock for scipy.cluster.hierarchy.dendrogram
def mock_dendrogram(*args, **kwargs):
    # Simulate a return value for `dendrogram`. It typically returns a dictionary
    # containing plot information. We provide a minimal structure for mocking.
    linkage_matrix = args[0]
    num_original_points = linkage_matrix.shape[0] + 1 if linkage_matrix.size > 0 else 0
    return {
        'color_list': ['b'] * max(1, num_original_points),
        'icoord': [[1,2,3,4]] * (linkage_matrix.shape[0] if linkage_matrix.size > 0 else 0),
        'dcoord': [[0,1,0,1]] * (linkage_matrix.shape[0] if linkage_matrix.size > 0 else 0),
        'leaves': list(range(num_original_points))
    }

# Test Cases

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.show')
@patch('scipy.cluster.hierarchy.dendrogram', side_effect=mock_dendrogram)
@patch('matplotlib.pyplot.axhline')
def test_plot_dendrogram_basic_functionality(mock_axhline, mock_sch_dendrogram, mock_plt_show, mock_plt_title, mock_plt_figure):
    """
    Test basic functionality: plotting a dendrogram without a specified cluster cut-off.
    Ensures core plotting functions are called and no axhline is drawn.
    """
    num_points = 5
    test_data = np.random.rand(num_points, 2)
    test_linkage_matrix = _generate_test_linkage_matrix(num_points)
    test_title = "Basic Dendrogram Test"

    plot_dendrogram(test_data, test_linkage_matrix, n_clusters_display=None, title=test_title)

    mock_plt_figure.assert_called_once()
    mock_plt_title.assert_called_once_with(test_title)
    # When n_clusters_display is None, dendrogram uses its default coloring (color_threshold=None)
    mock_sch_dendrogram.assert_called_once_with(test_linkage_matrix, color_threshold=None, above_threshold_color='k')
    mock_plt_show.assert_called_once()
    mock_axhline.assert_not_called()

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.show')
@patch('scipy.cluster.hierarchy.dendrogram', side_effect=mock_dendrogram)
@patch('matplotlib.pyplot.axhline')
def test_plot_dendrogram_with_cluster_cut_off(mock_axhline, mock_sch_dendrogram, mock_plt_show, mock_plt_title, mock_plt_figure):
    """
    Test functionality with a specified number of clusters to display, expecting a cut-off line.
    Ensures `dendrogram` is called with appropriate `color_threshold` and `axhline` is drawn.
    """
    num_points = 5
    test_data = np.random.rand(num_points, 2)
    test_linkage_matrix = _generate_test_linkage_matrix(num_points)
    n_clusters = 3
    test_title = f"Dendrogram with {n_clusters} Clusters"

    # Calculate the expected threshold distance from the linkage matrix.
    # This is the height of the (N-n_clusters+1)-th merge.
    expected_threshold_distance = test_linkage_matrix[-(n_clusters - 1), 2]

    plot_dendrogram(test_data, test_linkage_matrix, n_clusters_display=n_clusters, title=test_title)

    mock_plt_figure.assert_called_once()
    mock_plt_title.assert_called_once_with(test_title)
    # `dendrogram` should be called with the specific color_threshold
    mock_sch_dendrogram.assert_called_once_with(test_linkage_matrix, color_threshold=expected_threshold_distance, above_threshold_color='k')
    mock_plt_show.assert_called_once()
    # `axhline` should be called with the same threshold distance
    mock_axhline.assert_called_once_with(y=expected_threshold_distance, color='r', linestyle='--', label=f'{n_clusters} Clusters Cut-off')

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.show')
@patch('scipy.cluster.hierarchy.dendrogram') # Don't use side_effect=mock_dendrogram here to allow it to raise the actual error
@patch('matplotlib.pyplot.axhline')
def test_plot_dendrogram_empty_linkage_matrix_error(mock_axhline, mock_sch_dendrogram, mock_plt_show, mock_plt_title, mock_plt_figure):
    """
    Test behavior with an empty/invalid linkage matrix, expecting a ValueError from dendrogram.
    """
    num_points = 1 # Not enough points to form merges; linkage_matrix for 1 point is empty.
    test_data = np.random.rand(num_points, 2)
    test_linkage_matrix = np.array([]) # This will cause an error in dendrogram

    # Mock dendrogram to explicitly raise the expected error for an invalid input Z
    mock_sch_dendrogram.side_effect = ValueError("Input Z must be a 2-D array and have Z.shape[1] == 4.")

    test_title = "Empty Linkage Test"

    with pytest.raises(ValueError) as excinfo:
        plot_dendrogram(test_data, test_linkage_matrix, n_clusters_display=None, title=test_title)
    
    assert "Input Z must be a 2-D array and have Z.shape[1] == 4." in str(excinfo.value)
    # Figure and title might be created before `dendrogram` is called and raises an error
    mock_plt_figure.assert_called_once()
    mock_plt_title.assert_called_once_with(test_title)
    mock_sch_dendrogram.assert_called_once_with(test_linkage_matrix, color_threshold=None, above_threshold_color='k')
    mock_plt_show.assert_not_called() # No show if an error occurred
    mock_axhline.assert_not_called()

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.show')
@patch('scipy.cluster.hierarchy.dendrogram', side_effect=mock_dendrogram)
@patch('matplotlib.pyplot.axhline')
def test_plot_dendrogram_invalid_data_type_for_data(mock_axhline, mock_sch_dendrogram, mock_plt_show, mock_plt_title, mock_plt_figure):
    """
    Test behavior with an invalid data type for `data` (e.g., None), expecting a TypeError.
    The function expects `data` to be a `np.ndarray`.
    """
    test_data = None # Invalid data type for np.ndarray
    test_linkage_matrix = _generate_test_linkage_matrix(3)
    test_title = "Invalid Data Type Test"

    with pytest.raises(TypeError) as excinfo:
        plot_dendrogram(test_data, test_linkage_matrix, n_clusters_display=None, title=test_title)
    
    # Expected error if `len(data)` or `data.shape` is called on None
    assert isinstance(excinfo.value, TypeError) 
    # Mocks for plotting are not expected to be called if an error occurs early
    mock_plt_figure.assert_not_called()
    mock_plt_title.assert_not_called()
    mock_sch_dendrogram.assert_not_called()
    mock_plt_show.assert_not_called()
    mock_axhline.assert_not_called()

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.show')
@patch('scipy.cluster.hierarchy.dendrogram', side_effect=mock_dendrogram)
@patch('matplotlib.pyplot.axhline')
def test_plot_dendrogram_n_clusters_display_zero_or_one(mock_axhline, mock_sch_dendrogram, mock_plt_show, mock_plt_title, mock_plt_figure):
    """
    Test behavior when n_clusters_display is 0 or 1.
    For 0 clusters: `color_threshold=0` (all blue). No axhline for a cut-off.
    For 1 cluster: `color_threshold` effectively makes everything one color. No axhline for a cut-off.
    Let's test `n_clusters_display=0` as it's a clear edge case where no meaningful cut-off line exists.
    """
    num_points = 5
    test_data = np.random.rand(num_points, 2)
    test_linkage_matrix = _generate_test_linkage_matrix(num_points)
    test_title = "Dendrogram with Zero Clusters Cut"

    plot_dendrogram(test_data, test_linkage_matrix, n_clusters_display=0, title=test_title)

    mock_plt_figure.assert_called_once()
    mock_plt_title.assert_called_once_with(test_title)
    # When n_clusters_display is 0, it means no thresholding for coloring based on a specified number of clusters.
    # `scipy.cluster.hierarchy.dendrogram` treats `color_threshold <= 0` as no thresholding, making all lines blue by default.
    mock_sch_dendrogram.assert_called_once_with(test_linkage_matrix, color_threshold=0, above_threshold_color='k')
    mock_plt_show.assert_called_once()
    mock_axhline.assert_not_called() # No cut-off line for 0 clusters