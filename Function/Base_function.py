import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks
matplotlib.use("TkAgg")


def find_index(column_name, column_list):
    if column_name in column_list:
        return column_list.index(column_name)
    else:
        return None


def resample_size(data, target_length):
    """
    Resamples each series of angles to a target length.

    Parameters:
    - data: List or array representing a bambi series.
    - target_length: Target length to resample each series to.

    Returns:
    - resampled_data: Resampled data to the target length.
    """
    # Convert 'data' to a flat numpy array
    data = np.array(data).flatten()

    # Create an interpolation function for the series
    interp_function = interp1d(
        np.linspace(0, 1, len(data)), data, kind="linear", fill_value="extrapolate"
    )

    # Resample the series to the target length
    resampled_data = interp_function(np.linspace(0, 1, target_length))

    return resampled_data

def get_threshold_intervals(signal, threshold, mode="above"):
    """
    Detect continuous intervals where signal is above or below a threshold.

    Parameters:
        signal : 1D array of values (e.g. distance, position, etc.)
        threshold : Threshold value to compare against.
        mode :
            - "above": detects where signal > threshold
            - "below": detects where signal < threshold

    Returns:
        intervals : list of (start_idx, end_idx)
            List of (start, end) frame indices where the condition is met.
            Intervals are half-open: [start, end)
    """
    if mode == "above":
        condition = signal > threshold
    elif mode == "below":
        condition = signal < threshold
    else:
        raise ValueError("mode must be 'above' or 'below'")

    # Detect transitions
    transitions = np.diff(condition.astype(int))
    start_idxs = np.where(transitions == 1)[0] + 1
    end_idxs = np.where(transitions == -1)[0] + 1

    # Edge cases
    if condition[0]:
        start_idxs = np.insert(start_idxs, 0, 0)
    if condition[-1]:
        end_idxs = np.append(end_idxs, len(signal) -1)

    intervals = list(zip(start_idxs, end_idxs))
    return intervals

def intersect_intervals(intervals1, intervals2):
    """
    Find overlapping intervals between two lists of (start, end) tuples.
    Each interval is assumed to be in the form [start, end), i.e., end is exclusive.

    Returns a list of intersecting intervals.
    """
    intersections = []

    for a_start, a_end in intervals1:
        for b_start, b_end in intervals2:
            start = max(a_start, b_start)
            end = min(a_end, b_end)
            if start < end:  # valid overlap
                intersections.append((start, end))

    return intersections


def analyze_intervals_duration(intervals, time_vector):
    """
    Analyze a list of (start, end) intervals using a time vector.

    Parameters:
        intervals : list of (start_idx, end_idx) tuples
        time_vector : np.ndarray – time values in seconds

    Returns:
        summary : dict with:
            - number_of_event
            - time_in_contact
            - durations_per_event (list of durations in seconds)
    """
    durations_per_event = []

    for start, end in intervals:
        if end < len(time_vector):  # safety check
            duration = time_vector[end] - time_vector[start]
            durations_per_event.append(duration)

    total_time = np.sum(durations_per_event)
    count = len(durations_per_event)

    return {
        'number_of_event': count,
        'time_in_contact': total_time,
        'durations_per_event': durations_per_event
    }