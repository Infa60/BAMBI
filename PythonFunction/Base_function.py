import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import skew
from scipy.signal import hilbert, correlate, butter, filtfilt
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
            - "between": detects where low < signal < high

    Returns:
        intervals : list of (start_idx, end_idx)
            List of (start, end) frame indices where the condition is met.
            Intervals are half-open: [start, end)
    """
    if mode == "above":
        condition = signal > threshold
    elif mode == "below":
        condition = signal < threshold
    elif mode == "between":
        if not (isinstance(threshold, (tuple, list)) and len(threshold) == 2):
            raise ValueError("For 'between' mode, threshold must be a (low, high) tuple.")
        low, high = threshold
        condition = (signal > low) & (signal < high)
    else:
        raise ValueError("mode must be 'above', 'below', or 'between'")

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


def analyze_intervals_duration(
    intervals,
    time_vector: np.ndarray,
    distance: np.ndarray | None = None,
):
    """
    Analyse (start, end) intervals on a time vector.

    If *distance* is supplied, also compute the peak-to-peak amplitude of the
    distance signal within each interval.

    Parameters
    ----------
    intervals : list[tuple[int, int]]
        Index pairs delimiting each event (inclusive on both ends).

    time_vector : np.ndarray, shape (N,)
        Time stamps in seconds.

    distance : np.ndarray | None, shape (N,), optional
        Signal for which peak-to-peak amplitudes will be computed.
        If None (default), amplitude metrics are skipped.

    Returns
    -------
    dict
        Always contains:
            'number_of_event'
            'time_in_contact'
            'durations_per_event'
        Contains, when distance is provided:
            'amplitude_per_event'
    """
    durations_per_event = []
    amplitude_per_event = []          # will stay empty if distance is None

    n_samples = len(time_vector)
    use_distance = distance is not None

    for start, end in intervals:
        # --- basic sanity checks ---
        if start >= n_samples or end >= n_samples or start >= end:
            continue

        # duration in seconds
        duration = float(time_vector[end] - time_vector[start])
        durations_per_event.append(duration)

        # amplitude if requested
        if use_distance:
            seg = distance[start : end + 1]          # end inclusive
            amp = float(np.ptp(seg))                 # max - min
            amplitude_per_event.append(amp)

    summary = {
        "number_of_event"    : len(durations_per_event),
        "time_in_contact"    : float(np.sum(durations_per_event)),
        "durations_per_event": durations_per_event,
    }

    if use_distance:
        summary["amplitude_per_event"] = amplitude_per_event

    return summary

def plot_time_series(time_vector, title="Time Series Plot", ylabel="Value", **kwargs):
    """
    Plot multiple time series on a single graph.

    Parameters:
    - time_vector : array (n,) - time values
    - title : str - title of the plot
    - ylabel : str - label for the y-axis
    - kwargs : named series to plot.
        Each entry can be:
            - a vector of shape (n,) or (n, 1)
            - a scalar (expanded to a constant line)
        If the name contains 'threshold' or 'thresh', it will be plotted as a dashed line.
    """
    plt.figure(figsize=(10, 5))

    for name, data in kwargs.items():
        # Convert scalars to constant arrays
        if np.isscalar(data):
            data = np.full_like(time_vector, data, dtype=np.float64)

        # Remove singleton dimensions
        data = np.squeeze(data)

        if data.shape[0] != time_vector.shape[0]:
            raise ValueError(f"Dimension mismatch for '{name}': {data.shape[0]} vs {time_vector.shape[0]}")

        # Use dashed line for threshold-like labels
        linestyle = '--' if 'threshold' in name.lower() or 'thresh' in name.lower() else '-'
        plt.plot(time_vector, data, label=name, linestyle=linestyle)

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def get_leg_and_tibia_length(file_path, bambiID):
    # Load CSV with semicolon separator (common in French exports)
    df = pd.read_csv(file_path, sep=';')
    df.columns = df.columns.str.strip()  # clean up any whitespace in column names

    # Extract the numeric part from the Bambi ID
    participant_num = int(bambiID[-3:])

    # Select all rows for this participant (there may be two: left and right side)
    rows = df[df['participant_number'] == participant_num]
    if rows.empty:
        raise ValueError(f"No data found for participant {bambiID}")

    # Replace commas with dots for decimal conversion (e.g., "22,5" → "22.5")
    rows = rows.replace(',', '.', regex=True).infer_objects(copy=False)

    # Convert relevant columns to float (ignore conversion errors)
    for col in [
        'visit_total_left_leg_length', 'visit_total_right_leg_length',
        'visit_left_lower_leg_length', 'visit_right_lower_leg_length'
    ]:
        rows[col] = pd.to_numeric(rows[col], errors='coerce')

    # Combine all left/right leg length values and pick the first available
    leg_length = pd.concat([
        rows['visit_total_left_leg_length'],
        rows['visit_total_right_leg_length']
    ]).dropna().iloc[0]

    # Combine all left/right tibia length values and pick the first available
    tibia_length = pd.concat([
        rows['visit_left_lower_leg_length'],
        rows['visit_right_lower_leg_length']
    ]).dropna().iloc[0]
    leg_length = leg_length * 10
    tibia_length =  tibia_length * 10
    return leg_length, tibia_length


def butter_lowpass_filter(data, cutoff, fs, order=2):
    """
    Applies a low-pass Butterworth filter to a 1D or 2D signal.

    Parameters:
        data   : array-like, shape (n_samples,) or (n_samples, n_channels)
                 Input signal(s) to filter.
        cutoff : float
                 Cutoff frequency in Hz.
        fs     : float
                 Sampling frequency in Hz.
        order  : int
                 Order of the Butterworth filter (default: 2).

    Returns:
        filtered_data : np.ndarray, filtered signal(s), same shape as input.
    """
    nyq = 0.5 * fs                 # Nyquist frequency
    normal_cutoff = cutoff / nyq   # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)  # Zero-phase filtering
    return filtered_data

def add_summary_stats(
    dest: dict,
    prefix: str,
    values,
    nan_fill=np.nan,
    ndigits: int = 2
):
    """
    Append Min, Max, P5, P95, Mean, SD, Skewness to *dest* for the given *values*.

    Parameters
    ----------
    dest    : dict
        Dictionary that will be updated in place.

    prefix  : str
        Key prefix to use, e.g. "time_hand_hand_contact".
        Resulting keys will be:
            <prefix>_min, _max, _p05, _p95, _mean, _sd, _skew

    values  : array-like
        Iterable of numeric values. NaNs are ignored.

    nan_fill: float
        Value to write when *values* is empty or only NaNs (default = np.nan).
    """
    vals = np.asarray(values, dtype=float)

    # Remove NaNs for robust stats
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        stats = {k: nan_fill for k in
                 ("min", "max", "p05", "p95", "mean", "sd", "skew")}
    else:
        stats = {
            "min" : np.min(vals),
            "max" : np.max(vals),
            "p05" : np.percentile(vals, 5),
            "p95" : np.percentile(vals, 95),
            "mean": np.mean(vals),
            "sd"  : np.std(vals, ddof=0),          # population SD
            "skew": skew(vals, bias=False),
        }

    # Update destination dict with prefixed keys
    for k, v in stats.items():
        if np.isfinite(v):
            dest[f"{prefix}_{k}"] = round(float(v), ndigits)
        else:
            dest[f"{prefix}_{k}"] = nan_fill

def add_contact_metrics(
    dest: dict,
    prefix: str,
    durations_per_event,
    amplitude_per_event=None,
    nan_fill=np.nan,
    ndigits: int = 2
):
    """
    Update *dest* with all metrics for a contact type (left hand-mouth, right
    hand-hand, etc.).

    Parameters
    ----------
    dest : dict
        Row/dictionary updated in place.

    prefix : str
        Base prefix, e.g. "L_hand_mouth_contact".
        Keys created:
            number_of_<prefix>
            total_time_in_<prefix>
            time_<prefix>_<stat>      (stats on durations)
            distance_<prefix>_<stat>  (stats on amplitude, if provided)

    durations_per_event : array-like
        Durations (s) for each event.

    amplitude_per_event : array-like | None, optional
        Peak-to-peak amplitudes per event. If None, distance stats are skipped.

    nan_fill : float
        Value written when the supplied list is empty (default = np.nan).
    """
    # --- ensure numeric & drop NaNs ---
    durs = np.asarray(durations_per_event, dtype=float)
    durs = durs[np.isfinite(durs)]

    # Basic event metrics
    dest[f"number_of_{prefix}"]      = int(durs.size)
    dest[f"total_time_in_{prefix}"]  = round(float(np.sum(durs)),ndigits) if durs.size else nan_fill

    # Stats on durations
    add_summary_stats(
        dest   = dest,
        prefix = f"time_{prefix}",
        values = durs,
        nan_fill = nan_fill,
        ndigits = ndigits
    )

    # Stats on amplitude (optional)
    if amplitude_per_event is not None:
        amps = np.asarray(amplitude_per_event, dtype=float)
        amps = amps[np.isfinite(amps)]
        add_summary_stats(
            dest   = dest,
            prefix = f"distance_{prefix}",
            values = amps,
            nan_fill = nan_fill,
            ndigits = ndigits
        )


# === STEP 1: Compute angular threshold from cohort-wide shoulder widths ===

# Half of inter-shoulder distances (in meters) across baby
#inter_shoulder_half_lengths = np.array([...])

# 1. Input parameters
#desired_offset_m = 20

# 2. Compute the angle in radians using arctangent
#angle_threshold_rad = np.arctan(desired_offset_m / np.mean(inter_shoulder_half_lengths))

# 3. Convert to degrees for easier interpretation
#angle_threshold_deg = np.degrees(angle_threshold_rad)

# === STEP 2: For a specific baby, compute their personalized max offset ===

# Half shoulder width for the current subject
#half_shoulder_width_baby = 0.085  # replace with actual measurement

# Max lateral offset allowed before calling it "misalignment"
#max_offset = np.tan(angle_threshold_rad) * half_shoulder_width_baby