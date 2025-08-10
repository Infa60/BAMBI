import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import skew
from scipy.signal import hilbert, correlate, butter, filtfilt
import math

matplotlib.use("TkAgg")


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
            raise ValueError(
                "For 'between' mode, threshold must be a (low, high) tuple."
            )
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
        end_idxs = np.append(end_idxs, len(signal) - 1)

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
    reverse_from_threshold=None,
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
    amplitude_per_event = []  # will stay empty if distance is None

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
            seg = distance[start : end + 1]  # end inclusive
            amp = float(np.ptp(seg))  # max - min
            amplitude_per_event.append(amp)

    if not durations_per_event:  # aucune fenêtre valide
        durations_per_event = [float(0)]
    if use_distance and not amplitude_per_event:
        amplitude_per_event = [float(0)]

    summary = {
        "number_of_event": (
            len(durations_per_event) if durations_per_event[0] != 0 else 0
        ),
        "time_in_contact": float(np.sum(durations_per_event)),
        "durations_per_event": durations_per_event,
    }
    if use_distance:
        if reverse_from_threshold is not None:
            summary["amplitude_per_event"] = [
                reverse_from_threshold - val for val in amplitude_per_event
            ]
        else:
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
            raise ValueError(
                f"Dimension mismatch for '{name}': {data.shape[0]} vs {time_vector.shape[0]}"
            )

        # Use dashed line for threshold-like labels
        linestyle = (
            "--" if "threshold" in name.lower() or "thresh" in name.lower() else "-"
        )
        plt.plot(time_vector, data, label=name, linestyle=linestyle)

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def get_leg_and_tibia_length(file_path, bambiID):
    df = pd.read_csv(file_path, sep=",")
    df.columns = df.columns.str.strip()  # clean up any whitespace in column names

    leg_length = df[df["Inclusion number"] == bambiID]["TLL"].iloc[
        0
    ]  # 40% of body size
    tibia_length = df[df["Inclusion number"] == bambiID]["LLL"].iloc[
        0
    ]  # 40% of leg length

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
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)  # Zero-phase filtering
    return filtered_data


def add_summary_stats(
    dest: dict, prefix: str, values, nan_fill=np.nan, ndigits: int = 2
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
        stats = {
            k: nan_fill for k in ("min", "max", "p05", "p95", "mean", "sd", "skew")
        }
    else:
        stats = {
            "min": np.min(vals),
            "max": np.max(vals),
            "p05": np.percentile(vals, 5),
            "p95": np.percentile(vals, 95),
            "mean": np.mean(vals),
            "sd": np.std(vals, ddof=0),  # population SD
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
    ndigits: int = 2,
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
    if durs[0] != 0:
        dest[f"number_of_{prefix}"] = int(durs.size)
    else:
        dest[f"number_of_{prefix}"] = 0

    dest[f"total_time_in_{prefix}"] = (
        round(float(np.sum(durs)), ndigits) if durs.size else nan_fill
    )

    # Stats on durations
    add_summary_stats(
        dest=dest,
        prefix=f"time_{prefix}",
        values=durs,
        nan_fill=nan_fill,
        ndigits=ndigits,
    )

    # Stats on amplitude (optional)
    if amplitude_per_event is not None:
        amps = np.asarray(amplitude_per_event, dtype=float)
        amps = amps[np.isfinite(amps)]
        add_summary_stats(
            dest=dest,
            prefix=f"distance_{prefix}",
            values=amps,
            nan_fill=nan_fill,
            ndigits=ndigits,
        )


def compute_velocity(time, xyz):
    """
    Compute the instantaneous speed |v| for a 3-D trajectory.

    Parameters
    ----------
    time : (N,) array-like
        Time stamps in seconds.
    xyz : (N, 3) array-like
        Marker positions in meters.

    Returns
    -------
    speed : (N,) ndarray
        Speed (m/s) at each time step.
    """
    time = np.asarray(time)
    xyz = np.asarray(xyz)

    dt = np.gradient(time)  # Δt between samples
    velocity = np.gradient(xyz, axis=0) / dt[:, None]  # numerical derivative
    speed = np.linalg.norm(velocity, axis=1)  # magnitude of velocity

    return speed


def derivative(data, dt):
    """Return d(data)/dt."""
    return np.gradient(data, dt, axis=0)


def seconds_to_frames(intervals_s, freq, inclusive_right=False):
    """
    Convert a list of (t0, t1) time intervals in seconds to (f0, f1)
    frame intervals.

    Parameters
    ----------
    intervals_s : list[tuple[float, float]]
        Each tuple is (start_sec, end_sec).
    freq : float
        Sampling frequency in Hz (frames per second).
    inclusive_right : bool, optional
        If True, make the right boundary inclusive.

    Returns
    -------
    list[tuple[int, int]]
        Each tuple is (start_frame, end_frame).
    """
    frames = []
    for t0, t1 in intervals_s:
        f0 = math.floor(t0 * freq)  # inclusive left edge
        f1 = math.ceil(t1 * freq)  # exclusive right edge by default
        if inclusive_right:
            f1 += 1
        # On ignore les intervalles trop courts
        if (f1 - f0) >= 3:
            frames.append((f0, f1))
    return frames
