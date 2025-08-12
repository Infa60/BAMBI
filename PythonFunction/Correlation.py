import numpy as np
from numpy.linalg import svd, eig
from scipy.spatial.distance import cosine
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from scipy.signal import correlate
import matplotlib.pyplot as plt
from PythonFunction.Quantity_movement import *


def angle_projected(w, v, plane="xy"):
    """
    2-D angle (deg) between two 3-D vectors projected onto a plane.
    plane ∈ {'xy', 'xz', 'yz'}.
    """
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    # use list(idx) both times so NumPy treats it as 1-D fancy indexing
    w2 = w[list(idx)]
    v2 = v[list(idx)]
    # normalise
    w2 /= np.linalg.norm(w2)
    v2 /= np.linalg.norm(v2)
    cosang = np.clip(np.dot(w2, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def cca_first_component_sklearn(X, Y, *, plane="xy", max_iter=500, tol=1e-06):
    """
    First-component Canonical Correlation Analysis using scikit-learn.

    Parameters
    ----------
    X : (N, p) array_like
        First variable set (e.g. limb #1 positions XYZ).
    Y : (N, q) array_like
        Second variable set (e.g. limb #2 positions XYZ).
    plane : {'xy','xz','yz'}, optional
        Plane onto which the 2-D angle is projected (default 'xy').
    max_iter, tol : CCA solver parameters.

    Returns
    -------
    rho      : float          – first canonical correlation ρ₁
    w1       : (p,) ndarray   – canonical vector for X (unit length)
    v1       : (q,) ndarray   – canonical vector for Y (unit length)
    angle2D  : float          – 2-D angle (deg) between w1 & v1 in *plane*
    angle3D  : float          – 3-D angle (deg) between w1 & v1
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # scikit-learn CCA centres (and optionally scales) internally.
    cca = CCA(n_components=1, max_iter=max_iter, tol=tol, scale=True)
    cca.fit(X, Y)

    # Canonical variates → correlation ρ₁
    U, V = cca.transform(X, Y)
    rho, _ = pearsonr(U[:, 0], V[:, 0])

    # Canonical loading vectors
    w1 = cca.x_weights_[:, 0]
    v1 = cca.y_weights_[:, 0]
    w1 /= np.linalg.norm(w1)
    v1 /= np.linalg.norm(v1)

    # 3-D angle
    cos3 = 1.0 - cosine(w1, v1)
    angle3D = float(np.degrees(np.arccos(np.clip(cos3, -1.0, 1.0))))

    # 2-D projected angle
    angle2D = angle_projected(w1, v1, plane)

    return float(rho), w1, v1, angle2D, angle3D


def add_canonical_correlations_stat(pairs, ndigits, row):
    results_CCA = {}
    for name, (A, B) in pairs.items():
        rho, wA, wB, ang_xy, ang_3d = cca_first_component_sklearn(A, B, plane="xy")
        results_CCA[name] = dict(rho=rho, angle=ang_xy, angle_3d=ang_3d, wA=wA, wB=wB)

    for k, d in results_CCA.items():
        # print(f"{k}: ρ₁ = {d['rho']:.3f}, angleXY = {d['angle']:.1f}°, angle3D = {d['angle_3d']:.1f}°")
        row[f"{k}_rho"] = (
            round(float(d["rho"]), ndigits) if np.isfinite(d["rho"]) else np.nan
        )
        row[f"{k}_angle_xy"] = (
            round(float(d["angle"]), ndigits) if np.isfinite(d["angle"]) else np.nan
        )
        row[f"{k}_angle_3d"] = (
            round(float(d["angle_3d"]), ndigits)
            if np.isfinite(d["angle_3d"])
            else np.nan
        )


def optimal_lag_crosscorr(sig1, sig2, fs):
    """
    Compute the optimal lag between two signals using normalized cross-correlation.

    Parameters
    ----------
    sig1 : array-like
        First 1D signal.
    sig2 : array-like
        Second 1D signal, same length as sig1.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    lag_samples : int
        Optimal lag in samples (positive means sig1 leads sig2).
    lag_seconds : float
        Optimal lag in seconds.
    max_corr : float
        Maximum (absolute) normalized cross-correlation value.
    """
    sig1 = np.asarray(sig1)
    sig2 = np.asarray(sig2)
    n = len(sig1)
    # Z-score normalization
    sig1_z = (sig1 - sig1.mean()) / sig1.std(ddof=0)
    sig2_z = (sig2 - sig2.mean()) / sig2.std(ddof=0)
    # Full cross-correlation
    xcorr = correlate(sig1_z, sig2_z, mode="full")
    lags = np.arange(-n + 1, n)
    xcorr /= n - np.abs(lags)  # normalization by overlap
    # Optimal lag
    idx_max = np.argmax(np.abs(xcorr))
    lag_samples = lags[idx_max]
    lag_seconds = lag_samples / fs
    max_corr = xcorr[idx_max]
    return lag_seconds, max_corr


def signal_correlation_concatenate_segments(signal1, signal2, intervals, fs):
    """
    Compute the overall Pearson correlation and optimal lag between two signals
    by concatenating all specified intervals.

    Parameters
    ----------
    signal1 : array-like
        First 1D signal (e.g., joint angle, velocity, etc.).
    signal2 : array-like
        Second 1D signal, same length as signal1.
    intervals : list of (start, end) tuples
        List of intervals (start_idx, end_idx) to concatenate.
        Intervals should be in sample indices (Python-style: [start, end)).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    r : float
        Pearson correlation coefficient for concatenated segments.
    p : float
        Two-tailed p-value.
    lag_s : float
        Optimal lag in seconds (positive means signal1 leads signal2).

    Notes
    -----
    If `intervals` is empty, returns np.nan for all outputs.

    """
    if not intervals:
        return np.nan, np.nan, np.nan

    # 1 ) Concatenate all segments
    sig1_concat = np.concatenate([signal1[s:e] for s, e in intervals])
    sig2_concat = np.concatenate([signal2[s:e] for s, e in intervals])

    # 2 ) Pearson correlation
    r, p = pearsonr(sig1_concat, sig2_concat)

    # 3 ) Cross-correlation to find optimal lag (normalized)
    lag_seconds, max_corr = optimal_lag_crosscorr(sig1_concat, sig2_concat, fs)

    return r, p, lag_seconds


def add_correlations_stat(
    pairs, fs, ndigits, row, usage, intervals=None, method=None,
):
    """
    Compute and store correlation statistics (Pearson r, p-value, and optimal lag)
    for the tangential velocities between all specified pairs of markers.
    Optionally restricts the analysis to given intervals.

    Parameters
    ----------
    pairs : dict
        Dictionary where keys are pair names (e.g., 'knee_hip') and values are tuples
        of arrays: (marker_A_positions, marker_B_positions), each shape (n_samples, 3).
    fs : float
        Sampling frequency in Hz.
    ndigits : int
        Number of decimal digits to round the results.
    row : dict
        Dictionary to which results will be added as new keys.
    usage : str
        Choice between "velocity", "acceleration" or "jerk"
    intervals : list of (start, end) tuples, optional
        List of intervals (in sample indices) where correlation is computed.
        If None, computes correlation over the whole recording.
    method : "Intersection" or "Union" or empty

    Returns
    -------
    None (results are added in-place to `row`)
    """
    results_correlations = {}
    for name, (A, B) in pairs.items():

        velA, accA, jerkA = marker_pos_to_jerk(A, cutoff=6, fs=fs)
        velB, accB, jerkB = marker_pos_to_jerk(B, cutoff=6, fs=fs)

        if usage == "velocity":
            A_data, B_data = velA, velB
        elif usage == "acceleration":
            A_data, B_data = accA, accB
        elif usage == "jerk":
            A_data, B_data = jerkA, jerkB


        if intervals is not None:
            r, p_value, lag_s = signal_correlation_concatenate_segments(
                A_data, B_data, intervals, fs
            )
        else:
            method = "all_duration"
            r, p_value = pearsonr(A_data, B_data)
            lag_s, max_corr = optimal_lag_crosscorr(A_data, B_data, fs)

        results_correlations[name] = dict(corr=r, p_value=p_value, lag_s=lag_s)

    for k, d in results_correlations.items():
        row[f"{k}_{usage}_corr_{method}"] = (
            round(float(d["corr"]), ndigits) if np.isfinite(d["corr"]) else np.nan
        )
        row[f"{k}_{usage}_p_value_{method}"] = (
            round(float(d["p_value"]), ndigits) if np.isfinite(d["p_value"]) else np.nan
        )
        row[f"{k}_{usage}_lag_{method}"] = (
            round(float(d["lag_s"]), ndigits) if np.isfinite(d["lag_s"]) else np.nan
        )
