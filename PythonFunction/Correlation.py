import numpy as np
from numpy.linalg import svd, eig
from scipy.spatial.distance import cosine
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from scipy.signal import correlate
import matplotlib.pyplot as plt

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
        results_CCA[name] = dict(rho=rho, angle=ang_xy, angle_3d=ang_3d,
                                 wA=wA, wB=wB)

    for k, d in results_CCA.items():
        print(f"{k}: ρ₁ = {d['rho']:.3f}, angleXY = {d['angle']:.1f}°, angle3D = {d['angle_3d']:.1f}°")
        row[f"{k}_rho"] = round(float(d['rho']), ndigits) if np.isfinite(d['rho']) else np.nan
        row[f"{k}_angle_xy"] = round(float(d['angle']), ndigits) if np.isfinite(d['angle']) else np.nan
        row[f"{k}_rho_3d"] = round(float(d['angle_3d']), ndigits) if np.isfinite(d['angle_3d']) else np.nan


def knee_hip_correlation_concatenate_segment(knee_angle, hip_angle, kick_intervals, fs):
    """
    Compute the overall Pearson correlation and optimal lag between knee and hip angles
    by concatenating all kick intervals.

    A negative lag  → the hip leads the knee.
    A positive lag  → the knee leads the hip.
    """

    # 1 ) concatenate all segments
    knee_concat = np.concatenate([knee_angle[s:e] for s, e in kick_intervals])
    hip_concat = np.concatenate([hip_angle[s:e] for s, e in kick_intervals])

    # 2 ) correlation and p-value
    r, p = pearsonr(knee_concat, hip_concat)

    # 3 ) cross-correlation to find optimal lag (optionally normalised)
    k_z = (knee_concat - knee_concat.mean()) / knee_concat.std(ddof=0)
    h_z = (hip_concat - hip_concat.mean()) / hip_concat.std(ddof=0)
    n = len(k_z)
    xcorr = correlate(k_z, h_z, mode="full")
    lags = np.arange(-n + 1, n)
    xcorr /= (n - np.abs(lags))  # normalisation by overlap
    lag_opt = int(lags[np.argmax(np.abs(xcorr))])
    lag_s = lag_opt / fs

    return r, p, lag_s