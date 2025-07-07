import numpy as np
from numpy.linalg import svd, eig
from scipy.spatial.distance import cosine
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr


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