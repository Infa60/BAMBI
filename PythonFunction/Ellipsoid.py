import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator

matplotlib.use("TkAgg")


def ellipsoid_volume_and_points(points, confidence_threshold):
    """
    Computes the 90% ellipsoid volume and the points enclosed within that ellipsoid
    for a given marker.

    Returns:
    - volume_90: volume of the 90% ellipsoid
    - enclosed_points: points inside the ellipsoid
    - inside_mask: boolean mask of which points are enclosed
    """

    # Center the data
    mean = np.mean(points, axis=0)
    centered = points - mean

    # PCA to get axes
    pca = PCA(n_components=3)
    pca.fit(centered)
    transformed = pca.transform(centered)

    # Mahalanobis distance squared
    md_squared = np.sum((transformed / np.sqrt(pca.explained_variance_)) ** 2, axis=1)

    # Chi-squared threshold for 90% confidence in 3D
    threshold = chi2.ppf(confidence_threshold, df=3)
    inside = md_squared <= threshold
    enclosed_points = points[inside]

    # Volume of the ellipsoid
    axes_lengths = np.sqrt(pca.explained_variance_ * threshold)
    volume_90 = ((4 / 3) * np.pi * np.prod(axes_lengths))

    # Calculating statistics for output
    num_points = len(points)
    num_enclosed = inside.sum()
    percentage_enclosed = num_enclosed / num_points * 100

    # Store in a dictionary for later use
    stats_outcome = {
        "num_points": num_points,
        "num_enclosed": num_enclosed,
        "percentage_enclosed": round(percentage_enclosed, 3),
        "volume_90": round(volume_90, 3),
    }

    return (
        volume_90,
        enclosed_points,
        inside,
        mean,
        pca,
        threshold,
        stats_outcome,
        points,
    )


def plot_ellipsoid_and_points_stickman(
    point_of_interest,
    RANK,
    LANK,
    RKNE,
    LKNE,
    RPEL,
    LPEL,
    RSHO,
    LSHO,
    RELB,
    LELB,
    LWRA,
    RWRA,
    bambiID,
    folder_save_path,
    confidence_threshold,
    CSHD=None,
    FSHD=None,
    LSHD=None,
    RSHD=None,
    interactive=True,
    inside_point=False,
    outside_point=False,
):
    """
    Plot a 3D visualization combining:
    - The mean positions of key body markers,
    - A stickman model connecting those markers,
    - A 90% confidence ellipsoid around a specific marker,
    - And optionally the point cloud within or outside the ellipsoid.

    Head markers (CSHD, FSHD, LSHD, RSHD) are optional; if omitted, they won't be plotted.
    """

    markers_to_scale = [
        point_of_interest,
        RANK, LANK, RKNE, LKNE,
        RPEL, LPEL, RSHO, LSHO,
        RELB, LELB, LWRA, RWRA,
        CSHD, FSHD, LSHD, RSHD
    ]

    markers_to_scale = [m / 10 if m is not None else None for m in markers_to_scale]

    (
        point_of_interest,
        RANK, LANK, RKNE, LKNE,
        RPEL, LPEL, RSHO, LSHO,
        RELB, LELB, LWRA, RWRA,
        CSHD, FSHD, LSHD, RSHD
    ) = markers_to_scale
    os.makedirs(folder_save_path, exist_ok=True)

    # Get the ellipsoid volume and points
    (
        volume_90,
        enclosed_points,
        inside,
        mean,
        pca,
        threshold,
        stats_outcome,
        points,
    ) = ellipsoid_volume_and_points(
        point_of_interest, confidence_threshold=confidence_threshold
    )

    def safe_mean(arr):
        return None if arr is None else np.nanmean(arr, axis=0)

    # Calculate the mean positions for each (mandatory) marker
    mean_RANK = safe_mean(RANK)
    mean_LANK = safe_mean(LANK)
    mean_RKNE = safe_mean(RKNE)
    mean_LKNE = safe_mean(LKNE)
    mean_RSHO = safe_mean(RSHO)
    mean_LSHO = safe_mean(LSHO)
    mean_RPEL = safe_mean(RPEL)
    mean_LPEL = safe_mean(LPEL)
    mean_RELB = safe_mean(RELB)
    mean_LELB = safe_mean(LELB)
    mean_RWRA = safe_mean(RWRA)
    mean_LWRA = safe_mean(LWRA)

    # Optional head markers
    mean_CSHD = safe_mean(CSHD)
    mean_FSHD = safe_mean(FSHD)
    mean_LSHD = safe_mean(LSHD)
    mean_RSHD = safe_mean(RSHD)

    # Build marker dict only with available markers (not None and finite)
    def valid(p):
        return (p is not None) and np.isfinite(p).all()

    marker_dict = {
        name: pos
        for name, pos in [
            ("RANK", mean_RANK),
            ("LANK", mean_LANK),
            ("RKNE", mean_RKNE),
            ("LKNE", mean_LKNE),
            ("RSHO", mean_RSHO),
            ("LSHO", mean_LSHO),
            ("RPEL", mean_RPEL),
            ("LPEL", mean_LPEL),
            ("RELB", mean_RELB),
            ("LELB", mean_LELB),
            ("RWRA", mean_RWRA),
            ("LWRA", mean_LWRA),
            ("CSHD", mean_CSHD),
            ("FSHD", mean_FSHD),
            ("LSHD", mean_LSHD),
            ("RSHD", mean_RSHD),
        ]
        if valid(pos)
    }

    if len(marker_dict) == 0:
        raise ValueError("Aucun marqueur valide à tracer.")

    # Define stickman connections; include only connections whose endpoints exist
    base_connections = [
        ("LSHO", "RSHO"),
        ("LSHO", "LPEL"),
        ("RPEL", "RSHO"),
        ("LPEL", "RPEL"),
        ("LSHO", "LELB"),
        ("RSHO", "RELB"),
        ("LELB", "LWRA"),
        ("RELB", "RWRA"),
        ("LPEL", "LKNE"),
        ("RPEL", "RKNE"),
        ("LKNE", "LANK"),
        ("RKNE", "RANK"),
        # Head markers (optional)
        ("CSHD", "CSHD"),
        ("FSHD", "FSHD"),
        ("LSHD", "LSHD"),
        ("RSHD", "RSHD"),
    ]
    stickman_connections = [
        (a, b) for (a, b) in base_connections if (a in marker_dict and b in marker_dict)
    ]

    # Create the 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(azim=30)

    # Plot markers
    for name, pos in marker_dict.items():
        color = "red" if np.allclose(pos, mean, equal_nan=False) else "blue"
        ax.scatter(pos[0], pos[1], pos[2], color=color, s=100, label=name, marker="o")

    # Plot stickman segments
    for start, end in stickman_connections:
        p1, p2 = marker_dict[start], marker_dict[end]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black", linewidth=2)

    # Plot points (inside/outside) if requested
    if inside_point and points is not None and len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="blue", label="All points")
    if outside_point and enclosed_points is not None and len(enclosed_points) > 0:
        ax.scatter(
            enclosed_points[:, 0],
            enclosed_points[:, 1],
            enclosed_points[:, 2],
            color="green",
            label="Inside ellipsoid",
        )

    # Generate the ellipsoid surface around 'mean'
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ellipsoid_unit = np.stack((x, y, z), axis=-1)

    axes_lengths = np.sqrt(pca.explained_variance_ * threshold)
    ellipsoid_scaled = ellipsoid_unit * axes_lengths
    ellipsoid_rotated = np.einsum("ijk,lk->ijl", ellipsoid_scaled, pca.components_)

    x_e = ellipsoid_rotated[..., 0] + mean[0]
    y_e = ellipsoid_rotated[..., 1] + mean[1]
    z_e = ellipsoid_rotated[..., 2] + mean[2]
    ax.plot_surface(x_e, y_e, z_e, color="red", alpha=0.3)

    # Axis labels
    ax.set_xlabel("X Axis (cm)")
    ax.set_ylabel("Y Axis (cm)")
    ax.set_zlabel("Z Axis (cm)")

    # Compute axis limits from available markers
    all_points = np.vstack([pos for pos in marker_dict.values() if valid(pos)])
    x_limits = [np.nanmin(all_points[:, 0]), np.nanmax(all_points[:, 0])]
    y_limits = [np.nanmin(all_points[:, 1]), np.nanmax(all_points[:, 1])]
    z_limits = [np.nanmin(all_points[:, 2]), np.nanmax(all_points[:, 2])]

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    max_range = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    ) / 2

    if not np.isfinite([x_middle, y_middle, z_middle, max_range]).all():
        raise ValueError("Les coordonnées contiennent NaN/Inf – impossible de tracer.")

    ax.set_xlim(x_middle - max_range, x_middle + max_range)
    ax.set_ylim(y_middle - max_range, y_middle + max_range)
    ax.set_zlim(z_middle - max_range, z_middle + max_range)

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.zaxis.set_major_locator(MultipleLocator(100))
    ax.set_box_aspect([1, 1, 1])
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)

    # Show or save
    if interactive:
        plt.show()
    else:
        if np.allclose(mean, marker_dict.get("RANK", np.array([np.inf, np.inf, np.inf]))) or \
           np.allclose(mean, marker_dict.get("LANK", np.array([np.inf, np.inf, np.inf]))):
            region = "ankle"
        elif np.allclose(mean, marker_dict.get("RWRA", np.array([np.inf, np.inf, np.inf]))) or \
             np.allclose(mean, marker_dict.get("LWRA", np.array([np.inf, np.inf, np.inf]))):
            region = "wrist"
        else:
            region = "unknown"

        filename_to_save = f"{bambiID}_{region}_position_ellipsoid.png"
        save_path = os.path.join(folder_save_path, filename_to_save)
        fig.savefig(save_path, dpi=300, pad_inches=0)
        print(f"Static plot saved to {os.path.abspath(save_path)}")
        plt.close()

    return stats_outcome
