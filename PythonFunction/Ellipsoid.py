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
    volume_90 = (4 / 3) * np.pi * np.prod(axes_lengths)

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
    CSHD,
    FSHD,
    LSHD,
    RSHD,
    bambiID,
    folder_save_path,
    confidence_threshold,
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
    """
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

    # Calculate the mean positions for each marker (RANK, LANK, RKNE, etc.)
    mean_RANK = np.nanmean(RANK, axis=0)
    mean_LANK = np.nanmean(LANK, axis=0)
    mean_RKNE = np.nanmean(RKNE, axis=0)
    mean_LKNE = np.nanmean(LKNE, axis=0)
    mean_RSHO = np.nanmean(RSHO, axis=0)
    mean_LSHO = np.nanmean(LSHO, axis=0)
    mean_RPEL = np.nanmean(RPEL, axis=0)
    mean_LPEL = np.nanmean(LPEL, axis=0)
    mean_RELB = np.nanmean(RELB, axis=0)
    mean_LELB = np.nanmean(LELB, axis=0)
    mean_RWRA = np.nanmean(RWRA, axis=0)
    mean_LWRA = np.nanmean(LWRA, axis=0)
    mean_CSHD = np.nanmean(CSHD, axis=0)
    mean_FSHD = np.nanmean(FSHD, axis=0)
    mean_LSHD = np.nanmean(LSHD, axis=0)
    mean_RSHD = np.nanmean(RSHD, axis=0)

    # Create the 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(azim=30)  # Set the viewing angle for better visualization

    # Define all markers with their names and average 3D positions
    marker_dict = {
        "RANK": mean_RANK,
        "LANK": mean_LANK,
        "RKNE": mean_RKNE,
        "LKNE": mean_LKNE,
        "RSHO": mean_RSHO,
        "LSHO": mean_LSHO,
        "RPEL": mean_RPEL,
        "LPEL": mean_LPEL,
        "RELB": mean_RELB,
        "LELB": mean_LELB,
        "RWRA": mean_RWRA,
        "LWRA": mean_LWRA,
        "CSHD": mean_CSHD,
        "FSHD": mean_FSHD,
        "LSHD": mean_LSHD,
        "RSHD": mean_RSHD,
    }

    # Define stickman connections as pairs of (start, end) marker names
    stickman_connections = [
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
        ("CSHD", "CSHD"),
        ("FSHD", "FSHD"),
        ("LSHD", "LSHD"),
        ("RSHD", "RSHD"),
    ]

    # Plot each marker as a 3D scatter point
    for name, pos in marker_dict.items():
        color = "red" if np.allclose(pos, mean) else "blue"
        ax.scatter(pos[0], pos[1], pos[2], color=color, s=100, label=name, marker="o")

    # Plot all stickman segments
    for start, end in stickman_connections:
        p1, p2 = marker_dict[start], marker_dict[end]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black", linewidth=2
        )

    # Plot all points inside or outside the ellipsoid, depending on the flag
    if inside_point:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], color="blue", label="All points"
        )
    if outside_point:
        ax.scatter(
            enclosed_points[:, 0],
            enclosed_points[:, 1],
            enclosed_points[:, 2],
            color="green",
            label="Inside ellipsoid",
        )

    # Generate the ellipsoid surface
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ellipsoid_unit = np.stack((x, y, z), axis=-1)

    axes_lengths = np.sqrt(pca.explained_variance_ * threshold)
    ellipsoid_scaled = ellipsoid_unit * axes_lengths  # Scale the ellipsoid

    # Rotate the ellipsoid using PCA components
    ellipsoid_rotated = np.einsum("ijk,lk->ijl", ellipsoid_scaled, pca.components_)

    # Translate the ellipsoid to the mean position of the markers
    x_e = ellipsoid_rotated[..., 0] + mean[0]
    y_e = ellipsoid_rotated[..., 1] + mean[1]
    z_e = ellipsoid_rotated[..., 2] + mean[2]

    # Plot the ellipsoid surface
    ax.plot_surface(x_e, y_e, z_e, color="red", alpha=0.3)

    # Set axis labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Fix the aspect ratio so that the axes are equally scaled
    all_points = np.array(
        [
            mean_RANK,
            mean_LANK,
            mean_RKNE,
            mean_LKNE,
            mean_RSHO,
            mean_LSHO,
            mean_RPEL,
            mean_LPEL,
            mean_RELB,
            mean_LELB,
            mean_RWRA,
            mean_LWRA,
            mean_CSHD,
            mean_FSHD,
            mean_LSHD,
            mean_RSHD,
        ]
    )

    x_limits = [np.nanmin(all_points[:, 0]), np.nanmax(all_points[:, 0])]
    y_limits = [np.nanmin(all_points[:, 1]), np.nanmax(all_points[:, 1])]
    z_limits = [np.nanmin(all_points[:, 2]), np.nanmax(all_points[:, 2])]

    # Calculate the center and maximum radius for axis limits
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    max_range = (
        max(
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        )
        / 2
    )
    if not np.isfinite([x_middle, max_range]).all():
        raise ValueError("Les coordonnées contiennent NaN/Inf – impossible de tracer.")

    ax.set_xlim(x_middle - max_range, x_middle + max_range)
    ax.set_ylim(y_middle - max_range, y_middle + max_range)
    ax.set_zlim(z_middle - max_range, z_middle + max_range)

    # Adjust grid and axis ticks
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.zaxis.set_major_locator(MultipleLocator(100))

    ax.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for all axes

    # Adjust layout to remove margins
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)

    # Show interactive plot or save static image based on the parameter
    if interactive:
        plt.show()
    else:
        if np.allclose(mean, mean_RANK) or np.allclose(mean, mean_LANK):
            region = "ankle"
        elif np.allclose(mean, mean_RWRA) or np.allclose(mean, mean_LWRA):
            region = "wrist"
        else:
            region = "unknown"

        filename_to_save = f"{bambiID}_{region}_position_ellipsoid.png"
        save_path = os.path.join(folder_save_path, filename_to_save)
        fig.savefig(save_path, dpi=300, pad_inches=0)
        print(f"Static plot saved to {os.path.abspath(save_path)}")
        plt.close()

    return stats_outcome  # Return the computed stats outcome
