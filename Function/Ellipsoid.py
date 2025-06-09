import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.use("TkAgg")


def ellipsoid_volume_and_points(points):
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
    threshold = chi2.ppf(0.90, df=3)
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
        "percentage_enclosed": percentage_enclosed,
        "volume_90": volume_90,
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
    data,
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

    Parameters
    ----------
    data : Marker trajectory data used to fit the ellipsoid (frames × 3).

    RANK, LANK, RKNE, LKNE, RSHO, LSHO, RELB, LELB, RWRA, LWRA :
        3D coordinates (N_frames × 3) for each body marker .

    bambiID : Identifier of the trial or subject, used to name the saved figure.

    folder_save_path : Directory where the figure will be saved .

    interactive :  True, displays an interactive 3D plot; if False, saves a static image.

    inside_point : If True, shows all trajectory points.

    outside_point : If True, shows only the points inside the ellipsoid.
    """

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
    ) = ellipsoid_volume_and_points(data)

    # Calculate the mean positions for each marker (RANK, LANK, RKNE, etc.)
    mean_RANK = np.mean(RANK, axis=0)
    mean_LANK = np.mean(LANK, axis=0)
    mean_RKNE = np.mean(RKNE, axis=0)
    mean_LKNE = np.mean(LKNE, axis=0)
    mean_RSHO = np.mean(RSHO, axis=0)
    mean_LSHO = np.mean(LSHO, axis=0)
    mean_RPEL = np.mean(RPEL, axis=0)
    mean_LPEL = np.mean(LPEL, axis=0)
    mean_RELB = np.mean(RELB, axis=0)
    mean_LELB = np.mean(LELB, axis=0)
    mean_RWRA = np.mean(RWRA, axis=0)
    mean_LWRA = np.mean(LWRA, axis=0)

    # Create the 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(azim=30)  # Set the viewing angle for better visualization

    # Plot each key marker as a scatter plot point
    if mean[0] == mean_RANK[0]:
        ax.scatter(
            mean_RANK[0],
            mean_RANK[1],
            mean_RANK[2],
            color="red",
            s=100,
            label="RANK",
            marker="o",
        )
        ax.scatter(
            mean_LANK[0],
            mean_LANK[1],
            mean_LANK[2],
            color="blue",
            s=100,
            label="LANK",
            marker="o",
        )
    elif mean[0] == mean_LANK[0]:
        ax.scatter(
            mean_LANK[0],
            mean_LANK[1],
            mean_LANK[2],
            color="red",
            s=100,
            label="LANK",
            marker="o",
        )
        ax.scatter(
            mean_RANK[0],
            mean_RANK[1],
            mean_RANK[2],
            color="blue",
            s=100,
            label="RANK",
            marker="o",
        )

    # Plot the other body markers in the same way (RKNE, LKNE, RSHO, LSHO, etc.)
    ax.scatter(
        mean_RKNE[0],
        mean_RKNE[1],
        mean_RKNE[2],
        color="blue",
        s=100,
        label="RKNE",
        marker="o",
    )
    ax.scatter(
        mean_LKNE[0],
        mean_LKNE[1],
        mean_LKNE[2],
        color="blue",
        s=100,
        label="LKNE",
        marker="o",
    )
    ax.scatter(
        mean_RSHO[0],
        mean_RSHO[1],
        mean_RSHO[2],
        color="blue",
        s=100,
        label="RSHO",
        marker="o",
    )
    ax.scatter(
        mean_LSHO[0],
        mean_LSHO[1],
        mean_LSHO[2],
        color="blue",
        s=100,
        label="LSHO",
        marker="o",
    )
    ax.scatter(
        mean_RPEL[0],
        mean_RPEL[1],
        mean_RPEL[2],
        color="blue",
        s=100,
        label="RPEL",
        marker="o",
    )
    ax.scatter(
        mean_LPEL[0],
        mean_LPEL[1],
        mean_LPEL[2],
        color="blue",
        s=100,
        label="LPEL",
        marker="o",
    )
    ax.scatter(
        mean_RELB[0],
        mean_RELB[1],
        mean_RELB[2],
        color="blue",
        s=100,
        label="RELB",
        marker="o",
    )
    ax.scatter(
        mean_LELB[0],
        mean_LELB[1],
        mean_LELB[2],
        color="blue",
        s=100,
        label="LELB",
        marker="o",
    )
    ax.scatter(
        mean_RWRA[0],
        mean_RWRA[1],
        mean_RWRA[2],
        color="blue",
        s=100,
        label="RWRA",
        marker="o",
    )
    ax.scatter(
        mean_LWRA[0],
        mean_LWRA[1],
        mean_LWRA[2],
        color="blue",
        s=100,
        label="LWRA",
        marker="o",
    )

    # Create the stickman model by plotting lines between key body markers
    ax.plot(
        [mean_LSHO[0], mean_RSHO[0]],
        [mean_LSHO[1], mean_RSHO[1]],
        [mean_LSHO[2], mean_RSHO[2]],
        color="black",
        linewidth=2,
    )
    ax.plot(
        [mean_LSHO[0], mean_LPEL[0]],
        [mean_LSHO[1], mean_LPEL[1]],
        [mean_LSHO[2], mean_LPEL[2]],
        color="black",
        linewidth=2,
    )
    ax.plot(
        [mean_RPEL[0], mean_RSHO[0]],
        [mean_RPEL[1], mean_RSHO[1]],
        [mean_RPEL[2], mean_RSHO[2]],
        color="black",
        linewidth=2,
    )
    ax.plot(
        [mean_LPEL[0], mean_RPEL[0]],
        [mean_LPEL[1], mean_RPEL[1]],
        [mean_LPEL[2], mean_RPEL[2]],
        color="black",
        linewidth=2,
    )

    # Add other body segments using similar lines between markers (e.g., knees, elbows, wrists)

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
        ]
    )

    x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]

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
        filename_to_save = f"{bambiID}_ankle_position_ellipsoid.png"
        save_path = os.path.join(folder_save_path, filename_to_save)
        # fig.savefig(save_path, dpi=300, pad_inches=0)
        print(f"Static plot saved to {os.path.abspath(save_path)}")
        plt.close()

    return stats_outcome  # Return the computed stats outcome
