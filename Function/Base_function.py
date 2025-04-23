import numpy as np
from sklearn.decomposition import PCA
import scipy
from scipy.stats import chi2, shapiro
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import pandas as pd
from matplotlib.ticker import MultipleLocator

matplotlib.use('TkAgg')

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
    interp_function = interp1d(np.linspace(0, 1, len(data)), data, kind='linear', fill_value='extrapolate')

    # Resample the series to the target length
    resampled_data = interp_function(np.linspace(0, 1, target_length))

    return resampled_data

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
    md_squared = np.sum((transformed / np.sqrt(pca.explained_variance_))**2, axis=1)

    # Chi-squared threshold for 90% confidence in 3D
    threshold = chi2.ppf(0.90, df=3)
    inside = md_squared <= threshold
    enclosed_points = points[inside]

    # Volume of the ellipsoid
    axes_lengths = np.sqrt(pca.explained_variance_ * threshold)
    volume_90 = (4/3) * np.pi * np.prod(axes_lengths)

    # Calculating statistics for output
    num_points = len(points)
    num_enclosed = inside.sum()
    percentage_enclosed = num_enclosed / num_points * 100

    # Store in a dictionary for later use
    stats_outcome = {
        'num_points': num_points,
        'num_enclosed': num_enclosed,
        'percentage_enclosed': percentage_enclosed,
        'volume_90': volume_90
    }


    return volume_90, enclosed_points, inside, mean, pca, threshold, stats_outcome, points

def plot_ellipsoid_and_points(data,
                              bambiID,
                              folder_save_path,
                              interactive=True,
                              inside_point=False,
                              outside_point=False):
    """
    Plots the 3D points and 90% confidence ellipsoid around a marker.

    Parameters:
    - data: dataset dictionary
    - interactive: if True, shows interactive plot; if False, saves static PNG
    - save_path: path for saving the PNG if interactive is False
    """

    (volume_90,
     enclosed_points,
     inside,
     mean,
     pca,
     threshold,
     stats_outcome,
     points) = ellipsoid_volume_and_points(data)

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    if inside_point is True:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='All points')
    if outside_point is True:
        ax.scatter(enclosed_points[:, 0], enclosed_points[:, 1], enclosed_points[:, 2], color='green', label='Inside ellipsoid')

    # Generate ellipsoid surface
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ellipsoid_unit = np.stack((x, y, z), axis=-1)

    axes_lengths = np.sqrt(pca.explained_variance_ * threshold)
    ellipsoid_scaled = ellipsoid_unit * axes_lengths  # scale

    # Rotate using PCA components
    ellipsoid_rotated = np.einsum('ijk,lk->ijl', ellipsoid_scaled, pca.components_)

    # Translate to mean
    x_e = ellipsoid_rotated[..., 0] + mean[0]
    y_e = ellipsoid_rotated[..., 1] + mean[1]
    z_e = ellipsoid_rotated[..., 2] + mean[2]

    ax.plot_surface(x_e, y_e, z_e, color='red', alpha=0.3)

    ax.set_title(f"{bambiID}")
    ax.legend()

    # Adding axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Adjust layout and remove unnecessary margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if interactive:
        plt.show()
    else:

        filename_to_save = f"{bambiID}_ankle_position_ellipsoid.png"
        save_path = os.path.join(folder_save_path, filename_to_save)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Static plot saved to {os.path.abspath(save_path)}")
        plt.close()

    return stats_outcome

def plot_combined_pdf(hip_add_all, plot_name, folder_save_path, bins=50, color="lightcoral"):
    """
    Plot histogram + density estimate (PDF) of all hip angles combined across all bambis.

    Parameters:
    - hip_add_all: list of lists of angles (each sublist = one bambi's angle series)
    - bins: number of histogram bins (default=50)
    - color: color of the bars and curve
    """
    # Flatten the list of lists into one long list of angles
    all_angles = [angle for bambi in hip_add_all for angle in bambi]

    # Plot style
    sns.set(style="whitegrid")

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(all_angles, bins=bins, kde=True, stat="density",
                 color=color, edgecolor="black")

    # Title and axis labels
    plt.title(f"Distribution of Hip Angles {plot_name} - All Bambis")
    plt.xlabel("Hip Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.tight_layout()

    filename_to_save = f"{plot_name}_all_bambies.png"
    save_path = os.path.join(folder_save_path, filename_to_save)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mean_pdf(hip_add_all, plot_name, folder_save_path, grid_min=-90, grid_max=90, grid_points=500, show_std=True, all_line=True):
    """
    Plot the mean PDF of hip adduction/abduction angles across bambis.

    Parameters:
    - hip_add_all: list of lists. Each sublist contains angle values over time for one bambi.
    - grid_min, grid_max: range of values for KDE evaluation.
    - grid_points: number of points in the KDE grid.
    - show_std: whether to display ±1 standard deviation as a shaded area.
    """
    # Create evaluation grid
    grid = np.linspace(grid_min, grid_max, grid_points)

    # Estimate KDEs for each bambi
    kdes = []
    for bambi in hip_add_all:
        bambi_resample = resample_size(bambi, 7200)
        kde = gaussian_kde(bambi_resample)
        kdes.append(kde(grid))

    # Convert list to array for mean/std computation
    kdes = np.array(kdes)
    mean_pdf = np.mean(kdes, axis=0)
    std_pdf = np.std(kdes, axis=0)

    lower_bound = np.maximum(mean_pdf - std_pdf, 0)
    upper_bound = mean_pdf + std_pdf

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(grid, mean_pdf, label='Mean PDF', color='black', linewidth=2)

    if all_line:
        for kde in kdes:
            plt.plot(grid, kde, color='gray', alpha=0.2)

    if show_std:
        plt.fill_between(grid, lower_bound, upper_bound,
                         color='black', alpha=0.3, label='±1 Std Dev')

    plt.title(f"Mean PDF of Hip Angles {plot_name} Across Bambis")
    plt.xlabel("Hip Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename_to_save = f"{plot_name}_mean_across_bambies.png"
    save_path = os.path.join(folder_save_path, filename_to_save)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mean_pdf_stat(hip_add_all, bambiID_list, plot_name, folder_save_path, grid_min=-90, grid_max=90, grid_points=500, show_std=True,
                  all_line=True, target_length=7200):
    """
    Plot the mean PDF of hip adduction/abduction angles across bambis and save statistical analysis.

    Parameters:
    - hip_add_all: list of lists. Each sublist contains angle values over time for one bambi.
    - grid_min, grid_max: range of values for KDE evaluation.
    - grid_points: number of points in the KDE grid.
    - show_std: whether to display ±1 standard deviation as a shaded area.
    - all_line: whether to show all individual KDE lines.
    """
    # Create evaluation grid
    grid = np.linspace(grid_min, grid_max, grid_points)

    # Estimate KDEs for each bambi
    kdes = []
    stats_data = []  # For storing the stats (mean, std, skewness, kurtosis)
    hip_add_mean = []
    hip_add_std = []

    for idx, bambi in enumerate(hip_add_all, start=0):
        bambi_resample = resample_size(bambi, target_length)
        kde = gaussian_kde(bambi_resample)
        kde_values = kde(grid)
        kdes.append(kde_values)

        # Compute the statistics (skewness, kurtosis)
        skew_kde = skew(kde_values)
        kurt_kde = kurtosis(kde_values)

        # Add stats for the current bambi to the data (including an ID for each bambi)
        stats_data.append([f"{bambiID_list[idx]}", skew_kde, kurt_kde])

        hip_add_mean.append(np.mean(bambi))
        hip_add_std.append(np.std(bambi))


    # Convert list to array for mean/std computation
    kdes = np.array(kdes)
    mean_pdf = np.mean(kdes, axis=0)
    std_pdf = np.std(kdes, axis=0)

    lower_bound = np.maximum(mean_pdf - std_pdf, 0)
    upper_bound = mean_pdf + std_pdf

    # Compute stats for the mean PDF
    skew_mean_pdf = skew(mean_pdf)
    kurt_mean_pdf = kurtosis(mean_pdf)

    # Add the mean PDF stats to the data (mark this line as "Total Mean")
    stats_data.append(["Total Mean", skew_mean_pdf, kurt_mean_pdf])

    # Create the DataFrame for stats
    stats_df = pd.DataFrame(stats_data, columns=["Subject", "Skewness", "Kurtosis"])

    # Save the stats to a CSV file
    stats_filename = f"{plot_name}_stats_on_KDE_PDF.csv"
    stats_save_path = os.path.join(folder_save_path, stats_filename)
    # stats_df.to_csv(stats_save_path, index=False)

    mean_val = np.mean(hip_add_mean)
    std_val = np.std(hip_add_mean)


    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(grid, mean_pdf, label='Mean PDF', color='black', linewidth=2)

    if all_line:
        for kde in kdes:
            plt.plot(grid, kde, color='gray', alpha=0.2)

    if show_std:
        plt.fill_between(grid, lower_bound, upper_bound,
                         color='black', alpha=0.3, label='±1 Std Dev')

    plt.text(0.02, 0.965,
             f'Mean: {mean_val:.2f}\nSD: {std_val:.2f}',  # Ligne vide entre les deux
             transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             fontsize=12, linespacing=2,  # Augmente l’espacement des lignes
             bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.4', alpha=0.8))

    plt.title(f"Mean PDF of Hip Angles {plot_name} Across Bambis")
    plt.xlabel("Hip Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.ylim(0,0.25)

    filename_to_save = f"{plot_name}_mean_across_bambies.png"
    save_path = os.path.join(folder_save_path, filename_to_save)
    plt.savefig(save_path, dpi=300)
    plt.close()





def plot_ellipsoid_and_points_stickman(data, RANK, LANK, RKNE, LKNE, RPEL, LPEL, RSHO, LSHO,  RELB, LELB, LWRA, RWRA,
                              bambiID,
                              folder_save_path,
                              interactive=True,
                              inside_point=False,
                              outside_point=False):
    """
    Plots the 3D points and 90% confidence ellipsoid around a marker.

    Parameters:
    - data: dataset dictionary
    - interactive: if True, shows interactive plot; if False, saves static PNG
    - save_path: path for saving the PNG if interactive is False
    """

    (volume_90,
     enclosed_points,
     inside,
     mean,
     pca,
     threshold,
     stats_outcome,
     points) = ellipsoid_volume_and_points(data)

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
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(azim=30)

    if mean[0] == mean_RANK[0]:
        ax.scatter(mean_RANK[0], mean_RANK[1], mean_RANK[2], color='red', s=100, label='RANK', marker='o')
        ax.scatter(mean_LANK[0], mean_LANK[1], mean_LANK[2], color='blue', s=100, label='LANK', marker='o')

    elif mean[0] == mean_LANK[0]:
        ax.scatter(mean_LANK[0], mean_LANK[1], mean_LANK[2], color='red', s=100, label='LANK', marker='o')
        ax.scatter(mean_RANK[0], mean_RANK[1], mean_RANK[2], color='blue', s=100, label='RANK', marker='o')

    ax.scatter(mean_RKNE[0], mean_RKNE[1], mean_RKNE[2], color='blue', s=100, label='RKNE', marker='o')
    ax.scatter(mean_RKNE[0], mean_RKNE[1], mean_RKNE[2], color='blue', s=100, label='RKNE', marker='o')
    ax.scatter(mean_LKNE[0], mean_LKNE[1], mean_LKNE[2], color='blue', s=100, label='LKNE', marker='o')
    ax.scatter(mean_RSHO[0], mean_RSHO[1], mean_RSHO[2], color='blue', s=100, label='RSHO', marker='o')
    ax.scatter(mean_LSHO[0], mean_LSHO[1], mean_LSHO[2], color='blue', s=100, label='LSHO', marker='o')
    ax.scatter(mean_RPEL[0], mean_RPEL[1], mean_RPEL[2], color='blue', s=100, label='RPEL', marker='o')
    ax.scatter(mean_LPEL[0], mean_LPEL[1], mean_LPEL[2], color='blue', s=100, label='LPEL', marker='o')
    ax.scatter(mean_RELB[0], mean_RELB[1], mean_RELB[2], color='blue', s=100, label='RSHO', marker='o')
    ax.scatter(mean_LELB[0], mean_LELB[1], mean_LELB[2], color='blue', s=100, label='LSHO', marker='o')
    ax.scatter(mean_RWRA[0], mean_RWRA[1], mean_RWRA[2], color='blue', s=100, label='RPEL', marker='o')
    ax.scatter(mean_LWRA[0], mean_LWRA[1], mean_LWRA[2], color='blue', s=100, label='LPEL', marker='o')

    ax.plot([mean_LSHO[0], mean_RSHO[0]],
            [mean_LSHO[1], mean_RSHO[1]],
            [mean_LSHO[2], mean_RSHO[2]],
            color='black', linewidth=2)
    ax.plot([mean_LSHO[0], mean_LPEL[0]],
            [mean_LSHO[1], mean_LPEL[1]],
            [mean_LSHO[2], mean_LPEL[2]],
            color='black', linewidth=2)
    ax.plot([mean_RPEL[0], mean_RSHO[0]],
            [mean_RPEL[1], mean_RSHO[1]],
            [mean_RPEL[2], mean_RSHO[2]],
            color='black', linewidth=2)
    ax.plot([mean_LPEL[0], mean_RPEL[0]],
            [mean_LPEL[1], mean_RPEL[1]],
            [mean_LPEL[2], mean_RPEL[2]],
            color='black', linewidth=2)

    ax.plot([mean_LPEL[0], mean_LKNE[0]],
            [mean_LPEL[1], mean_LKNE[1]],
            [mean_LPEL[2], mean_LKNE[2]],
            color='black', linewidth=2)
    ax.plot([mean_RKNE[0], mean_RPEL[0]],
            [mean_RKNE[1], mean_RPEL[1]],
            [mean_RKNE[2], mean_RPEL[2]],
            color='black', linewidth=2)

    ax.plot([mean_LANK[0], mean_LKNE[0]],
            [mean_LANK[1], mean_LKNE[1]],
            [mean_LANK[2], mean_LKNE[2]],
            color='black', linewidth=2)
    ax.plot([mean_RKNE[0], mean_RANK[0]],
            [mean_RKNE[1], mean_RANK[1]],
            [mean_RKNE[2], mean_RANK[2]],
            color='black', linewidth=2)

    ax.plot([mean_LELB[0], mean_LSHO[0]],
            [mean_LELB[1], mean_LSHO[1]],
            [mean_LELB[2], mean_LSHO[2]],
            color='black', linewidth=2)
    ax.plot([mean_RELB[0], mean_RSHO[0]],
            [mean_RELB[1], mean_RSHO[1]],
            [mean_RELB[2], mean_RSHO[2]],
            color='black', linewidth=2)

    ax.plot([mean_LELB[0], mean_LWRA[0]],
            [mean_LELB[1], mean_LWRA[1]],
            [mean_LELB[2], mean_LWRA[2]],
            color='black', linewidth=2)
    ax.plot([mean_RELB[0], mean_RWRA[0]],
            [mean_RELB[1], mean_RWRA[1]],
            [mean_RELB[2], mean_RWRA[2]],
            color='black', linewidth=2)


    # Plot all points
    if inside_point is True:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='All points')
    if outside_point is True:
        ax.scatter(enclosed_points[:, 0], enclosed_points[:, 1], enclosed_points[:, 2], color='green', label='Inside ellipsoid')

    # Generate ellipsoid surface
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ellipsoid_unit = np.stack((x, y, z), axis=-1)

    axes_lengths = np.sqrt(pca.explained_variance_ * threshold)
    ellipsoid_scaled = ellipsoid_unit * axes_lengths  # scale

    # Rotate using PCA components
    ellipsoid_rotated = np.einsum('ijk,lk->ijl', ellipsoid_scaled, pca.components_)

    # Translate to mean
    x_e = ellipsoid_rotated[..., 0] + mean[0]
    y_e = ellipsoid_rotated[..., 1] + mean[1]
    z_e = ellipsoid_rotated[..., 2] + mean[2]

    ax.plot_surface(x_e, y_e, z_e, color='red', alpha=0.3)

    # ax.set_title(f"{bambiID}")
    # ax.legend()

    # Adding axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Fix equal aspect ratio
    all_points = np.array([
        mean_RANK, mean_LANK,
        mean_RKNE, mean_LKNE,
        mean_RSHO, mean_LSHO,
        mean_RPEL, mean_LPEL,
        mean_RELB, mean_LELB,
        mean_RWRA, mean_LWRA
    ])

    x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]

    # Calcul du centre et du rayon maximal
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    max_range = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0]
    ) / 2

    ax.set_xlim(x_middle - max_range, x_middle + max_range)
    ax.set_ylim(y_middle - max_range, y_middle + max_range)
    ax.set_zlim(z_middle - max_range, z_middle + max_range)

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.zaxis.set_major_locator(MultipleLocator(100))

    # set_axes_equal(ax)

    ax.set_box_aspect([1, 1, 1])

    # Adjust layout and remove unnecessary margins
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)

    if interactive:
        plt.show()
    else:

        filename_to_save = f"{bambiID}_ankle_position_ellipsoid.png"
        save_path = os.path.join(folder_save_path, filename_to_save)
        fig.savefig(save_path, dpi=300, pad_inches=0)
        print(f"Static plot saved to {os.path.abspath(save_path)}")
        plt.close()

    return stats_outcome