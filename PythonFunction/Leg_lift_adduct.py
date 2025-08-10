import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import skew, kurtosis
import pandas as pd
from PythonFunction.Base_function import (
    resample_size,
    get_threshold_intervals,
    intersect_intervals,
    analyze_intervals_duration,
)

matplotlib.use("TkAgg")


def plot_combined_pdf(
    hip_add_all,
    plot_name,
    folder_save_path,
    bins=50,
    color="lightcoral",
    plot_save=False,
):
    """
    Plot a combined histogram and kernel density estimate (PDF) of hip angles
    pooled from all subjects.

    Parameters:
    ----------
    hip_add_all : list of lists -- Hip angles for each subject (one list per subject).

    plot_name : Base name used for the saved figure.

    folder_save_path : Path to the folder where the figure will be saved.

    bins : Number of histogram bins (default: 50).

    color : Color used for both the histogram bars and KDE curve.
    """

    # Fusionne toutes les valeurs d’angle en une seule liste
    all_angles = [angle for bambi in hip_add_all for angle in bambi]

    # Applique un style de tracé épuré
    sns.set(style="whitegrid")

    # Crée la figure avec histogramme + estimation de densité (KDE)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        all_angles, bins=bins, kde=True, stat="density", color=color, edgecolor="black"
    )

    # Titres et étiquettes
    plt.title(f"Distribution of Hip Angles {plot_name} - All Bambis")
    plt.xlabel("Hip Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.tight_layout()

    # Enregistre la figure
    filename_to_save = f"{plot_name}_all_bambies.png"
    save_path = os.path.join(folder_save_path, filename_to_save)
    if plot_save:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mean_pdf_stat(
    hip_add_all,
    bambiID_list,
    plot_name,
    folder_save_path,
    grid_min=-90,
    grid_max=90,
    grid_points=500,
    show_std=True,
    all_line=True,
    target_length=7200,
    plot_save=False,
):
    """
    This function computes and plots the average Kernel Density Estimate (KDE)
    of hip adduction/abduction angles across multiple subjects ("bambis"). It also
    computes basic statistics (skewness and kurtosis) on the KDEs and saves them
    in a CSV file. The final plot includes the mean PDF and optionally displays
    individual subject curves and a ±1 standard deviation shaded region.
    """

    # Create evaluation grid for KDE
    grid = np.linspace(grid_min, grid_max, grid_points)

    kdes = []  # List to store KDE values for each subject
    stats_data = []  # To hold skewness and kurtosis values
    hip_add_mean = []  # Mean angle for each subject
    hip_add_std = []  # Standard deviation for each subject

    for idx, bambi in enumerate(hip_add_all, start=0):
        # Resample signal to a common length
        bambi_resample = resample_size(bambi, target_length)

        # Estimate KDE over the grid
        kde = gaussian_kde(bambi_resample)
        kde_values = kde(grid)
        kdes.append(kde_values)

        # Compute skewness and kurtosis of the estimated PDF
        skew_kde = skew(kde_values)
        kurt_kde = kurtosis(kde_values)

        # Store statistics along with subject ID
        stats_data.append([f"{bambiID_list[idx]}", skew_kde, kurt_kde])

        # Store mean and std for plotting text summary
        hip_add_mean.append(np.mean(bambi))
        hip_add_std.append(np.std(bambi))

    # Convert KDE list to array to compute mean and standard deviation PDF
    kdes = np.array(kdes)
    mean_pdf = np.mean(kdes, axis=0)
    std_pdf = np.std(kdes, axis=0)

    # Define bounds for ±1 std dev shading
    lower_bound = np.maximum(mean_pdf - std_pdf, 0)
    upper_bound = mean_pdf + std_pdf

    # Compute statistics on the mean PDF
    skew_mean_pdf = skew(mean_pdf)
    kurt_mean_pdf = kurtosis(mean_pdf)
    stats_data.append(["Total Mean", skew_mean_pdf, kurt_mean_pdf])

    # Create a DataFrame of all the statistics
    stats_df = pd.DataFrame(stats_data, columns=["Subject", "Skewness", "Kurtosis"])

    # Save statistics to CSV file
    stats_filename = f"{plot_name}_stats_on_KDE_PDF.csv"
    stats_save_path = os.path.join(folder_save_path, stats_filename)
    # stats_df.to_csv(stats_save_path, index=False)  # Uncomment to save

    # Compute mean and std of mean hip angles (not PDFs)
    mean_val = np.mean(hip_add_mean)
    std_val = np.std(hip_add_mean)

    # ---- PLOTTING ----
    plt.figure(figsize=(8, 5))
    plt.plot(grid, mean_pdf, label="Mean PDF", color="black", linewidth=2)

    # Optionally show individual PDFs
    if all_line:
        for kde in kdes:
            plt.plot(grid, kde, color="gray", alpha=0.2)

    # Optionally show ±1 std deviation as shaded area
    if show_std:
        plt.fill_between(
            grid, lower_bound, upper_bound, color="black", alpha=0.3, label="±1 Std Dev"
        )

    # Add box with mean and std of original signals
    plt.text(
        0.02,
        0.965,
        f"Mean: {mean_val:.2f}\nSD: {std_val:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=12,
        linespacing=2,
        bbox=dict(
            facecolor="white",
            edgecolor="lightgrey",
            boxstyle="round,pad=0.4",
            alpha=0.8,
        ),
    )

    # Final plot formatting
    plt.title(f"Mean PDF of Hip Angles {plot_name} Across Bambis")
    plt.xlabel("Hip Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.ylim(0, 0.25)

    # Save figure
    filename_to_save = f"{plot_name}_mean_across_bambies.png"
    save_path = os.path.join(folder_save_path, filename_to_save)
    if plot_save:
        plt.savefig(save_path, dpi=300)
    plt.close()


def ankle_high(
    ankle_marker,
    pelvis_marker,
    ankle_marker_world,
    time_vector,
    leg_length,
    high_threshold,
    max_flexion,
    folder_outcome,
    plot_name,
    plot=False,
):

    thigh_proportion = 55
    thigh_length = leg_length * thigh_proportion / 100

    ankle_high_from_ground = ankle_marker_world[:, 2]

    range_extension = np.sqrt(
        thigh_length**2
        + (leg_length - thigh_length) ** 2
        - 2
        * thigh_length
        * (leg_length - thigh_length)
        * np.cos(np.radians(180 - max_flexion))
    )  # add *0.9

    # Compute the Euclidean distance between pelvis and ankle at each time frame
    distance_pelv_ank = np.linalg.norm(pelvis_marker - ankle_marker, axis=1)

    close_to_max_extension_interval = get_threshold_intervals(
        distance_pelv_ank, range_extension, mode="above"
    )

    ankle_in_elevation_interval = get_threshold_intervals(
        ankle_high_from_ground, high_threshold, mode="above"
    )

    common_intervals = intersect_intervals(
        close_to_max_extension_interval, ankle_in_elevation_interval
    )

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(time_vector, ankle_high_from_ground, label="Ankle high (mm)", color="blue")
        plt.plot(
            time_vector, distance_pelv_ank, label="Distance Pelvis Ankle", color="red"
        )

        for start, end in common_intervals:
            plt.axvspan(time_vector[start], time_vector[end], color="orange", alpha=0.3)

        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_outcome, f"{plot_name}_leg_lift.png"), dpi=300)
        plt.close()

    lift_with_leg_extend = analyze_intervals_duration(
        common_intervals, time_vector, ankle_high_from_ground
    )

    return lift_with_leg_extend, distance_pelv_ank


def compute_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance frame-by-frame between two marker arrays.
    coords1, coords2: shape (N, 3)
    Returns distances: shape (N,)
    """
    return np.linalg.norm(coords1 - coords2, axis=1)


def compute_hip_adduction_angles(
    RPEL: np.ndarray,
    LPEL: np.ndarray,
    RANK: np.ndarray,
    LANK: np.ndarray,
    RSHO: np.ndarray,
    LSHO: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    Compute hip adduction (+)/abduction (-) angles in degrees for right and left legs.
    Midpoint of shoulders is calculated internally.
    """
    # compute shoulder midpoint
    midShoulder = (RSHO + LSHO) / 2

    # reference vectors
    v_ref_r = RPEL - LPEL
    v_ref_l = LPEL - RPEL

    # thigh vectors
    thigh_R = RANK - RPEL
    thigh_L = LANK - LPEL

    # trunk plane normal per frame
    normal = np.cross(RPEL - midShoulder, LPEL - midShoulder)
    norm_sq = np.einsum("ij,ij->i", normal, normal)

    # project thigh vectors onto trunk plane
    proj_R = (
        thigh_R - (np.einsum("ij,ij->i", thigh_R, normal) / norm_sq)[:, None] * normal
    )
    proj_L = (
        thigh_L - (np.einsum("ij,ij->i", thigh_L, normal) / norm_sq)[:, None] * normal
    )

    # compute cosines
    cos_r = np.einsum("ij,ij->i", v_ref_r, proj_R) / (
        np.linalg.norm(v_ref_r, axis=1) * np.linalg.norm(proj_R, axis=1)
    )
    cos_l = np.einsum("ij,ij->i", v_ref_l, proj_L) / (
        np.linalg.norm(v_ref_l, axis=1) * np.linalg.norm(proj_L, axis=1)
    )

    # clamp and convert to angles, offset by -90°
    angle_r = np.degrees(np.arccos(np.clip(cos_r, -1, 1))) - 90
    angle_l = np.degrees(np.arccos(np.clip(cos_l, -1, 1))) - 90

    return angle_r, angle_l


def compute_hip_flexion_angles(
    RPEL: np.ndarray,
    LPEL: np.ndarray,
    RANK: np.ndarray,
    LANK: np.ndarray,
    RSHO: np.ndarray,
    LSHO: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    Compute hip flexion/extension angles in degrees for right and left legs.
    Midpoints of shoulders and pelvis are calculated internally.
    """
    # compute midpoints
    midShoulder = (RSHO + LSHO) / 2
    midPelvis = (RPEL + LPEL) / 2

    # trunk plane normal per frame
    normal = LPEL - RPEL

    # reference vector from pelvis midpoint to shoulder midpoint
    v_ref = midShoulder - midPelvis

    # thigh vectors
    thigh_R = RANK - RPEL
    thigh_L = LANK - LPEL

    # project thigh vectors onto trunk plane
    norm_sq = np.einsum("ij,ij->i", normal, normal)
    proj_R = (
        thigh_R - (np.einsum("ij,ij->i", thigh_R, normal) / norm_sq)[:, None] * normal
    )
    proj_L = (
        thigh_L - (np.einsum("ij,ij->i", thigh_L, normal) / norm_sq)[:, None] * normal
    )

    # compute cosines
    cos_r = np.einsum("ij,ij->i", v_ref, proj_R) / (
        np.linalg.norm(v_ref, axis=1) * np.linalg.norm(proj_R, axis=1)
    )
    cos_l = np.einsum("ij,ij->i", v_ref, proj_L) / (
        np.linalg.norm(v_ref, axis=1) * np.linalg.norm(proj_L, axis=1)
    )

    # clamp and convert to angles
    angle_r = np.degrees(np.arccos(np.clip(cos_r, -1, 1)))
    angle_l = np.degrees(np.arccos(np.clip(cos_l, -1, 1)))

    return angle_r, angle_l


def compute_pdf_hist_kde(data: np.ndarray, bin_width: float = 1.0):
    """
    Compute histogram-based PDF and KDE for data.
    Returns bin_centers, hist_pdf (density), kde_x, kde_pdf
    """
    min_val, max_val = np.min(data), np.max(data)
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    counts, edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    kde = gaussian_kde(
        data, bw_method=2 / bin_width
    )  # approximate bandwidth from bin_width
    kde_x = np.linspace(min_val, max_val, 200)
    kde_pdf = kde(kde_x)

    return bin_centers, counts, kde_x, kde_pdf


def plot_hist_kde(
    data_left: np.ndarray,
    data_right: np.ndarray,
    label_left: str,
    label_right: str,
    bin_width: float,
    plot_name: str,
    folder_outcome,
    color_left: str = "b",
    color_right: str = "r",
    bandwidth: float = None,
    plot=False,
):
    """
    Plot side-by-side histograms and KDE curves for two data series..
    """
    # Prepare data
    left = np.asarray(data_left, dtype=float)
    right = np.asarray(data_right, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]

    # Compute plot layout
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, data, label, color in [
        (axL, left, label_left, color_left),
        (axR, right, label_right, color_right),
    ]:
        # Histogram (counts normalized by total counts)
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
        counts, edges = np.histogram(data, bins=bins, density=False)
        pdf = counts / counts.sum()
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, pdf, width=bin_width, alpha=0.5, label="Hist")

        # KDE
        kde = gaussian_kde(data, bw_method=bandwidth)
        xi = np.linspace(data.min(), data.max(), 200)
        ax.plot(xi, kde(xi), "-", linewidth=2, color=color, label="KDE")

        ax.set_xlabel(label)
        ax.set_ylabel("Probability Density")
        ax.grid(True)
        ax.legend()

    if plot:
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(folder_outcome, f"{plot_name}_KDE.png"), dpi=300)
        plt.close()


def plot_cdf(
    series_dict: dict[str, np.ndarray],
    plot_name: str,
    folder_outcome,
    linewidth: float = 2.0,
    plot=True,
):
    """
    Plot empirical CDFs for one or more data series.
    """
    if plot:
        plt.figure()
        for label, data in series_dict.items():
            data = np.asarray(data, dtype=float)
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            sorted_data = np.sort(data)
            cdf_vals = np.arange(1, sorted_data.size + 1) / sorted_data.size
            plt.plot(sorted_data, cdf_vals, label=label, linewidth=linewidth)
        plt.xlabel("Angle (°)")
        plt.ylabel("CDF")
        plt.grid(True)
        plt.legend()
        if plot:
            plt.savefig(os.path.join(folder_outcome, f"{plot_name}_CDF.png"), dpi=300)
            # plt.show()
            plt.close()


def add_stats(
    row: dict, prefix: str, data: np.ndarray, bin_width: float = 1.0, ndigits: int = 2
):
    """
    Compute mean, std, skewness, kurtosis and histogram-based mode for data,
    round them, and store in row under keys:
      mean_<prefix>, std_<prefix>, skew_<prefix>, mode_<prefix>.
    """
    d = np.asarray(data, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        vals = dict(mean=np.nan, std=np.nan, skew=np.nan, kurt=np.nan, mode=np.nan)
    else:
        vals = {
            "mean": np.nanmean(d),
            "std": np.nanstd(d),
            "skew": skew(d, nan_policy="omit", bias=False),
        }
        # histogram-based mode
        bins = np.arange(d.min(), d.max() + bin_width, bin_width)
        counts, edges = np.histogram(d, bins=bins)
        centers = edges[:-1] + np.diff(edges) / 2
        vals["mode"] = centers[np.argmax(counts)] if counts.sum() > 0 else np.nan

    # round & store
    for k, v in vals.items():
        row[f"{k}_{prefix}"] = round(float(v), ndigits) if np.isfinite(v) else np.nan
