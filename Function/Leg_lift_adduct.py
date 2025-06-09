import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import skew, kurtosis
import pandas as pd
from Function.Base_function import resample_size, get_threshold_intervals, intersect_intervals, analyze_intervals_duration

matplotlib.use("TkAgg")

def plot_combined_pdf(
    hip_add_all, plot_name, folder_save_path, bins=50, color="lightcoral"
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
):
    """
        This function computes and plots the average Kernel Density Estimate (KDE)
        of hip adduction/abduction angles across multiple subjects ("bambis"). It also
        computes basic statistics (skewness and kurtosis) on the KDEs and saves them
        in a CSV file. The final plot includes the mean PDF and optionally displays
        individual subject curves and a ±1 standard deviation shaded region.

    Parameters:
        hip_add_all : list of lists -- Hip angles over time for each subject (one list per subject).

        bambiID_list : list of str -- Subject identifiers, in the same order as hip_add_all.

        plot_name : Base name for the output files (plot and CSV).

        folder_save_path : Folder path to save the outputs.

        grid_min, grid_max : Range of values for KDE evaluation (default: -90 to 90).

        grid_points : Number of points in the KDE grid.

        show_std : If True, displays ±1 standard deviation shading.

        all_line : If True, overlays all individual PDFs (default: True).

        target_length : Resampling length for all signals before KDE.
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
    plt.savefig(save_path, dpi=300)
    plt.close()


def ankle_high(ankle_marker, pelvis_marker, time_vector, leg_length, high_threshold, max_flexion, plot = False):

    thigh_proportion = 55
    thigh_length = leg_length * thigh_proportion/100

    range_extension = np.sqrt(thigh_length**2 + (leg_length-thigh_length)**2 - 2 * thigh_length * (leg_length-thigh_length) * np.cos(np.radians(180-max_flexion)))

    # Compute the Euclidean distance between pelvis and ankle at each time frame
    distance_pelv_ank = np.linalg.norm(pelvis_marker - ankle_marker, axis=1)

    close_to_max_extension_interval = get_threshold_intervals(distance_pelv_ank, range_extension, mode="above")

    ankle_in_elevation_interval = get_threshold_intervals(ankle_marker[:,2], high_threshold, mode="above")

    common_intervals = intersect_intervals(close_to_max_extension_interval, ankle_in_elevation_interval)

    if plot == True:
        plt.figure(figsize=(12, 4))
        plt.plot(time_vector, ankle_marker[:,2], label="Ankle", color='blue')
        plt.plot(time_vector, distance_pelv_ank, label="Distance Pelvis Ankle", color='red')

        for start, end in common_intervals:
            plt.axvspan(time_vector[start], time_vector[end], color='orange', alpha=0.3)

        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    lift_with_leg_extend = analyze_intervals_duration(common_intervals, time_vector)

    return lift_with_leg_extend