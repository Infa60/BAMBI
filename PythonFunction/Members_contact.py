from scipy.stats import gaussian_kde, skew, kurtosis
import os
from PythonFunction.Base_function import *
matplotlib.use("TkAgg")


def distance_foot_foot(LANK, RANK, LKNE, RKNE, threshold_ankle, threshold_knee, time_vector, plot=False):
    """
    Analyze foot-to-foot contact events when knees are sufficiently apart.

    Measures when both feet come close together (ankle distance below threshold),
    while knees are separated (knee distance above threshold).
    Returns the number of events, total time, and per-event durations.
    Optionally plots both distance signals over time.
    """

    # 1. Compute frame-by-frame distances
    distance_foot_foot = np.linalg.norm(LANK - RANK, axis=1)
    distance_knee_knee = np.linalg.norm(LKNE - RKNE, axis=1)

    # 2. Detect contact and separation intervals
    foot_foot_interval = get_threshold_intervals(distance_foot_foot, threshold_ankle, "below")
    knee_knee_interval = get_threshold_intervals(distance_knee_knee, threshold_knee, "above")

    # 3. Get overlapping intervals (feet close + knees apart)
    plantar_plantar_contact_intervals = intersect_intervals(knee_knee_interval, foot_foot_interval)

    # 4.1. Analyze timing and duration of foot foot contact intervals
    foot_foot_contact_outcomes = analyze_intervals_duration(foot_foot_interval, time_vector, distance_foot_foot)

    # 4.2. Analyze timing and duration of plantar plantar contact intervals
    plantar_plantar_contact_outcomes = analyze_intervals_duration(plantar_plantar_contact_intervals, time_vector)

    # 5. Optional plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_foot_foot, label="Foot-to-Foot Distance")
        plt.plot(time_vector, distance_knee_knee, label="Knee-to-Knee Distance")
        plt.axhline(threshold_ankle, color='red', linestyle='--', label="Ankle Threshold")
        plt.axhline(threshold_knee, color='green', linestyle='--', label="Knee Threshold")
        for start, end in plantar_plantar_contact_intervals:
            plt.axvspan(time_vector[start], time_vector[end], color='orange', alpha=0.3)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (mm)")
        plt.title("Foot and Knee Distances Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 6. Return summary dictionary
    return plantar_plantar_contact_outcomes, foot_foot_contact_outcomes

def distance_hand_hand(LWRA, RWRA, threshold, time_vector, plot=False):
    """
    Analyze hand-to-hand proximity events.

    Measures how often the two wrists come close together (below a distance threshold),
    and returns event count, total time, and durations. Optionally plots the distance over time.
    """

    # 1. Compute frame-by-frame Euclidean distance between the wrists
    distance_hand_hand = np.linalg.norm(LWRA - RWRA, axis=1)

    # 2. Detect intervals where hands are close (below threshold)
    hand_hand_interval = get_threshold_intervals(distance_hand_hand, threshold, "below")

    # 3. Analyze duration and count of close-contact events
    hand_hand_contact_outcomes = analyze_intervals_duration(hand_hand_interval, time_vector, distance_hand_hand)

    # 4. Optional plot
    if plot:
        plot_time_series(time_vector, Hand_to_Hand_Distance=distance_hand_hand, Threshold=threshold,
                         ylabel="Distance (mm)", title="Distance Between Left and Right Hands Over Time")

    # 5. Return summary dictionary
    return hand_hand_contact_outcomes



def distance_hand_foot(LANK, RANK, LWRA, RWRA, threshold, time_vector, plot=False):
    """
    Analyze hand-foot proximity events.

    Measures how often each hand comes close to each foot, based on a distance threshold.
    Returns the number of events, total time in contact, and duration of each event.
    Optionally plots all four hand-foot distance signals.
    """

    # 1. Compute distances between each hand and each foot
    distance_handR_footR = np.linalg.norm(RWRA - RANK, axis=1)
    distance_handR_footL = np.linalg.norm(RWRA - LANK, axis=1)
    distance_handL_footR = np.linalg.norm(LWRA - RANK, axis=1)
    distance_handL_footL = np.linalg.norm(LWRA - LANK, axis=1)

    # 2. Find intervals where distance is below the threshold (contact)
    handR_footR_interval = get_threshold_intervals(distance_handR_footR, threshold, "below")
    handR_footL_interval = get_threshold_intervals(distance_handR_footL, threshold, "below")
    handL_footR_interval = get_threshold_intervals(distance_handL_footR, threshold, "below")
    handL_footL_interval = get_threshold_intervals(distance_handL_footL, threshold, "below")

    ipsilateral_intervals = handR_footR_interval + handL_footL_interval
    contralateral_intervals = handR_footL_interval + handL_footR_interval

    # 3. Analyze duration and count of each contact type
    handR_footR_interval_contact_outcomes = analyze_intervals_duration(handR_footR_interval, time_vector)
    handR_footL_interval_contact_outcomes = analyze_intervals_duration(handR_footL_interval, time_vector)
    handL_footR_interval_contact_outcomes = analyze_intervals_duration(handL_footR_interval, time_vector)
    handL_footL_interval_contact_outcomes = analyze_intervals_duration(handL_footL_interval, time_vector)

    ipsilateral_contact_outcomes = analyze_intervals_duration(ipsilateral_intervals, time_vector)
    contralateral_contact_outcomes = analyze_intervals_duration(contralateral_intervals, time_vector)


    # 4. Optional plot of distances over time
    if plot:
        plot_time_series(time_vector, Right_Hand_Right_Foot=distance_handR_footR,
                         Right_Hand_Left_Foot=distance_handR_footL, Left_Hand_Right_Foot = distance_handL_footR,
                         Left_Hand_Left_Foot = distance_handL_footL, Threshold=threshold, ylabel="Distance (mm)",
                         title="Hand-Foot Distances Over Time")

    # 5. Return summary of contact intervals for each pair
    return {
        'ipsilateral_contact_outcomes': ipsilateral_contact_outcomes,
        'contralateral_contact_outcomes': contralateral_contact_outcomes,
        'handR_footR_interval_contact': handR_footR_interval_contact_outcomes,
        'handR_footL_interval_contact': handR_footL_interval_contact_outcomes,
        'handL_footR_interval_contact': handL_footR_interval_contact_outcomes,
        'handL_footL_interval_contact': handL_footL_interval_contact_outcomes
    }


def plot_mean_pdf_contact(
        outcomes_total,
        bambiID_list,
        plot_name,
        folder_save_path,
        grid_min=0.0,
        grid_max=6.0,
        grid_points=500,
        show_std=True,
        all_line=True,
        bandwidth='scott',
):
    """
    Trace la PDF moyenne des durées de contact pied-pied sur un ensemble de bébés
    et enregistre des statistiques (skewness, kurtosis) dans un CSV.
    """

    # Grille d'évaluation
    grid = np.linspace(grid_min, grid_max, grid_points)

    kdes = []
    stats_data = []
    n_events_list = []
    time_contact_list = []

    for idx, outcome in enumerate(outcomes_total):
        durations = np.asarray(outcome["durations_per_event"], dtype=float)

        # ---------- cas sans événement ----------
        if durations.size == 0:
            # 1) on enregistre quand même des stats "vides" pour le CSV
            stats_data.append([bambiID_list[idx], np.nan, np.nan])

            # 2) on comptabilise le nombre d’événements (=0) et le temps de contact (=0)
            n_events_list.append(0)
            time_contact_list.append(0.0)

            # 3) pas de KDE → on passe au suivant
            continue

        kde = gaussian_kde(durations, bw_method=bandwidth)
        kde_values = kde(grid)
        kdes.append(kde_values)

        # Skew & kurtosis sur la PDF estimée
        stats_data.append([
            bambiID_list[idx],
            skew(kde_values),
            kurtosis(kde_values)
        ])

        # Statistiques descriptives additionnelles
        n_events_list.append(outcome["number_of_event"])
        time_contact_list.append(outcome["time_in_contact"])

    # ---------- PDF moyenne ----------
    kdes = np.array(kdes)
    if len(kdes) == 0:
        raise ValueError("Aucun sujet n’a de foot-foot contact ; rien à tracer.")

    mean_pdf = kdes.mean(axis=0)
    std_pdf = kdes.std(axis=0)
    lower_bound = np.clip(mean_pdf - std_pdf, 0, None)
    upper_bound = mean_pdf + std_pdf

    # Skew & kurtosis de la PDF moyenne
    stats_data.append(["Total Mean", skew(mean_pdf), kurtosis(mean_pdf)])

    stats_df = pd.DataFrame(stats_data,
                            columns=["Subject", "Skewness", "Kurtosis"])

    # ---------- Sauvegarde CSV ----------
    csv_path = os.path.join(folder_save_path,
                            f"{plot_name}_stats_on_KDE_PDF.csv")
    stats_df.to_csv(csv_path, index=False)

    # ---------- Stats globales "brutes" ----------
    mean_n_events = np.mean(n_events_list)
    std_n_events = np.std(n_events_list)
    mean_time_cont = np.mean(time_contact_list)
    std_time_cont = np.std(time_contact_list)

    # ---------- Plot ----------
    plt.figure(figsize=(8, 5))
    plt.plot(grid, mean_pdf, color="black", linewidth=2, label="Mean PDF")

    if all_line:
        for kde in kdes:
            plt.plot(grid, kde, color="gray", alpha=0.25)

    if show_std:
        plt.fill_between(grid, lower_bound, upper_bound,
                         color="black", alpha=0.3, label="±1 SD")

    # Boîte récapitulative
    text_box = (
        f"Events : {mean_n_events:.1f} ± {std_n_events:.1f}\n"
        f"T_contact : {mean_time_cont:.2f} ± {std_time_cont:.2f} s"
    )
    plt.text(
        0.02, 0.965, text_box,
        transform=plt.gca().transAxes,
        va="top", ha="center", fontsize=11,
        bbox=dict(facecolor="white",
                  edgecolor="lightgrey",
                  boxstyle="round,pad=0.4",
                  alpha=0.8)
    )

    plt.title(f"Mean PDF of Foot-Foot Contact Durations – {plot_name}")
    plt.xlabel("Duration per event (s)")
    plt.ylabel("Probability Density")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(folder_save_path,
                            f"{plot_name}_mean_pdf_durations.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()