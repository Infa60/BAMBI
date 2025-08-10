from scipy.stats import gaussian_kde, skew, kurtosis
import os
from PythonFunction.Base_function import *

matplotlib.use("TkAgg")


def distance_foot_foot(
    LANK,
    RANK,
    LKNE,
    RKNE,
    threshold_ankle,
    threshold_knee,
    time_vector,
    folder_outcome,
    plot_name,
    bambi_indiv_interval,
    plot=False,
):
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
    foot_foot_interval = get_threshold_intervals(
        distance_foot_foot, threshold_ankle, "below"
    )
    knee_knee_interval = get_threshold_intervals(
        distance_knee_knee, threshold_knee, "above"
    )

    # 3. Get overlapping intervals (feet close + knees apart)
    plantar_plantar_contact_intervals = intersect_intervals(
        knee_knee_interval, foot_foot_interval
    )

    add_in_mat_file_interval(foot_foot_interval,"foot_foot_and_knee_knee", bambi_indiv_interval, distance_foot_foot, distance_knee_knee)


    # 4.1. Analyze timing and duration of foot foot contact intervals
    foot_foot_contact_outcomes = analyze_intervals_duration(
        foot_foot_interval,
        time_vector,
        distance_foot_foot,
        reverse_from_threshold=threshold_ankle,
    )

    # 4.2. Analyze timing and duration of plantar plantar contact intervals
    plantar_plantar_contact_outcomes = analyze_intervals_duration(
        plantar_plantar_contact_intervals, time_vector
    )

    # 5. Optional plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_foot_foot, label="Foot-to-Foot Distance")
        plt.plot(time_vector, distance_knee_knee, label="Knee-to-Knee Distance")
        plt.axhline(
            threshold_ankle, color="red", linestyle="--", label="Ankle Threshold"
        )
        plt.axhline(
            threshold_knee, color="green", linestyle="--", label="Knee Threshold"
        )
        for start, end in plantar_plantar_contact_intervals:
            plt.axvspan(time_vector[start], time_vector[end], color="orange", alpha=0.3)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (mm)")
        plt.title("Foot and Knee Distances Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_outcome, f"{plot_name}_foot_foot.png"), dpi=300)
        plt.close()

    # 6. Return summary dictionary
    return plantar_plantar_contact_outcomes, foot_foot_contact_outcomes


def distance_hand_hand(
    LWRA, RWRA, threshold, time_vector, folder_outcome, plot_name, bambi_indiv_interval, plot=False
):
    """
    Analyze hand-to-hand proximity events.

    Measures how often the two wrists come close together (below a distance threshold),
    and returns event count, total time, and durations. Optionally plots the distance over time.
    """

    # 1. Compute frame-by-frame Euclidean distance between the wrists
    distance_hand_hand = np.linalg.norm(LWRA - RWRA, axis=1)

    # 2. Detect intervals where hands are close (below threshold)
    hand_hand_interval = get_threshold_intervals(distance_hand_hand, threshold, "below")

    add_in_mat_file_interval(hand_hand_interval,"hand_hand", bambi_indiv_interval, distance_hand_hand)

    # 3. Analyze duration and count of close-contact events
    hand_hand_contact_outcomes = analyze_intervals_duration(
        hand_hand_interval,
        time_vector,
        distance_hand_hand,
        reverse_from_threshold=threshold,
    )

    # 4. Optional plot
    if plot:
        plot_time_series(
            time_vector,
            Hand_to_Hand_Distance=distance_hand_hand,
            Threshold=threshold,
            ylabel="Distance (mm)",
            title="Distance Between Left and Right Hands Over Time",
        )
        plt.savefig(os.path.join(folder_outcome, f"{plot_name}_hand_hand.png"), dpi=300)
        plt.close()

    # 5. Return summary dictionary
    return hand_hand_contact_outcomes


def distance_hand_foot(
    LANK,
    RANK,
    LWRA,
    RWRA,
    threshold,
    time_vector,
    folder_outcome,
    plot_name,
    bambi_indiv_interval,
    plot=False,
):
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
    handR_footR_interval = get_threshold_intervals(
        distance_handR_footR, threshold, "below"
    )
    handR_footL_interval = get_threshold_intervals(
        distance_handR_footL, threshold, "below"
    )
    handL_footR_interval = get_threshold_intervals(
        distance_handL_footR, threshold, "below"
    )
    handL_footL_interval = get_threshold_intervals(
        distance_handL_footL, threshold, "below"
    )

    ipsilateral_intervals = handR_footR_interval + handL_footL_interval
    contralateral_intervals = handR_footL_interval + handL_footR_interval

    add_in_mat_file_interval(handL_footL_interval,"handL_footL", bambi_indiv_interval, distance_handL_footL)
    add_in_mat_file_interval(handL_footR_interval,"handL_footR", bambi_indiv_interval, distance_handL_footR)
    add_in_mat_file_interval(handR_footR_interval,"handR_footR", bambi_indiv_interval, distance_handR_footR)
    add_in_mat_file_interval(handR_footL_interval,"handR_footL", bambi_indiv_interval, distance_handR_footL)


    # 3. Analyze duration and count of each contact type
    handR_footR_interval_contact_outcomes = analyze_intervals_duration(
        handR_footR_interval, time_vector
    )
    handR_footL_interval_contact_outcomes = analyze_intervals_duration(
        handR_footL_interval, time_vector
    )
    handL_footR_interval_contact_outcomes = analyze_intervals_duration(
        handL_footR_interval, time_vector
    )
    handL_footL_interval_contact_outcomes = analyze_intervals_duration(
        handL_footL_interval, time_vector
    )

    ipsilateral_contact_outcomes = analyze_intervals_duration(
        ipsilateral_intervals, time_vector
    )
    contralateral_contact_outcomes = analyze_intervals_duration(
        contralateral_intervals, time_vector
    )

    # 4. Optional plot of distances over time
    if plot:
        plot_time_series(
            time_vector,
            Right_Hand_Right_Foot=distance_handR_footR,
            Right_Hand_Left_Foot=distance_handR_footL,
            Left_Hand_Right_Foot=distance_handL_footR,
            Left_Hand_Left_Foot=distance_handL_footL,
            Threshold=threshold,
            ylabel="Distance (mm)",
            title="Hand-Foot Distances Over Time",
        )

        plt.savefig(os.path.join(folder_outcome, f"{plot_name}_hand_foot.png"), dpi=300)
        plt.close()

    # 5. Return summary of contact intervals for each pair
    return {
        "ipsilateral_contact_outcomes": ipsilateral_contact_outcomes,
        "contralateral_contact_outcomes": contralateral_contact_outcomes,
        "handR_footR_interval_contact": handR_footR_interval_contact_outcomes,
        "handR_footL_interval_contact": handR_footL_interval_contact_outcomes,
        "handL_footR_interval_contact": handL_footR_interval_contact_outcomes,
        "handL_footL_interval_contact": handL_footL_interval_contact_outcomes,
    }


def plot_mean_pdf_contact(
    outcomes_total,
    bambiID_list,
    plot_name,
    folder_save_path,
    field="durations_per_event",
    grid_min=None,
    grid_max=None,  # if None, auto-compute from data
    grid_points=500,
    show_std=True,
    all_line=True,
    bandwidth="scott",
):
    """
    Plot the average PDF of a given contact metric (durations or amplitudes)
    across subjects, save skewness/kurtosis to CSV, and save the figure.

    If grid_max is provided explicitly, the x-axis and PDF curves are
    inverted so that x=0 corresponds to *full* contact and x=grid_max
    corresponds to *start* of contact. If grid_max is None, it's taken
    as the max of all data and no inversion occurs.
    """
    if field not in ("durations_per_event", "amplitude_per_event"):
        raise ValueError("field must be 'durations_per_event' or 'amplitude_per_event'")

    os.makedirs(folder_save_path, exist_ok=True)

    explicit_max = grid_max is not None

    # auto-compute grid_max if needed
    if not explicit_max:
        all_vals = np.hstack([o.get(field, []) for o in outcomes_total], dtype=float)
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            raise ValueError(
                f"No valid data for field '{field}' – cannot determine grid_max."
            )
        grid_max = all_vals.max()

    explicit_min = grid_min is not None

    # auto-compute grid_min if needed
    if not explicit_min:
        all_vals = np.hstack([o.get(field, []) for o in outcomes_total], dtype=float)
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            raise ValueError(
                f"No valid data for field '{field}' – cannot determine grid_min."
            )
        grid_min = all_vals.min()

    # choose labels
    if field == "durations_per_event":
        metric_name = "Contact Durations"
        x_label = "Duration per event (s)"
        summary_lbl = "T_contact (s)"
    else:
        metric_name = "Contact Amplitudes"
        x_label = "Amplitude per event"
        summary_lbl = "Amplitude"

    # evaluation grid
    grid = np.linspace(grid_min, grid_max, grid_points)

    kdes = []
    stats = []
    counts = []
    sums = []

    for idx, outcome in enumerate(outcomes_total):
        vals = np.asarray(outcome.get(field, []), float)
        vals = vals[np.isfinite(vals)]

        counts.append(vals.size)
        sums.append(
            vals.sum()
            if field == "durations_per_event"
            else (vals.mean() if vals.size else 0.0)
        )

        if vals.size < 2:
            stats.append([bambiID_list[idx], np.nan, np.nan])
            continue

        kde = gaussian_kde(vals, bw_method=bandwidth)
        pdf_vals = kde(grid)
        kdes.append(pdf_vals)
        stats.append([bambiID_list[idx], skew(pdf_vals), kurtosis(pdf_vals)])

    if not kdes:
        print(f"Warning: no subject has ≥2 data points for '{field}'; skipping plot.")
        return

    kdes = np.array(kdes)
    mean_pdf = kdes.mean(axis=0)
    std_pdf = kdes.std(axis=0)
    lower = np.clip(mean_pdf - std_pdf, 0, None)
    upper = mean_pdf + std_pdf

    # invert axis & curves if grid_max was explicit
    # if explicit_max:
    #    grid      = (grid_max - grid)[::-1]
    #    mean_pdf  = mean_pdf[::-1]
    #    lower     = lower[::-1]
    #    upper     = upper[::-1]
    #    kdes      = kdes[:, ::-1]

    # append overall stats
    stats.append(["Total Mean", skew(mean_pdf), kurtosis(mean_pdf)])
    df = pd.DataFrame(stats, columns=["Subject", "Skewness", "Kurtosis"])
    csv_path = os.path.join(folder_save_path, f"{plot_name}_stats_on_KDE_PDF.csv")
    df.to_csv(csv_path, index=False)

    # global summaries
    mean_n = np.mean(counts)
    std_n = np.std(counts)
    mean_s = np.mean(sums)
    std_s = np.std(sums)

    # --- plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(grid, mean_pdf, "k-", lw=2, label="Mean PDF")
    if all_line:
        for subj_pdf in kdes:
            plt.plot(grid, subj_pdf, color="gray", alpha=0.25)
    if show_std:
        plt.fill_between(grid, lower, upper, color="black", alpha=0.3, label="±1 SD")

    summary = (
        f"Events: {mean_n:.1f} ± {std_n:.1f}\n"
        f"{summary_lbl}: {mean_s:.2f} ± {std_s:.2f}"
    )
    # plt.text(0.02, 0.965, summary, transform=plt.gca().transAxes,
    # va="top", ha="left", fontsize=11,
    # bbox=dict(facecolor="white", edgecolor="lightgrey",
    # boxstyle="round,pad=0.4", alpha=0.8))

    plt.title(f"Mean PDF of {metric_name} – {plot_name}")
    plt.xlabel(x_label)
    plt.ylabel("Probability Density")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(folder_save_path, f"{plot_name}_mean_pdf_{field}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Figure saved → {fig_path}")
    print(f"CSV saved    → {csv_path}")


def add_in_mat_file_interval(intervals, contact_name, store, distance1, distance2=None):
    """
    Store for each interval: start index, end index, mean(distance1), mean(distance2 or NaN).
    - intervals : list of (start, end) in Python convention [start, end) (0-based, end exclusive)
    - store     : dict to later pass to savemat
    """
    n = len(intervals)
    arr = np.full((n, 4), np.nan, dtype=float)  # columns: [start, end, mean_d1, mean_d2]

    for i, (s, e) in enumerate(intervals):
        m1 = float(np.mean(distance1[s:e])) if e > s else np.nan
        m2 = float(np.mean(distance2[s:e])) if (distance2 is not None and e > s) else np.nan

        arr[i, 0] = int(s)  # start index
        arr[i, 1] = int(e)  # end index
        arr[i, 2] = round(m1, 3)      # mean distance1
        arr[i, 3] = round(m2, 3)      # mean distance2 or NaN

    store[contact_name] = arr
    return store