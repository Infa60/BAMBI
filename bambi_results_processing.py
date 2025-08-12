import scipy.io
from scipy.io import savemat

from scipy.stats import skew
from collections import Counter
from PythonFunction.Ellipsoid import plot_ellipsoid_and_points_stickman
from PythonFunction.Leg_lift_adduct import (
    plot_combined_pdf,
    plot_mean_pdf_stat,
    ankle_high,
)
from PythonFunction.Kicking_function import *
from PythonFunction.Members_contact import *
from PythonFunction.Body_symmetry import *
from PythonFunction.Head_contact_orientation import *
from PythonFunction.Base_function import *
from PythonFunction.TEST import *
from PythonFunction.Leg_lift_adduct import *
from PythonFunction.Quantity_movement import *
from PythonFunction.Correlation import *

# Set matplotlib backend
matplotlib.use("TkAgg")

# Set path and load .mat file
path = "/Users/mathieubourgeois/Documents/BAMBI_Data"
outcome_path = f"{path}/Outcome_v2"
result_file = f"{path}/resultats_v2.mat"
anthropo_file = f"{path}/3_months_validity_and_reliability.csv"

data = scipy.io.loadmat(result_file)

# Access the structured results
results_struct = data["results"][0, 0]

foot_foot_threshold = 100
hand_hand_threshold = 100
ankle_high_treshold = 100
hand_mouth_threshold = 100
knee_knee_threshold = 300
hand_foot_threshold = 100


# Lists to collect data for global analysis
data_hand_hand_row = []
data_foot_foot_row = []
data_hand_foot_row = []
data_hand_mouth_row = []
data_leg_lift_row = []
data_leg_kick_row = []
data_leg_flex_add_row = []
data_ank_mouv_row = []
data_marker_trajectory_row = []
data_marker_velocity_row = []
data_marker_acceleration_row = []
data_marker_jerk_row = []
data_correlation_all_duration_row = []
data_correlation_union_row = []
data_correlation_intersection_row = []
data_wrist_mouv_row = []


hip_add_all = []
hip_flex_all = []
pdf_list = []
x_list = []
foot_outcomes_total = []
hand_outcomes_total = []
hand_foot_contra_outcomes_total = []
hand_foot_ipsi_outcomes_total = []
mouth_handR_outcomes_total = []
mouth_handL_outcomes_total = []
legR_lift_outcomes_total = []
legL_lift_outcomes_total = []

bambiID_list = results_struct.dtype.names  # Extract all Bambi IDs

hand_hand_path = os.path.join(outcome_path, "hand_hand")
foot_foot_path = os.path.join(outcome_path, "foot_foot")
hand_foot_path = os.path.join(outcome_path, "hand_foot")
hand_mouth_path = os.path.join(outcome_path, "hand_mouth")
leg_lift_path = os.path.join(outcome_path, "leg_lift")
leg_kick_path = os.path.join(outcome_path, "leg_kick")
leg_flex_add_path = os.path.join(outcome_path, "leg_flex_add")
ank_mouv_path = os.path.join(outcome_path, "ank_mouv")
marker_movement_path = os.path.join(outcome_path, "marker_movement")
correlation_path = os.path.join(outcome_path, "correlation")
wrist_mouv_path = os.path.join(outcome_path, "wrist_mouv")

mat_file_interval = {}


for folder in [
    hand_hand_path,
    foot_foot_path,
    hand_foot_path,
    hand_mouth_path,
    leg_lift_path,
    leg_kick_path,
    leg_flex_add_path,
    ank_mouv_path,
    marker_movement_path,
    correlation_path,
    wrist_mouv_path
]:
    os.makedirs(folder, exist_ok=True)

# Iterate through each Bambi ID
for i, bambiID in enumerate(results_struct.dtype.names):
    #if results_struct[bambiID]["marker_category"][0][0][0] != "full":
        #continue

    bambi_indiv_interval = {}

    # if bambiID != "BAMBI004_3M_Supine1_LH":
    #    continue
    print(f"{bambiID} is running")
    bambi_name = bambiID.split("_", 1)[0]

    laterality = results_struct[bambiID]["laterality"][0][0][0][0][0]

    time_duration = results_struct[bambiID]["time_duration"][0][0][0]
    while isinstance(time_duration, np.ndarray) and time_duration.ndim > 1:
        time_duration = time_duration[0]
    cycle_durations = np.diff(time_duration)
    freq = int(1 / cycle_durations[0])

    total_time_sec = time_duration[-1]
    total_time_min = total_time_sec / 60.0

    bambi_folder = os.path.join(outcome_path, "individual_plot", bambi_name)
    os.makedirs(bambi_folder, exist_ok=True)

    leg_length, shank_length = get_leg_and_tibia_length(anthropo_file, bambi_name)

    hand_hand_row = {}
    foot_foot_row = {}
    hand_foot_row = {}
    hand_mouth_row = {}
    leg_lift_row = {}
    leg_kick_row = {}
    leg_flex_add_row = {}
    ank_mouv_row = {}
    marker_trajectory_row = {}
    marker_velocity_row = {}
    marker_acceleration_row = {}
    marker_jerk_row = {}
    correlation_all_duration_row = {}
    correlation_union_row = {}
    correlation_intersection_row = {}
    wrist_mouv_row = {}

    all_rows = [
        hand_hand_row,
        foot_foot_row,
        hand_foot_row,
        hand_mouth_row,
        leg_lift_row,
        leg_kick_row,
        leg_flex_add_row,
        ank_mouv_row,
        marker_trajectory_row,
        marker_velocity_row,
        marker_acceleration_row,
        marker_jerk_row,
        correlation_all_duration_row,
        correlation_union_row,
        correlation_intersection_row,
        wrist_mouv_row
    ]

    # bambiID est défini juste avant
    for row in all_rows:
        row["bambiID"] = bambiID
        row["laterality"] = laterality
        row["Frequence"] = freq
        row["Total time (minute)"] = total_time_min

    ## Retrieve joint positions
    pos_ankle = results_struct[bambiID]["ankle_pos"][0, 0]
    RANK = results_struct[bambiID]["RANK"][0, 0]
    LANK = results_struct[bambiID]["LANK"][0, 0]
    RKNE = results_struct[bambiID]["RKNE"][0, 0]
    LKNE = results_struct[bambiID]["LKNE"][0, 0]
    LPEL = results_struct[bambiID]["LPEL"][0, 0]
    RPEL = results_struct[bambiID]["RPEL"][0, 0]
    LSHO = results_struct[bambiID]["LSHO"][0, 0]
    RSHO = results_struct[bambiID]["RSHO"][0, 0]
    LELB = results_struct[bambiID]["LELB"][0, 0]
    RELB = results_struct[bambiID]["RELB"][0, 0]
    LWRA = results_struct[bambiID]["LWRA"][0, 0]
    RWRA = results_struct[bambiID]["RWRA"][0, 0]

    if results_struct[bambiID]["marker_category"][0][0][0] == "full":

        CSHD = results_struct[bambiID]["CSHD"][0, 0]
        FSHD = results_struct[bambiID]["FSHD"][0, 0]
        LSHD = results_struct[bambiID]["LSHD"][0, 0]
        RSHD = results_struct[bambiID]["RSHD"][0, 0]

    else:
        CSHD = None
        FSHD = None
        LSHD = None
        RSHD = None

    RANK_global = results_struct[bambiID]["RANK_global_frame"][0, 0]
    LANK_global = results_struct[bambiID]["LANK_global_frame"][0, 0]

    ## Trajectory and velocity outside mean and std
    marker_for_trajectory_outside_mean_std = {
        "RWRA": RWRA,
        "LWRA": LWRA,
        "RANK": RANK,
        "LANK": LANK,
    }

    compute_area_outside_mean_std(
        markers=marker_for_trajectory_outside_mean_std,
        time=time_duration,
        freq=freq,
        row=marker_trajectory_row,
        win_mult=2,
        k=1,
        save_path=bambi_folder,
        plot_name=bambiID,
        data_type="Trajectory",
    )

    compute_area_outside_mean_std(
        markers=marker_for_trajectory_outside_mean_std,
        time=time_duration,
        freq=freq,
        row=marker_velocity_row,
        win_mult=2,
        k=1,
        save_path=bambi_folder,
        plot_name=bambiID,
        data_type="Velocity",
    )

    ## Canonical Correlation Analysis
    marker_for_correlations = {
        "RWRA–LANK": (RWRA, LANK),
        "RWRA–RANK": (RWRA, RANK),
        "LWRA–LANK": (LWRA, LANK),
        "LWRA–RANK": (LWRA, RANK),
        "RWRA–LWRA": (RWRA, LWRA),
        "LANK–RANK": (LANK, RANK),
    }

    add_canonical_correlations_stat(
        pairs=marker_for_correlations, ndigits=3, row=correlation_all_duration_row
    )

    ## Marker to analyse
    marker_to_velocity_compute = {
        "RWRA": RWRA,
        "LWRA": LWRA,
        "RANK": RANK,
        "LANK": LANK,
        "RKNE": RKNE,
        "LKNE": LKNE,
        "RELB": RELB,
        "LELB": LELB,
    }

    for marker_name, marker_xyz in marker_to_velocity_compute.items():

        marker_vel, marker_acc, marker_jerk = marker_pos_to_jerk(marker_xyz, cutoff=6, fs=freq)

        marker_outcome(
            marker_vel,
            row=marker_velocity_row,
            marker_name=marker_name,
            type_value="velocity",
        )

        marker_outcome(
            marker_acc,
            row=marker_acceleration_row,
            marker_name=marker_name,
            type_value="acceleration",
        )

        marker_outcome(
            marker_jerk,
            row=marker_jerk_row,
            marker_name=marker_name,
            type_value="jerk",
        )


    ## Leg adduction flexion
    hip_add_r, hip_add_l = compute_hip_adduction_angles(
        RPEL, LPEL, RANK, LANK, RSHO, LSHO
    )
    hip_flex_r, hip_flex_l = compute_hip_flexion_angles(
        RPEL, LPEL, RANK, LANK, RSHO, LSHO
    )

    # Append angle data to list for further PDF plotting
    if laterality == "right":
        hip_add_all.append(hip_add_r)
        hip_flex_all.append(hip_flex_r)
    else:
        hip_add_all.append(hip_add_l)
        hip_flex_all.append(hip_flex_l)

    plot_cdf(
        {"Right Hip Add": hip_add_r, "Left Hip Add": hip_add_l},
        plot_name=f"{bambiID}_Adduction",
        folder_outcome=bambi_folder,
        plot=True,
    )
    plot_hist_kde(
        data_left=hip_add_l,
        data_right=hip_add_r,
        label_left="Left Hip Add",
        label_right="Right Hip Add",
        bin_width=1.0,
        plot_name=f"{bambiID}_Adduction",
        folder_outcome=bambi_folder,
        color_left="b",
        color_right="r",
        bandwidth=2.0,
        plot=True,
    )

    plot_cdf(
        {"Right Hip Flex": hip_flex_r, "Left Hip Flex": hip_flex_l},
        plot_name=f"{bambiID}_Flexion",
        folder_outcome=bambi_folder,
        plot=True,
    )
    plot_hist_kde(
        data_left=hip_flex_l,
        data_right=hip_flex_r,
        label_left="Left Hip Flex",
        label_right="Right Hip FLex",
        bin_width=1.0,
        plot_name=f"{bambiID}_Flexion",
        folder_outcome=bambi_folder,
        color_left="b",
        color_right="r",
        bandwidth=2.0,
        plot=True,
    )

    add_stats(row=leg_flex_add_row, prefix="Right_add", data=hip_add_r)
    add_stats(row=leg_flex_add_row, prefix="Left_add", data=hip_add_l)

    add_stats(row=leg_flex_add_row, prefix="Right_flex", data=hip_flex_r)
    add_stats(row=leg_flex_add_row, prefix="Left_flex", data=hip_flex_l)

    ## Leg lifting
    right_lift_with_leg_extend, distance_pelv_ank_right = ankle_high(
        RANK,
        RPEL,
        RANK_global,
        time_vector=time_duration,
        leg_length=leg_length,
        high_threshold=ankle_high_treshold,
        max_flexion=45,
        folder_outcome=bambi_folder,
        plot_name=f"{bambiID}_Right",
        plot=True,
    )
    left_lift_with_leg_extend, distance_pelv_ank_left = ankle_high(
        LANK,
        LPEL,
        LANK_global,
        time_vector=time_duration,
        leg_length=leg_length,
        high_threshold=ankle_high_treshold,
        max_flexion=45,
        folder_outcome=bambi_folder,
        plot_name=f"{bambiID}_Left",
        plot=True,
    )
    add_contact_metrics(
        dest=leg_lift_row,
        prefix="right_lift_with_leg_extend",
        durations_per_event=right_lift_with_leg_extend["durations_per_event"],
        amplitude_per_event=right_lift_with_leg_extend["amplitude_per_event"],
    )
    add_contact_metrics(
        dest=leg_lift_row,
        prefix="left_lift_with_leg_extend",
        durations_per_event=left_lift_with_leg_extend["durations_per_event"],
        amplitude_per_event=left_lift_with_leg_extend["amplitude_per_event"],
    )
    legR_lift_outcomes_total.append(right_lift_with_leg_extend)
    legL_lift_outcomes_total.append(left_lift_with_leg_extend)

    # Calculate total distance traveled by the ankle
    distances_ankle_travel = np.linalg.norm(np.diff(pos_ankle, axis=0), axis=1)
    ank_mouv_row["distance_travel_ankle (mm)"] = np.sum(distances_ankle_travel)

    # Call function to plot ellipsoid and stickman, retrieve geometric stats
    stats_ankle_ellipsoid = plot_ellipsoid_and_points_stickman(
        pos_ankle,
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
        folder_save_path=ank_mouv_path,
        confidence_threshold=0.90,
        CSHD=CSHD,
        FSHD=FSHD,
        LSHD=LSHD,
        RSHD=RSHD,
        interactive=False,
        inside_point=False,
        outside_point=False,
    )

    # Save geometric and velocity distribution stats
    ank_mouv_row["num_points"] = stats_ankle_ellipsoid["num_points"]
    ank_mouv_row["num_enclosed"] = stats_ankle_ellipsoid["num_enclosed"]
    ank_mouv_row["percentage_enclosed"] = stats_ankle_ellipsoid["percentage_enclosed"]
    ank_mouv_row["volume_90 (cm3)"] = stats_ankle_ellipsoid["volume_90"]

    ## Hand to mouth contact
    if results_struct[bambiID]["marker_category"][0][0][0] == "full":

        R_hand_mouth_contact, L_hand_mouth_contact = distance_hand_mouth(
            LWRA,
            RWRA,
            CSHD,
            FSHD,
            LSHD,
            RSHD,
            threshold=hand_mouth_threshold,
            time_vector=time_duration,
            folder_outcome=bambi_folder,
            plot_name=f"{bambiID}_Right_left",
            bambi_indiv_interval=bambi_indiv_interval,
            plot=True,
        )
        add_contact_metrics(
            dest=hand_mouth_row,
            prefix="R_hand_mouth_contact",
            durations_per_event=R_hand_mouth_contact["durations_per_event"],
            amplitude_per_event=R_hand_mouth_contact["amplitude_per_event"],
        )
        add_contact_metrics(
            dest=hand_mouth_row,
            prefix="L_hand_mouth_contact",
            durations_per_event=L_hand_mouth_contact["durations_per_event"],
            amplitude_per_event=L_hand_mouth_contact["amplitude_per_event"],
        )
        mouth_handR_outcomes_total.append(R_hand_mouth_contact)
        mouth_handL_outcomes_total.append(L_hand_mouth_contact)

    ## Hand-hand contact
    hand_hand_contact = distance_hand_hand(
        LWRA,
        RWRA,
        threshold=hand_hand_threshold,
        time_vector=time_duration,
        folder_outcome=bambi_folder,
        plot_name=f"{bambiID}_Right_left",
        bambi_indiv_interval=bambi_indiv_interval,
        plot=True,
    )
    add_contact_metrics(
        dest=hand_hand_row,
        prefix="hand_hand_contact",
        durations_per_event=hand_hand_contact["durations_per_event"],
        amplitude_per_event=hand_hand_contact["amplitude_per_event"],
    )
    hand_outcomes_total.append(hand_hand_contact)

    ## Foot-foot contact
    plantar_plantar_contact_outcomes, foot_foot_contact_outcomes = distance_foot_foot(
        LANK,
        RANK,
        LKNE,
        RKNE,
        threshold_ankle=foot_foot_threshold,
        threshold_knee=knee_knee_threshold,
        time_vector=time_duration,
        folder_outcome=bambi_folder,
        plot_name=f"{bambiID}_Right_left",
        bambi_indiv_interval=bambi_indiv_interval,
        plot=True,
    )
    add_contact_metrics(
        dest=foot_foot_row,
        prefix="foot_foot_contact",
        durations_per_event=foot_foot_contact_outcomes["durations_per_event"],
        amplitude_per_event=foot_foot_contact_outcomes["amplitude_per_event"],
    )
    foot_outcomes_total.append(foot_foot_contact_outcomes)

    ## Hand-foot contact
    hand_foot_contact_outcomes = distance_hand_foot(
        LANK,
        RANK,
        LWRA,
        RWRA,
        threshold=hand_foot_threshold,
        time_vector=time_duration,
        folder_outcome=bambi_folder,
        plot_name=f"{bambiID}_Right_left",
        bambi_indiv_interval=bambi_indiv_interval,
        plot=True,
    )
    add_contact_metrics(
        dest=hand_foot_row,
        prefix="foot_hand_contact_contralateral",
        durations_per_event=hand_foot_contact_outcomes[
            "contralateral_contact_outcomes"
        ]["durations_per_event"],
    )
    add_contact_metrics(
        dest=hand_foot_row,
        prefix="foot_hand_contact_ipsilateral",
        durations_per_event=hand_foot_contact_outcomes["ipsilateral_contact_outcomes"][
            "durations_per_event"
        ],
    )
    hand_foot_contra_outcomes_total.append(
        hand_foot_contact_outcomes["contralateral_contact_outcomes"]
    )
    hand_foot_ipsi_outcomes_total.append(
        hand_foot_contact_outcomes["contralateral_contact_outcomes"]
    )

    ## Wrist ellipsoid
    stats_right_wrist_ellipsoid = plot_ellipsoid_and_points_stickman(
        RWRA,
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
        folder_save_path=f"{outcome_path}/Right_Wrist_Outcomes_plot",
        confidence_threshold=0.90,
        CSHD=CSHD,
        FSHD=FSHD,
        LSHD=LSHD,
        RSHD=RSHD,
        interactive=False,
        inside_point=False,
        outside_point=False,
    )

    # Save geometric and velocity distribution stats
    wrist_mouv_row["R_num_points"] = stats_right_wrist_ellipsoid["num_points"]
    wrist_mouv_row["R_num_enclosed"] = stats_right_wrist_ellipsoid["num_enclosed"]
    wrist_mouv_row["R_percentage_enclosed"] = stats_right_wrist_ellipsoid["percentage_enclosed"]
    wrist_mouv_row["R_volume_90 (cm3)"] = stats_right_wrist_ellipsoid["volume_90"]

    stats_left_wrist_ellipsoid = plot_ellipsoid_and_points_stickman(
        LWRA,
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
        folder_save_path=f"{outcome_path}/Left_Wrist_Outcomes_plot",
        confidence_threshold=0.90,
        CSHD=CSHD,
        FSHD=FSHD,
        LSHD=LSHD,
        RSHD=RSHD,
        interactive=False,
        inside_point=False,
        outside_point=False,
    )

    # Save geometric and velocity distribution stats
    wrist_mouv_row["L_num_points"] = stats_left_wrist_ellipsoid["num_points"]
    wrist_mouv_row["L_num_enclosed"] = stats_left_wrist_ellipsoid["num_enclosed"]
    wrist_mouv_row["L_percentage_enclosed"] = stats_left_wrist_ellipsoid["percentage_enclosed"]
    wrist_mouv_row["L_volume_90 (cm3)"] = stats_left_wrist_ellipsoid["volume_90"]

    ## Head rotation
    # head_rotation(CSHD, FSHD, LSHD, RSHD, LSHO, RSHO, LPEL, RPEL, threshold=(-5, 5), time_vector=time_duration, plot=False)

    ## Body symmetry
    # body_symmetry(LPEL, RPEL, LSHO, RSHO, 40, time_vector=time_duration, plot=False)

    ## Kicking
    kicking_cycle_outcomes_left, distance_kicking_left, kick_intervals_left = kicking(
        LPEL, LANK, time_duration, leg_length, LKNE, LPEL, LANK, freq, plot=False
    )
    mean_std_kicking_values_left = get_mean_and_std(kicking_cycle_outcomes_left)

    kicking_cycle_outcomes_right, distance_kicking_right, kick_intervals_right = (
        kicking(
            RPEL, RANK, time_duration, leg_length, RKNE, RPEL, RANK, freq, plot=False
        )
    )
    mean_std_kicking_values_right = get_mean_and_std(kicking_cycle_outcomes_right)

    knee_angle_right, hip_angle_right = synchro_hip_knee(
        time_duration, RPEL, RKNE, RSHO, RANK, plot=False
    )

    # plot_time_series(knee_angle_right, hip_angle_right=hip_angle_right)
    # plt.show()

    knee_angle_left, hip_angle_left = synchro_hip_knee(
        time_duration, LPEL, LKNE, LSHO, LANK, plot=False
    )

    mean_corr_right, std_corr_right, mean_lags_right, std_lags_right = (
        knee_hip_correlation_individual_segment(
            knee_angle_right, hip_angle_right, kick_intervals_right
        )
    )
    mean_corr_left, std_corr_left, mean_lags_left, std_lags_left = (
        knee_hip_correlation_individual_segment(
            knee_angle_left, hip_angle_left, kick_intervals_left
        )
    )

    # classification_results = classify_kicks(kick_intervals_right, kick_intervals_left, knee_angle_right, knee_angle_left, fs=freq)

    # type_counts = Counter(r['type'] for r in classification_results)
    # for k, v in type_counts.items():
    # print(f"{k}: {v}")

    # plot_kick_classification_with_bars(knee_angle_right, knee_angle_left, classification_results, freq)

    # Save List
    # labeled_kicks = []
    # for (start, end) in kick_intervals_right:
    # label_and_save_kick(knee_angle_right, knee_angle_left, start, end, 'right', labeled_kicks)
    # plot_kick_classification_with_bars(knee_angle_right, knee_angle_left, labeled_kicks, freq)

    leg_kick_row["number_of_right_kick"] = len(kick_intervals_right)

    leg_kick_row["mean_flexion_amplitude_right"] = mean_std_kicking_values_right["mean"]["flexion_amplitude"]
    leg_kick_row["std_flexion_amplitude_right"] = mean_std_kicking_values_right["std"]["flexion_amplitude"]
    leg_kick_row["mean_extension_amplitude_right"] = mean_std_kicking_values_right["mean"]["extension_amplitude"]
    leg_kick_row["std_extension_amplitude_right"] = mean_std_kicking_values_right["std"]["extension_amplitude"]
    leg_kick_row["mean_duration_right"] = mean_std_kicking_values_right["mean"]["duration"]
    leg_kick_row["std_duration_right"] = mean_std_kicking_values_right["std"]["duration"]
    leg_kick_row["mean_steepness_right"] = mean_std_kicking_values_right["mean"]["steepness"]
    leg_kick_row["std_steepness_right"] = mean_std_kicking_values_right["std"]["steepness"]
    leg_kick_row["mean_flexion_speed_right"] = mean_std_kicking_values_right["mean"]["flexion_speed"]
    leg_kick_row["std_flexion_speed_right"] = mean_std_kicking_values_right["std"]["flexion_speed"]
    leg_kick_row["mean_extension_speed_right"] = mean_std_kicking_values_right["mean"]["extension_speed"]
    leg_kick_row["std_extension_speed_right"] = mean_std_kicking_values_right["std"]["extension_speed"]

    leg_kick_row["number_of_left_kick"] = len(kick_intervals_left)

    leg_kick_row["mean_flexion_amplitude_left"] = mean_std_kicking_values_left["mean"]["flexion_amplitude"]
    leg_kick_row["std_flexion_amplitude_left"] = mean_std_kicking_values_left["std"]["flexion_amplitude"]
    leg_kick_row["mean_extension_amplitude_left"] = mean_std_kicking_values_left["mean"]["extension_amplitude"]
    leg_kick_row["std_extension_amplitude_left"] = mean_std_kicking_values_left["std"]["extension_amplitude"]
    leg_kick_row["mean_duration_left"] = mean_std_kicking_values_left["mean"]["duration"]
    leg_kick_row["std_duration_left"] = mean_std_kicking_values_left["std"]["duration"]
    leg_kick_row["mean_steepness_left"] = mean_std_kicking_values_left["mean"]["steepness"]
    leg_kick_row["std_steepness_left"] = mean_std_kicking_values_left["std"]["steepness"]
    leg_kick_row["mean_flexion_speed_left"] = mean_std_kicking_values_left["mean"]["flexion_speed"]
    leg_kick_row["std_flexion_speed_left"] = mean_std_kicking_values_left["std"]["flexion_speed"]
    leg_kick_row["mean_extension_speed_left"] = mean_std_kicking_values_left["mean"]["extension_speed"]
    leg_kick_row["std_extension_speed_left"] = mean_std_kicking_values_left["std"]["extension_speed"]

    leg_kick_row["mean_corr_right"] = mean_corr_right
    leg_kick_row["std_corr_right"] = std_corr_right
    leg_kick_row["mean_lags_right"] = mean_lags_right
    leg_kick_row["std_lags_right"] = std_lags_right

    leg_kick_row["mean_corr_left"] = mean_corr_left
    leg_kick_row["std_corr_left"] = std_corr_left
    leg_kick_row["mean_lags_left"] = mean_lags_left
    leg_kick_row["std_lags_left"] = std_lags_left

    ## Common movement extremities -- interval intersection

    interval_movement_common_intersection_second = plot_multi_markers_speed_color(
        time=time_duration,
        fs=freq,
        thr=0.15,
        gap_tol=0.5,
        cutoff=6,
        show_common="intersection",
        save_path=bambi_folder,
        bambiID=bambiID,
        RANK=RANK,
        LANK=LANK,
        RKNE=RKNE,
        LKNE=LKNE,
    )
    interval_movement_common_intersection_frame = seconds_to_frames(
        interval_movement_common_intersection_second, freq
    )

    ## Common movement extremities -- interval union

    interval_movement_common_union_second = plot_multi_markers_speed_color(
        time=time_duration,
        fs=freq,
        thr=0.15,
        gap_tol=0.5,
        cutoff=6,
        show_common="union",
        save_path=bambi_folder,
        bambiID=bambiID,
        RANK=RANK,
        LANK=LANK,
        RWRA=RWRA,
        LKNE=LWRA,
    )
    interval_movement_common_union_frame = seconds_to_frames(
        interval_movement_common_union_second, freq
    )

    ## Correlation hip/knee angle based on interval intersection

    corr_right, p_corr_right, corr_right_lag_s = signal_correlation_concatenate_segments(
            knee_angle_right, hip_angle_right, interval_movement_common_intersection_frame, freq
    )
    corr_left, p_corr_left, corr_left_lag_s = signal_correlation_concatenate_segments(
        knee_angle_left, hip_angle_left, interval_movement_common_intersection_frame, freq
    )

    correlation_intersection_row["corr_right_hip_knee_movement"] = corr_left
    correlation_intersection_row["p_value_right_hip_knee_movement"] = p_corr_right
    correlation_intersection_row["lags_right_hip_knee_movement"] = corr_right_lag_s

    correlation_intersection_row["corr_left_hip_knee_movement"] = corr_left
    correlation_intersection_row["p_value_left_hip_knee_movement"] = p_corr_left
    correlation_intersection_row["lags_left_hip_knee_movement"] = corr_left_lag_s



    ## Correlation beteween marker velocity/acceleration/jerk during interval activation -- union

    for usage in ("velocity", "acceleration", "jerk"):
        add_correlations_stat(
            marker_for_correlations,
            fs=freq,
            ndigits=2,
            row=correlation_union_row,
            usage=usage,
            intervals=interval_movement_common_union_frame,
            method="union"
        )

        add_correlations_stat(
            marker_for_correlations,
            fs=freq,
            ndigits=2,
            row=correlation_intersection_row,
            usage=usage,
            intervals=interval_movement_common_intersection_frame,
            method="intersection"
        )

        add_correlations_stat(
            marker_for_correlations,
            fs=freq,
            ndigits=2,
            row=correlation_all_duration_row,
            usage=usage,
            intervals=None,
        )

    # Add the row to the list
    data_hand_hand_row.append(hand_hand_row)
    data_foot_foot_row.append(foot_foot_row)
    data_hand_foot_row.append(hand_foot_row)
    data_hand_mouth_row.append(hand_mouth_row)
    data_leg_lift_row.append(leg_lift_row)
    data_leg_kick_row.append(leg_kick_row)
    data_leg_flex_add_row.append(leg_flex_add_row)
    data_ank_mouv_row.append(ank_mouv_row)
    data_marker_trajectory_row.append(marker_trajectory_row)
    data_marker_velocity_row.append(marker_velocity_row)
    data_marker_acceleration_row.append(marker_acceleration_row)
    data_marker_jerk_row.append(marker_jerk_row)
    data_correlation_all_duration_row.append(correlation_all_duration_row)
    data_correlation_union_row.append(correlation_union_row)
    data_correlation_intersection_row.append(correlation_intersection_row)
    data_wrist_mouv_row.append(wrist_mouv_row)

    mat_file_interval[bambiID] = bambi_indiv_interval

mat_interval_path = os.path.join(outcome_path, 'interval_outcomes.mat')

savemat(mat_interval_path, mat_file_interval)

## Foot foot plot
plot_mean_pdf_contact (foot_outcomes_total, bambiID_list, 'foot_foot', foot_foot_path, field="durations_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)
plot_mean_pdf_contact (foot_outcomes_total, bambiID_list, 'foot_foot', foot_foot_path, field="amplitude_per_event",
        grid_min=None, grid_max=foot_foot_threshold, grid_points=500)

## Hand hand plot
plot_mean_pdf_contact (hand_outcomes_total, bambiID_list, 'hand_hand', hand_hand_path, field="durations_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)
plot_mean_pdf_contact (hand_outcomes_total, bambiID_list, 'hand_hand', hand_hand_path, field="amplitude_per_event",
        grid_min=None, grid_max=hand_hand_threshold, grid_points=500)

## Hand mouth plot
plot_mean_pdf_contact (mouth_handR_outcomes_total, bambiID_list, 'mouth_hand_R', hand_mouth_path, field="durations_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)
plot_mean_pdf_contact (mouth_handR_outcomes_total, bambiID_list, 'mouth_hand_R', hand_mouth_path, field="amplitude_per_event",
        grid_min=None, grid_max=hand_mouth_threshold, grid_points=500)
plot_mean_pdf_contact (mouth_handL_outcomes_total, bambiID_list, 'mouth_hand_L', hand_mouth_path, field="durations_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)
plot_mean_pdf_contact (mouth_handL_outcomes_total, bambiID_list, 'mouth_hand_L', hand_mouth_path, field="amplitude_per_event",
        grid_min=None, grid_max=hand_mouth_threshold, grid_points=500)

## Leg lift plot
plot_mean_pdf_contact (legR_lift_outcomes_total, bambiID_list, 'leg_lift_R', leg_lift_path, field="durations_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)
plot_mean_pdf_contact (legR_lift_outcomes_total, bambiID_list, 'leg_lift_R', leg_lift_path, field="amplitude_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)
plot_mean_pdf_contact (legL_lift_outcomes_total, bambiID_list, 'leg_lift_L', leg_lift_path, field="durations_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)
plot_mean_pdf_contact (legL_lift_outcomes_total, bambiID_list, 'leg_lift_L', leg_lift_path, field="amplitude_per_event",
        grid_min=0.0, grid_max=None, grid_points=500)

## Hand foot plot
plot_mean_pdf_contact (hand_foot_contra_outcomes_total, bambiID_list, 'hand_foot_contra', hand_foot_path, field="durations_per_event",
        grid_min=0.0, grid_max=hand_foot_threshold, grid_points=500)
plot_mean_pdf_contact (hand_foot_ipsi_outcomes_total, bambiID_list, 'hand_foot_ipsi', hand_foot_path, field="durations_per_event",
        grid_min=0.0, grid_max=hand_foot_threshold, grid_points=500)


data_map = {
    "hand_hand":    (data_hand_hand_row,    hand_hand_path),
    "foot_foot":    (data_foot_foot_row,    foot_foot_path),
    "hand_foot":    (data_hand_foot_row,    hand_foot_path),
    "hand_mouth":   (data_hand_mouth_row,   hand_mouth_path),
    "leg_lift":     (data_leg_lift_row,     leg_lift_path),
    "leg_kick":     (data_leg_kick_row,     leg_kick_path),
    "leg_flex_add": (data_leg_flex_add_row, leg_flex_add_path),
    "ank_mouv":     (data_ank_mouv_row,     ank_mouv_path),
    "marker_trajectory": (data_marker_trajectory_row, marker_movement_path),
    "marker_velocity": (data_marker_velocity_row, marker_movement_path),
    "marker_acceleration": (data_marker_acceleration_row, marker_movement_path),
    "marker_jerk": (data_marker_jerk_row, marker_movement_path),
    "correlation_all_duration": (data_correlation_all_duration_row, correlation_path),
    "correlation_intersection": (data_correlation_intersection_row, correlation_path),
    "correlation_union": (data_correlation_union_row, correlation_path),
    "wrist_mouv": (data_wrist_mouv_row, wrist_mouv_path),

}

# 5) Write one CSV per folder
for name, (rows, folder) in data_map.items():
    if not rows:
        print(f"⚠️  `{name}` list is empty → skipping")
        continue
    df = pd.DataFrame(rows)
    csv_file = os.path.join(folder, f"{name}_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"✅  Saved `{name}` results to {csv_file}")


# Plot PDFs and mean statistics for hip adduction-abduction
plot_combined_pdf(
    hip_add_all,
    "Adduction-Abduction",
    folder_save_path=leg_flex_add_path,
    plot_save=False,
)
plot_mean_pdf_stat(
    hip_add_all,
    bambiID_list,
    "Adduction-Abduction",
    folder_save_path=leg_flex_add_path,
    grid_min=-70,
    grid_max=45,
    grid_points=500,
    plot_save=True,
)

# Plot PDFs and mean statistics for hip flexion-extension
plot_combined_pdf(
    hip_flex_all,
    "Flexion-Extension",
    folder_save_path=leg_flex_add_path,
    plot_save=False,
)
plot_mean_pdf_stat(
    hip_flex_all,
    bambiID_list,
    "Flexion-Extension",
    folder_save_path=leg_flex_add_path,
    grid_min=120,
    grid_max=190,
    grid_points=500,
    plot_save=True,
)
