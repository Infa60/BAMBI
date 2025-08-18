import scipy.io

from PythonFunction.TEST import *
from collections import defaultdict

from PythonFunction.Correlation import *
from PythonFunction.Concatenate_supine_bambi import *




# Set matplotlib backend
# matplotlib.use("TkAgg")

# Set path and load .mat file
path = "/Users/mathieubourgeois/Documents/BAMBI_Data"

anthropo_file = f"{path}/3_months_validity_and_reliability.csv"

outcome_type = "no"
children_type = "TD"

if outcome_type == "ESMAC":
    point_of_vue = True
    outcome_path = os.path.join(path, "Outcome_v2bis_ESMAC")
    result_file = os.path.join(path, "resultats_v2_ESMAC.mat")
    ankle_high_distance_mean = True
    ellipsoid_size_extract = True
elif children_type == "TD":
    point_of_vue = False
    outcome_path = os.path.join(path, "Outcome_v2bis_TD")
    result_file = os.path.join(path, "resultats_v2_TD.mat")
    ankle_high_distance_mean = False
    ellipsoid_size_extract = False
    kinematics_by_baby = defaultdict(dict)

elif children_type == "HR":
    point_of_vue = False
    outcome_path = os.path.join(path, "Outcome_v2_HR")
    result_file = os.path.join(path, "resultats_v2_HR.mat")
    ankle_high_distance_mean = False
    ellipsoid_size_extract = False
    kinematics_by_baby = defaultdict(dict)

else:
    raise ValueError(f"Unknown outcome_type {outcome_type} or children_type: {children_type}")

data = scipy.io.loadmat(result_file)

# Access the structured results
results_struct = data["results"][0, 0]



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
data_marker_concatenate_per_bambi_velocity_row = []
data_marker_concatenate_per_supine_velocity_row = []
data_marker_concatenate_per_bambi_acceleration_row = []
data_marker_concatenate_per_supine_acceleration_row = []
data_marker_concatenate_per_bambi_jerk_row = []
data_marker_concatenate_per_supine_jerk_row = []

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
concatenate_per_bambi_path = os.path.join(outcome_path, "concatenate_per_bambi")
concatenate_per_supine_path = os.path.join(outcome_path, "concatenate_per_supine")

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
    wrist_mouv_path,
    concatenate_per_bambi_path,
    concatenate_per_supine_path,
]:
    os.makedirs(folder, exist_ok=True)


# Iterate through each Bambi ID
for i, bambiID in enumerate(results_struct.dtype.names):
    #if results_struct[bambiID]["marker_category"][0][0][0] != "full":
        #continue

    bambi_indiv_interval = {}
    kinematics_by_marker = {}
    #if bambiID != "BAMBI043_3M_Supine1_MC":
       #continue

    garder = {
        "BAMBI050_3M_Supine1_cropped_LH",
        "BAMBI050_3M_Supine3_cropped_LH",
        "BAMBI051_3M_Supine1_cropped_LH_A",
        "BAMBI051_3M_Supine1_cropped_LH_B",
        "BAMBI051_3M_Supine1_cropped_LH_C",
        "BAMBI051_3M_Supine2_cropped_LH_A",
        "BAMBI051_3M_Supine2_cropped_LH_B",
        "BAMBI051_3M_Supine2_cropped_LH_C",
    }
    if bambiID not in garder:
        continue

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

        kinematics_by_marker[marker_name] = {
            "velocity": marker_vel,
            "acceleration": marker_acc,
            "jerk": marker_jerk,
        }




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
    kinematics_by_baby[bambiID] = kinematics_by_marker


bambi_unique_ID = sorted({ run_id.split('_', 1)[0] for run_id in kinematics_by_baby })



collector_by_metric_per_bambi = {
    "velocity":     data_marker_concatenate_per_bambi_velocity_row,
    "acceleration": data_marker_concatenate_per_bambi_acceleration_row,
    "jerk":         data_marker_concatenate_per_bambi_jerk_row,
}


collector_by_metric_per_supine = {
    "velocity":     data_marker_concatenate_per_supine_velocity_row,
    "acceleration": data_marker_concatenate_per_supine_acceleration_row,
    "jerk":         data_marker_concatenate_per_supine_jerk_row,
}

concatenate_compute_store(kinematics_by_baby, collector_by_metric_per_bambi, collector_by_metric_per_supine)






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
    "per_bambi_velocity": (data_marker_concatenate_per_bambi_velocity_row, concatenate_per_bambi_path),
    "per_bambi_acceleration": (data_marker_concatenate_per_bambi_acceleration_row, concatenate_per_bambi_path),
    "per_bambi_jerk": (data_marker_concatenate_per_bambi_jerk_row, concatenate_per_bambi_path),
    "per_supine_velocity": (data_marker_concatenate_per_supine_velocity_row, concatenate_per_supine_path),
    "per_supine_acceleration": (data_marker_concatenate_per_supine_acceleration_row, concatenate_per_supine_path),
    "per_supine_jerk": (data_marker_concatenate_per_supine_jerk_row, concatenate_per_supine_path),
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


