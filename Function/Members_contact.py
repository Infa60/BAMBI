import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Function.Base_function import get_threshold_intervals, analyze_intervals_duration, intersect_intervals
matplotlib.use("TkAgg")


def distance_foot_foot(LANK, RANK, LKNE, RKNE, threshold_ankle, threshold_knee, time_vector, plot=False):
    """
    Analyse the distance between left and right ankles:
    - Counts how many times the feet come close together (below threshold)
    - Computes total and per-event durations

    Parameters:
        LANK, RANK: 3D positions of each ankle
        threshold: distance threshold (e.g., 10 mm)
        time_vector: timestamps in seconds

    Returns:
        count_close: number of close-contact events
        total_time_under_thresh: total time under threshold
        durations_per_event: durations for each event
    """

    # 1. Compute the frame-by-frame Euclidean distance
    distance_foot_foot = np.linalg.norm(LANK - RANK, axis=1)

    distance_knee_knee = np.linalg.norm(LKNE - RKNE, axis=1)

    foot_foot_interval = get_threshold_intervals(distance_foot_foot, threshold_ankle, "below")

    knee_knee_interval = get_threshold_intervals(distance_knee_knee, threshold_knee, "above")

    common_intervals = intersect_intervals(knee_knee_interval, foot_foot_interval)


    members_contact = analyze_intervals_duration(common_intervals, time_vector)

    if plot == True:
        # 7. Optionnal plot of distance foot-foot
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_foot_foot, label="Foot-to-Foot Distance")
        plt.plot(time_vector, distance_knee_knee, label="Knee-to-knee Distance")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Distance Between Left and Right Ankles Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return members_contact

def distance_hand_hand(LWRA, RWRA, threshold, time_vector, plot=False):
    """
    Analyse the distance between left and right wrist:
    - Counts how many times the wrist come close together (below threshold)
    - Computes total and per-event durations

    Parameters:
        LWRA, RWRA: 3D positions of each wrist
        threshold: distance threshold (e.g., 10 mm)
        time_vector: timestamps in seconds

    Returns:
        count_close: number of close-contact events
        total_time_under_thresh: total time under threshold
        durations_per_event: durations for each event
    """

    # 1. Compute the frame-by-frame Euclidean distance
    distance_hand_hand = np.linalg.norm(LWRA - RWRA, axis=1)

    hand_hand_interval = get_threshold_intervals(distance_hand_hand, threshold, "below")

    # 4. Count events
    members_contact = analyze_intervals_duration(hand_hand_interval, time_vector)

    if plot == True:
        # 7. Optionnal plot of distance hand-hand
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_hand_hand, label="Hand-to-Hand Distance")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Distance Between Left and Right Ankles Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return members_contact


def distance_hand_foot(LANK, RANK, LWRA, RWRA, threshold, time_vector, plot=False):
    """
    Analyse the distance between left and right ankles:
    - Counts how many times the feet come close together (below threshold)
    - Computes total and per-event durations

    Parameters:
        LANK, RANK: 3D positions of each ankle
        threshold: distance threshold (e.g., 10 mm)
        time_vector: timestamps in seconds

    Returns:
        count_close: number of close-contact events
        total_time_under_thresh: total time under threshold
        durations_per_event: durations for each event
    """

    # 1. Compute the frame-by-frame Euclidean distance
    distance_handR_footR = np.linalg.norm(RWRA - RANK, axis=1)

    distance_handR_footL = np.linalg.norm(RWRA - LANK, axis=1)

    distance_handL_footR = np.linalg.norm(LWRA - RANK, axis=1)

    distance_handL_footL = np.linalg.norm(LWRA - LANK, axis=1)

    handR_footR_interval = get_threshold_intervals(distance_handR_footR, threshold, "below")
    handR_footL_interval = get_threshold_intervals(distance_handR_footL, threshold, "below")
    handL_footR_interval = get_threshold_intervals(distance_handL_footR, threshold, "below")
    handL_footL_interval = get_threshold_intervals(distance_handL_footL, threshold, "below")

    handR_footR_interval_contact = analyze_intervals_duration(handR_footR_interval, time_vector)
    handR_footL_interval_contact = analyze_intervals_duration(handR_footL_interval, time_vector)
    handL_footR_interval_contact = analyze_intervals_duration(handL_footR_interval, time_vector)
    handL_footL_interval_contact = analyze_intervals_duration(handL_footL_interval, time_vector)


    if plot == True:
        # 7. Optionnal plot of distance foot-foot
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_handR_footR, label="RR Distance")
        plt.plot(time_vector, distance_handR_footL, label="RL Distance")
        plt.plot(time_vector, distance_handL_footR, label="LR Distance")
        plt.plot(time_vector, distance_handL_footL, label="LL Distance")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Distance Between Left and Right Ankles Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return (handR_footR_interval_contact,
            handR_footL_interval_contact,
            handL_footR_interval_contact,
            handL_footL_interval_contact)