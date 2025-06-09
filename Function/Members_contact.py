import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Function.Base_function import get_threshold_intervals, analyze_intervals_duration
matplotlib.use("TkAgg")


def distance_foot_foot(LANK, RANK, LKNE, RKNE, threshold, time_vector, plot=False):
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

    foot_foot_interval = get_threshold_intervals(distance_foot_foot, threshold, "below")

    members_contact = analyze_intervals_duration(foot_foot_interval, time_vector)

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