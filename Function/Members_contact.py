import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Function.Base_function import get_threshold_intervals, analyze_intervals_duration, intersect_intervals
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
    common_intervals = intersect_intervals(knee_knee_interval, foot_foot_interval)

    # 4. Analyze timing and duration of those intervals
    members_contact = analyze_intervals_duration(common_intervals, time_vector)

    # 5. Optional plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_foot_foot, label="Foot-to-Foot Distance")
        plt.plot(time_vector, distance_knee_knee, label="Knee-to-Knee Distance")
        plt.axhline(threshold_ankle, color='red', linestyle='--', label="Ankle Threshold")
        plt.axhline(threshold_knee, color='green', linestyle='--', label="Knee Threshold")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Foot and Knee Distances Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 6. Return summary dictionary
    return members_contact

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
    members_contact = analyze_intervals_duration(hand_hand_interval, time_vector)

    # 4. Optional plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_hand_hand, label="Hand-to-Hand Distance")
        plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Distance Between Left and Right Hands Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 5. Return summary dictionary
    return members_contact



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

    # 3. Analyze duration and count of each contact type
    handR_footR_interval_contact = analyze_intervals_duration(handR_footR_interval, time_vector)
    handR_footL_interval_contact = analyze_intervals_duration(handR_footL_interval, time_vector)
    handL_footR_interval_contact = analyze_intervals_duration(handL_footR_interval, time_vector)
    handL_footL_interval_contact = analyze_intervals_duration(handL_footL_interval, time_vector)

    # 4. Optional plot of distances over time
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_handR_footR, label="Right Hand - Right Foot")
        plt.plot(time_vector, distance_handR_footL, label="Right Hand - Left Foot")
        plt.plot(time_vector, distance_handL_footR, label="Left Hand - Right Foot")
        plt.plot(time_vector, distance_handL_footL, label="Left Hand - Left Foot")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Hand-Foot Distances Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 5. Return summary of contact intervals for each pair
    return {
        'handR_footR_interval_contact': handR_footR_interval_contact,
        'handR_footL_interval_contact': handR_footL_interval_contact,
        'handL_footR_interval_contact': handL_footR_interval_contact,
        'handL_footL_interval_contact': handL_footL_interval_contact
    }