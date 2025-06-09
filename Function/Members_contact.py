import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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

    # 2. Identify frames under the threshold
    foot_in_contact = distance_foot_foot < threshold

    # 3. Detect transitions using difference
    transitions = np.diff(foot_in_contact.astype(int))
    start_idxs = np.where(transitions == 1)[0] + 1  # +1 because diff shifts index
    end_idxs = np.where(transitions == -1)[0] + 1

    # Handle edge cases (if starts under threshold or ends under threshold)
    if foot_in_contact[0]:
        start_idxs = np.insert(start_idxs, 0, 0)
    if foot_in_contact[-1]:
        end_idxs = np.append(end_idxs, len(foot_in_contact) - 1)

    # 4. Count events
    count_close = len(start_idxs)

    # 5. Durations per event
    durations_per_event = []
    for start, end in zip(start_idxs, end_idxs):
        duration = time_vector[end] - time_vector[start]
        durations_per_event.append(duration)

    # 6. Total time under threshold
    total_time_under_thresh = np.sum(durations_per_event)

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

    return count_close, total_time_under_thresh, durations_per_event

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

    # 2. Identify frames under the threshold
    hand_in_contact = distance_hand_hand < threshold

    # 3. Detect transitions using difference
    transitions = np.diff(hand_in_contact.astype(int))
    start_idxs = np.where(transitions == 1)[0] + 1  # +1 because diff shifts index
    end_idxs = np.where(transitions == -1)[0] + 1

    # Handle edge cases (if starts under threshold or ends under threshold)
    if hand_in_contact[0]:
        start_idxs = np.insert(start_idxs, 0, 0)
    if hand_in_contact[-1]:
        end_idxs = np.append(end_idxs, len(hand_in_contact) - 1)

    # 4. Count events
    count_close = len(start_idxs)

    # 5. Durations per event
    durations_per_event = []
    for start, end in zip(start_idxs, end_idxs):
        duration = time_vector[end] - time_vector[start]
        durations_per_event.append(duration)

    # 6. Total time under threshold
    total_time_under_thresh = np.sum(durations_per_event)


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

    return count_close, total_time_under_thresh, durations_per_event