import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PythonFunction.Base_function import (
    get_threshold_intervals,
    analyze_intervals_duration,
)

matplotlib.use("TkAgg")


def body_symmetry(LPEL, RPEL, LSHO, RSHO, threshold, time_vector, plot=False):
    """
    Analyze foot-to-foot contact events when knees are sufficiently apart.

    Measures when both feet come close together (ankle distance below threshold),
    while knees are separated (knee distance above threshold).
    Returns the number of events, total time, and per-event durations.
    Optionally plots both distance signals over time.
    """

    # 1. Compute frame-by-frame distances
    distance_shoulder_pelvis_right = np.linalg.norm(LPEL - LSHO, axis=1)
    distance_shoulder_pelvis_left = np.linalg.norm(RPEL - RSHO, axis=1)

    # 2. Compute absolute difference between left and right distances
    distance_diff_right_left = np.abs(
        distance_shoulder_pelvis_right - distance_shoulder_pelvis_left
    )

    # 3. Detect contact and separation intervals
    diff_right_left_interval = get_threshold_intervals(
        distance_diff_right_left, threshold, "above"
    )

    # 4. Analyze timing and duration of those intervals
    shoulder_pelvis_contact = analyze_intervals_duration(
        diff_right_left_interval, time_vector
    )

    # 5. Optional plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(
            time_vector, distance_diff_right_left, label="Difference right left side"
        )
        plt.axhline(threshold, color="green", linestyle="--", label="Threshold")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (mm)")
        plt.title("Body symmetry")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return shoulder_pelvis_contact
