import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
matplotlib.use("TkAgg")


def kicking(
        pelvis_marker,
        ankle_marker,
        time_duration,
        leg_length,
        plot=False
            ):
    """
        Analyze kicking movements based on the distance between pelvis and ankle markers.

        This function identifies kicking cycles (from peak to peak), and extracts for each cycle:
        - Amplitude
        - Duration
        - Steepness of extension
        - Mean flexion speed (based on point-wise gradient)
        - Mean extension speed (based on point-wise gradient)

        Parameters:
            pelvis_marker:
            ankle_marker:
            time_duration: time vector (in seconds)
            leg_length:  subject-specific leg length for normalization

        Returns:
            kicking_cycle_data: list of dicts – one per cycle with extracted features
            distance_pelv_ank_norm: normalized pelvis-ankle distance over time
        """
    # Compute the Euclidean distance between pelvis and ankle at each time frame
    distance_pelv_ank = np.linalg.norm(pelvis_marker - ankle_marker, axis=1)

    # Normalize the distance by the leg length to account for subject variability
    distance_pelv_ank_norm = distance_pelv_ank / leg_length

    # Detect peaks in the normalized distance signal (corresponding to extension phases)
    # Thresholds are adaptively set based on the 50th percentile and minimum peak prominence
    peaks, _ = find_peaks(
        distance_pelv_ank_norm,
        height=np.percentile(distance_pelv_ank_norm, 50),
        distance=20,          # Minimum number of frames between two kicks
        prominence=10         # Minimum required prominence to filter out noise
    )

    kicking_cycle_data = []

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_duration, distance_pelv_ank_norm, label='Normalized pelvis-ankle distance', color='blue')
        plt.scatter(time_duration[peaks], distance_pelv_ank_norm[peaks], color='green', marker='o',
                    label='Detected peaks')
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized distance (pelvis to ankle)")
        plt.title("Detection of leg extension phases (kicks)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Analyze each kick cycle between two consecutive extension peaks
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        # Extract the segment of the signal and corresponding time
        segment = distance_pelv_ank_norm[start:end]
        segment_time = time_duration[start:end]

        if len(segment) < 2:
            continue  # Skip cycles that are too short

        # 1. Compute the amplitude of the kicking cycle
        # Defined as the distance from the peak to the minimum (flexion phase)
        min_idx = np.argmin(segment)
        min_val = segment[min_idx]
        amplitude = distance_pelv_ank_norm[start] - min_val

        # 2. Compute the duration of the cycle (in seconds)
        duration = segment_time[-1] - segment_time[0]

        # 3. Compute average steepness of the extension phase (slope between flexion and extension)
        if min_idx < len(segment) - 1:
            ext_segment = segment[min_idx:]
            ext_time = segment_time[min_idx:]
            steepness = (ext_segment[-1] - ext_segment[0]) / (ext_time[-1] - ext_time[0])
        else:
            steepness = np.nan

        # 4. Compute mean velocity (using point-wise gradient) for flexion phase
        if min_idx > 1:
            flex_segment = segment[:min_idx + 1]
            flex_time = segment_time[:min_idx + 1]
            flex_velocity = np.gradient(flex_segment, flex_time)
            flexion_speed = np.mean(np.abs(flex_velocity))
        else:
            flexion_speed = np.nan

        # Compute mean velocity (using point-wise gradient) for extension phase
        if min_idx < len(segment) - 2:
            ext_segment = segment[min_idx:]
            ext_time = segment_time[min_idx:]
            ext_velocity = np.gradient(ext_segment, ext_time)
            extension_speed = np.mean(np.abs(ext_velocity))
        else:
            extension_speed = np.nan

        # 5. Store the extracted features for this kicking cycle
        kicking_cycle_data.append({
            'amplitude': amplitude,
            'duration': duration,
            'steepness': steepness,
            'flexion_speed': flexion_speed,
            'extension_speed': extension_speed
        })

    # Return list of cycle-level features and the full normalized distance signal
    return kicking_cycle_data, distance_pelv_ank_norm

def get_mean_and_std(kicking_cycle_data):
    # Convert the list of dicts to a DataFrame
    df_kicking_cycle = pd.DataFrame(kicking_cycle_data)

    # Compute mean and standard deviation for numeric columns
    mean_values_kicking = df_kicking_cycle.mean(numeric_only=True)
    std_values_kicking = df_kicking_cycle.std(numeric_only=True)

    # Combine into a single DataFrame
    mean_std_kicking_values = pd.DataFrame({
        "mean": mean_values_kicking,
        "std": std_values_kicking
    })

    return mean_std_kicking_values