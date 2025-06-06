import numpy as np
import matplotlib
from scipy.signal import find_peaks
matplotlib.use("TkAgg")


def kicking(
        pelvis_marker,
        ankle_marker,
        time_duration,
        leg_length
            ):
    distance_pelv_ank = np.linalg.norm(pelvis_marker - ankle_marker, axis=1)
    distance_pelv_ank_norm = distance_pelv_ank/leg_length
    peaks, _ = find_peaks(
                        distance_pelv_ank_norm,
                        height=np.percentile(distance_pelv_ank_norm, 50),
                        distance=20,
                        prominence=10
                        )

    kicking_cycle_data = []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        segment = distance_pelv_ank_norm[start:end]
        segment_time = time_duration[start:end]

        if len(segment) < 2:
            continue

        # 1. Amplitude (mm)
        min_idx = np.argmin(segment)
        min_val = segment[min_idx]
        amplitude = distance_pelv_ank_norm[start] - min_val

        # 2. Durée du cycle (en secondes)
        duration = segment_time[-1] - segment_time[0]

        # 3. Vitesse (dérivée)
        velocity = np.gradient(segment, segment_time)
        max_velocity = np.max(np.abs(velocity))  # vitesse maximale absolue

        # 4. Pente moyenne (sur la phase extension ou flexion)
        # Exemple : extension = du min vers le pic
        if min_idx < len(segment) - 1:
            ext_segment = segment[min_idx:]
            ext_time = segment_time[min_idx:]
            steepness = (ext_segment[-1] - ext_segment[0]) / (ext_time[-1] - ext_time[0])
        else:
            steepness = np.nan

        # 5. Stocker les résultats
        kicking_cycle_data.append({
            'cycle_start_time': segment_time[0],
            'cycle_end_time': segment_time[-1],
            'amplitude': amplitude,
            'duration': duration,
            'max_velocity': max_velocity,
            'steepness': steepness
        })


    return kicking_cycle_data