import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

matplotlib.use("TkAgg")

def extract_kick_intervals(
    distance_signal,
    time_vector,
    peaks,
    min_drop=0.1,
    max_duration=5.0,
    max_jump=4
):
    intervals = []
    i = 0

    while i < len(peaks) - 1:
        start = peaks[i]
        start_val = distance_signal[start]
        found = False

        best_j = None
        best_end = None
        best_end_val = -np.inf

        for offset in range(1, min(max_jump + 1, len(peaks) - i)):
            j = i + offset
            end = peaks[j]
            duration = time_vector[end] - time_vector[start]
            drop = start_val - np.min(distance_signal[start:end + 1])

            if drop >= min_drop and duration <= max_duration:
                # Check absence of kick intermédiaire
                no_intermediate_kick = True
                for k in range(i + 1, j):
                    a, b = peaks[k - 1], peaks[k]
                    drop_k = distance_signal[a] - np.min(distance_signal[a:b + 1])
                    dur_k = time_vector[b] - time_vector[a]
                    if drop_k >= min_drop and dur_k <= max_duration:
                        no_intermediate_kick = False
                        break
                if not no_intermediate_kick:
                    continue

                # Vérifie qu’aucun pic intermédiaire n’est plus haut que le début ou la fin
                intermediate_higher_peak = False
                for k in range(i + 1, j):
                    if distance_signal[peaks[k]] > max(distance_signal[start], distance_signal[end]):
                        intermediate_higher_peak = True
                        break
                if intermediate_higher_peak:
                    continue  # ne valide pas ce kick

                # On garde le j le plus haut possible (pic final le plus élevé)
                if distance_signal[end] > best_end_val:
                    best_j = j
                    best_end = end
                    best_end_val = distance_signal[end]

        if best_j is not None:
            j = best_j
            end = best_end
            while (j + 1 < len(peaks) and (j - i) < max_jump):
                next_end = peaks[j + 1]
                if distance_signal[next_end] <= distance_signal[end]:
                    break
                drop_between = distance_signal[end] - np.min(distance_signal[end:next_end + 1])
                dur_between = time_vector[next_end] - time_vector[end]
                if drop_between >= min_drop and dur_between <= max_duration:
                    break
                drop_total = start_val - np.min(distance_signal[start:next_end + 1])
                dur_total = time_vector[next_end] - time_vector[start]
                # --- Même contrainte sur les pics intermédiaires dans l’extension !
                intermediate_higher_peak = False
                for k in range(i + 1, j + 1):
                    if distance_signal[peaks[k]] > max(distance_signal[start], distance_signal[next_end]):
                        intermediate_higher_peak = True
                        break
                if drop_total >= min_drop and dur_total <= max_duration and not intermediate_higher_peak:
                    j += 1
                    end = next_end
                    continue
                break
            intervals.append((start, end))
            i = j
        else:
            i += 1

    return intervals


def refine_kick_starts(
    distance_signal,
    time_vector,
    peaks,
    intervals,
    min_drop=0.1,
    max_duration=5.0,
    min_ratio=0.90,
    margin=0.10
):
    """
    For each interval (start, end), searches for a better kick start among peaks in [start, end),
    keeping only candidates with height >= min_ratio * original_start and that satisfy
    all criteria (duration, drop, no intermediate higher peak).

    Returns: list of (final_start, end) tuples
    """
    refined = []
    for start, end in intervals:
        start_val = distance_signal[start]
        # List of all peaks between start and end (inclusive start, exclusive end)
        candidate_starts = [p for p in peaks if start <= p < end]
        # Sort candidates by closeness to end (favor closer)
        candidate_starts = sorted(candidate_starts, key=lambda x: end - x)
        best_start = start
        for cand_start in candidate_starts:
            # Condition 1 : hauteur suffisante
            if distance_signal[cand_start] < start_val * min_ratio:
                continue
            # Condition 2 : durée max
            duration = time_vector[end] - time_vector[cand_start]
            if duration > max_duration:
                continue
            # Condition 3 : delta min_drop
            drop = distance_signal[cand_start] - np.min(distance_signal[cand_start:end + 1])
            if drop < min_drop:
                continue
            # Condition 4 : aucun pic intermédiaire trop haut
            threshold_start = distance_signal[cand_start] * (1 + margin)
            threshold_end = distance_signal[end]
            higher_peak = False
            for k in [p for p in peaks if cand_start < p < end]:
                if distance_signal[k] > max(threshold_start, threshold_end):
                    higher_peak = True
                    break
            if higher_peak:
                continue
            # Tout est OK, on prend ce nouveau start plus proche
            best_start = cand_start
            break  # On s'arrête au plus proche valide
        refined.append((best_start, end))
    return refined

def refine_kick_ends(
    distance_signal,
    time_vector,
    peaks,
    intervals,
    max_peaks_ahead=3,
    max_creux_ratio=0.10,
    max_duration=5.0
):
    """
    Pour chaque (start, end), regarde dans les max_peaks_ahead pics qui suivent end si
    un pic plus haut peut devenir le nouveau kick end, à condition qu'il n'y ait pas un creux
    (min) supérieur à max_creux_ratio de l'amplitude du kick actuel.
    Renvoie la liste raffinée des (start, end).
    """
    refined = []
    n_peaks = len(peaks)
    for start, end in intervals:
        # Recherche la position de end dans la liste des pics
        if end not in peaks:
            refined.append((start, end))
            continue  # Securité : ne devrait jamais arriver
        idx = np.where(peaks == end)[0][0]
        kick_amplitude = distance_signal[start] - np.min(distance_signal[start:end+1])
        best_end = end
        best_end_val = distance_signal[end]
        for offset in range(1, max_peaks_ahead + 1):
            next_idx = idx + offset
            if next_idx >= n_peaks:
                break
            candidate = peaks[next_idx]
            # Il faut que le pic soit plus haut que end actuel
            if distance_signal[candidate]*1.1 <= best_end_val:
                continue
            # Il ne doit pas y avoir de creux supérieur à 10% de l'amplitude
            min_between = np.min(distance_signal[end:candidate+1])
            creux = best_end_val - min_between
            if creux > max_creux_ratio * kick_amplitude:
                continue
            # Optionnel : durée totale ne doit pas excéder max_duration
            duration = time_vector[candidate] - time_vector[start]
            if duration > max_duration:
                continue
            # Nouveau end accepté (on prend le premier admissible le plus proche)
            best_end = candidate
            best_end_val = distance_signal[candidate]
            # break pour prendre le plus proche, sinon continue pour prendre le plus haut
            break
        refined.append((start, best_end))
    return refined


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
        distance=3,
        prominence=0.02 * (np.max(distance_pelv_ank_norm) - np.min(distance_pelv_ank_norm))

    )

    kick_intervals = extract_kick_intervals(distance_pelv_ank_norm, time_duration, peaks)
    refined_intervals_start = refine_kick_starts(distance_pelv_ank_norm,  time_duration, peaks, kick_intervals)
    refined_intervals_start_end = refine_kick_ends(distance_pelv_ank_norm,  time_duration, peaks, refined_intervals_start)

    kicking_cycle_data = []

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_duration, distance_pelv_ank_norm, label='Normalized pelvis-ankle distance', color='blue')
        plt.scatter(time_duration[peaks], distance_pelv_ank_norm[peaks], color='orange', marker='o',
                    label='Detected peaks')
        for i, (start, end) in enumerate(refined_intervals_start_end):
            plt.scatter(time_duration[start], distance_pelv_ank_norm[start], color='yellow', label='Kick start' if i == 0 else "",
                        zorder=3)
            plt.scatter(time_duration[end], distance_pelv_ank_norm[end], color='purple', label='Kick end' if i == 0 else "", zorder=3)


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

        # 1. Identify flexion and extension amplitudes
        # Flexion: from initial extension peak to minimum
        # Extension: from minimum to next extension peak

        start_val = distance_pelv_ank_norm[start]
        end_val = distance_pelv_ank_norm[end]

        min_idx = np.argmin(segment)
        min_val = segment[min_idx]

        flexion_amplitude = start_val - min_val
        extension_amplitude = end_val - min_val

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
            'flexion_amplitude': flexion_amplitude,
            'extension_amplitude': extension_amplitude,
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

def shoudler_knee_distance(LKNE, RKNE, LSHO, RSHO):
    distance_knee_shoulder_right = np.linalg.norm(LKNE - LSHO, axis=1)
    distance_knee_shoulder_left = np.linalg.norm(RKNE - RSHO, axis=1)
    return distance_knee_shoulder_right, distance_knee_shoulder_left

