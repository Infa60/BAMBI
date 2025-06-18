from PythonFunction.Base_function import *
from scipy.signal import find_peaks
from sklearn.cross_decomposition import CCA
matplotlib.use("TkAgg")

def extract_kick_intervals(
    distance_signal,
    time_vector,
    peaks,
    min_drop=20,
    max_duration=4.0,
    max_jump=4,
    min_ratio=0.90
):
    intervals = []
    i = 0

    while i < len(peaks) - 1:
        start = peaks[i]
        start_val = distance_signal[start]

        found_end = False
        for offset in range(1, min(max_jump + 1, len(peaks) - i)):
            j = i + offset
            end = peaks[j]
            end_val = distance_signal[end]
            duration = time_vector[end] - time_vector[start]
            min_in_interval = np.min(distance_signal[start:end + 1])
            drop_start = start_val - min_in_interval
            drop_end = end_val - min_in_interval

            if drop_start >= min_drop and drop_end >= min_drop and duration <= max_duration:
                # Tentative d'extension
                k = j
                while k + 1 < len(peaks) and (k - i) < max_jump:
                    next_end = peaks[k + 1]
                    next_end_val = distance_signal[next_end]
                    dur_total = time_vector[next_end] - time_vector[start]
                    if dur_total > max_duration:
                        break

                    if next_end_val <= distance_signal[end]:
                        break

                    duration_next = time_vector[next_end] - time_vector[end]
                    min_between = np.min(distance_signal[end:next_end + 1])
                    drop_between = distance_signal[end] - min_between
                    drop_next = next_end_val - min_between

                    if drop_between >= min_drop and drop_next >= min_drop and duration_next <= max_duration:
                        break

                    end = next_end
                    end_val = next_end_val
                    k += 1

                # ====================
                # Recherche meilleur start entre start et end
                candidate_starts = [p for p in peaks if start < p < end]
                candidate_starts = sorted(candidate_starts, key=lambda x: end - x)  # plus proche de la fin d'abord
                best_start = start
                for cand_start in candidate_starts:
                    cand_val = distance_signal[cand_start]
                    if cand_val < start_val * min_ratio:
                        continue
                    # Vérifie critères
                    duration_cand = time_vector[end] - time_vector[cand_start]
                    min_in_cand_interval = np.min(distance_signal[cand_start:end + 1])
                    drop_cand_start = cand_val - min_in_cand_interval
                    drop_cand_end = end_val - min_in_cand_interval
                    if duration_cand > max_duration or drop_cand_start < min_drop or drop_cand_end < min_drop:
                        continue
                    # Aucun pic intermédiaire plus haut que cand_start ou end
                    higher_peak = False
                    for pk in [p for p in peaks if cand_start < p < end]:
                        if distance_signal[pk] > max(cand_val, end_val):
                            higher_peak = True
                            break
                    if higher_peak:
                        continue
                    # Ce candidat est valide
                    best_start = cand_start
                    break  # On prend le plus proche de la fin qui convient

                intervals.append((best_start, end))
                i = np.where(peaks == end)[0][0]
                found_end = True
                break

        if not found_end:
            i += 1

    return intervals


def kicking(
        pelvis_marker,
        ankle_marker,
        time_duration,
        leg_length,
        KNE,
        PEL,
        ANK,
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
    distance_pelv_ank_norm_nofilt = distance_pelv_ank / leg_length

    knee_angle = 180 - angle_from_vector(KNE, PEL, ANK)

    # Apply Butterworth filter (6 Hz cutoff)
    cutoff = 6  # Cutoff frequency (Hz)
    knee_angle_filt = butter_lowpass_filter(knee_angle, cutoff, 200, order=2)
    distance_pelv_ank_norm = butter_lowpass_filter(distance_pelv_ank_norm_nofilt, cutoff, 200, order=2)

    # Detect peaks in the normalized distance signal (corresponding to extension phases)
    peaks, _ = find_peaks(
        knee_angle_filt,
        distance=3,
        prominence=0.1 * (np.max(knee_angle_filt) - np.min(knee_angle_filt))
    )

    kick_intervals = extract_kick_intervals(knee_angle_filt, time_duration, peaks, min_drop=20)

    kicking_cycle_data = []

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_duration, knee_angle_filt, label='Normalized pelvis-ankle distance', color='blue')
        plt.scatter(time_duration[peaks], knee_angle_filt[peaks], color='orange', marker='o',
                    label='Detected peaks')
        for i, (start, end) in enumerate(kick_intervals):
            plt.scatter(time_duration[start], knee_angle_filt[start], color='green', label='Kick start' if i == 0 else "",
                        zorder=3)
            plt.scatter(time_duration[end], knee_angle_filt[end], color='red', label='Kick end' if i == 0 else "", zorder=3)


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
        segment = knee_angle_filt[start:end]
        segment_time = time_duration[start:end]

        if len(segment) < 2:
            continue  # Skip cycles that are too short

        # 1. Identify flexion and extension amplitudes
        # Flexion: from initial extension peak to minimum
        # Extension: from minimum to next extension peak

        start_val = knee_angle_filt[start]
        end_val = knee_angle_filt[end]

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
    return kicking_cycle_data, distance_pelv_ank, kick_intervals


def knee_hip_correlation(knee_angle, hip_angle, kick_intervals):
    """
    A lag < 0 means that the hip precedes the knee (the hip “moves” before the knee in the cycle).
    A lag > 0 means that the knee precedes the hip (the knee “moves” before the hip).
    """
    correlations = []
    lags = []
    for i, (start, end) in enumerate(kick_intervals):
        # Extract the knee and hip angle segments for the current interval
        knee_segment = knee_angle[start:end]
        knee_segment_norm = resample_size(knee_segment, 100)  # Resample to 100 points

        hip_segment = hip_angle[start:end]
        hip_segment_norm = resample_size(hip_segment, 100)  # Resample to 100 points

        # Compute the Pearson correlation coefficient (classical, direct alignment)
        corr = np.corrcoef(knee_segment_norm, hip_segment_norm)[0, 1]
        correlations.append(corr)

        # Compute the cross-correlation to find the optimal lag (temporal shift)
        cross_corr = np.correlate(
            knee_segment_norm - np.mean(knee_segment_norm),
            hip_segment_norm - np.mean(hip_segment_norm),
            mode='full'
        )
        # Array of lag values corresponding to the cross-correlation output
        lags_arr = np.arange(-len(knee_segment_norm) + 1, len(knee_segment_norm))
        # Find the lag with the maximum absolute correlation value
        lag_opt = lags_arr[np.argmax(np.abs(cross_corr))]
        lags.append(lag_opt)

    # Convert correlations list to a NumPy array for easier computation
    correlations = np.array(correlations)
    lags = np.array(lags)

    # Calculate mean and standard deviation of the correlations and lags
    mean_corr = np.nanmean(correlations)
    std_corr = np.nanstd(correlations)
    mean_lags = np.nanmean(lags)
    std_lags = np.nanstd(lags)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram for correlations
    axs[0].hist(correlations, bins=20)
    axs[0].set_xlabel("Knee-Hip Correlation")
    axs[0].set_ylabel("Number of kicks")
    axs[0].set_title("Distribution of correlations per cycle")

    # Histogram for lags
    axs[1].hist(lags, bins=10)
    axs[1].set_xlabel("Lag (in samples)")
    axs[1].set_ylabel("Number of kicks")
    axs[1].set_title("Distribution of optimal lags")

    plt.tight_layout()
    plt.show()


def classify_kicks(
    kick_intervals_d, kick_intervals_g, knee_angle_d, knee_angle_g, fs, simult_threshold_pct=0.1, window=5
):
    """
    Classifies each detected kick as single, alternate, or simultaneous,
    using a biomechanical definition (alternate = l'autre jambe en phase opposée).

    Parameters:
    - kick_intervals_d: list of (start, end) for right knee kicks (indices)
    - kick_intervals_g: list of (start, end) for left knee kicks (indices)
    - knee_angle_d: array of right knee angles
    - knee_angle_g: array of left knee angles
    - fs: sampling frequency (Hz)
    - simult_threshold_pct: time threshold (fraction of cycle duration, e.g., 0.1 for 10%)
    - window: number of samples before/after the kick peak for phase estimation

    Returns:
    - List of dictionaries, one per kick, each with classification info
    """

    # 1. Trouver le pic de chaque kick et stocker
    kicks = []
    for (start, end) in kick_intervals_d:
        idx_peak = start + np.argmax(knee_angle_d[start:end])
        kicks.append({'side': 'right', 'index': idx_peak, 'start': start, 'end': end})
    for (start, end) in kick_intervals_g:
        idx_peak = start + np.argmax(knee_angle_g[start:end])
        kicks.append({'side': 'left', 'index': idx_peak, 'start': start, 'end': end})

    kicks = sorted(kicks, key=lambda x: x['index'])
    n = len(kicks)

    # 2. Pré-classification : simultaneous (moins de X% du cycle)
    simult_groups = []
    assigned = np.zeros(n, dtype=bool)
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        idx_i = kicks[i]['index']
        side_i = kicks[i]['side']
        duration_i = kicks[i]['end'] - kicks[i]['start']
        for j in range(i+1, n):
            if assigned[j]:
                continue
            idx_j = kicks[j]['index']
            duration_j = kicks[j]['end'] - kicks[j]['start']
            # Use average duration of two kicks
            cycle_duration = int(np.mean([duration_i, duration_j]))
            threshold = simult_threshold_pct * cycle_duration
            if abs(idx_i - idx_j) <= threshold:
                group.append(j)
        if len(group) > 1:
            for k in group:
                assigned[k] = True
            simult_groups.append(group)

    # Marquer tous les kicks simultanés
    labels = [None]*n
    for group in simult_groups:
        for idx in group:
            labels[idx] = 'simultaneous'

    # 3. Alternate: une jambe en flexion, l'autre en extension
    # Pour tous les kicks non déjà marqués simultanés
    for i in range(n):
        if labels[i] is not None:
            continue
        my_kick = kicks[i]
        my_side = my_kick['side']
        my_idx = my_kick['index']
        # Déterminer l'angle opposé à ce moment
        if my_side == 'right':
            other_angle = knee_angle_g
        else:
            other_angle = knee_angle_d
        # Fenêtre de quelques points avant/après
        w = window
        before = np.mean(other_angle[max(my_idx-w,0): my_idx])
        after = np.mean(other_angle[my_idx: my_idx+w])
        slope = after - before

        # On considère "alternate" si la jambe qui fait un kick a une dérivée opposée à celle de l'autre jambe
        # On checke si, autour du pic, l'autre jambe a un mouvement opposé (ou une amplitude opposée)
        # Ici : slope négatif -> l'autre jambe part en extension pendant que celle-ci est en flexion
        if slope < 0:
            labels[i] = 'alternate'

    # Pour que les alternates soient bien marqués par paires (voire plus : séquences alternées !)
    # On relit et groupe toutes les alternances proches
    already_marked = set()
    for i in range(n):
        if labels[i] == 'alternate' and i not in already_marked:
            # Cherche tous les alternates proches dans le temps (dans la même séquence d'alternance)
            group = [i]
            idx_i = kicks[i]['index']
            side_i = kicks[i]['side']
            for j in range(i+1, n):
                if labels[j] == 'alternate' and kicks[j]['side'] != side_i:
                    idx_j = kicks[j]['index']
                    # Si moins de demi-cycle d'écart → même alternance
                    if abs(idx_i - idx_j) < 0.5 * (kicks[i]['end']-kicks[i]['start']):
                        group.append(j)
            # Marque le groupe
            for k in group:
                already_marked.add(k)
                labels[k] = 'alternate'

    # 4. Les autres sont des singles
    for i in range(n):
        if labels[i] is None:
            labels[i] = 'single'

    # 5. Construction du résultat
    results = []
    for i in range(n):
        results.append({
            'type': labels[i],
            'side': kicks[i]['side'],
            'index': kicks[i]['index'],
            'start': kicks[i]['start'],
            'end': kicks[i]['end'],
        })
    return results

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

def synchro_hip_knee(time_vector, PEL, KNE, SHO, ANK, plot=False):

    hip_angle = angle_from_vector(PEL, SHO, KNE)
    knee_angle = angle_from_vector(KNE, PEL, ANK)

    if plot:
        plot_time_series(time_vector, knee_angle=knee_angle, hip_angle=hip_angle, title="Angle synchro", ylabel="Angle (°)")

    return knee_angle, hip_angle

def angle_from_vector(mid, prox, dist):
    """
    Computes the hip angle at each frame using pelvis, knee, and shoulder 3D coordinates.
    """
    vec_thigh = mid - dist
    vec_body = prox - mid

    # Normalize vectors
    vec_thigh_norm = vec_thigh / np.linalg.norm(vec_thigh, axis=1, keepdims=True)
    vec_body_norm = vec_body / np.linalg.norm(vec_body, axis=1, keepdims=True)

    # Angle using dot product
    dot = np.sum(vec_thigh_norm * vec_body_norm, axis=1)
    dot = np.clip(dot, -1, 1)  # Numerical safety
    angles = np.degrees(np.arccos(dot))

    return angles


