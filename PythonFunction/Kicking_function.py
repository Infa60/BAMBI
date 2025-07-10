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
        freq,
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
    knee_angle_filt = butter_lowpass_filter(knee_angle, cutoff, freq, order=2)
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


def knee_hip_correlation_individual_segment(knee_angle, hip_angle, kick_intervals, plot=False):
    """
    A lag < 0 means that the hip precedes the knee (the hip “moves” before the knee in the cycle).
    A lag > 0 means that the knee precedes the hip (the knee “moves” before the hip).
    """
    correlations = []
    lags = []
    for i, (start, end) in enumerate(kick_intervals):
        if end - start < 3:
            continue

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
    mean_corr = round(np.nanmean(correlations),2)
    std_corr = round(np.nanstd(correlations),2)
    mean_lags = round(np.nanmean(lags),2)
    std_lags = round(np.nanstd(lags),2)

    if plot:

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram for correlations
        axs[0].hist(correlations, bins=20)
        axs[0].set_xlabel("Knee-Hip Correlation")
        axs[0].set_ylabel("Number of kicks")
        axs[0].set_title("Distribution of correlations per cycle")

        # Histogram for lags
        axs[1].hist(lags, bins=20)
        axs[1].set_xlabel("Lag (in samples)")
        axs[1].set_ylabel("Number of kicks")
        axs[1].set_title("Distribution of optimal lags")

        plt.tight_layout()
        plt.show()

    return mean_corr, std_corr, mean_lags, std_lags


def classify_kicks(
    kick_intervals_d, kick_intervals_g, knee_angle_d, knee_angle_g, fs,
    simult_threshold_s=0.15,
    overlap_threshold_s=0.25,
    plot=False
):
    """
    Classifies each detected kick as single, alternate, or simultaneous,
    using explicit matching but only annotates the current kick (not its pair).
    """

    # Helper: find min value index in an interval
    def find_min_idx(signal, start, end):
        return start + np.argmin(signal[start:end])

    # Convert interval lists to list of dicts
    kicks_d = []
    for (start, end) in kick_intervals_d:
        idx_min = find_min_idx(knee_angle_d, start, end)
        kicks_d.append({'side': 'right', 'start': start, 'end': end, 'min': idx_min})

    kicks_g = []
    for (start, end) in kick_intervals_g:
        idx_min = find_min_idx(knee_angle_g, start, end)
        kicks_g.append({'side': 'left', 'start': start, 'end': end, 'min': idx_min})

    results = []

    # Annotate right kicks
    for kd in kicks_d:
        # Find best matching left kick
        match = None
        min_tdiff = None
        for kg in kicks_g:
            # Overlap or close enough in time
            overlap = (kd['start'] <= kg['end'] and kd['end'] >= kg['start']) or \
                      (abs(kd['min'] - kg['min'])/fs < overlap_threshold_s)
            if overlap:
                tdiff = abs(kd['min'] - kg['min'])/fs
                if min_tdiff is None or tdiff < min_tdiff:
                    min_tdiff = tdiff
                    match = kg

        if match is not None:
            # Simultaneous if peaks are close enough
            if min_tdiff < simult_threshold_s:
                results.append({'type': 'simultaneous', 'side': 'right', 'start': kd['start'], 'end': kd['end'], 'index': kd['min']})
            else:
                results.append({'type': 'alternate', 'side': 'right', 'start': kd['start'], 'end': kd['end'], 'index': kd['min']})
        else:
            results.append({'type': 'single', 'side': 'right', 'start': kd['start'], 'end': kd['end'], 'index': kd['min']})

    # Annotate left kicks
    for kg in kicks_g:
        match = None
        min_tdiff = None
        for kd in kicks_d:
            overlap = (kg['start'] <= kd['end'] and kg['end'] >= kd['start']) or \
                      (abs(kg['min'] - kd['min'])/fs < overlap_threshold_s)
            if overlap:
                tdiff = abs(kg['min'] - kd['min'])/fs
                if min_tdiff is None or tdiff < min_tdiff:
                    min_tdiff = tdiff
                    match = kd

        if match is not None:
            if min_tdiff < simult_threshold_s:
                results.append({'type': 'simultaneous', 'side': 'left', 'start': kg['start'], 'end': kg['end'], 'index': kg['min']})
            else:
                results.append({'type': 'alternate', 'side': 'left', 'start': kg['start'], 'end': kg['end'], 'index': kg['min']})
        else:
            results.append({'type': 'single', 'side': 'left', 'start': kg['start'], 'end': kg['end'], 'index': kg['min']})

    # Optionally, sort by time
    results = sorted(results, key=lambda x: x['index'])
    return results


def plot_kick_classification_with_bars(knee_angle_d, knee_angle_g, results, fs):
    # Color code for bars
    bar_colors = {'single': 'gold', 'alternate': 'violet', 'simultaneous': 'deepskyblue'}
    label_map = {'single': 'Single', 'alternate': 'Alternate', 'simultaneous': 'Simultaneous'}

    t = np.arange(len(knee_angle_d)) / fs
    plt.figure(figsize=(14, 8))

    # Plot knee angles
    plt.plot(t, knee_angle_d, label='Right Knee', color='tab:green', alpha=0.7)
    plt.plot(t, knee_angle_g, label='Left Knee', color='tab:orange', alpha=0.7)

    # Lignes verticales début/fin
    jitter = 0.003
    shown_labels = set()
    for r in results:
        if r['side'] == 'right':
            x_start = r['start'] / fs - jitter
            x_end = r['end'] / fs + jitter
            plt.axvline(x_start, color='green', linestyle='-', linewidth=1.4, alpha=0.7,
                        label='Right Start' if 'Right Start' not in shown_labels else "")
            plt.axvline(x_end, color='red', linestyle='-', linewidth=1.4, alpha=0.7,
                        label='Right End' if 'Right End' not in shown_labels else "")
            shown_labels.update(['Right Start', 'Right End'])
        else:
            x_start = r['start'] / fs - jitter
            x_end = r['end'] / fs + jitter
            plt.axvline(x_start, color='green', linestyle='--', linewidth=1.4, alpha=0.7,
                        label='Left Start' if 'Left Start' not in shown_labels else "")
            plt.axvline(x_end, color='red', linestyle='--', linewidth=1.4, alpha=0.7,
                        label='Left End' if 'Left End' not in shown_labels else "")
            shown_labels.update(['Left Start', 'Left End'])

    # Bands for type (top for right, bottom for left)
    ylim = plt.ylim()
    bar_height = (ylim[1] - ylim[0]) * 0.04
    bar_y_top = ylim[1] + bar_height * 1.1
    bar_y_bot = ylim[0] - bar_height * 2.1

    for r in results:
        color = bar_colors[r['type']]
        x0 = r['start'] / fs
        x1 = r['end'] / fs
        if r['side'] == 'right':
            plt.fill_betweenx([bar_y_top, bar_y_top + bar_height], x0, x1, color=color, alpha=0.95, linewidth=0)
        else:
            plt.fill_betweenx([bar_y_bot, bar_y_bot + bar_height], x0, x1, color=color, alpha=0.95, linewidth=0)

    # Légende pour les bandes
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=bar_colors['single'], label='Single'),
        mpatches.Patch(color=bar_colors['alternate'], label='Alternate'),
        mpatches.Patch(color=bar_colors['simultaneous'], label='Simultaneous')
    ]
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(legend_patches + list(by_label.values()), [p.get_label() for p in legend_patches] + list(by_label.keys()), loc="upper right")

    plt.ylim(bar_y_bot - bar_height, bar_y_top + bar_height*2)
    plt.xlabel("Time (s)")
    plt.ylabel("Knee Angle (deg)")
    plt.title("Knee angles with labeled kicks and color code bars (top: Right, bottom: Left)")
    plt.tight_layout()
    plt.show()

def get_mean_and_std(
    kicking_cycle_data,
    *,
    fill_value: float | int = 0.0
) -> pd.DataFrame:
    """
    Aggregate a list of per-kick dictionaries into mean / std statistics.
    If the list is empty (→ no kicks detected) the function returns a
    DataFrame filled with *fill_value* instead of raising an error.

    Parameters
    ----------
    kicking_cycle_data : list[dict]
        One dict per detected kick. Keys are metric names, values numeric.
    fill_value : float, default 0.0
        Value used when no kicks are available or when a metric is all-NaN.

    Returns
    -------
    pd.DataFrame
        Two columns ("mean", "std") indexed by the expected metric names.
    """

    # ------------------------------------------------------------------
    # Default numeric metrics expected in every summary
    # ------------------------------------------------------------------
    expected_cols = [
        "flexion_amplitude",
        "extension_amplitude",
        "duration",
        "steepness",
        "flexion_speed",
        "extension_speed",
        "peak_speed",
    ]

    # ------------------------------------------------------------------
    # 1) Empty input  ⟹  return zeros immediately
    # ------------------------------------------------------------------
    if not kicking_cycle_data:
        zeros = pd.Series(fill_value, index=expected_cols, dtype=float)
        return pd.DataFrame({"mean": zeros, "std": zeros})

    # ------------------------------------------------------------------
    # 2) Normal aggregation path
    # ------------------------------------------------------------------
    df = pd.DataFrame(kicking_cycle_data)

    # Ensure every expected column exists and is numeric
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan                        # create missing column

    df[expected_cols] = df[expected_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    mean_vals = df[expected_cols].mean().fillna(fill_value)
    std_vals  = df[expected_cols].std(ddof=1).fillna(fill_value)

    return pd.DataFrame({"mean": mean_vals, "std": std_vals})

def synchro_hip_knee(time_vector, PEL, KNE, SHO, ANK, plot=False):

    hip_angle = 180 - angle_from_vector(PEL, SHO, KNE)
    knee_angle = 180 - angle_from_vector(KNE, PEL, ANK)

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


