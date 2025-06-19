import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from PythonFunction.Base_function import *

def extract_kick_intervals(
    distance_signal,
    time_vector,
    peaks,
    min_drop=20,
    max_duration=4.0,
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
    min_drop=20,
    max_duration=4.0,
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
    min_drop=20,
    max_duration=4.0
):
    """
    Pour chaque (start, end), regarde dans les max_peaks_ahead pics qui suivent end si
    un pic plus haut peut devenir le nouveau kick end, à condition qu'il n'y ait pas un creux
    (min) supérieur à min_drop de l'amplitude du kick actuel.
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
            if creux > min_drop * kick_amplitude:
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

def label_and_save_kick(knee_angle_d, knee_angle_g, start, end, kick_side, save_list=None, fs=100):
    """
    Affiche les angles genou droit/gauche sur un intervalle de kick élargi pour contexte,
    permet de choisir un label par bouton (single, alternate, simultaneous),
    ajoute les valeurs et le label à save_list.
    Affiche le temps en secondes sur l'axe x.
    """
    if save_list is None:
        save_list = []

    # Calcul de l'extension de la fenêtre
    kick_len = end - start
    ext = kick_len // 2
    N = len(knee_angle_d)

    start_ext = max(0, start - ext)
    end_ext = min(N, end + ext)

    # Indice du minimum dans l'intervalle de kick
    if kick_side == 'right':
        idx_min = start + np.argmin(knee_angle_d[start:end])
    else:
        idx_min = start + np.argmin(knee_angle_g[start:end])

    fig, ax = plt.subplots(figsize=(10, 5))

    # Temps en secondes pour l'affichage
    t_ext = np.arange(start_ext, end_ext) / fs
    t_start = start / fs
    t_end = end / fs
    t_min = idx_min / fs

    ax.plot(t_ext, knee_angle_d[start_ext:end_ext], label='Right knee', color='tab:green')
    ax.plot(t_ext, knee_angle_g[start_ext:end_ext], label='Left knee', color='tab:orange')

    # Ligne verticale sur le minimum du kick sélectionné (en secondes)
    ax.axvline(t_min, color='red', linestyle='--', linewidth=2, label='Kick minimum')

    # Mise en valeur de la zone du kick sélectionné (en secondes)
    ax.axvspan(t_start, t_end, color='grey', alpha=0.15, label='Kick interval')

    ax.set_title(f'Label {kick_side.capitalize()} Kick | Interval: {t_start:.2f}-{t_end:.2f} s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Knee angle (deg)')
    ax.legend()

    plt.subplots_adjust(bottom=0.25)

    label_dict = {'label': None}


    def on_label(label):
        label_dict['label'] = label
        plt.close(fig)
        save_list.append({
            'side': kick_side,
            'start': start,
            'end': end,
            'knee_angle_d': knee_angle_d[start:end].copy(),
            'knee_angle_g': knee_angle_g[start:end].copy(),
            'label': label
        })

    # Add buttons
    ax_single = plt.axes([0.15, 0.05, 0.2, 0.09])
    ax_alternate = plt.axes([0.4, 0.05, 0.2, 0.09])
    ax_simul = plt.axes([0.65, 0.05, 0.2, 0.09])

    b_single = Button(ax_single, 'Single', color='gold', hovercolor='orange')
    b_alternate = Button(ax_alternate, 'Alternate', color='violet', hovercolor='purple')
    b_simul = Button(ax_simul, 'Simultaneous', color='deepskyblue', hovercolor='blue')

    b_single.on_clicked(lambda event: on_label('single'))
    b_alternate.on_clicked(lambda event: on_label('alternate'))
    b_simul.on_clicked(lambda event: on_label('simultaneous'))

    plt.show()

    return save_list