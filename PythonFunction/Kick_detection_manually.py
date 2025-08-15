import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence
from matplotlib.widgets import Button

LABEL_COLORS = {
    None: "black",
    "start": "green",
    "end": "red",
    "end_start": "orange",
    "hidden": "lightgrey",
}
LABEL_KEYS = {"a": "start", "z": "end", "e": "end_start", "d": "hidden"}


def interactive_label_peaks(
    signal: Sequence[float], peaks_idx: Sequence[int], out_path: str, fs: float = 1.0
):
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    signal = np.asarray(signal, dtype=float)
    N = len(peaks_idx)
    out_path = Path(out_path)
    labels = [None] * N
    hidden = np.zeros(N, dtype=bool)

    # Load existing annotation if file exists
    if out_path.exists():
        df = pd.read_csv(out_path)
        for col, lbl in (("start", "start"), ("end", "end")):
            if col in df.columns:
                for v in df[col].dropna():
                    if int(v) in peaks_idx:
                        idx = np.where(peaks_idx == int(v))[0][0]
                        if labels[idx] is None:
                            labels[idx] = lbl
                        elif labels[idx] != lbl:
                            labels[idx] = "end_start"
    current = 0

    fig, ax = plt.subplots(figsize=(12, 6))
    t = np.arange(signal.size) / fs
    ax.plot(t, signal, "-", lw=0.8, label="Signal")
    scat = ax.scatter(
        t[peaks_idx],
        signal[peaks_idx],
        c=[LABEL_COLORS[labels[i]] for i in range(N)],
        s=70,
        zorder=10,
        picker=7,
    )
    (sel_dot,) = ax.plot(
        [t[peaks_idx[current]]],
        [signal[peaks_idx[current]]],
        "o",
        ms=18,
        mfc="none",
        mec="black",
        mew=2,
        zorder=15,
    )
    ax.set_title(
        "←/→ navigate | a=start (green), z=end (red), e=end+start (orange), d=hide, m=save"
    )
    fig.tight_layout()
    plt.subplots_adjust(top=0.93)

    def redraw():
        # Update scatter colors and selection dot
        scat.set_color(
            [
                LABEL_COLORS["hidden"] if hidden[i] else LABEL_COLORS[labels[i]]
                for i in range(N)
            ]
        )
        sel_dot.set_data([t[peaks_idx[current]]], [signal[peaks_idx[current]]])
        fig.canvas.draw_idle()

    def show_warning(msg):
        print(msg)
        fig.suptitle(msg, color="crimson", fontsize=13)
        plt.pause(1.2)
        fig.suptitle("")  # Clear
        fig.canvas.draw_idle()

    def select_peak(idx):
        nonlocal current
        if 0 <= idx < N:
            current = idx
            redraw()

    def next_unhidden(direction=1):
        # Move to next non-hidden peak in given direction
        idx = current
        for _ in range(N):
            idx = (idx + direction) % N
            if not hidden[idx]:
                return idx
        return current

    def on_pick(event):
        # Mouse click on peak
        select_peak(event.ind[0])

    def valid_alternance(new_label, idx):
        # Check label alternation logic
        valid_idx = [
            i
            for i, l in enumerate(labels)
            if l is not None and not hidden[i] and i < idx
        ]
        seq = [labels[i] for i in valid_idx]
        if not seq:
            if new_label != "start":
                return False, "The first peak must be a start."
            return True, None
        last = seq[-1]
        # After a start → end or end_start only
        if last == "start" and new_label not in ("end", "end_start"):
            return False, "After start, must label end or end+start."
        # After end → only start allowed
        if last == "end" and new_label != "start":
            return False, "After end, must label start."
        # After end_start → only end or end_start allowed
        if last == "end_start" and new_label not in ("end", "end_start"):
            return False, "After end+start, must label end or end+start."
        # Never two consecutive starts
        if last == "start" and new_label == "start":
            return False, "Impossible: two consecutive starts."
        return True, None

    def on_key(event):
        nonlocal current
        if event.key in ("left", "right"):
            direction = 1 if event.key == "right" else -1
            idx = current
            for _ in range(N):
                idx = (idx + direction) % N
                if not hidden[idx]:
                    current = idx
                    break
            redraw()
            return

        if event.key in ("a", "z", "e"):
            lbl = LABEL_KEYS[event.key]
            ok, msg = valid_alternance(lbl, current)
            if not ok:
                show_warning(msg)
                return
            labels[current] = lbl
            redraw()
            idx = next_unhidden(1)
            if idx != current:
                current = idx
            redraw()
            return

        if event.key == "d":
            # Hide the current peak
            hidden[current] = True
            redraw()
            idx = next_unhidden(1)
            if idx != current:
                current = idx
            redraw()
            return

        if event.key == "m":
            # Export only labeled and visible peaks with strict alternation
            valid_idx = [
                i for i, l in enumerate(labels) if l is not None and not hidden[i]
            ]
            seq = [labels[i] for i in valid_idx]
            peaks_seq = [peaks_idx[i] for i in valid_idx]
            starts, ends = [], []
            if not seq:
                show_warning("No labeled peak!")
                return
            expecting = "start"
            for lbl, idx_val in zip(seq, peaks_seq):
                if expecting == "start" and lbl == "start":
                    starts.append(idx_val)
                    expecting = "end"
                elif expecting == "end" and lbl == "end":
                    ends.append(idx_val)
                    expecting = "start"
                elif expecting == "end" and lbl == "end_start":
                    ends.append(idx_val)
                    starts.append(idx_val)
                    # Still expect end or end_start after this
                    expecting = "end"
                else:
                    show_warning(
                        "Export error: sequence is not valid. Check alternation!"
                    )
                    return
            df = pd.DataFrame(
                {
                    "start": pd.Series(starts, dtype=float),
                    "end": pd.Series(ends, dtype=float),
                }
            )
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path} (start: {len(starts)}, end: {len(ends)})")

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


def label_and_save_kick(
    knee_angle_d, knee_angle_g, start, end, kick_side, save_list=None, fs=100
):
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
    if kick_side == "right":
        idx_min = start + np.argmin(knee_angle_d[start:end])
    else:
        idx_min = start + np.argmin(knee_angle_g[start:end])

    fig, ax = plt.subplots(figsize=(10, 5))

    # Temps en secondes pour l'affichage
    t_ext = np.arange(start_ext, end_ext) / fs
    t_start = start / fs
    t_end = end / fs
    t_min = idx_min / fs

    ax.plot(
        t_ext, knee_angle_d[start_ext:end_ext], label="Right knee", color="tab:green"
    )
    ax.plot(
        t_ext, knee_angle_g[start_ext:end_ext], label="Left knee", color="tab:orange"
    )

    # Ligne verticale sur le minimum du kick sélectionné (en secondes)
    ax.axvline(t_min, color="red", linestyle="--", linewidth=2, label="Kick minimum")

    # Mise en valeur de la zone du kick sélectionné (en secondes)
    ax.axvspan(t_start, t_end, color="grey", alpha=0.15, label="Kick interval")

    ax.set_title(
        f"Label {kick_side.capitalize()} Kick | Interval: {t_start:.2f}-{t_end:.2f} s"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Knee angle (deg)")
    ax.legend()

    plt.subplots_adjust(bottom=0.25)

    label_dict = {"type": None}

    def on_label(label):
        label_dict["type"] = label
        plt.close(fig)
        save_list.append(
            {
                "side": kick_side,
                "start": start,
                "end": end,
                "knee_angle_d": knee_angle_d[start:end].copy(),
                "knee_angle_g": knee_angle_g[start:end].copy(),
                "type": label,
            }
        )

    # Add buttons
    ax_single = plt.axes([0.15, 0.05, 0.2, 0.09])
    ax_alternate = plt.axes([0.4, 0.05, 0.2, 0.09])
    ax_simul = plt.axes([0.65, 0.05, 0.2, 0.09])

    b_single = Button(ax_single, "Single", color="gold", hovercolor="orange")
    b_alternate = Button(ax_alternate, "Alternate", color="violet", hovercolor="purple")
    b_simul = Button(ax_simul, "Simultaneous", color="deepskyblue", hovercolor="blue")

    b_single.on_clicked(lambda event: on_label("single"))
    b_alternate.on_clicked(lambda event: on_label("alternate"))
    b_simul.on_clicked(lambda event: on_label("simultaneous"))

    plt.show()

    return save_list