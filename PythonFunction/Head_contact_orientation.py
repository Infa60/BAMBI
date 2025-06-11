import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation as R
from PythonFunction.Base_function import get_threshold_intervals, analyze_intervals_duration

matplotlib.use("TkAgg")

def get_trunk_rot_matrix(LSHO, RSHO, LPEL, RPEL):
    # --- Origine tronc : milieu des épaules ---
    Origin_trunk = 0.5 * (LSHO + RSHO)

    # --- Axe Y_trunk : de droite vers gauche ---
    Y_trunk = LSHO - RSHO
    Y_trunk /= np.linalg.norm(Y_trunk, axis=1, keepdims=True)

    # --- Axe Z_trunk : du bassin vers les épaules ---
    mid_hips = 0.5 * (LPEL + RPEL)
    Z_trunk = Origin_trunk - mid_hips
    Z_trunk /= np.linalg.norm(Z_trunk, axis=1, keepdims=True)

    # --- Axe X_trunk : perpendiculaire (main droite) ---
    X_trunk = np.cross(Y_trunk, Z_trunk)
    X_trunk /= np.linalg.norm(X_trunk, axis=1, keepdims=True)

    # --- Ortho Z_trunk corrigé (recalé pour ⟂ à X, Y) ---
    Z_trunk = np.cross(X_trunk, Y_trunk)
    Z_trunk /= np.linalg.norm(Z_trunk, axis=1, keepdims=True)

    # --- Matrice de rotation du tronc (shape: n_frames × 3 × 3) ---
    R_trunk = np.stack((X_trunk, Y_trunk, Z_trunk), axis=-1)  # axes en colonnes
    return R_trunk

def get_head_rot_matrix_and_mouth_pos(CSHD, LSHD, RSHD):
    Origin_glabelle = 0.5 * (LSHD + RSHD)

    # Axe Y (droite → gauche)
    Y = LSHD - RSHD
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    # Axe Z (haut) : sommet – origine, projeté pour être ⟂ à Y
    Z = CSHD - Origin_glabelle
    dot_ZY = np.sum(Z * Y, axis=1)
    Z_proj = dot_ZY[:, np.newaxis] * Y
    Z = Z - Z_proj
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    # Axe X (avant) : X = Y × Z
    X = np.cross(Y, Z)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    head_radius = (np.linalg.norm(LSHD - RSHD, axis=1)) / 2
    mean_head_radius = np.mean(head_radius)

    # Offsets
    dz = 0.035 + 0.016  # glabella → stomion
    dx = mean_head_radius

    Mouth_position = Origin_glabelle - dz * Z + dx * X

    # --- Matrice de rotation de la tête (axes déjà normalisés) ---
    R_head = np.stack((X, Y, Z), axis=-1)  # shape: (n_frames, 3, 3)
    return Mouth_position, R_head

def distance_hand_mouth(LWRA, RWRA, CSHD, FSHD, LSHD, RSHD, threshold, time_vector, plot=False):
    """
    Analyze hand-to-hand proximity events.

    Measures how often the two wrists come close together (below a distance threshold),
    and returns event count, total time, and durations. Optionally plots the distance over time.
    """

    mouth_pos, matrix_rot_head = get_head_rot_matrix_and_mouth_pos(CSHD, LSHD, RSHD)

    # 1. Compute frame-by-frame Euclidean distance between the wrists
    distance_handR_mouth = np.linalg.norm(mouth_pos - RWRA, axis=1)
    distance_handL_mouth = np.linalg.norm(mouth_pos - LWRA, axis=1)

    # 2. Detect intervals where hands are close (below threshold)
    handR_mouth_interval = get_threshold_intervals(distance_handR_mouth, threshold, "below")
    handL_mouth_interval = get_threshold_intervals(distance_handL_mouth, threshold, "below")

    # 3. Analyze duration and count of close-contact events
    R_hand_contact = analyze_intervals_duration(handR_mouth_interval, time_vector)
    L_hand_contact = analyze_intervals_duration(handL_mouth_interval, time_vector)

    # 4. Optional plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, distance_handR_mouth, label="Right")
        plt.plot(time_vector, distance_handL_mouth, label="Left")
        plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (mm)")
        plt.title("Distance Between Left and Right Hands Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 5. Return summary dictionary
    return R_hand_contact, L_hand_contact

def head_rotation(CSHD, FSHD, LSHD, RSHD, LSHO, RSHO, LPEL, RPEL, threshold, time_vector, plot=False):
    """
    Analyze hand-to-hand proximity events.

    Measures how often the two wrists come close together (below a distance threshold),
    and returns event count, total time, and durations. Optionally plots the distance over time.
    """

    mouth_pos, matrix_rot_head = get_head_rot_matrix_and_mouth_pos(CSHD, LSHD, RSHD)

    matrix_rot_trunk = get_trunk_rot_matrix(LSHO, RSHO, LPEL, RPEL)

    R_rel = np.einsum('nij,njk->nik', np.transpose(matrix_rot_trunk, (0, 2, 1)), matrix_rot_head)

    euler_rel = R.from_matrix(R_rel).as_euler('ZYX', degrees=True)
    yaw, pitch, roll = euler_rel[:, 0], euler_rel[:, 1], euler_rel[:, 2]


    # 2. Detect intervals where hands are close (below threshold)
    head_centered_interval = get_threshold_intervals(yaw, threshold, "between")

    # 3. Analyze duration and count of close-contact events
    head_centered_contact = analyze_intervals_duration(head_centered_interval, time_vector)

    # 4. Optional plot
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, yaw, label='Yaw (rotation horizontale)')
        # plt.plot(time_vector, pitch, label='Pitch (flexion/extension)')
        # plt.plot(time_vector, roll, label='Roll (inclinaison latérale)')
        low, high = threshold
        plt.axhline(low, color='red', linestyle='--', label="Seuil bas")
        plt.axhline(high, color='orange', linestyle='--', label="Seuil haut")
        plt.xlabel("Temps (s)")
        plt.ylabel("Angle relatif tête vs tronc (°)")
        plt.title("Angles de rotation de la tête par rapport au tronc")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 5. Return summary dictionary
    return head_centered_contact