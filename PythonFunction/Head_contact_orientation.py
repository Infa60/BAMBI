import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from PythonFunction.Base_function import get_threshold_intervals, analyze_intervals_duration, plot_time_series

matplotlib.use("TkAgg")

def get_trunk_rot_matrix(LSHO, RSHO, LPEL, RPEL):
    # --- Trunk origin: midpoint between shoulders ---
    Origin_trunk = 0.5 * (LSHO + RSHO)

    # --- Y_trunk axis: from right to left shoulder ---
    Y_trunk = LSHO - RSHO
    Y_trunk /= np.linalg.norm(Y_trunk, axis=1, keepdims=True)

    # --- Z_trunk axis: from pelvis to shoulders (vertical trunk direction) ---
    mid_hips = 0.5 * (LPEL + RPEL)
    Z_trunk = Origin_trunk - mid_hips
    Z_trunk /= np.linalg.norm(Z_trunk, axis=1, keepdims=True)

    # --- X_trunk axis: orthogonal to Y and Z (points forward) ---
    X_trunk = np.cross(Y_trunk, Z_trunk)
    X_trunk /= np.linalg.norm(X_trunk, axis=1, keepdims=True)

    # --- Re-orthogonalize Z_trunk to ensure orthonormal basis ---
    Z_trunk = np.cross(X_trunk, Y_trunk)
    Z_trunk /= np.linalg.norm(Z_trunk, axis=1, keepdims=True)

    # --- Trunk rotation matrix (shape: n_frames × 3 × 3) ---
    R_trunk = np.stack((X_trunk, Y_trunk, Z_trunk), axis=-1)

    return R_trunk

def get_head_rot_matrix_and_mouth_pos(CSHD, LSHD, RSHD, FSHD):
    # --- Glabella origin approximation: midpoint between left and right side of head ---
    Origin_glabelle = 0.5 * (LSHD + RSHD)

    # --- Y axis (right to left) ---
    Y = LSHD - RSHD
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    # --- Z axis (upward): from top of head to glabella, projected to be orthogonal to Y ---
    Z = CSHD - Origin_glabelle
    dot_ZY = np.sum(Z * Y, axis=1)
    Z_proj = dot_ZY[:, np.newaxis] * Y
    Z = Z - Z_proj
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    # --- X axis (forward): cross product of Y and Z ---
    X = np.cross(Y, Z)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # --- Head radius approximation (1/3 of head width) ---
    head_radius = np.linalg.norm(LSHD - RSHD, axis=1) / 3
    mean_head_radius = np.mean(head_radius)

    # --- Offsets from glabella to mouth ---
    dz = 100  # distance from glabella to mouth (stomion), in mm (to adjust)
    dx = mean_head_radius

    # --- Estimate mouth position in world coordinates ---
    Mouth_position = Origin_glabelle - dz * Z + dx * X

    # --- Head rotation matrix (shape: n_frames × 3 × 3), axes are column vectors ---
    R_head = np.stack((X, Y, Z), axis=-1)

    return Mouth_position, R_head

def distance_hand_mouth(LWRA, RWRA, CSHD, FSHD, LSHD, RSHD, threshold, time_vector, folder_outcome, plot_name, plot=False):
    """
    Analyze hand-to-hand proximity events.

    Measures how often the two wrists come close together (below a distance threshold),
    and returns event count, total time, and durations. Optionally plots the distance over time.
    """

    mouth_pos, matrix_rot_head = get_head_rot_matrix_and_mouth_pos(CSHD, LSHD, RSHD, FSHD)

    # 1. Compute frame-by-frame Euclidean distance between the wrists
    distance_handR_mouth = np.linalg.norm(mouth_pos - RWRA, axis=1)
    distance_handL_mouth = np.linalg.norm(mouth_pos - LWRA, axis=1)

    # 2. Detect intervals where hands are close (below threshold)
    handR_mouth_interval = get_threshold_intervals(distance_handR_mouth, threshold, "below")
    handL_mouth_interval = get_threshold_intervals(distance_handL_mouth, threshold, "below")

    # 3. Analyze duration and count of close-contact events
    R_hand_contact = analyze_intervals_duration(handR_mouth_interval, time_vector, distance_handR_mouth)
    L_hand_contact = analyze_intervals_duration(handL_mouth_interval, time_vector, distance_handL_mouth)

    # 4. Optional plot
    if plot:
        plot_time_series(time_vector, Right=distance_handR_mouth, Left=distance_handL_mouth, threshold=threshold,
                         ylabel="Distance (mm)", title="Distance Between Mouth and Left / Right Hands Over Time")
        plt.savefig(os.path.join(folder_outcome, f"{plot_name}_hand_mouth.png"), dpi=300)
        plt.close()



    # 5. Return summary dictionary
    return R_hand_contact, L_hand_contact

def head_rotation(CSHD, FSHD, LSHD, RSHD, LSHO, RSHO, LPEL, RPEL, threshold, time_vector, plot=False):
    """
    Analyze hand-to-hand proximity events.

    Measures how often the two wrists come close together (below a distance threshold),
    and returns event count, total time, and durations. Optionally plots the distance over time.
    """

    mouth_pos, matrix_rot_head = get_head_rot_matrix_and_mouth_pos(CSHD, LSHD, RSHD, FSHD)

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