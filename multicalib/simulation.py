"""Synthetic SE(3) regression tests, not a substitute for physical camera checks."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import make_T, inv, error_T
from .dataset import save_sample
from .solver import solve_wrist, solve_external


def random_motion(rng, position_center, position_spread=.07, rot_max_deg=35):
    rotvec = rng.normal(size=3)
    rotvec = rotvec / np.linalg.norm(rotvec) * np.deg2rad(rng.uniform(0, rot_max_deg))
    return make_T(Rotation.from_rotvec(rotvec).as_matrix(),
                  np.array(position_center) + rng.uniform(-position_spread, position_spread, size=3))


def perturb(rng, T, noise_pos_m=.0006, noise_rot_deg=.12):
    d = make_T(Rotation.from_rotvec(rng.normal(0, np.deg2rad(noise_rot_deg), size=3)).as_matrix(),
               rng.normal(0, noise_pos_m, size=3))
    return T @ d


def make_synthetic(data_dir, seed=42, n_wrist=32, n_cameras=12, noisy=True):
    rng = np.random.default_rng(seed)
    X = make_T(Rotation.from_euler('xyz', [12, -8, 19], degrees=True).as_matrix(), [.025, -.018, .09])
    Y = make_T(Rotation.from_euler('xyz', [5, 12, 3], degrees=True).as_matrix(), [.45, .06, .11])
    C1 = make_T(Rotation.from_euler('xyz', [12, 25, -20], degrees=True).as_matrix(), [.9, -.3, .6])
    C2 = make_T(Rotation.from_euler('xyz', [-20, 15, 35], degrees=True).as_matrix(), [.7, .45, .65])
    noisep, noiser = (.0006, .12) if noisy else (0, 0)
    for i in range(n_wrist):
        BG = random_motion(rng, [.40, -.04, .36], .1, rot_max_deg=70)
        WP = inv(BG @ X) @ Y
        WP = perturb(rng, WP, noisep, noiser)
        save_sample(data_dir, 'wrist', {'T_B_G': BG, 'T_W_P': WP}, {}, {'synthetic': True})
    for camera, C in [('cam1', C1), ('cam2', C2)]:
        for i in range(n_cameras):
            BG = random_motion(rng, [.40, -.04, .36], .12, rot_max_deg=45)
            # In Stage 2, the board can move independently at every capture.
            board_pose = random_motion(rng, [.46, .03, .18], .10, rot_max_deg=30)
            WP = perturb(rng, inv(BG @ X) @ board_pose, noisep, noiser)
            CP = perturb(rng, inv(C) @ board_pose, noisep, noiser)
            save_sample(data_dir, camera, {'T_B_G': BG, 'T_W_P': WP, 'T_C_P': CP}, {}, {'synthetic': True})
    return X, C1, C2


def run_synthetic(cfg):
    # Never modify a user's real calibration samples/results.
    data = str(Path(cfg['paths']['dataset']) / 'synthetic_demo')
    out = str(Path(cfg['paths']['output']) / 'synthetic_demo')
    if (Path(data) / 'samples').exists():
        raise FileExistsError(f"Synthetic dataset already exists: {data}; delete explicitly to regenerate")
    X, C1, C2 = make_synthetic(data)
    estimated_X, _, _ = solve_wrist(data, out)
    est_C1, _ = solve_external(data, out, 'cam1')
    est_C2, _ = solve_external(data, out, 'cam2')
    print('\nSYNTHETIC GROUND TRUTH ERRORS (calibration simulation only)')
    for name, reference, estimate in [('T_gripper_wrist', X, estimated_X), ('T_base_cam1', C1, est_C1), ('T_base_cam2', C2, est_C2)]:
        t, r = error_T(reference, estimate)
        print(f"{name:20s} {1000*t:.3f} mm  {r:.4f} deg")
    print(f"Synthetic results at: {out}")
