"""Stage 1 eye-in-hand and Stage 2 camera-to-base pose estimation.

Convention:
 T_B_G robot FK; T_W_P ChArUco PnP; X=T_G_W hand-eye.
 Fixed board Stage 1: T_B_P = T_B_G X T_W_P.
 Movable board Stage 2: T_B_C = T_B_G X T_W_P inv(T_C_P).
"""
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import inv, make_T, T_to_vector, vector_to_T, estimate_pose, pose_errors, summary_errors
from .dataset import load_samples, save_transform, read_transform


def _print_metrics(title, m):
    print(f"\n{title}: {json.dumps(m, indent=2)}")


def _write_report(output_dir, name, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _xy_from_samples(samples):
    return [(s['poses']['T_B_G'], s['poses']['T_W_P']) for s in samples]


def _excitation_check(pairs):
    # Dynamic rotary excitation needed. These are diagnostics, not formal observability tests.
    rot = [Rotation.from_matrix(p[0][:3, :3]) for p in pairs]
    first = rot[0]
    degrees = np.array([(first.inv() * r).magnitude() * 180 / np.pi for r in rot])
    if max(degrees) < 15:
        raise ValueError("Insufficient rotation variation (max <15 deg); acquire diverse EEF angles.")
    axes = []
    for r in rot[1:]:
        rv = (first.inv() * r).as_rotvec()
        if np.linalg.norm(rv) > np.deg2rad(5):
            axes.append(rv / np.linalg.norm(rv))
    if len(axes) < 3:
        raise ValueError("Need more informative non-translation-only wrist motions.")
    s = np.linalg.svd(np.array(axes), compute_uv=False)
    if s[1] < 0.15:
        raise ValueError("Rotation axes nearly collinear; rotate around multiple axes.")
    return {"max_rotation_from_first_deg": float(max(degrees)), "rotation_axes_singular_values": s.tolist()}


def _initial_handeye(pairs):
    R_BG = [p[0][:3, :3] for p in pairs]
    t_BG = [p[0][:3, 3] for p in pairs]
    R_WP = [p[1][:3, :3] for p in pairs]
    t_WP = [p[1][:3, 3] for p in pairs]
    solutions = []
    for name, method in [("PARK", cv2.CALIB_HAND_EYE_PARK), ("TSAI", cv2.CALIB_HAND_EYE_TSAI)]:
        try:
            R, t = cv2.calibrateHandEye(R_BG, t_BG, R_WP, t_WP, method=method)
            X = make_T(R, t)
            if np.all(np.isfinite(X)) and abs(np.linalg.det(R) - 1) < 1e-3:
                board_poses = [a @ X @ b for a, b in pairs]
                Y, _, errors = estimate_pose(board_poses)
                score = np.median(errors[:, 0] / 0.005 + errors[:, 1] / 1.0)
                solutions.append((score, name, X, Y))
        except cv2.error as exc:
            print(f"[warning] HandEye {name} failed: {exc}")
    if not solutions:
        raise RuntimeError("Both Park and Tsai hand-eye estimation failed; check FK/PnP and excitation.")
    solutions.sort(key=lambda x: x[0])
    return solutions[0]


def _refine_handeye(pairs, X_init, Y_init, mask=None):
    if mask is None:
        mask = np.ones(len(pairs), dtype=bool)
    scale_m = 0.005
    scale_r = np.deg2rad(1.0)

    def fun(v):
        X = vector_to_T(v[:6]); Y = vector_to_T(v[6:])
        values = []
        for keep, (T_B_G, T_W_P) in zip(mask, pairs):
            if not keep:
                continue
            delta = inv(Y) @ T_B_G @ X @ T_W_P
            values.extend((delta[:3, 3] / scale_m).tolist())
            values.extend((Rotation.from_matrix(delta[:3, :3]).as_rotvec() / scale_r).tolist())
        return np.array(values)

    start = np.r_[T_to_vector(X_init), T_to_vector(Y_init)]
    fit = least_squares(fun, start, loss="huber", f_scale=1.0, max_nfev=400)
    return vector_to_T(fit.x[:6]), vector_to_T(fit.x[6:])


def _holdout_count(n):
    return max(2, int(n * .2)) if n >= 12 else 0


def solve_wrist(data_dir, result_dir, min_samples=8):
    samples = load_samples(data_dir, 'wrist')
    if len(samples) < min_samples:
        raise ValueError(f"Wrist needs >= {min_samples} valid poses, found {len(samples)}")
    pairs = _xy_from_samples(samples)
    n_holdout = _holdout_count(len(pairs))
    train = pairs[:-n_holdout] if n_holdout else pairs
    test = pairs[-n_holdout:] if n_holdout else []
    excitation = _excitation_check(train)
    _, method, X0, Y0 = _initial_handeye(train)
    X, Y = _refine_handeye(train, X0, Y0)
    inliers = np.ones(len(train), dtype=bool)
    for _ in range(3):
        board_poses = [a @ X @ b for a, b in train]
        errors = pose_errors(Y, board_poses)
        med_t, med_r = np.median(errors, axis=0)
        mad_t = 1.4826 * np.median(abs(errors[:, 0] - med_t))
        mad_r = 1.4826 * np.median(abs(errors[:, 1] - med_r))
        candidate = (errors[:, 0] <= max(0.004, med_t + 3 * mad_t)) & (errors[:, 1] <= max(0.7, med_r + 3 * mad_r))
        if candidate.sum() < max(8, int(.65 * len(train))):
            break
        if np.array_equal(candidate, inliers):
            break
        inliers = candidate
        X, Y = _refine_handeye(train, X, Y, inliers)
    train_errors = pose_errors(Y, [a @ X @ b for a, b in train])
    test_errors = pose_errors(Y, [a @ X @ b for a, b in test]) if test else np.empty((0, 2))
    report = {
        'stage': 'wrist', 'initial_handeye_method': method,
        'total_captures': len(samples), 'train_captures': len(train),
        'train_inliers': int(inliers.sum()), 'heldout_captures': len(test),
        'excitation': excitation,
        'training_error': summary_errors(train_errors[inliers]),
        'holdout_error': summary_errors(test_errors),
        'note': 'Error measures consistency of a stationary board under FK+handeye+PnP, not absolute accuracy.'
    }
    save_transform(result_dir, 'T_gripper_wrist', X, report)
    save_transform(result_dir, 'T_base_board_stage1', Y, report)
    _write_report(result_dir, 'wrist', report)
    _print_metrics('Wrist hand-eye', report)
    return X, Y, report


def solve_external(data_dir, result_dir, camera, min_samples=1):
    if camera not in ('cam1', 'cam2'):
        raise ValueError(camera)
    X = read_transform(result_dir, 'T_gripper_wrist')
    samples = load_samples(data_dir, camera)
    if len(samples) < min_samples:
        raise ValueError(f"{camera}: need >= {min_samples} captures, found {len(samples)}")
    poses = []
    for sample in samples:
        s = sample['poses']
        poses.append(s['T_B_G'] @ X @ s['T_W_P'] @ inv(s['T_C_P']))
    n_holdout = max(2, len(poses)//5) if len(poses) >= 8 else (1 if len(poses) >= 3 else 0)
    train = poses[:-n_holdout] if n_holdout else poses
    heldout = poses[-n_holdout:] if n_holdout else []
    C, mask, e = estimate_pose(train)
    test_err = pose_errors(C, heldout) if heldout else np.empty((0, 2))
    report = {
        'stage': camera, 'total_captures': len(samples),
        'train_captures': len(train), 'train_inliers': int(mask.sum()),
        'holdout_captures': len(heldout),
        'training_error': summary_errors(e[mask]),
        'holdout_error': summary_errors(test_err),
        'warning': 'Single-image result has no independent accuracy estimate. Collect 5-10 diverse board poses.' if len(poses) <= 2 else None,
        'note': 'Each Stage 2 capture may use a different stationary board placement; board must be motionless within each capture.'
    }
    save_transform(result_dir, f'T_base_{camera}', C, report)
    _write_report(result_dir, camera, report)
    _print_metrics(f'{camera} -> Robot Base', report)
    return C, report


def inspect_all(result_dir):
    result = Path(result_dir)
    for name in ['T_gripper_wrist', 'T_base_board_stage1', 'T_base_cam1', 'T_base_cam2']:
        path = result / (name + '.npy')
        if path.exists():
            print(f"\n{name}\n{np.array2string(np.load(path), precision=6, suppress_small=True)}")
        else:
            print(f"{name}: not yet available")
