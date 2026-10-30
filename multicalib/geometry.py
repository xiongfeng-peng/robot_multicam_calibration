"""Rigid transforms; T_A_B maps coordinates in B to coordinates in A."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def make_T(R=None, t=None):
    T = np.eye(4, dtype=float)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    return make_T(R.T, -(R.T @ T[:3, 3]))


def vector_to_T(v):
    v = np.asarray(v, dtype=float)
    return make_T(Rotation.from_rotvec(v[3:]).as_matrix(), v[:3])


def T_to_vector(T):
    T = np.asarray(T, dtype=float)
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]


def error_T(reference, measured):
    """Return translation and rotation error, for two absolute transforms in same frame."""
    D = inv(reference) @ measured
    return float(np.linalg.norm(D[:3, 3])), float(np.linalg.norm(Rotation.from_matrix(D[:3, :3]).as_rotvec()) * 180 / np.pi)


def pose_errors(reference, candidates):
    return np.array([error_T(reference, v) for v in candidates], dtype=float)


def estimate_pose(transforms, translation_scale=0.005, rotation_scale_deg=1.0, max_iter=4):
    """Robust SE(3) center with hard outlier rejection, returns T, inlier mask, errors.

    We compare poses in the *same* source/target frames. Independent dimensions
    are normalized by typical expected errors, with a Huber robust loss.
    """
    transforms = [np.asarray(T, float).reshape(4, 4) for T in transforms]
    if not transforms:
        raise ValueError("No transforms")
    if len(transforms) == 1:
        return transforms[0].copy(), np.ones(1, bool), np.zeros((1, 2))
    ts = np.array([T[:3, 3] for T in transforms])
    R = Rotation.from_matrix([T[:3, :3] for T in transforms])
    center = make_T(R.mean().as_matrix(), np.median(ts, axis=0))
    mask = np.ones(len(transforms), bool)
    rs = np.deg2rad(rotation_scale_deg)

    for _ in range(max_iter):
        def residual(v):
            C = vector_to_T(v)
            rr = []
            for take, T in zip(mask, transforms):
                if not take:
                    continue
                delta = inv(C) @ T
                rr.extend((delta[:3, 3] / translation_scale).tolist())
                rr.extend((Rotation.from_matrix(delta[:3, :3]).as_rotvec() / rs).tolist())
            return np.asarray(rr)

        sol = least_squares(residual, T_to_vector(center), loss="huber", f_scale=1.0, max_nfev=150)
        center = vector_to_T(sol.x)
        errors = pose_errors(center, transforms)
        if len(transforms) < 5:
            break
        trans_med = np.median(errors[mask, 0]); rot_med = np.median(errors[mask, 1])
        trans_mad = 1.4826 * np.median(abs(errors[mask, 0] - trans_med))
        rot_mad = 1.4826 * np.median(abs(errors[mask, 1] - rot_med))
        trans_thresh = max(0.004, trans_med + 3 * trans_mad)
        rot_thresh = max(0.7, rot_med + 3 * rot_mad)
        new_mask = (errors[:, 0] <= trans_thresh) & (errors[:, 1] <= rot_thresh)
        # Do not allow an aggressive rejection to destroy solvability.
        if new_mask.sum() < max(3, int(np.ceil(len(transforms) * .6))):
            break
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    return center, mask, pose_errors(center, transforms)


def summary_errors(errors):
    errors = np.asarray(errors, float).reshape(-1, 2)
    if len(errors) == 0:
        return {"n": 0}
    return {
        "n": len(errors),
        "translation_mm_median": round(float(np.median(errors[:, 0]) * 1000), 3),
        "translation_mm_p95": round(float(np.percentile(errors[:, 0], 95) * 1000), 3),
        "rotation_deg_median": round(float(np.median(errors[:, 1])), 4),
        "rotation_deg_p95": round(float(np.percentile(errors[:, 1], 95)), 4),
    }
