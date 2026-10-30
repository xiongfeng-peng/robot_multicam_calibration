"""Live/offline independent sanity check: project the robot EEF pose to each RAW RGB camera.

T_A_B maps B -> A.  Uses OpenCV optical coordinates and calibrated Brown-Conrady K,D.
This is a VISUAL sanity check, not an independent metric calibration certification.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .charuco import intrinsics_from_file
from .geometry import inv
from .sources import Inputs

CAMERAS = ('wrist', 'cam1', 'cam2')
AXES = (('X', (0, 0, 255)), ('Y', (0, 180, 0)), ('Z', (255, 70, 0)))  # BGR


def _valid_T(T, label):
    a = np.asarray(T, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError(f'{label} must be finite 4x4')
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-3) or np.linalg.det(a[:3, :3]) < .99:
        raise ValueError(f'{label} is not SE(3)')
    return a


def load_extrinsics(output_dir):
    out = Path(output_dir)
    X = _valid_T(np.load(out / 'T_gripper_wrist.npy'), 'T_G_W')
    T1 = _valid_T(np.load(out / 'T_base_cam1.npy'), 'T_B_C1')
    T2 = _valid_T(np.load(out / 'T_base_cam2.npy'), 'T_B_C2')
    return X, T1, T2


def camera_base_poses(T_B_G, T_G_W, T_B_C1, T_B_C2):
    return {'wrist': T_B_G @ T_G_W, 'cam1': T_B_C1, 'cam2': T_B_C2}


def project_eef(K, D, T_B_C, T_B_G, origin_offset_m=(0, 0, 0), axis_length_m=.05, extra_points=None):
    """Project G-frame origin and local XYZ endpoints (or TCP offset). Return pixel positions and camera depth.

    origin_offset_m is a known *point in G* (e.g. physical TCP), with axes parallel to G.
    If the tool has a separate rotated TCP frame, pass its full orientation after
    composing T_B_G @ T_G_TCP before calling this function.
    """
    T_B_C = _valid_T(T_B_C, 'T_B_C'); T_B_G = _valid_T(T_B_G, 'T_B_G')
    offset = np.asarray(origin_offset_m, float).reshape(3)
    if not np.isfinite(offset).all() or not np.isfinite(axis_length_m) or axis_length_m <= 0:
        raise ValueError('Origin offset / axis length invalid')
    p_G = [offset]
    p_G.extend([offset + np.eye(3)[i] * axis_length_m for i in range(3)])
    keys = ['origin', 'X', 'Y', 'Z']
    for name, p in (extra_points or {}).items():
        if name in keys:
            raise ValueError(f'duplicate name {name}')
        q = np.asarray(p, float).reshape(3)
        if not np.isfinite(q).all():
            raise ValueError(f'invalid point: {name}')
        keys.append(str(name)); p_G.append(q)
    p_G = np.asarray(p_G)
    T_C_G = inv(T_B_C) @ T_B_G
    p_C = (T_C_G[:3, :3] @ p_G.T).T + T_C_G[:3, 3]
    front = p_C[:, 2] > 1e-5
    uv = np.full((len(p_G), 2), np.nan, dtype=float)
    if np.any(front):
        pts, _ = cv2.projectPoints(np.asarray(p_C[front], dtype=np.float64),
                                   np.zeros(3), np.zeros(3), np.asarray(K, float), np.asarray(D, float))
        uv[front] = pts.reshape(-1, 2)
    return {name: {'uv': uv[i].tolist() if np.isfinite(uv[i]).all() else None, 'depth_m': float(p_C[i, 2]), 'in_front': bool(front[i])}
            for i, name in enumerate(keys)}


def _put_text(img, txt, x, y, size=.55, color=(240, 245, 248)):
    cv2.putText(img, str(txt), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, str(txt), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def render_pose_overlay(image, K, D, T_B_C, T_B_G, label, origin_offset_m=(0, 0, 0), axis_length_m=.05, points=None):
    """Draw projected position and orientation; report visibility, and pixel/depth positions."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'{label}: expected BGR HxWx3 raw camera image')
    out = image.copy()
    h, w = out.shape[:2]
    projected = project_eef(K, D, T_B_C, T_B_G, origin_offset_m, axis_length_m, points)
    origin = projected['origin']
    def inside(item):
        if item['uv'] is None:
            return False
        x, y = item['uv']
        return bool(item['in_front'] and np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h)
    origin_visible = inside(origin)
    visible_axes = []
    if origin_visible:
        cx, cy = map(lambda x: int(round(x)), origin['uv'])
        cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 9, (0, 0, 0), 1, cv2.LINE_AA)
        for axis, bgr in AXES:
            end = projected[axis]
            if end['in_front'] and end['uv'] is not None and all(np.isfinite(end['uv'])):
                ex, ey = map(lambda x: int(np.clip(round(x), -100000, 100000)), end['uv'])
                cv2.line(out, (cx, cy), (ex, ey), bgr, 3, cv2.LINE_AA)
                if inside(end):
                    _put_text(out, axis, ex + 6, ey - 7, .65, bgr)
                    visible_axes.append(axis)
        _put_text(out, 'EEF', cx + 10, cy + 17, .5, (0, 255, 255))
    for name, item in projected.items():
        if name in ('origin', 'X', 'Y', 'Z') or not inside(item):
            continue
        x, y = map(lambda a: int(round(a)), item['uv'])
        cv2.drawMarker(out, (x, y), (255, 240, 40), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        _put_text(out, name, x + 7, y - 8, .4)
    R_B_G = T_B_G[:3, :3]
    euler = Rotation.from_matrix(R_B_G).as_euler('xyz', degrees=True)
    p_B = T_B_G[:3, 3]
    status = ('VISIBLE' if origin_visible else ('BEHIND CAMERA' if not origin['in_front'] else 'OUTSIDE IMAGE'))
    cv2.rectangle(out, (0, 0), (min(w, 680), min(h, 96)), (9, 23, 40), -1)
    _put_text(out, f'{label} | EEF pose projection | {status}', 9, 23, .56)
    _put_text(out, f'Base position [m] {p_B[0]:+.3f} {p_B[1]:+.3f} {p_B[2]:+.3f}', 9, 47, .48)
    _put_text(out, f'Base XYZ Euler [deg] {euler[0]:+.1f} {euler[1]:+.1f} {euler[2]:+.1f}', 9, 70, .48)
    if h > 114:
        _put_text(out, 'XYZ axes: RED / GREEN / BLUE (OpenCV optical frame)', 9, h - 12, .44)
    return out, {'camera': label, 'status': status, 'origin_in_image': origin_visible,
                 'visible_axes': visible_axes, 'projections': projected, 'image_size': [w, h],
                 'eef_position_base_m': p_B.tolist(), 'eef_euler_xyz_extrinsic_deg': euler.tolist()}


def _read_extra_points(path):
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Point file must be {"name": [x,y,z], ...} in gripper frame, meters')
    return raw


def _load_pose_json(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return _valid_T(raw['T_base_gripper'], 'T_base_gripper')


def _load_cameras(cfg):
    cams = {}
    for name in CAMERAS:
        K, D, size = intrinsics_from_file(cfg['cameras'][name]['intrinsics'])
        cams[name] = (K, D, size)
    return cams


def _validate_sizes(frames, cameras):
    for name, frame in frames.items():
        expected = cameras[name][2]
        if expected is not None and frame is not None and (frame.shape[1], frame.shape[0]) != tuple(expected):
            raise ValueError(f'{name} image {frame.shape[1]}x{frame.shape[0]} does not match intrinsics {expected}. No resizing allowed.')


def _draw_all(frames, T_B_G, cameras, extrinsics, origin_offset_m, axis_length_m, points):
    T_G_W, T_B_C1, T_B_C2 = extrinsics
    T_by_cam = camera_base_poses(T_B_G, T_G_W, T_B_C1, T_B_C2)
    overlays = {}; reports = {}
    for name in CAMERAS:
        im = frames.get(name)
        if im is None:
            reports[name] = {'status': 'NO FRAME'}
            continue
        K, D, _ = cameras[name]
        overlays[name], reports[name] = render_pose_overlay(
            im, K, D, T_by_cam[name], T_B_G, name, origin_offset_m, axis_length_m, points)
    return overlays, reports


def _save_snapshot(output, raw, overlays, reports, T_B_G):
    folder = Path(output) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    for name in CAMERAS:
        if name in raw and raw[name] is not None:
            cv2.imwrite(str(folder / f'{name}_raw.png'), raw[name])
        if name in overlays:
            cv2.imwrite(str(folder / f'{name}_eef_overlay.png'), overlays[name])
    (folder / 'validation_report.json').write_text(json.dumps({
        'T_base_gripper': T_B_G.tolist(), 'results': reports,
        'caveat': 'Visual FK-to-image projection sanity check only; no independently measured 2D/3D error.',
    }, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return folder


def validate_projection(cfg, images_dir=None, pose_file=None, save_dir=None,
                        origin_offset_m=(0, 0, 0), axis_length_m=.05, points_json=None):
    """Offline: image_dir/{wrist,cam1,cam2}.png + pose JSON. Online: GUI cameras + live FK."""
    cameras = _load_cameras(cfg)
    extrinsics = load_extrinsics(cfg['paths']['output'])
    points = _read_extra_points(points_json)
    save_dir = Path(save_dir or Path(cfg['paths']['output']) / 'projection_validation')
    if images_dir:
        folder = Path(images_dir)
        raw = {}
        for name in CAMERAS:
            f = folder / f'{name}.png'
            if not f.exists():
                raise FileNotFoundError(f'{f} missing. Offline mode requires wrist.png, cam1.png, cam2.png')
            raw[name] = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if raw[name] is None:
                raise ValueError(f'Could not read {f}')
        _validate_sizes(raw, cameras)
        pose_filename = pose_file or cfg['robot'].get('pose_file')
        if not pose_filename:
            raise ValueError('Offline mode requires --pose-file file.json with T_base_gripper')
        T_B_G = _load_pose_json(pose_filename)
        overlays, reports = _draw_all(raw, T_B_G, cameras, extrinsics, origin_offset_m, axis_length_m, points)
        folder_saved = _save_snapshot(save_dir, raw, overlays, reports, T_B_G)
        for name, report in reports.items():
            print(f'{name}: {report["status"]}')
        print(f'Saved overlays and results: {folder_saved}')
        return folder_saved

    print('Live EEF overlay: [c] save current snapshot, [q] quit. Stop the robot to inspect an aligned pose.')
    print('XYZ in projected robot G frame. For physical TCP, supply --origin-offset-m (G coordinates).')
    inputs = Inputs(cfg, stage='wrist', camera_names=CAMERAS, require_robot=True)
    try:
        while True:
            raw = {}
            valid = True
            for name in CAMERAS:
                frame, stamp = inputs.frame(name)
                if stamp is None or time.monotonic() - stamp > 1.0:
                    frame = None
                    valid = False
                raw[name] = frame
            try:
                T_B_G = inputs.fk()
                _validate_sizes(raw, cameras)
                overlays, reports = _draw_all(raw, T_B_G, cameras, extrinsics, origin_offset_m, axis_length_m, points)
            except Exception as exc:
                valid = False
                print(f'[overlay] waiting for valid FK/images: {exc}', end='\r')
                overlays = {}; reports = {}
                T_B_G = None
            for name in CAMERAS:
                image = overlays.get(name)
                if image is None:
                    image = np.zeros((360, 480, 3), dtype=np.uint8)
                    _put_text(image, f'{name}: waiting for frame / FK', 16, 48)
                cv2.imshow(f'EEF projection | {name}', image)
            key = cv2.waitKey(40) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                if T_B_G is None or not valid:
                    print('\n[REJECTED] Not all image streams and FK are valid.')
                else:
                    folder_saved = _save_snapshot(save_dir, raw, overlays, reports, T_B_G)
                    print(f'\n[SAVED] {folder_saved}')
    finally:
        inputs.close()
        cv2.destroyAllWindows()
