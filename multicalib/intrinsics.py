"""ChArUco monocular RGB intrinsic calibration for wrist/cam1/cam2.

This stage needs images ONLY, never robot FK. Images may be acquired by moving
camera or board, but must depict multiple board orientations and screen positions.
We intentionally use cv2.calibrateCameraExtended + CharucoBoard.matchImagePoints:
these APIs remain available when cv2.aruco.calibrateCameraCharucoExtended is not.
"""
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np
import yaml

from .charuco import create_board
from .sources import Inputs


@dataclass
class IntrinsicDetection:
    obj: np.ndarray
    img: np.ndarray
    corners: np.ndarray
    ids: np.ndarray
    number: int
    bbox_fraction: float
    centroid: tuple


class IntrinsicDetector:
    """Detect corners without needing an existing K or distortion estimate."""
    def __init__(self, board_conf, min_corners=12):
        self.board = create_board(board_conf)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.min_corners = int(min_corners)

    def detect(self, bgr):
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(bgr)
        preview = bgr.copy()
        if marker_ids is not None and len(marker_ids):
            cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
        count = 0 if ids is None else len(ids)
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
        if count < self.min_corners:
            return None, preview, count
        obj, img = self.board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < self.min_corners:
            return None, preview, count
        xy = np.asarray(img, np.float32).reshape(-1, 2)
        wh = np.array([bgr.shape[1], bgr.shape[0]], dtype=float)
        span = np.ptp(xy, axis=0)
        bbox_fraction = float(np.prod(span / wh))
        center = tuple(map(float, np.mean(xy, axis=0) / wh))
        result = IntrinsicDetection(np.asarray(obj, np.float32).reshape(-1, 1, 3),
                                    np.asarray(img, np.float32).reshape(-1, 1, 2),
                                    corners, ids, len(obj), bbox_fraction, center)
        return result, preview, count


def intrinsic_image_dir(data_dir, camera):
    if camera not in ('wrist', 'cam1', 'cam2'):
        raise ValueError(f'Unknown camera: {camera}')
    return Path(data_dir) / 'intrinsics' / camera


def _read_images(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'No intrinsic image folder: {root}')
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
    if not paths:
        raise ValueError(f'No image files in {root}; capture with intrinsics capture first')
    return paths


def _banner(bgr, lines, ok):
    frame = bgr.copy()
    if frame.shape[1] > 900:
        frame = cv2.resize(frame, (900, round(frame.shape[0] * 900 / frame.shape[1])))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], min(92, frame.shape[0])), (24, 24, 24), -1)
    color = (90, 225, 90) if ok else (90, 100, 240)
    for i, line in enumerate(lines):
        cv2.putText(frame, str(line)[:110], (12, 24+22*i), cv2.FONT_HERSHEY_SIMPLEX, .57,
                    color, 1, cv2.LINE_AA)
    return frame


def capture_intrinsics(cfg, camera):
    """Image-only live collection; uses same OpenCV / ROS2 source as extrinsics."""
    options = cfg.get('intrinsic_calibration', {})
    minimum = int(options.get('min_charuco_corners', 12))
    detector = IntrinsicDetector(cfg['board'], min_corners=minimum)
    root = intrinsic_image_dir(cfg['paths']['dataset'], camera)
    root.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(cfg, camera, camera_names=[camera], require_robot=False)
    taken = len(_read_images(root)) if any(root.iterdir()) else 0
    wait_s = float(cfg.get('calibration', {}).get('max_frame_age_s', 1.))
    print(f'\n[intrinsics {camera}] c: capture, q/ESC: quit; images in {root}')
    print('Move the ChArUco board to diverse locations/tilts. Robot FK is NOT required.')
    print('Use ORIGINAL raw RGB resolution and fixed focus/exposure/zoom settings.')
    try:
        while True:
            img, stamp = inputs.frame(camera)
            observation = None
            if img is None:
                preview = np.zeros((360, 640, 3), np.uint8)
                preview = _banner(preview, [f'{camera}: waiting for camera frame...'], False)
            else:
                observation, preview, count = detector.detect(img)
                age = float('inf') if stamp is None else time.monotonic()-stamp
                good = observation is not None and age <= wait_s
                desc = (f'valid={count} bbox={observation.bbox_fraction:.3f}'
                        if observation is not None else f'corners={count} (need >= {minimum})')
                preview = _banner(preview, [f'{camera} | {desc}',
                                                    f'age={age:.2f}s saved={taken} [c] capture [q] quit'], good)
            cv2.imshow(f'intrinsics: {camera}', preview)
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), 27):
                break
            if key != ord('c'):
                continue
            # Re-read to avoid saving an arbitrarily old frame.
            fresh, fresh_stamp = inputs.frame(camera)
            if fresh is None or fresh_stamp is None or time.monotonic()-fresh_stamp > wait_s:
                print('[REJECTED] Missing/stale image')
                continue
            det, _, count = detector.detect(fresh)
            if det is None:
                print(f'[REJECTED] Need {minimum} visible ChArUco corners; found {count}')
                continue
            names = [int(p.stem.split('_')[-1]) for p in root.glob('sample_*.png')
                     if p.stem.split('_')[-1].isdigit()]
            n = max(names, default=-1) + 1
            filename = root / f'sample_{n:04d}.png'
            if not cv2.imwrite(str(filename), fresh):
                print(f'[REJECTED] Failed to write {filename}')
                continue
            info = {'camera': camera, 'filename': filename.name, 'timestamp_unix_s': time.time(),
                    'resolution_wh': [fresh.shape[1], fresh.shape[0]], 'n_corners': det.number,
                    'bbox_fraction': det.bbox_fraction, 'center_fraction': det.centroid}
            (root / f'sample_{n:04d}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
            taken += 1
            print(f'[SAVED] {filename} corners={det.number}, bbox={det.bbox_fraction:.3f}')
    finally:
        inputs.close()
        cv2.destroyAllWindows()


def _calibrate(detections, size, flags=0):
    obj = [d.obj for d in detections]
    img = [d.img for d in detections]
    rms, K, D, rvecs, tvecs, std_int, std_ext, per_view = cv2.calibrateCameraExtended(
        obj, img, size, None, None,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-9))
    error = []
    for ob, r, t in zip(detections, rvecs, tvecs):
        projected, _ = cv2.projectPoints(ob.obj, r, t, K, D)
        diffs = projected.reshape(-1, 2) - ob.img.reshape(-1, 2)
        error.append(float(np.sqrt(np.mean(np.sum(diffs**2, axis=1)))))
    return float(rms), K, D.reshape(-1), error, np.asarray(std_int).reshape(-1)


def solve_intrinsics(cfg, camera, image_dir=None, overwrite=False):
    """Estimate pinhole+5 Brown-Conrady coefficients and write compatible YAML.

    Protect existing non-placeholder intrinsics unless overwrite=True.
    """
    if camera not in cfg['cameras']:
        raise ValueError(camera)
    options = cfg.get('intrinsic_calibration', {})
    required = int(options.get('min_images', 15))
    min_corners = int(options.get('min_charuco_corners', 12))
    max_rms = float(options.get('max_rms_px', 1.5))
    max_drop_frac = float(options.get('max_outlier_fraction', .2))
    root = Path(image_dir) if image_dir is not None else intrinsic_image_dir(cfg['paths']['dataset'], camera)
    target = Path(cfg['cameras'][camera]['intrinsics'])
    if target.exists():
        old = yaml.safe_load(target.read_text(encoding='utf-8')) if target.suffix != '.json' else json.loads(target.read_text(encoding='utf-8'))
        if not old.get('placeholder', False) and not overwrite:
            raise FileExistsError(f'{target} already contains calibrated intrinsics. Add --overwrite to recalibrate.')
    det = IntrinsicDetector(cfg['board'], min_corners)
    paths = _read_images(root)
    detections = []
    accepted_paths = []
    rejected = []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            rejected.append({'image': str(path), 'reason': 'unreadable'})
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            raise ValueError(f'Mixed image resolution: {path} is {size}, expected {image_size}')
        obs, _, n = det.detect(image)
        if obs is None:
            rejected.append({'image': str(path), 'reason': f'too few ChArUco corners ({n})'})
            continue
        detections.append(obs)
        accepted_paths.append(path)
    if len(detections) < required:
        raise ValueError(f'{camera}: only {len(detections)} valid images of {len(paths)}; '
                         f'need >= {required} from varied positions and tilts')
    # Keep a simple 5-coefficient pinhole model (k1,k2,p1,p2,k3) so downstream
    # OpenCV solvePnP sees the exact model it expects.
    flags = 0
    remaining = np.arange(len(detections))
    max_drops = min(max(0, len(detections)-required), int(np.floor(len(detections)*max_drop_frac)))
    dropped = []
    for iteration in range(max_drops+1):
        use = [detections[i] for i in remaining]
        try:
            rms, K, D, errors, std_int = _calibrate(use, image_size, flags)
        except cv2.error as exc:
            raise RuntimeError(f'OpenCV intrinsic solve failed; add more board tilts/corner coverage: {exc}') from exc
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)) or K[0,0]<=0 or K[1,1]<=0:
            raise RuntimeError('Invalid intrinsic solve: camera matrix/distortion not finite or focal length negative')
        if iteration == max_drops:
            break
        med = float(np.median(errors))
        mad = 1.4826*float(np.median(np.abs(np.asarray(errors)-med)))
        worst = int(np.argmax(errors))
        threshold = max(.6, med+3*mad)
        if errors[worst] <= threshold:
            break
        dropped.append({'image': str(accepted_paths[remaining[worst]]), 'rmse_px': errors[worst]})
        remaining = np.delete(remaining, worst)
    # Intrinsic source resolution MUST agree with source nominal resolution where provided.
    src = cfg['cameras'][camera]['source']
    if src.get('width') and src.get('height'):
        expected = (int(src['width']), int(src['height']))
        if image_size != expected:
            raise ValueError(f'{camera}: captured images {image_size} vs configured source {expected}. '
                             'Update source resolution or recapture. No resized images are supported.')
    if rms > max_rms:
        raise ValueError(f'{camera}: calibration RMS={rms:.3f}px > max_rms_px={max_rms:.3f}px; '
                         'no intrinsics written. Add sharper and more diverse images, or review the limit.')
    # Soft diagnostics: warnings are informational and do NOT imply accuracy certification.
    centroids = np.array([detections[i].centroid for i in remaining])
    spread = np.ptp(centroids, axis=0)
    cover = [detections[i].bbox_fraction for i in remaining]
    warnings = []
    if float(max(cover)) < .12:
        warnings.append('Calibration board stays small in image; move closer for some frames.')
    if float(min(spread)) < .12:
        warnings.append('Board centers cover a narrow image region; also sample edges/corners.')
    if float(np.max(abs(D[:2]))) > 2.0:
        warnings.append('Very large radial distortion coefficients; verify view diversity/lens model.')
    report = {
        'camera': camera, 'method': 'ChArUco + cv2.calibrateCameraExtended',
        'model': 'pinhole_plumb_bob_5', 'resolution_wh': list(image_size),
        'rms_px': rms, 'max_accepted_rms_px': max_rms,
        'K': K.reshape(3, 3).tolist(), 'D': D.tolist(),
        'images_total': len(paths), 'images_detected': len(detections), 'images_used': len(remaining),
        'images_rejected_detection': rejected, 'images_rejected_outlier': dropped,
        'per_image': [{'path': str(accepted_paths[i]), 'rmse_px': float(err),
                       'corner_count': detections[i].number,
                       'bbox_fraction': detections[i].bbox_fraction}
                      for i,err in zip(remaining, errors)],
        'intrinsic_stddev': std_int.tolist(), 'centroid_spread_fraction': spread.tolist(),
        'max_board_bbox_fraction': max(cover), 'warnings': warnings,
        'note': ('RMS is image-plane reprojection fit; it does not certify 3D depth accuracy or robot-base accuracy. '
                 'Capture images with focus/zoom/resolution fixed and use raw unrectified RGB.')
    }
    # Only write once fully validated. Atomic write to avoid downstream reading partial K.
    raw = {'width': int(image_size[0]), 'height': int(image_size[1]),
           'image_width': int(image_size[0]), 'image_height': int(image_size[1]),
           'camera_name': camera, 'distortion_model': 'plumb_bob',
           'K': K.reshape(-1).tolist(), 'D': D.tolist(),
           'camera_matrix': {'rows': 3, 'cols': 3, 'data': K.reshape(-1).tolist()},
           'distortion_coefficients': {'rows': 1, 'cols': len(D), 'data': D.tolist()},
           'calibration_rms_px': rms, 'calibration_images_used': len(remaining),
           'board': cfg['board']}
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and overwrite:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = target.with_name(target.name + f'.bak_{timestamp}')
        shutil.copy2(target, backup)
        print(f'[BACKUP] Previous intrinsics saved to {backup}')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix=target.suffix,
                                     prefix='.intrinsics_', dir=target.parent, delete=False) as f:
        temp_path = Path(f.name)
        if target.suffix == '.json':
            json.dump(raw, f, indent=2)
        else:
            yaml.safe_dump(raw, f, sort_keys=False)
    try:
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    report_dir = Path(cfg['paths']['output']) / 'intrinsics'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f'{camera}_report.json'
    report_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Human-inspectable rectification preview; do not use this processed image in PnP.
    raw_preview = cv2.imread(str(accepted_paths[remaining[-1]]))
    if raw_preview is not None:
        rectified = cv2.undistort(raw_preview, K, D)
        raw_small = raw_preview
        if raw_preview.shape[1] > 1000:
            target_w = 1000
            target_h = round(raw_preview.shape[0] * target_w / raw_preview.shape[1])
            raw_small = cv2.resize(raw_preview, (target_w, target_h))
            rectified = cv2.resize(rectified, (target_w, target_h))
        side_by_side = np.concatenate((raw_small, rectified), axis=1)
        preview_path = report_dir / f'{camera}_raw_vs_undistorted.png'
        cv2.imwrite(str(preview_path), side_by_side)
        print(f'  distortion preview (left raw / right undistorted): {preview_path}')
    print(f'[INTRINSICS SOLVED] camera={camera}, used={len(remaining)}, rms={rms:.3f}px, '
          f'fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, cx={K[0,2]:.3f}, cy={K[1,2]:.3f}')
    print(f'  calibration: {target}\n  report: {report_file}')
    for w in warnings:
        print(f'  [WARNING] {w}')
    return K, D, report
