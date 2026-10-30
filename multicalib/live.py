"""Operator-assisted calibration capture: live preview and single-key sample acquisition."""
import time
import numpy as np
import cv2
from .sources import Inputs
from .charuco import CharucoPose
from .geometry import error_T
from .dataset import save_sample, load_samples
from .solver import solve_wrist, solve_external


def _banner(image, lines, valid=True):
    scale = min(1.0, 850 / image.shape[1])
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    cv2.rectangle(image, (0, 0), (image.shape[1], 70), (20, 20, 20), -1)
    for idx, line in enumerate(lines):
        color = (110, 220, 110) if valid else (90, 90, 255)
        cv2.putText(image, str(line), (12, 23 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)
    return image


def _view(image, title):
    cv2.imshow(title, image)


def live_capture(cfg, stage):
    conf = cfg['calibration']
    names = ['wrist'] if stage == 'wrist' else ['wrist', stage]
    detectors = {name: CharucoPose(cfg['board'], cfg['cameras'][name]['intrinsics'],
                                 conf.get('min_charuco_corners', 8), conf.get('max_pnp_rmse_px', 2.0))
                 for name in names}
    inputs = Inputs(cfg, stage)
    print(f"\nLive stage={stage}: press [c] to capture, [s] to solve now, [q] to quit.")
    print("Important: stop robot and hold ChArUco board completely still before c.")
    print("If PnP result is invalid / camera frame stale / FK stale, capture is rejected.")
    capture_count = len(load_samples(cfg['paths']['dataset'], stage))
    latest = {}
    loop_idx = 0
    try:
        while True:
            loop_idx += 1
            for name in names:
                frame, stamp = inputs.frame(name)
                if frame is None:
                    blank = np.zeros((320, 480, 3), np.uint8)
                    latest[name] = (None, None, None)
                    _view(name, _banner(blank, [f"{name}: waiting for frames"], False))
                    continue
                try:
                    obs, annotated, count = detectors[name].estimate(frame)
                    freshness = time.monotonic() - stamp if stamp is not None else float('inf')
                    good = obs is not None and freshness <= conf.get('max_frame_age_s', 1.)
                    if obs is not None:
                        msg = f"corners={count} rmse={obs.reproj_rmse_px:.2f}px area={obs.cover_fraction:.3f}"
                    else:
                        msg = f"corners={count} INVALID (min corners / reproj RMSE)"
                    latest[name] = (frame, obs, stamp)
                    _view(name, _banner(annotated, [f"{name} | {msg}", f"age={freshness:.2f}s samples={capture_count} [c]save [s]solve [q]quit"], good))
                except (cv2.error, ValueError) as exc:
                    latest[name] = (frame, None, stamp)
                    _view(name, _banner(frame.copy(), [f"{name}: ERROR: {str(exc)[:80]}"], False))
            key = cv2.waitKey(30) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                try:
                    if stage == 'wrist':
                        solve_wrist(cfg['paths']['dataset'], cfg['paths']['output'])
                    else:
                        solve_external(cfg['paths']['dataset'], cfg['paths']['output'], stage)
                    print('[SOLVED] Saved matrices and metrics; you can collect more captures if needed.')
                except Exception as exc:
                    print(f'[SOLVE FAILED] {exc}')
                continue
            if key != ord('c'):
                continue
            if any(name not in latest or latest[name][1] is None for name in names):
                print("[REJECTED] Need valid ChArUco PnP in each camera.")
                continue
            stamps = [latest[name][2] for name in names]
            age = max(time.monotonic() - stamp for stamp in stamps)
            if age > conf.get('max_frame_age_s', 1.0):
                print(f"[REJECTED] Stale frame age={age:.3f}s")
                continue
            delta = max(stamps) - min(stamps)
            if delta > conf.get('max_pair_timestamp_delta_s', .2):
                print(f"[REJECTED] Frame acquisition delta {delta:.3f}s too large")
                continue
            try:
                T_B_G1 = inputs.fk()
                time.sleep(conf.get('settle_check_s', .3))
                T_B_G2 = inputs.fk()
                t_err, r_err = error_T(T_B_G1, T_B_G2)
                if t_err > conf.get('max_still_translation_m', .0005) or r_err > conf.get('max_still_rotation_deg', .15):
                    print(f"[REJECTED] EEF moved during stillness check {t_err*1000:.2f}mm / {r_err:.2f}deg")
                    continue
                # Refresh both frames *after* FK stillness check, re-run detections.
                fresh = {}
                for name in names:
                    img, stamp = inputs.frame(name)
                    if img is None or time.monotonic() - stamp > conf.get('max_frame_age_s', 1.):
                        raise RuntimeError(f"Stale {name} frame")
                    obs, _, count = detectors[name].estimate(img)
                    if obs is None:
                        raise RuntimeError(f"Invalid {name} ChArUco PnP, corners={count}")
                    fresh[name] = (img, stamp, obs)
                stamps2 = [fresh[name][1] for name in names]
                if max(stamps2) - min(stamps2) > conf.get('max_pair_timestamp_delta_s', .2):
                    raise RuntimeError('Frame delta exceeds configured limit')
                T_B_G = inputs.fk()
                t_err2, r_err2 = error_T(T_B_G2, T_B_G)
                if t_err2 > conf.get('max_still_translation_m', .0005) or r_err2 > conf.get('max_still_rotation_deg', .15):
                    raise RuntimeError('Robot moved just before camera capture')
                poses = {'T_B_G': T_B_G, 'T_W_P': fresh['wrist'][2].T_C_P}
                if stage != 'wrist':
                    poses['T_C_P'] = fresh[stage][2].T_C_P
                meta = {'timestamp_unix_s': time.time(), 'frame_skew_s': max(stamps2)-min(stamps2),
                        'cameras': {name: {'rmse_px': fresh[name][2].reproj_rmse_px,
                                           'corner_count': fresh[name][2].n_corners,
                                           'cover_fraction': fresh[name][2].cover_fraction} for name in names}}
                directory = save_sample(cfg['paths']['dataset'], stage, poses,
                                        {name: fresh[name][0] for name in names}, meta)
                capture_count += 1
                print(f"[SAVED] {directory}: FK static delta={t_err2*1000:.3f}mm/{r_err2:.3f}deg")
            except Exception as exc:
                print(f"[REJECTED] {exc}")
    finally:
        inputs.close()
        cv2.destroyAllWindows()
