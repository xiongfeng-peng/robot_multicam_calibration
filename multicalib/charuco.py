"""ChArUco target detection and calibrated PnP. Camera frames are optical frames."""
from dataclasses import dataclass
import numpy as np
import cv2
from .geometry import make_T


@dataclass
class Observation:
    T_C_P: np.ndarray
    reproj_rmse_px: float
    n_corners: int
    annotated: np.ndarray
    cover_fraction: float


def intrinsics_from_file(path):
    import json
    import yaml
    from pathlib import Path
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    if raw.get('placeholder', False):
        raise ValueError(f"Intrinsics file is a placeholder: {path}. Replace it with the REAL camera calibration and remove placeholder: true.")
    # Supports native K/distortion or ROS CameraInfo YAML fields.
    K = raw.get("K", raw.get("camera_matrix"))
    if isinstance(K, dict):
        K = K["data"]
    D = raw.get("D", raw.get("distortion_coefficients", [0, 0, 0, 0, 0]))
    if isinstance(D, dict):
        D = D["data"]
    K = np.asarray(K, dtype=float).reshape(3, 3)
    D = np.asarray(D, dtype=float).ravel()
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"Bad intrinsics K: {path}")
    width = raw.get("width", raw.get("image_width"))
    height = raw.get("height", raw.get("image_height"))
    return K, D, (int(width), int(height)) if width and height else None


def create_board(board_conf):
    """Construct a single consistent ChArUco pattern for intrinsics and extrinsics."""
    aruco = cv2.aruco
    if not hasattr(aruco, 'CharucoDetector'):
        raise RuntimeError('opencv-contrib-python >= 4.8 is required')
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, board_conf['dictionary']))
    return aruco.CharucoBoard(
        (int(board_conf['squares_x']), int(board_conf['squares_y'])),
        float(board_conf['square_length_m']), float(board_conf['marker_length_m']), dictionary)


class CharucoPose:
    def __init__(self, board_conf, camera_intrinsics, min_corners=8, max_rmse_px=2.0):
        self.board = create_board(board_conf)
        if self.board.getChessboardCorners().shape[0] < min_corners:
            raise ValueError("Board has fewer corners than min_corners")
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.K, self.D, self.expected_wh = intrinsics_from_file(camera_intrinsics)
        self.min_corners = min_corners
        self.max_rmse_px = max_rmse_px

    def estimate(self, bgr):
        """Detect and return T_camera_optical_board or None if invalid."""
        if self.expected_wh is not None and (bgr.shape[1], bgr.shape[0]) != self.expected_wh:
            raise ValueError(f"Frame is {bgr.shape[1]}x{bgr.shape[0]}, intrinsics expect {self.expected_wh}.")
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(bgr)
        annotated = bgr.copy()
        if marker_ids is not None and len(marker_ids):
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
        if ids is None or len(ids) < self.min_corners:
            return None, annotated, int(0 if ids is None else len(ids))
        cv2.aruco.drawDetectedCornersCharuco(annotated, corners, ids)
        obj, img = self.board.matchImagePoints(corners, ids)
        if len(obj) < self.min_corners:
            return None, annotated, len(obj)
        success, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None, annotated, len(obj)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)
        rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
        if rmse > self.max_rmse_px or float(tvec.reshape(-1)[2]) <= 0:
            return None, annotated, len(obj)
        R, _ = cv2.Rodrigues(rvec)
        T = make_T(R, tvec.reshape(3))
        cv2.drawFrameAxes(annotated, self.K, self.D, rvec, tvec, float(self.board.getSquareLength()) * 2)
        # Fraction of image covered by detected corners bbox, a quick quality diagnostic.
        pts = img.reshape(-1, 2)
        box = np.ptp(pts, axis=0)
        area_fraction = float(np.prod(box) / (bgr.shape[1] * bgr.shape[0]))
        obs = Observation(T, rmse, len(obj), annotated, area_fraction)
        return obs, annotated, len(obj)


def generate_board_image(board_conf, out_path, square_pixels=240):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_conf["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (int(board_conf["squares_x"]), int(board_conf["squares_y"])),
        float(board_conf["square_length_m"]), float(board_conf["marker_length_m"]), dictionary
    )
    img = board.generateImage((int(board_conf["squares_x"] * square_pixels),
                               int(board_conf["squares_y"] * square_pixels)), marginSize=0)
    cv2.imwrite(str(out_path), img)
    return img
