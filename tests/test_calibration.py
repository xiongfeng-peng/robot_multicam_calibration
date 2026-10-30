import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from multicalib.geometry import make_T, inv, error_T
from multicalib.simulation import make_synthetic
from multicalib.solver import solve_wrist, solve_external
from multicalib.charuco import generate_board_image, CharucoPose, intrinsics_from_file
from multicalib.intrinsics import IntrinsicDetector, solve_intrinsics
import cv2
import yaml


class CalibrationTest(unittest.TestCase):
    def test_inverse(self):
        T = make_T(Rotation.from_euler('xyz', [12, 23, -7], degrees=True).as_matrix(), [.1, .3, .5])
        np.testing.assert_allclose(T @ inv(T), np.eye(4), atol=1e-10)

    def test_synthetic_exact(self):
        with tempfile.TemporaryDirectory() as d:
            data, output = Path(d) / 'data', Path(d) / 'out'
            X, C1, C2 = make_synthetic(data, noisy=False)
            XX, _, _ = solve_wrist(data, output)
            Z1, _ = solve_external(data, output, 'cam1')
            Z2, _ = solve_external(data, output, 'cam2')
            for truth, est in [(X, XX), (C1, Z1), (C2, Z2)]:
                t, r = error_T(truth, est)
                self.assertLess(t, 1e-5)
                self.assertLess(r, .0001)

    def test_synthetic_noise(self):
        with tempfile.TemporaryDirectory() as d:
            data, output = Path(d) / 'data', Path(d) / 'out'
            X, C1, C2 = make_synthetic(data, noisy=True)
            XX, _, _ = solve_wrist(data, output)
            Z1, _ = solve_external(data, output, 'cam1')
            Z2, _ = solve_external(data, output, 'cam2')
            for truth, est in [(X, XX), (C1, Z1), (C2, Z2)]:
                t, r = error_T(truth, est)
                self.assertLess(t, .008)
                self.assertLess(r, 1.0)

    def test_charuco_pnp_rendered_image(self):
        board = dict(dictionary='DICT_5X5_1000', squares_x=8, squares_y=6,
                     square_length_m=.035, marker_length_m=.025)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            image = generate_board_image(board, d / 'board.png', 120)
            K = np.array([[650., 0, 320], [0, 650., 240], [0, 0, 1.]])
            truth = make_T(Rotation.from_euler('xyz', [7, -10, 5], degrees=True).as_matrix(),
                           [-.14, -.105, .52])
            corners3d = np.array([[0, 0, 0], [.28, 0, 0], [.28, .21, 0], [0, .21, 0]], float)
            uv, _ = cv2.projectPoints(corners3d, cv2.Rodrigues(truth[:3,:3])[0],
                                      truth[:3,3], K, np.zeros(5))
            corners2d = np.array([[0, 0], [image.shape[1]-1, 0],
                                  [image.shape[1]-1, image.shape[0]-1],
                                  [0, image.shape[0]-1]], np.float32)
            warp = cv2.getPerspectiveTransform(corners2d, uv.reshape(4,2).astype(np.float32))
            frame = cv2.warpPerspective(image, warp, (640, 480), borderValue=255)
            (d/'intrinsics.yaml').write_text(yaml.safe_dump({
                'width': 640, 'height': 480, 'K': K.reshape(-1).tolist(), 'D': [0]*5}))
            detector = CharucoPose(board, d/'intrinsics.yaml')
            pose, _, n = detector.estimate(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            self.assertIsNotNone(pose)
            self.assertGreater(n, 20)
            translation_error, rotation_error = error_T(truth, pose.T_C_P)
            self.assertLess(translation_error, .004)
            self.assertLess(rotation_error, .6)


    def test_intrinsics_from_rendered_charuco_images(self):
        """End-to-end image read -> ChArUco corners -> K/D YAML -> external PnP load."""
        rng = np.random.default_rng(543)
        board = dict(dictionary='DICT_5X5_1000', squares_x=8, squares_y=6,
                     square_length_m=.035, marker_length_m=.025)
        true_K = np.array([[680., 0, 480.], [0, 675., 360.], [0, 0, 1.]])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames = root / 'data' / 'intrinsics' / 'wrist'
            frames.mkdir(parents=True)
            board_img = generate_board_image(board, root / 'board.png', 100)
            source_uv = np.float32([[0, 0], [board_img.shape[1]-1, 0],
                                    [board_img.shape[1]-1, board_img.shape[0]-1],
                                    [0, board_img.shape[0]-1]])
            board_xyz = np.float32([[0, 0, 0], [.28, 0, 0], [.28, .21, 0], [0, .21, 0]])
            detector = IntrinsicDetector(board)
            saved = 0
            for _ in range(32):
                eul = rng.uniform([-35, -35, -50], [35, 35, 50])
                R = Rotation.from_euler('xyz', eul, degrees=True).as_matrix()
                t = (np.array([rng.uniform(-.13, .13), rng.uniform(-.12, .12), rng.uniform(.52, .85)])
                     - R @ np.array([.14, .105, 0]))
                uv, _ = cv2.projectPoints(board_xyz, cv2.Rodrigues(R)[0], t, true_K, np.zeros(5))
                uv = uv.reshape(4,2)
                if (uv[:,0].min() < 24 or uv[:,0].max() > 935 or
                    uv[:,1].min() < 24 or uv[:,1].max() > 695):
                    continue
                warp = cv2.getPerspectiveTransform(source_uv, uv.astype(np.float32))
                image = cv2.warpPerspective(board_img, warp, (960, 720), borderValue=255)
                det, _, _ = detector.detect(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                if det is not None:
                    cv2.imwrite(str(frames / f'sample_{saved:04d}.png'), image)
                    saved += 1
            self.assertGreaterEqual(saved, 20)
            intr_file = root / 'wrist_intrinsics.yaml'
            intr_file.write_text('placeholder: true\n', encoding='utf-8')
            config = {'board': board,
                      'paths': {'dataset': str(root / 'data'), 'output': str(root / 'output')},
                      'cameras': {'wrist': {'intrinsics': str(intr_file),
                                            'source': {'type': 'opencv', 'width': 960, 'height': 720}}},
                      'intrinsic_calibration': {'min_images': 15, 'max_rms_px': 1.5}}
            K, D, report = solve_intrinsics(config, 'wrist')
            self.assertLess(report['rms_px'], .5)
            self.assertLess(np.max(np.abs(K - true_K)), 5.)
            self.assertEqual(D.size, 5)
            K2, D2, wh = intrinsics_from_file(intr_file)
            np.testing.assert_allclose(K, K2)
            np.testing.assert_allclose(D, D2)
            self.assertEqual(wh, (960, 720))
            self.assertTrue((root / 'output' / 'intrinsics' / 'wrist_report.json').exists())
            with self.assertRaises(FileExistsError):
                solve_intrinsics(config, 'wrist')
            # Verify safe explicit overwrite creates backup of earlier calibration.
            solve_intrinsics(config, 'wrist', overwrite=True)
            self.assertEqual(len(list(root.glob('wrist_intrinsics.yaml.bak_*'))), 1)
            # Reject configuration/image dimension mismatch without writing intrinsics.
            config['cameras']['wrist']['source']['width'] = 640
            with self.assertRaises(ValueError):
                solve_intrinsics(config, 'wrist', overwrite=True)

    def test_generate_board(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'board.png'
            img = generate_board_image(dict(dictionary='DICT_5X5_1000', squares_x=8, squares_y=6,
                                            square_length_m=.035, marker_length_m=.025), p, 80)
            self.assertEqual(img.shape, (480, 640))
            self.assertTrue(p.is_file())


if __name__ == '__main__':
    unittest.main()
