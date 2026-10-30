"""Numerical EEF reprojection validation tests, no camera or robot hardware required."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from multicalib.geometry import inv, make_T
from multicalib.projection import (camera_base_poses, load_extrinsics,
                                   project_eef, render_pose_overlay, validate_projection)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]], float)
        self.D = np.zeros(5)

    def test_perspective_origin_and_axes(self):
        T_B_C = make_T(t=[0, 0, -1])
        T_B_G = np.eye(4)
        projected = project_eef(self.K, self.D, T_B_C, T_B_G, axis_length_m=.1)
        np.testing.assert_allclose(projected['origin']['uv'], [320, 240], atol=1e-7)
        np.testing.assert_allclose(projected['X']['uv'], [390, 240], atol=1e-7)
        np.testing.assert_allclose(projected['Y']['uv'], [320, 310], atol=1e-7)
        np.testing.assert_allclose(projected['Z']['uv'], [320, 240], atol=1e-7)
        self.assertAlmostEqual(projected['origin']['depth_m'], 1)

    def test_wrist_composition(self):
        T_B_G = make_T(t=[.2, .1, .3])
        T_G_W = make_T(t=[0, 0, -.2])
        transforms = camera_base_poses(T_B_G, T_G_W, np.eye(4), np.eye(4))
        np.testing.assert_allclose(inv(transforms['wrist']) @ T_B_G, inv(T_G_W), atol=1e-10)

    def test_behind_camera_and_bad_offsets(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        render, report = render_pose_overlay(img, self.K, self.D, np.eye(4),
                                             make_T(t=[0, 0, -1]), 'cam1')
        self.assertEqual(report['status'], 'BEHIND CAMERA')
        self.assertFalse(report['origin_in_image'])
        self.assertEqual(render.shape, img.shape)
        with self.assertRaises(ValueError):
            project_eef(self.K, self.D, np.eye(4), np.eye(4), axis_length_m=-.1)

    def test_offline_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'output').mkdir()
            (root/'image').mkdir()
            for f in ('T_gripper_wrist.npy', 'T_base_cam1.npy', 'T_base_cam2.npy'):
                # place the optical cameras 1m behind EEF along Z
                np.save(root/'output'/f, make_T(t=[0, 0, -1]))
            for name in ('wrist','cam1','cam2'):
                cv2.imwrite(str(root/'image'/f'{name}.png'), np.ones((480, 640, 3), np.uint8)*175)
                (root/f'{name}.yaml').write_text('width: 640\nheight: 480\nK: [700,0,320,0,700,240,0,0,1]\nD: [0,0,0,0,0]\n')
            (root/'pose.json').write_text(json.dumps({'T_base_gripper': np.eye(4).tolist()}))
            cfg = {'paths': {'output': str(root/'output')},
                   'cameras': {name: {'intrinsics': str(root/f'{name}.yaml')} for name in ('wrist','cam1','cam2')},
                   'robot': {'pose_file': str(root/'pose.json')}}
            path = validate_projection(cfg, images_dir=root/'image', save_dir=root/'check', axis_length_m=.05)
            self.assertTrue((path/'cam1_eef_overlay.png').exists())
            self.assertTrue((path/'cam2_eef_overlay.png').exists())
            self.assertTrue((path/'wrist_eef_overlay.png').exists())
            report = json.loads((path/'validation_report.json').read_text())
            self.assertEqual(report['results']['cam1']['status'], 'VISIBLE')
            self.assertTrue(report['results']['wrist']['origin_in_image'])
            # An invisible EEF should still produce a valid strict-JSON report.
            (root/'pose.json').write_text(json.dumps({'T_base_gripper': make_T(t=[0, 0, -2]).tolist()}))
            path2 = validate_projection(cfg, images_dir=root/'image', save_dir=root/'check')
            report2 = json.loads((path2/'validation_report.json').read_text())
            self.assertEqual(report2['results']['cam1']['status'], 'BEHIND CAMERA')
            self.assertIsNone(report2['results']['cam1']['projections']['origin']['uv'])


if __name__ == '__main__':
    unittest.main()
