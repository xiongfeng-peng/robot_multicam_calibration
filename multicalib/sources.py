"""Online camera/FK adapters. No automatic robot motion is issued by this package."""
import json
import threading
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from .geometry import make_T


class OpenCVCamera:
    """Asynchronous latest-frame capture (useful for USB cameras and RTSP)."""
    def __init__(self, config):
        device = config['device']
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {device}")
        for attr, key in [(cv2.CAP_PROP_FRAME_WIDTH, 'width'), (cv2.CAP_PROP_FRAME_HEIGHT, 'height')]:
            if config.get(key):
                self.cap.set(attr, int(config[key]))
        self.frame = None; self.stamp = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.stamp = time.monotonic()
            else:
                time.sleep(.05)

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.stamp

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


class PoseFileFK:
    """Read current FK from JSON file updated atomically by user robot adapter."""
    def __init__(self, path):
        self.path = Path(path)

    def latest(self):
        j = json.loads(self.path.read_text(encoding='utf-8'))
        T = np.asarray(j['T_base_gripper'], dtype=float).reshape(4, 4)
        stamp = float(j.get('stamp_unix_s', 0.0))
        if stamp == 0 or (abs(time.time() - stamp) > 3.0):
            raise RuntimeError("FK pose_file stamp_unix_s is missing/stale (>3s). Update atomically while sampling.")
        if not np.all(np.isfinite(T)) or abs(np.linalg.det(T[:3,:3]) - 1) > 1e-3:
            raise RuntimeError("FK is not a valid rigid transform.")
        return T

    def close(self):
        pass


class RosBridge:
    """ROS2 images and latest TF; spinning executor in a background thread.

    Use ROS2 environment with rclpy, tf2_ros, cv_bridge, sensor_msgs available.
    ROS2 header timestamps are not used to interpolate robot poses. Calibration
    captures must be made with the robot, board and cameras stationary.
    """
    def __init__(self, camera_sources, base_frame, gripper_frame):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image, CompressedImage
            from cv_bridge import CvBridge
            from tf2_ros import Buffer, TransformListener
            from rclpy.time import Time
        except ImportError as exc:
            raise RuntimeError("ROS2 adapter requires system ROS2 Python packages; source /opt/ros/.../setup.bash") from exc
        self.rclpy = rclpy; self.Time = Time
        if not rclpy.ok():
            rclpy.init()
            self.owns_ros = True
        else:
            self.owns_ros = False
        self.node = Node('robot_multicam_online_calibration')
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self.node)
        self.base_frame = base_frame; self.gripper_frame = gripper_frame
        self.bridge = CvBridge()
        self.images = {}
        self.lock = threading.Lock()
        self.subscribers = []
        for name, spec in camera_sources.items():
            compressed = bool(spec.get('compressed', False))
            msg_type = CompressedImage if compressed else Image
            def callback(msg, name=name, compressed=compressed):
                try:
                    image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') if compressed else \
                            self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    with self.lock:
                        self.images[name] = (image, time.monotonic())
                except Exception as e:
                    self.node.get_logger().error(f"{name} image decode error: {e}")
            sub = self.node.create_subscription(msg_type, str(spec['topic']), callback, qos_profile_sensor_data)
            self.subscribers.append(sub)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def latest(self, name):
        with self.lock:
            item = self.images.get(name)
            return (item[0].copy(), item[1]) if item is not None else (None, None)

    def fk_latest(self):
        if self.base_frame is None or self.gripper_frame is None:
            raise RuntimeError('This camera-only ROS bridge has no FK interface')
        tf = self.tf.lookup_transform(self.base_frame, self.gripper_frame, self.Time())
        t = tf.transform.translation
        q = tf.transform.rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return make_T(R, [t.x, t.y, t.z])

    def close(self):
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2)
        self.node.destroy_node()
        if self.owns_ros and self.rclpy.ok():
            self.rclpy.shutdown()


class Inputs:
    def __init__(self, config, stage, camera_names=None, require_robot=True):
        self.names = list(camera_names) if camera_names is not None else (['wrist'] if stage == 'wrist' else ['wrist', stage])
        self.cameras = {}
        self.bridge = None
        cam_configs = config['cameras']
        ros_specs = {name: cam_configs[name]['source'] for name in self.names if cam_configs[name]['source']['type'] == 'ros2'}
        robot_cfg = config['robot']
        if ros_specs or (require_robot and robot_cfg['type'] == 'ros2_tf'):
            self.bridge = RosBridge(ros_specs,
                                    robot_cfg['base_frame'] if require_robot else None,
                                    robot_cfg['gripper_frame'] if require_robot else None)
        for name in self.names:
            source = cam_configs[name]['source']
            if source['type'] == 'opencv':
                self.cameras[name] = OpenCVCamera(source)
            elif source['type'] != 'ros2':
                raise ValueError(f"Camera source type unsupported: {source['type']}")
        self.robot = PoseFileFK(robot_cfg['pose_file']) if require_robot and robot_cfg['type'] == 'pose_file' else None
        if require_robot and not self.robot and robot_cfg['type'] != 'ros2_tf':
            raise ValueError(f"Unsupported robot type: {robot_cfg['type']}")

    def frame(self, name):
        return self.cameras[name].latest() if name in self.cameras else self.bridge.latest(name)

    def fk(self):
        if self.robot is not None:
            return self.robot.latest()
        if self.bridge is None or self.bridge.base_frame is None:
            raise RuntimeError('FK is not configured for image-only intrinsic calibration')
        return self.bridge.fk_latest()

    def close(self):
        for camera in self.cameras.values():
            camera.close()
        if self.robot: self.robot.close()
        if self.bridge: self.bridge.close()
