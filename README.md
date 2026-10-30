# Robot Multi-Camera Calibration v3（在线三相机内参 + 外参标定 + EEF 投影验证）

**目标：** 三台相机首先独立完成 **RGB 内参 K + 畸变 D 标定**，随后两个固定第三视角相机 `cam1` / `cam2` + 一个 Eye-in-Hand `wrist`，全部使用 **Robot Base** 坐标系。支持 Stage 2A/2B 移动 Wrist，亦允许 **每次采集之后移动 ChArUco 标定板**。机器人运动必须由操作员或已有控制器完成；此工程 **不会自动驱动机器人**。

## 0. 完整运行顺序：先内参，后外参（v2 新增）

```
Stage 0A: Wrist RGB     ChArUco 内参 -> configs/wrist_intrinsics.yaml
Stage 0B: Third Cam1 RGB ChArUco 内参 -> configs/cam1_intrinsics.yaml
Stage 0C: Third Cam2 RGB ChArUco 内参 -> configs/cam2_intrinsics.yaml
        ↓ 3 份 YAML 自动供外参阶段使用
Stage 1: Wrist + EEF 手眼外参 -> output/T_gripper_wrist.npy
Stage 2A: Wrist + Cam1       -> output/T_base_cam1.npy
Stage 2B: Wrist + Cam2       -> output/T_base_cam2.npy
Stage 3: FK + K/D + 所有外参 -> 三路 RGB 上投影 EEF Position/Rotation
```

### Stage 0：每个 RGB 相机分别标定 K、D（不需要 FK，不需要机器人控制）

**准备：** 使用本工程同一块打印板；每台相机分别拍摄**约 25–40 张**清晰、不同朝向/距离/画面位置的 ChArUco 图像。板可以在图片之间移动。每帧应保持清晰且至少能看到 12 个 ChArUco 角点；尽量覆盖画面中央、四角、上下左右边缘，并加入 ±20–45° 的倾角。相机镜头焦距、对焦、裁剪、分辨率必须和后续外参标定一致。

推荐在项目根目录运行（USB/RTSP 配置见 `configs/example.yaml`，ROS2 图像见 `configs/ros2_example.yaml`）：

```bash
# 0A: 手腕相机 RGB 内参：移动标定板，或在视野中移动相机
python -m multicalib intrinsics capture --camera wrist
python -m multicalib intrinsics solve --camera wrist

# 0B: 第三视角 1 的 RGB 内参
python -m multicalib intrinsics capture --camera cam1
python -m multicalib intrinsics solve --camera cam1

# 0C: 第三视角 2 的 RGB 内参
python -m multicalib intrinsics capture --camera cam2
python -m multicalib intrinsics solve --camera cam2

# 三组图像均已采集完：也可以一次求解全部内参
python -m multicalib intrinsics solve --camera all
```

实时预览按 **`c`** 采集一张原始分辨率图像，按 **`q`/Esc** 结束。采集只启用对应的一个相机，**不需要任何 FK/EEF 数据，也不要求先进行 Wrist 手眼标定**。每次按 `c` 都要移动标定板到新位置/倾角后再采集，避免多张几乎相同的视角造成退化。每个 Stage 0 可以独立进行；Cam1 与 Cam2 无需视野重叠。

- 原始图像：`data/intrinsics/wrist/sample_0000.png`、`data/intrinsics/cam1/...`、`data/intrinsics/cam2/...`，每张旁边自动存 `sample_0000.json` 记录角点数量/画面覆盖。
- 输出位置：**直接写入**配置文件 `cameras.<camera>.intrinsics` 指定的 `.yaml` 或 `.json`；默认是 `configs/*_intrinsics.yaml`。自动替换 `placeholder: true` 占位文件，因此无需人工填 K/D。
- 报告：`output/intrinsics/wrist_report.json`、`cam1_report.json`、`cam2_report.json`：包括图像数量、相机矩阵 K、畸变 D、每张图的重投影 RMSE、自动剔除的异常帧和分布警告，另保存 `{camera}_raw_vs_undistorted.png` 左右对比预览。
- 内参模型：OpenCV **普通针孔相机 + Brown–Conrady 5 系数畸变**：`fx, fy, cx, cy` 与 `D=[k1,k2,p1,p2,k3]`，输出兼容现有 `intrinsics_from_file()`、Stage 1 / Stage 2 `solvePnP()`。
- 防误操作：已有非占位内参默认不会被覆盖。确需重新标定，加 `--overwrite`，原文件将自动备份为 `.bak_日期时间`。
- 当全局重投影 RMS 超过 `intrinsic_calibration.max_rms_px`（示例默认 1.5 px）时不会输出内参。可以补采更清晰、更丰富姿态的图像或根据相机成像条件调整阈值；不能只用低 RMS 断定三维精度。

**不连接实际摄像头的离线模式：** 先将各相机的原始 ChArUco PNG/JPG 图片放到对应 `data/intrinsics/<camera>` 文件夹，再直接运行上面的 `intrinsics solve`。也支持单相机指定图片目录：

```bash
python -m multicalib intrinsics solve --camera wrist --image-dir /absolute/path/to/raw_rgb_images
python -m multicalib intrinsics solve --camera wrist --overwrite
```

**重要限定：** 这里标定的是**未去畸变/未校正的原始 RGB 图像的内参**，不是 RGB-D 相机独立的 Depth 内参，也不求 RGB→Depth 的跨传感器外参。若后续使用的是对齐到 RGB 的深度图/点云，必须使用设备 SDK 中与该对齐方式一致的标定参数；如果 SDK/ROS2 已提供可靠的 `CameraInfo`，可直接使用，不必重复标定。超广角 fisheye/equidistant 镜头不适合默认 5 系数模型，需另做鱼眼模型适配。建议 **Stage 0 内参先完成再采 Stage 1/2**；如果更新了内参，旧外参样本内保存的 PnP 不会自动刷新，需重新采集外参样本或基于原始图像重新计算 PnP。外参阶段输入图像应和标定内参的图像处理方式一致，避免对已经 rectified 的画面再使用原始 `D`。

---

## 1. 坐标定义（必须先核对）

`T_A_B` 将 B 坐标系的点变换到 A，即 `p_A = T_A_B @ [p_B, 1]`，旋转是正交矩阵，平移单位始终为**米**。图像的相机坐标系使用 OpenCV optical frame（X 向右、Y 向下、Z 向前）。

- `T_B_G`：机器人 FK 输出的 **base ← gripper（tool0/flange）**，不要无意中用 TCP 代替 flange；必须始终同一个 G。
- `T_W_P`：Wrist 相机观测 ChArUco 标定板的 `wrist optical ← board`，由 OpenCV PnP 得到。
- `T_C_P`：某个第三视角相机观测板的 `camera optical ← board`，由 PnP 得到。
- `T_G_W`：Stage 1 求得的固定 `gripper ← wrist optical`。
- `T_B_C1`、`T_B_C2`：Stage 2 求得的固定 `base ← third camera optical`。
- Wrist 实时外参：`T_B_W(t) = T_B_G(t) @ T_G_W`。

### Stage 1：必须固定标定板，移动机器人 EEF

采集 20–30 组 `{T_B_G, T_W_P}`，至少 8 组才能运行求解，并且必须有多轴旋转激励。通过 `cv2.calibrateHandEye()` 初值和多姿态非线性优化求 `T_G_W`，并同时反算固定板的 `T_B_P`。如果板动了，本阶段的数学前提失效，必须重采。

### Stage 2A / Stage 2B：可改变标定板位置

针对 `cam1` 或 `cam2`：**每一次配对采集**，Wrist + 目标第三视角相机都需要能看到**同一块**标定板，机器人与板在采集瞬间保持静止。可以在相邻样本间移动板，也可以移动机器人去增大视野重叠。无需两个第三视角互相看见对方。

每组采集 `{T_B_G, T_W_P, T_C_P}`，计算：

```
T_B_C = T_B_G @ T_G_W @ T_W_P @ inverse(T_C_P)
```

1 组可以计算初始外参；建议 8–12 组不同的板位姿（其中自动留 2 组做简单留出验证）。使用鲁棒位姿估计、异常剔除，保存 `.npy` 和 `.json`。**Stage 2 移动板之前，必须完成该组的双图像采集。**

## 2. 安装

```bash
cd robot_multicam_calibration
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
```

ROS2 `rclpy`, `tf2_ros`, `sensor_msgs`, `cv_bridge` 应通过系统 ROS2 安装并 source 对应环境；它们不在 pip 依赖中。ROS2 环境更适合使用 ROS2 系统 Python 或启用 `--system-site-packages` 的 venv。

OpenCV USB/RTSP 相机只需要 `opencv-contrib-python`。机器 FK 可以用 ROS2 `/tf`，也可由现有机器人 SDK 写到一个小型 JSON 文件（见 `examples/update_pose_file.py`）。

## 3. 打印标定板 & 内参

```bash
python -m multicalib --config configs/example.yaml board --out board.png
```

也可以直接使用工程内的 `assets/charuco_8x6_280x210mm.png`。示例板为 8×6 方格、每格 35 mm、marker 25 mm，**实际物理尺寸 280 mm × 210 mm**。打印时不要自动缩放！建议 A3 纸、刚性底板；测量实际格长并改配置（不能直接按 PNG dpi 假设其实际尺寸）。

编辑 `configs/example.yaml` 的相机 index/ROS topics、robot FK 设置，**优先使用 Stage 0 直接生成三份 `configs/*_intrinsics.yaml`**；如果已有可靠的原始内参（例如 ROS2 `CameraInfo`），也可以手动填入真实参数后删除 `placeholder: true`。相机内参必须对应**实际原始 RGB 分辨率、镜头变焦、畸变和裁剪状态**。不同相机不可共用示例内参。

ROS2 模式请参考 `configs/ros2_example.yaml`，注意 `base_frame` / `gripper_frame`。

## 4. 运行在线外参标定（请先完成 Stage 0 内参；有 GUI 的 Linux 桌面）

```bash
# Stage 1: 固定板，移动 EEF 20–30 个姿态
python -m multicalib --config configs/example.yaml capture --stage wrist
python -m multicalib --config configs/example.yaml solve --stage wrist

# Stage 2A: Cam1 和 wrist 在每次采集时同时看到同一块板
python -m multicalib --config configs/example.yaml capture --stage cam1
python -m multicalib --config configs/example.yaml solve --stage cam1

# Stage 2B: 移动 Wrist 到 Cam2 重叠区域，重复
python -m multicalib --config configs/example.yaml capture --stage cam2
python -m multicalib --config configs/example.yaml solve --stage cam2

# 查看所有矩阵
python -m multicalib --config configs/example.yaml inspect
```

GUI **`c`** = 保存本次样本；**`s`** = 在窗口中直接求解并保存；**`q`/ESC** = 结束当前采集。画面标注 ChArUco 角点数量、PnP 重投影 RMS、图像新鲜度、已采集数。仅当全部所需相机有效，机器人保持静止，图像时间戳足够接近时保存。每条样本会自动保存原始 PNG、FK、PnP、评估信息；同一 `data/samples/stage` 已有样本时会继续追加，不会清空。

### Pose-file 方式接入自定义机器人

自定义机器人进程在每个控制/观测循环取得真实 `T_base_gripper`，调用：

```python
from examples.update_pose_file import write_pose
write_pose(T_base_gripper, filename='examples/robot_pose.json')
```

更新频率建议 ≥10Hz，必须原子写入并包含当前 Unix 时间 `stamp_unix_s`。示例 JSON 中的 FK **不是真实 FK**，运行采集前必须连接实际机器人 SDK / FK。若旧于 3 秒，采集会拒绝。

### ROS2 方式

```bash
source /opt/ros/<distro>/setup.bash
python -m multicalib --config configs/ros2_example.yaml capture --stage wrist
```

ROS2 订阅 `sensor_msgs/Image` 或 `sensor_msgs/CompressedImage`（`compressed: true`），并从 tf2 查询 `lookup_transform(base_frame, gripper_frame, latest)`。这要求你的 TF tree 正确地包含机器人运动学；**不能把未标定 wrist optical TF 当成 FK 末端 TF 使用**。本工程采用“停止机器人后采集”的方式，不做运动中的 TF 精确时间同步；有高速动态标定需求需要改为图像时间戳 + TF 插值。

## 5. 输出

```
data/samples/wrist/sample_0000/{observation.json,wrist.png}
                 ...
 data/samples/cam1/sample_0000/{observation.json,wrist.png,cam1.png}
 data/samples/cam2/sample_0000/{observation.json,wrist.png,cam2.png}
 output/T_gripper_wrist.npy    # gripper <- wrist optical
 output/T_base_board_stage1.npy
 output/T_base_cam1.npy       # robot base <- cam1 optical
 output/T_base_cam2.npy       # robot base <- cam2 optical
 output/{wrist,cam1,cam2}_report.json
 data/intrinsics/{wrist,cam1,cam2}/sample_*.png
 output/intrinsics/{wrist,cam1,cam2}_report.json
 configs/{wrist,cam1,cam2}_intrinsics.yaml
```

每个 `.npy` 旁还有同名 `.json`，写明单位和方向，可直接用 NumPy 加载：

```python
T_B_C1 = np.load('output/T_base_cam1.npy')
p_cam1 = np.array([0.02, 0.01, 0.6, 1.0])
p_base = T_B_C1 @ p_cam1
# wrist optical -> robot base changes with robot pose
T_B_W = T_B_G_current @ np.load('output/T_gripper_wrist.npy')
```

## 6. 先不连接机器人：运行仿真回归

```bash
python -m multicalib simulate
python -m unittest discover -s tests -v
```

这会在 `data/synthetic_demo` / `output/synthetic_demo` 内创建各阶段的随机已知外参样本，用 OpenCV 手眼标定 + 优化复原，显示与真值之间的平移和旋转误差。**synthetic_demo 不使用真实图片，也不验证你的驱动、内参或实体安装精度**。

## 7. 精度验证与常见错误

- 优先对 Stage 1 采集不同旋转轴的 EEF 变化，只有平移或只有一个旋转轴可能退化。
- Stage 1 标定板严格不动；Stage 2 每对图像采集期间严格不动，采集之后可随时移动。
- GUI 每按一次 `c` 记录一个 EEF 姿态 + 当前相机图像（第三视角阶段是两张图）。`solve` 不要求固定相机随 Wrist 移动。
- `*_report.json` 包含训练与留出样本的多帧位姿一致性误差，但**不是对绝对世界坐标真值的直接检验**。
- 建议拿未参与标定的板位姿、三相机点云或已知尺寸标定物实测 base 下 3D 误差。要求 1–2 mm 时应额外考虑 RGB-D 深度偏差、板平整度、FK 定义、安装刚性与角点噪声。
- `PnP` 需要正确的内参和实际打印后的 square_length_m；不要对缩放后的 RGB 使用原始内参。
- robot gripper frame 必须对应 FK 输出（例如 tool0 / flange），不要把未申明的 TCP offset 混入。
- ROS2 图像 `bgr8`、camera optical frame、ROS `camera_link` 常有固定光学框架旋转；不要忽略。
- 多组配对图像只要求同一块板，并且每次对应同一物理时刻、板静止。与把固定相机分别做 eye-to-hand 的方法相比，这个两阶段方法会传递 Wrist 手眼外参误差。
- 结果如果误差较大，可先排查时序 / 画面分辨率 / gripper 选择，必要时进一步做联合标定或 point cloud refinement。

## 8. 重要边界

这是**可以直接运行的通用在线内参及外参标定工程**，包含 OpenCV 和 ROS2 输入；但是无法预先猜测你的 Franka / Agilex / OHT 机器人控制 API、Orbbec 设备序号或 ROS topic，必须修改配置或提供 `T_base_gripper` 适配。程序只采集与计算，不会运动机械臂，避免未经安全检查的自动运动。

## 9. Stage 3：把 EEF Position + Rotation 投影到三个相机图像（v2 扩展）

标定完成后，**建议用实际 EEF/TCP 位姿投影验证最终的相机外参**。本工程新增 `multicalib.projection` 和 `python -m multicalib validate`，读取三份 RGB 内参、三个已求得的外参，以及机器人当前 `T_base_gripper`，在三路 **原始 BGR RGB 图像**中绘制 EEF 原点及其红 X / 绿 Y / 蓝 Z 旋转坐标轴。

坐标关系：

```
T_B_W(t) = T_B_G(t) @ T_G_W
T_Ci_G(t) = inverse(T_B_Ci(t)) @ T_B_G(t)
p_Ci = R_Ci_G @ p_G + t_Ci_G
(u, v) = cv2.projectPoints(p_Ci, K_i, D_i)   # RAW 图像
```

Cam1/Cam2 的 `T_B_Ci` 是固定的；Wrist 的 `T_B_W` 随 FK 更新。由于相机与 Gripper 刚性连接，`T_W_G = inverse(T_G_W)` 为常量：Wrist 中机器人 G 坐标轴不随机器人整体运动而移动，是正常现象。

### 在线实时验证（默认）

```bash
# 操作员先将机器人停止到一个姿态；程序本身不驱动机器人
python -m multicalib --config configs/example.yaml validate

# G（flange/tool0）原点若在 Wrist 画面之外，则使用实际 TCP 在 G 下的偏移
# 下述 0.12 m 只是示例！必须替换为真实几何参数。
python -m multicalib validate --origin-offset-m 0 0 0.12 --axis-length-m 0.05

# 也可标注夹爪上的已知物理特征点（坐标系是 G，单位为米）
# 示例文件的坐标是说明性占位，须用真实测量值替换
python -m multicalib validate --points-json configs/physical_gripper_points_example.json
```

GUI 的 `c` 保存本次三路 raw PNG + 投影 overlay PNG + FK 与投影像素的 `validation_report.json`；`q`/Esc 退出。默认保存到 `output/projection_validation/<timestamp>/`。当 FK、图像无效或过时，程序会显示等待状态并拒绝保存。使用 RAW 图像和对应内参 `K/D`，不可以在已 rectified 的图像上再次应用原始 `D`。

### 离线验证

```bash
# offline_images 目录必须有 wrist.png、cam1.png、cam2.png
# pose_snapshot.json 内包含所拍摄时刻的 T_base_gripper (4x4)
python -m multicalib validate \
  --images-dir offline_images \
  --pose-file pose_snapshot.json \
  --save-dir output/offline_pose_check
```

**注意物理可见性：** flange / tool0 常被夹爪挡住或位于 Wrist 相机后方，导致 `OUTSIDE IMAGE` / `BEHIND CAMERA` 提示，这不直接代表标定错误。请指定真实 TCP 的 `--origin-offset-m X Y Z`（仅改变投影原点，轴方向仍为 G）或实测的夹爪可见特征点。如果希望绘制 TCP 本身的旋转坐标轴，必须额外考虑完整的 `T_G_TCP` 旋转；仅用平移偏移不够。

**如何判断：** 在 Cam1 和 Cam2 的多个静止机器人姿态下，投影位置与夹爪对应物理位置一致、XYZ 轴方向随真实运动合理；Wrist 中相对安装位置保持不变。投影叠加是 **定性 sanity check**，不是独立的毫米级精度测量。为得到定量结果，应在未参与标定的姿态下标注真实夹爪关键点（比较 2D 重投影误差），或使用已知精确几何 / 外部测量获得 3D 真值。对于存在图像时间戳偏差的动态运动，应先停止机器人再验证，本工程不做高速图像/FK 插值同步。

HTML 技术文档：`calibration_method_guide.html`，可直接浏览器打开（含 Stage 0–3 原理、公式、流程图、命令、验证和检查清单）。
