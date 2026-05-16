# Real-time Tracking System

> 
> 
> 
> The project is to **design a real-time tracking system for a robot vehicle**, using a camera with an embedded device (Nvidia Jetson Orin).
> 
> In this project, you will be developing a real-world machine learning model to **process real-time video input to perform the tracking**. You will need to **develop experiments to validate model performance**. You may also need to **design software simulation to emulate the robot vehicle behaviour**.
> 

# Hardware

# Camera

collect low latency and high definition real-time video

# Brain

## Nvidia Jetson Orin - edge computing device

put huge AI computing power on the robot body which is the edge

all the camera processing and target tracking calculations can be completed instantly locally without Internet connection and with low latency

 In this tracking project, it plays a role of command center.

- eye input: the camera captures the image and transmits it instantly to Orin via a cable
- thinking: the GPU in Orin runs at high speed. It runs the YOLO model once, discovering something like “the target is the upper left corner of the screen”
- neural command: the CPU calculates something like “the robot needs to turn 15 degrees to the left” and sends an electrical signal to the robot.
- Loop: The process above needs to be repeated n times per second (n FPS) to achieve smooth real-time tracking

# Software

# Object Detection

- goal: find the target the robot wants to track in each frame of the camera and box it
- some techniques: YOLO

# Multi-Object Tracking (MOT)

- goal: the tracking algorithm is responsible for assigning a unique ID to the target in video frames so that the target can be captured even if it is briefly hidden.
- some techniques: DeepSORT, ByteTrack

# Deployment

model compression and acceleration

Nvidia’s TensorRT - slim down and re-format the deep learning model, enabling it to run at the highest efficiency on Orin’s GPU

# Simulation and Experiment Verification

- Task: Build a 3D physical world in a computer and integrate the tracking algorithm into a virtual robot and see if it loses track when meeting turns and obstacles
- usual tools: ROS (Robot Operating system), Gazebo, Isaac Sim

# Validation

- **FPS (Frames Per Second)：** Operating Speed
- **mAP：** accuracy of detection
- **MOTA / IDF1：** accuracy of tracking (such as mistakenly identified or lost)

# Task

## process real-time video input to perform the tracking

develop a ML model for tracking

1. use YOLO to detect objects and use ByteTrack/DeepSORT to track
2. prepare data and label
3. training
4. convert the model into a format that Orin can  run via Nvidia’s TensorRT
- input: real-time video, dataset
- output: class, bounding box, tracking id, confidence score

## develop experiments to validate model performance

use datasets under different lighting and occlusion conditions to test the robustness

record FPS, accuracy … to write report and make graphs

- input: model, test dataset, ground truth
- output: FPS, mAP, …

## design software simulation to emulate vehicle performance

1. use simulation software to set up virtural environment 
2. use ROS to convert the model outputs to the message that robot can understand
3. design the control logic to adjust the speed and direction
4. operation and debugging
- input: position of the target, virtual environment parameters, control algorithm parameters
- output: control commands, system status feedback, output of validation

# Techniques

- Language: Python (process data, training model, draw graphs…), C++ (deployment)
- OS: Linux (Ubuntu)
- Artificial Intelligence & Computer Vision: Pytorch (deep learning framework), **YOLO, ByteTrack / DeepSORT / BoT-SORT, OpenCV (process data)**
- Deployment and Accelerate: CUDA, TensorRT, DeepStream (audio and video stream processing framework)
- Robotics and Simulation: ROS (communication framework), Gazebo/Nvidia Isaac Sim (simulation software), PID (control algorithm)

## Week3-Week5

### Dataset - GTSDB(German Traffic Sign Detection Benchmark, YOLO format)

![1_jpg.rf.0cf79f481ce5ae6e41d388f1234da1d6.jpg](Real-time%20Tracking%20System/1_jpg.rf.0cf79f481ce5ae6e41d388f1234da1d6.jpg)

![2_jpg.rf.d212c51d9ce1f036e6accb896915df49.jpg](Real-time%20Tracking%20System/2_jpg.rf.d212c51d9ce1f036e6accb896915df49.jpg)

![15_jpg.rf.8d19b664e2aeae351fc383b6a35967d8.jpg](Real-time%20Tracking%20System/15_jpg.rf.8d19b664e2aeae351fc383b6a35967d8.jpg)

![23_jpg.rf.42d12e208bc74a9a4947cc2fe83142f0.jpg](Real-time%20Tracking%20System/23_jpg.rf.42d12e208bc74a9a4947cc2fe83142f0.jpg)

![截屏2026-03-31 16.14.48.png](Real-time%20Tracking%20System/%E6%88%AA%E5%B1%8F2026-03-31_16.14.48.png)

![截屏2026-03-31 16.16.23.png](Real-time%20Tracking%20System/%E6%88%AA%E5%B1%8F2026-03-31_16.16.23.png)

### Comparison

| model | mAP50 | mAP50-95 | precision | recall | params(M) | GFLOPs | size(MB) | latency(ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 0.85382 | 0.41975 | **0.93069** | 0.75494 | **2.51** | **7.2** | **5.03** | 65.88 | 15.2 |
| YOLOv8 | **0.87251** | **0.44256** | 0.86024 | **0.83822** | 3.01 | 8.2 | 5.96 | **61.07** | **16.4** |
| YOLOv9 | 0.86198 | 0.43053 | **0.93949** | 0.76777 | **2.01** | 7.9 | **4.42** | 88.05 | 11.4 |
| YOLOv10 | 0.82018 | 0.40244 | 0.82256 | 0.78616 | 2.71 | 8.4 | 5.49 | 85.76 | 11.7 |
| **YOLOv11** | **0.88073** | **0.43988** | 0.88308 | **0.80693** | 2.59 | **6.4** | 5.22 | **59.1** | **16.9** |

![image.png](Real-time%20Tracking%20System/image.png)

![image.png](Real-time%20Tracking%20System/image%201.png)

![104_jpg.rf.ba00725b00d47611b4fb78b8f1b25441.jpg](Real-time%20Tracking%20System/104_jpg.rf.ba00725b00d47611b4fb78b8f1b25441.jpg)

![120_jpg.rf.383619b82a1d227974d9756d0645cdf5.jpg](Real-time%20Tracking%20System/120_jpg.rf.383619b82a1d227974d9756d0645cdf5.jpg)

![135_jpg.rf.b4c6716a333493ed1121c35553d66469.jpg](Real-time%20Tracking%20System/135_jpg.rf.b4c6716a333493ed1121c35553d66469.jpg)

![185_jpg.rf.614aa19cf387462effc9992da2d301a0.jpg](Real-time%20Tracking%20System/185_jpg.rf.614aa19cf387462effc9992da2d301a0.jpg)

## Week5-Week7

---

### 摄像机选型 | Camera Selection

项目最开始用的是 Logitech USB 网络摄像头，但测试发现它是固定焦距，即使安装到最高位置，视角也不够大，无法拍到整张赛道桌面。因此换成了 GoPro，它的广角镜头从单个顶部机位就能覆盖整个赛道，这个问题就解决了。

We initially tested a Logitech USB webcam, but even at maximum mount height its fixed focal length couldn't cover the full table. We switched to a GoPro — its wide-angle lens covers the entire track surface from a single overhead position.

GoPro 有三种视野模式：Linear 畸变最小，是离线分析的首选；但当 GoPro 通过 USB 接电脑做实时推流时，Linear 模式不可用，只能选 Wide 或 SuperView。SuperView 畸变过于严重，所以实时场景固定使用 Wide 模式。

GoPro has three FOV modes. Linear has minimal distortion and is ideal for offline work, but it's unavailable in USB Webcam mode. SuperView has severe barrel distortion, so Wide mode is the only practical choice for real-time use.

---

### 畸变校正 | Distortion Correction

Wide 模式存在轻微的桶形畸变（直线看起来略微弯曲）。为此专门写了一个手动调参工具，左右两边同时显示原图和校正后的图，并叠加绿色网格线，通过拖动滑块调节 K1、K2、焦距缩放三个参数，直到网格线看起来笔直为止。最终手动估算出来的参数是 k1 = -0.462，k2 = -0.054。

Wide mode has slight barrel distortion. We built a manual tuning tool that shows the original and corrected images side by side with a green grid overlay. By adjusting sliders for K1, K2, and focal scale until the grid lines appear straight, we estimated k1 = -0.462 and k2 = -0.054.

更精确的做法是棋盘格标定——打印棋盘格，从不同角度拍摄 20–30 张，OpenCV 自动计算精确的相机矩阵和畸变系数，这套流程也已经写好了（`calibrate_gopro.py`），待图像采集完成后执行。

A more rigorous approach is checkerboard calibration: print a checkerboard, photograph it from 20–30 angles, and OpenCV computes precise coefficients automatically. This pipeline is already written and ready to run once the calibration images are collected.

不过实际测试发现，在当前摄像机安装高度下，Wide 模式的畸变已经非常轻微，网格线肉眼几乎看不出弯曲。强行应用手动估算的参数反而让图像变得更差，所以目前畸变校正是关闭的，用原始图像反而更干净。

In practice though, at our mount height the Wide distortion is so subtle the grid lines look nearly straight already. Applying the manually estimated correction actually made the image worse, so distortion correction is currently disabled — the raw image is cleaner.

---

### 标记系统选型 | Choosing a Marker System

定位车辆和校准视角需要用到视觉标记，我们对比了 AprilTag 和 ArUco 两套方案。二者本质上是同类技术——印刷出来的方形码，可被摄像机识别和定位——但有几个实际区别。AprilTag 在 Python 下需要单独安装第三方库，检测速度相对较慢，文档主要是学术论文。ArUco 直接内置于 OpenCV，不需要任何额外安装，检测速度更快，官方文档完整。在这个项目里，相机距离近、室内环境可控，ArUco 的精度完全够用，没必要为 AprilTag 更强的远距离鲁棒性付出额外的依赖成本，所以选择了 ArUco。

We needed fiducial markers for vehicle localisation and perspective correction, so we compared AprilTag and ArUco. Both are the same class of technology — printed square codes that cameras can detect and locate — but practically: AprilTag requires a separate Python library and is slower; ArUco is built directly into OpenCV with no extra installs, faster detection, and complete official documentation. For a close-range, controlled indoor setup, ArUco accuracy is more than sufficient, so we chose ArUco.

具体使用的是 `DICT_4X4_50` 字典，这是最简单的图案（16 个数据位），近距离最容易被检测到，50 个 ID 也完全够用。共生成 5 枚标记：桌子四个角各一枚（ID 0–3），车顶一枚（ID 4）。

We used the `DICT_4X4_50` dictionary — the simplest pattern (16 data bits), easiest to detect at close range, with 50 unique IDs which is more than enough. Five markers were generated: one at each table corner (ID 0–3) and one mounted on the car (ID 4).

---

### 透视变换 | Perspective Warp

摄像机从斜上方拍摄时，赛道在画面里是梯形的，不是正射的俯视图。检测到四个角标之后，用 `cv2.getPerspectiveTransform` 计算单应性矩阵，把原始画面"拉正"成一个 600×600 px 的标准顶视图，消除摄像机角度带来的透视变形。这个矩阵每帧都重新计算，所以摄像机轻微晃动时系统会自动补偿，不需要固定摄像机位置一分不差。

The camera views the track at an angle, so the raw image is geometrically distorted. Once the four corner markers are detected, `cv2.getPerspectiveTransform` computes a homography matrix that warps every frame into a 600×600 px orthographic top-down view. The matrix is recalculated each frame, so minor camera movement is automatically corrected — the camera doesn't need to be perfectly fixed.

---

### 赛道检测与在轨判断 | Track Detection and On/Off Track Logic

赛道是白底黑轨，特征很明确。对顶视图做颜色阈值处理，把图像二值化，提取黑色区域，生成一张"赛道掩膜"——本质上是一张同尺寸的黑白图，轨道位置标白，其余标黑，保存为 `track_mask.png`。

The track is black on a white background, which is easy to isolate. We threshold and binarise the top-down view to extract the black track region, producing a saved track mask — a same-size black-and-white image where white pixels represent the track — saved as `track_mask.png`.

判断小车是否在轨的基本思路是：找到车辆标记（ID=4）在顶视图中的位置，在那个坐标周围取半径 20px 的小区域，查掩膜里有多少是轨道像素，超过 10% 就判断为在轨。

The on/off track logic is straightforward: find the car marker (ID 4) in the warped view, sample a 20 px radius region around it, check what percentage of those pixels are track pixels in the saved mask — above 10% means ON TRACK.

但这里有一个关键问题：车顶贴的 ArUco 标记是黑白方块，它会物理遮住车下面的赛道。如果实时去读那个位置的像素颜色，看到的全是标记的图案，根本看不到赛道，导致系统始终判断为出轨。

There's a key problem here: the ArUco marker on the car is a black-and-white square that physically covers the track beneath it. Any real-time pixel check at the car's position reads the marker pattern, not the track — causing the system to always report OFF TRACK.

解决方法是：先不放车，在干净的赛道上按 `c` 键拍一张无遮挡的掩膜并保存；然后再把车放上去。后续所有判断都查这张预先保存的干净掩膜，而不是实时画面。车放上去之后遮住了什么根本不影响结果，因为查的是放车之前的那张图。

The fix: before placing the car, press `c` to capture a clean unobstructed track mask and save it. Then place the car. All subsequent checks query this pre-saved mask rather than the live image — whatever the car covers at runtime is irrelevant, because we're checking what was there before it was placed.

最后还加了一个防抖滤波：车在赛道边缘行驶时，判断结果容易在在轨和出轨之间快速跳变、一直闪烁。方法是维护最近 8 帧的记录，超过 60% 的帧都同意才更新状态——单帧的噪声被多数帧"投票否决"掉了。

Finally, a temporal debounce filter was added: near the track edge, results can flicker rapidly between ON and OFF. We keep a rolling window of the last 8 frames and only update the displayed state when 60% of frames agree — single-frame noise gets outvoted.

---

### 下一步 | Next Steps

当前系统已能稳定检测车辆位置并判断在轨状态。接下来要做的是：采集棋盘格图像完成正式相机标定、将标记安装到实体车上做运动测试、在顶视图中加入道路标志识别（STOP 标志和限速 55 标志）。

The current system reliably detects the vehicle position and determines on/off track status. Next steps are: collect checkerboard images to complete formal camera calibration, attach the marker to the physical car for motion testing, and add road sign detection (STOP and speed limit 55) into the warped top-down view.

## Week7-Week9

# 两周工作总结：YOLO 标志检测集成

## 一、背景与问题定义

本项目搭建了一套**微型赛道实时追踪系统**，通过 GoPro 相机俯拍赛道，结合 ArUco 标记与透视变换，实时输出 600×600 像素的正射俯视图。

在此基础上，导师要求识别赛道上的三种标志：

- **STOP 停止标志**（赛道底部）
- **限速 55 标志**（赛道右侧）
- **LED 交通灯**（赛道左侧，三种状态：灭/绿/红）

---

## 二、第一阶段：HSV 颜色检测（初始方案，被放弃）

### 方案原理

在透视矫正图中划定三个固定 ROI（感兴趣区域），用 HSV 颜色阈值判断标志是否可见：

- STOP / 55 牌：检测 ROI 内红色像素占比是否超过阈值
- 交通灯：检测绿色/红色像素占比

### ROI 坐标（最终校准值，单位：像素，基于 600×600 俯视图）

| 标志 | x1 | y1 | x2 | y2 |
| --- | --- | --- | --- | --- |
| LED 灯 | 10 | 252 | 72 | 315 |
| STOP | 305 | 472 | 402 | 550 |
| 限速 55 | 468 | 283 | 568 | 362 |

### 遇到的问题

1. **ROI 坐标标定繁琐**：需要对照实际截图反复调整，初始估值偏差较大（STOP 框偏右约 55px）
2. **鲁棒性差**：光照变化、阴影、小车遮挡等场景下误检率高
3. **灯光检测不稳定**：LED 实际颜色与 HSV 阈值边界模糊，绿灯/红灯经常混淆
4. **根本缺陷**：颜色阈值是人工经验值，无法泛化到不同光照或相机角度

**结论：放弃 HSV 方案，改用 YOLO 深度学习检测。**

---

## 三、第二阶段：YOLO 集成（迁移原有模型，失败）

### 尝试方案

直接使用已有的 YOLOv11n 模型（原本训练在真实道路标志数据集上）进行检测。

### 失败原因：域迁移不匹配（Domain Mismatch）

- 原模型训练数据：真实街道上的标准交通标志（高清、大尺寸、多角度）
- 实际检测目标：打印在 A4 纸上的**微型仿制标志**（约 3×3cm，俯视压缩，分辨率 ~100×80px）

两者外观差异极大，模型完全无法识别赛道上的玩具标志。

---

## 四、第三阶段：赛道专用 YOLO 模型训练（核心工作）

### 4.1 整体思路

**"用旧的 HSV 检测来自动生成训练标注，再用标注好的数据训练 YOLO"**

这是一个自举（bootstrap）策略：HSV 本身精度不高，但足以生成粗略的标注框，配合人工微调，形成高质量训练集，再让 YOLO 学习更鲁棒的特征。

### 4.2 数据采集

- 运行 `aruco_detect.py`，进入俯视图，按 `s` 键保存截图
- 在不同光照、小车位置、灯光状态下采集
- 最终数据集：**训练集 44 张，验证集 12 张**（600×600 px 俯视图）
- 目录结构：
    
    `track_dataset/
    ├── images/
    │   ├── train/   (44张)
    │   └── val/     (12张)
    └── labels/
        ├── train/
        └── val/`
    

### 4.3 标注工具：`auto_label.py`

自行开发的半自动标注工具，核心功能：

**自动检测阶段**

- 读入截图，在三个 ROI 用 HSV 颜色检测标志是否可见
- 自动生成 YOLO 格式的标注（归一化坐标 xc, yc, w, h）

**5 类标签**（最终版本，增加了灯光三态）

| 类别 ID | 名称 | 含义 |
| --- | --- | --- |
| 0 | stop | 停止标志可见 |
| 1 | speed_55 | 限速55标志可见 |
| 2 | light_off | 交通灯熄灭 |
| 3 | light_green | 绿灯亮 |
| 4 | light_red | 红灯亮 |

**交互控制界面**

每张图弹出 600×600 预览窗口，显示自动检测的标注框，支持实时调整：

| 按键 | 功能 |
| --- | --- |
| Enter | 保存当前标注，下一张 |
| `1` | 切换 STOP 框显示/隐藏 |
| `2` | 切换 55 框显示/隐藏 |
| `3` | 循环切换灯光状态（灭→绿→红→灭） |
| `e` | 进入鼠标编辑模式（用 `cv2.selectROI` 手动拖框） |
| `d` | 丢弃该图（不保存标注） |
| `q` | 退出 |

**关键 Bug 修复（标注框重置问题）**

开发过程中发现：用 `e` 鼠标编辑好框后，按 Enter 保存再看下一张时，上一张编辑的框坐标会影响到下一张（变量未隔离）。

根本原因：每次循环都从全局 `AUTO_BOXES` 读取初始坐标，导致同一图片多次循环中编辑结果被覆盖。

修复方案：每张图初始化独立的 `custom_boxes` 字典，所有编辑写回 `custom_boxes`，循环从 `custom_boxes` 读取（而非 `AUTO_BOXES`）。

### 4.4 训练配置：`train_track.py` + `track_data.yaml`

`# track_data.yaml
path: track_dataset
train: images/train
val:   images/val
nc: 5
names: ['stop', 'speed_55', 'light_off', 'light_green', 'light_red']`

训练参数：

- 基础权重：`yolo11n.pt`（ImageNet 预训练，不使用旧的道路标志模型）
- Epochs：50（针对小数据集调低，避免过拟合）
- Batch：8（CPU 友好）
- Patience：15（早停）
- 输出路径：`runs/train/track_signs/`（独立目录，不覆盖原模型）

**训练结果**

| 指标 | 数值 |
| --- | --- |
| mAP50 | **0.995** |
| mAP50-95 | 0.873 |
| Precision | ~0.97 |
| Recall | ~0.96 |

---

## 五、第四阶段：集成到主程序（多次调试）

### 5.1 第一次集成（失败）

在 `aruco_detect.py` 中，`detect_signs()` 函数将三个 ROI 区域**分别裁剪**后送入 YOLO：

`stop_crop = warped[472:550, 305:402]   # 97×78 px
model(stop_crop, conf=0.35)`

**问题**：模型在全图（600×600）上训练，标志只占整图的一小块。裁剪后送入 YOLO，模型看到的是"标志占满整个画面"，与训练时的视觉上下文完全不同，导致**置信度极低甚至零检测**。

将置信度阈值从 0.35 降到 0.20 也无效，`conf=0.05` 也没有任何输出。

通过编写 `debug_yolo.py` 确认了这一点：

`── ROI [stop]  pixels=97x78
   No detections at conf>=0.05    ← 极低阈值仍然无输出`

### 5.2 根本原因分析

训练时：YOLO 看到完整的 600×600 俯视图，标志作为小目标存在于图中
推理时：裁剪后的小图送入 YOLO，标志充满整个输入帧

两种输入的视觉分布截然不同，模型的特征提取完全失效。

### 5.3 修复方案（成功）

**改为全图推理 + ROI 过滤**：

1. 将完整 600×600 warped 图送入 YOLO（与训练时完全一致）
2. YOLO 输出所有检测框的绝对坐标
3. 计算每个检测框的中心点，判断是否落在对应标志的 ROI 内
4. 同一类别保留置信度最高的一个检测结果

`# 核心逻辑
for box, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
    name = model.names[int(c)]
    bx = (box[0] + box[2]) / 2   # box centre x
    by = (box[1] + box[3]) / 2   # box centre y
    if roi[0] <= bx <= roi[2] and roi[1] <= by <= roi[3]:
        best[name] = (conf, box)   # keep highest conf`

全图推理后 debug 输出：

`DETECTED: stop       conf=0.907  centre=(335,530)  ✓ inside ROI
DETECTED: light_off  conf=0.825  centre=(62,289)   ✓ inside ROI
DETECTED: speed_55   conf=0.475  centre=(533,307)  ✓ inside ROI`

### 5.4 可视化：检测框叠加

最终在俯视图上实时绘制 YOLO 检测框，含类别名称和置信度，颜色编码：

- 红色框：STOP
- 橙色框：限速 55
- 灰/绿/深红框：交通灯状态

---

## 六、附：GoPro 连接稳定性修复

调试过程中发现 GoPro USB Webcam 在 Mac 上频繁断连，排查后修复：

1. 所有 `cv2.VideoCapture` 调用统一加 `cv2.CAP_AVFOUNDATION` 参数（Mac 原生后端）
2. `list_cameras()` 扫描时每次 open/release 后 `sleep(0.3)`，让系统释放设备
3. 正式打开摄像头前 `sleep(1.0)` 等待设备就绪
4. 热身帧从 5 帧增加到 10 帧

---

## 七、最终系统架构

`GoPro Webcam
    │
    ▼
畸变矫正（可选，已标定参数）
    │
    ▼
ArUco 检测（ID 0-3 → 4 角定位）
    │
    ▼
透视变换 → 600×600 正射俯视图
    │
    ├── 小车追踪（ArUco ID 4 → 坐标 + ON/OFF TRACK）
    │
    └── YOLO 标志检测（全图推理 + ROI 过滤）
            ├── STOP：是/否
            ├── 限速55：是/否
            └── 交通灯：灭/绿/红
                    （实时边界框 + HUD 显示）`

---

## 八、新增文件清单

| 文件 | 用途 |
| --- | --- |
| `auto_label.py` | 半自动标注工具，HSV 初检 + 鼠标微调 |
| `train_track.py` | 赛道专用 YOLO 训练脚本 |
| `track_data.yaml` | YOLO 数据集配置（5类，train/val 路径） |
| `debug_yolo.py` | 检测调试工具，定位推理失败根因 |
| `gopro_latency.py` | GoPro 帧率与延迟测量工具 |

## Week9-Week11

## 自动轨道识别模块开发报告

---

### 一、背景与目标

在此之前，系统需要用户手动按键（`c`）触发一次轨道掩膜（track mask）的采集，每次换桌子或摄像头角度改变后都需重新操作。导师要求实现**全自动轨道识别**：系统启动后无需人工干预，自动检测出轨道位置，并在摄像头视角变化时重新检测。

---

### 二、核心技术：自动轨道掩膜检测

### 2.1 算法整体流程

`原始帧 → 透视矫正(600×600) → 预处理遮挡 → 下采样(300×300)
→ 高斯模糊 → Otsu 双极性阈值 → 形态学清洗 → 上采样(600×600)
→ sign ROI 桥接修复 → 保存 track_mask.png`

### 2.2 Otsu 自适应阈值（双极性策略）

普通 Otsu 只尝试一种二值化方向，但实际拍摄中不确定轨道是"暗背景亮轨道"还是"亮背景暗轨道"。因此采用**双极性检测（dual-polarity detection）**：

`mask_inv, _ = _segment(cv2.THRESH_BINARY_INV)   # 深色→白（轨道为深色）
mask_nor, _ = _segment(cv2.THRESH_BINARY)         # 亮色→白

ratio_inv = mask_inv.sum() / 255 / mask_inv.size
ratio_nor = mask_nor.sum() / 255 / mask_nor.size
TRACK_MIN, TRACK_MAX = 0.05, 0.70`

- 若某方向的掩膜白色像素比在 5%–70% 之间，认为是合理的轨道分割结果
- 若两者都在范围内，选择比例更接近 0.30（典型椭圆轨道占比）的那个
- 这使系统无需手动设阈值，适应不同光照和桌面颜色

### 2.3 形态学清洗

`# 形态学闭运算：填补轨道断点（closing）
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

# 开运算：去除细小噪点（opening）
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

# 面积过滤：去除面积 < 1% 的连通域
min_area = 300 * 300 * 0.01`

### 2.4 半分辨率处理（Half-Resolution Processing）

轨道检测在 **300×300** 下运行（原始 warped 图为 600×600），处理完后再上采样回 600×600。计算量减少约 75%，让背景线程能快速完成而不阻塞主循环。

---

### 三、遮挡处理：贴纸与标识干扰

### 3.1 问题

赛道上放有三处标识：红绿灯（ROI_LIGHT）、停车标志（ROI_STOP）、限速 55（ROI_55）。标识底部贴有**黑色贴纸**，与轨道颜色相同，导致 Otsu 将其当作轨道像素，形态学 close 后在那些位置出现矩形大色块而非细线。

### 3.2 尝试过的方案（演进过程）

| 版本 | 策略 | 问题 |
| --- | --- | --- |
| v1 | Otsu 前将 ROI 区域涂成桌面色 → 造成缺口 → 补线 | `ROI_55` 覆盖右侧轨道，导致右边轨道丢失；补线产生 Y/T 形伪影 |
| v2 | 完全不涂 ROI，只涂小车 ArUco | 贴纸仍被识别为粗轨道块 |
| v3 | `_bridge_sign_roi()`：Otsu 后对每个 ROI 清除色块，扫描 ROI 外缘找入射/出射点，连直线 | 顶部仍有凸起（非 ROI 位置的干扰） |
| v4（失败） | 形态学骨架化（skeletonization）+ 枝干剪枝（branch pruning） | 轨道若在图像边界有缺口，剪枝会从端点向内侵蚀，导致完全检测不到 |
| 当前 | 恢复 v3（`_bridge_sign_roi`） | 保持可用，ROI 内细线效果良好 |

### 3.3 `_bridge_sign_roi` 实现原理

`def _bridge_sign_roi(mask, rx1, ry1, rx2, ry2, line_width=10):
    # 1. 扫描 ROI 外围 18px 条带，找到轨道从哪几个方向进入 ROI
    # 2. 清除 ROI 内所有像素（去除贴纸色块）
    # 3. 在所有入射点中，选择间距最大的两点连线
    #    （选最长对，避免连出 Y/T 形——只画一条线）
    best_d, best_pair = -1, ...
    for i, j in all_pairs:
        d = (xi-xj)² + (yi-yj)²
        if d > best_d: best_pair = (i, j)
    cv2.line(mask, best_pair[0], best_pair[1], 255, line_width)`

---

### 四、背景线程与非阻塞检测

### 4.1 问题

`auto_detect_track_mask()` 在 CPU 上运行约 200–400ms。若在主循环中同步调用，整个视频流会冻结。

### 4.2 解决方案：Daemon Thread + 共享容器

`_mask_running = False
_mask_result  = [None]   # 单元素列表作为共享容器（线程安全写入）

def _run_mask_detection(warped_snap, car_pos_snap, M_snap):
    global _mask_running, _mask_M
    result = auto_detect_track_mask(warped_snap, car_pos_snap)
    _mask_result[0] = result          # 写入结果
    if result is not None:
        _mask_M = M_snap              # 记录本次检测时的透视矩阵
    _mask_running = False

def _trigger_mask(warped_now, markers_now):
    global _mask_running
    if _mask_running: return          # 已在检测中，跳过
    _mask_running = True
    threading.Thread(target=_run_mask_detection, ..., daemon=True).start()`

主循环每帧检查 `_mask_result[0]`，有结果就取走，不阻塞。

---

### 五、透视变化自动重检测

### 5.1 问题

摄像头视角或轨道位置改变后，旧掩膜与新画面不匹配，需要重新检测。

### 5.2 `_perspective_changed()` 原理

记录上一次检测时用的透视矩阵 `_mask_M`。每帧将当前矩阵 `M` 与旧矩阵比较：把图像的 4 个角点用两个矩阵分别变换，计算对应点的最大位移。

`def _perspective_changed(M_new):
    corners = np.array([[0,0],[600,0],[600,600],[0,600]], dtype=np.float32)
    p_old = cv2.perspectiveTransform(corners, _mask_M)
    p_new = cv2.perspectiveTransform(corners, M_new)
    max_shift = float(np.max(np.linalg.norm(p_new - p_old, axis=2)))
    return max_shift > MASK_REPRO_THR   # 阈值 15px`

若任一角点位移超过 15px，触发重新检测。

### 5.3 首次检测触发逻辑

`# 连续 30 帧稳定检测到 4 个角点后，自动触发第一次检测
if show_warp and M is not None and not _mask_running and _M_stable_frames >= 30:
    need_detect = (not _first_detect_done) or \
                  (_first_detect_done and _perspective_changed(M))
    if need_detect:
        _first_detect_done = True
        _trigger_mask(warped_now, markers)`

---

### 六、YOLO 标识检测优化

### 6.1 全图推理（关键修正）

早期将 ROI 区域裁剪后送入 YOLO，检测率极低。原因：模型在完整 600×600 俯视图上训练，裁剪后标识填满整帧，**域不匹配（domain mismatch）**。

正确做法：**YOLO 在完整 600×600 warped 图上推理**，再用 ROI 中心坐标过滤结果：

`for r in model(warped, verbose=False, conf=SIGN_CONF):
    for box, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        bx = float((box[0] + box[2]) / 2)
        by = float((box[1] + box[3]) / 2)
        if roi[0] <= bx <= roi[2] and roi[1] <= by <= roi[3]:
            best[name] = (conf_f, box)   # 只保留中心落在 ROI 内的检测`

修正后立即检测成功：`stop=0.907, light_off=0.825, speed_55=0.475`

### 6.2 帧跳缓存（Frame-Skip Caching）

YOLO 在 CPU 上单次推理约 100–300ms，每帧推理会使主循环降至 3–5 FPS。

`SIGN_EVERY_N = 8   # 每 8 帧推理一次
_sign_cache  = {'light': 'OFF', 'stop': False, 'speed': False, 'boxes': []}

_sign_frame_cnt += 1
if _sign_frame_cnt % SIGN_EVERY_N == 0:
    _sign_cache = detect_signs(warped)
signs = _sign_cache   # 中间帧复用上次结果`

有效帧率从 ~5 FPS 恢复到 ~25 FPS。

---

### 七、摄像头稳定性（GoPro on Mac）

GoPro 通过 USB 以 Webcam 模式接入 Mac 时，偶尔无法打开或推流未就绪。

**修复措施：**

`def open_camera(source, retries=5):
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)  # Mac 专用 backend
        if not cap.isOpened():
            time.sleep(3.0); continue
        time.sleep(2.0)    # 等待硬件开始推流
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        cap.release(); time.sleep(3.0)`

同时：仅在未指定 `--source` 时才扫描可用摄像头（避免扫描过程干扰 GoPro 驱动状态）。

### 主循环帧丢失容错

`_read_failures = 0
_MAX_FAILURES  = 300   # 连续失败 300 帧（约 10s）才退出

while True:
    ret, frame = cap.read()
    if not ret:
        _read_failures += 1
        if _read_failures >= _MAX_FAILURES: break
        cv2.waitKey(1); continue
    _read_failures = 0`

GoPro 偶发丢帧不再导致系统直接退出。

---

### 八、`is_on_track()` ROI 短路逻辑

小车若压在标识区域（贴纸覆盖轨道处），掩膜像素不连续可能判定为"脱轨"。

`def is_on_track(saved_mask, wx, wy, check_r=20):
    # 标识 ROI 下方必然是轨道，直接返回 True
    for rx1, ry1, rx2, ry2 in [ROI_LIGHT, ROI_STOP, ROI_55]:
        if rx1 <= wx <= rx2 and ry1 <= wy <= ry2:
            return True
    # 正常掩膜像素检查
    region = saved_mask[y1:y2, x1:x2]
    track_ratio = float(np.sum(region > 0)) / region.size
    ...`

---

### 九、工作量总结

| 模块 | 关键技术 |
| --- | --- |
| 自动轨道分割 | Otsu 双极性、形态学 close/open、半分辨率处理 |
| 遮挡修复 | `_bridge_sign_roi`：ROI 外缘扫描 + 最长对连线 |
| 非阻塞检测 | Python daemon thread + 共享列表容器 |
| 视角变化重检 | 透视矩阵角点位移比较（15px 阈值） |
| YOLO 推理 | 全图推理 + ROI 中心过滤 + 帧跳缓存（每8帧） |
| 摄像头稳定 | CAP_AVFOUNDATION + 重试循环 + 帧丢失计数器 |
| 脱轨判断 | 防抖（debounce 8帧） + ROI 短路 |