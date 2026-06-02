#!/usr/bin/env python3
"""
pc_controller.py — 电脑端远程控制器
运行于有 GPU / CPU 的电脑

功能:
  1. 接收小车摄像头视频流 → 原画展示 + YOLO 实时识别
  2. 接收小车传感器数据 (灰度、超声波、舵机角度)
  3. 网页前端:
     - 两个视频窗口: 原视频流 / YOLO 识别结果
     - 传感器数据实时展示
     - 按键/按钮控制小车
  4. 控制指令转发至小车

架构:
  小车 (car_server.py)  ──TCP+HTTP──>  本程序 (pc_controller.py)  ──Flask──> 浏览器
                                        │
                                        └── YOLO (ultralytics)
                                        │
                                        └── 网页前端

启动:
  # CPU 模式 (默认)
  python pc_controller.py --car-host 192.168.x.x

  # GPU 模式
  python pc_controller.py --car-host 192.168.x.x --model yolov10n.pt

  # NPU 模式 (Ascend)
  python pc_controller.py --car-host 192.168.x.x --npu --om-model yolov10n.om
"""

import cv2
import numpy as np
import socket
import threading
import time
import argparse
import sys
import os
import json

from flask import Flask, Response, request, render_template_string


# ═══════════════════════════════════════════════════════════
#  COCO 类别 (YOLO)
# ═══════════════════════════════════════════════════════════
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


def _random_colors(n, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 255, size=(n, 3)).tolist()


COLORS = _random_colors(len(COCO_CLASSES))


# ═══════════════════════════════════════════════════════════
#  YOLO Detector 抽象
# ═══════════════════════════════════════════════════════════
class Detector:
    """基类: 输入 BGR frame → 返回标注后的 frame"""
    def detect(self, frame):
        raise NotImplementedError


class UltralyticsDetector(Detector):
    """Ultralytics YOLO (CPU / GPU / MPS)"""
    def __init__(self, model_path, conf=0.25, iou=0.45):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        print(f"[YOLO] Ultralytics 模型已加载: {model_path}")

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        return results[0].plot()


class AscendNPUDetector(Detector):
    """Ascend NPU (ais_bench) 推理"""
    def __init__(self, om_path, conf_threshold=0.25, nms_threshold=0.45):
        self.om_path = om_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._session = None
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._session is not None:
            return
        with self._init_lock:
            if self._session is not None:
                return
            from ais_bench.infer.interface import InferSession
            self._session = InferSession(device_id=0, model_path=self.om_path)
            print(f"[NPU] InferSession ready")

    @staticmethod
    def _preprocess(frame):
        h, w = frame.shape[:2]
        length = max(h, w)
        canvas = np.zeros((length, length, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        scale = length / 640.0
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255, (640, 640), swapRB=True)
        return blob, scale, (h, w)

    def _postprocess(self, outputs, scale, orig_shape, draw_target):
        preds = np.array([cv2.transpose(outputs[0][0])])
        rows = preds.shape[1]
        boxes, scores, class_ids = [], [], []
        for i in range(rows):
            classes_scores = preds[0][i][4:]
            minScore, maxScore, minClassLoc, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
            if maxScore < self.conf_threshold:
                continue
            box = [
                preds[0][i][0] - 0.5 * preds[0][i][2],
                preds[0][i][1] - 0.5 * preds[0][i][3],
                preds[0][i][2], preds[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(int(maxClassIndex))
        if not boxes:
            return
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold, 0.5)
        for idx in result_boxes.flatten() if len(result_boxes) > 0 else []:
            x = round(boxes[idx][0] * scale)
            y = round(boxes[idx][1] * scale)
            x2 = round((boxes[idx][0] + boxes[idx][2]) * scale)
            y2 = round((boxes[idx][1] + boxes[idx][3]) * scale)
            cls_id = class_ids[idx]
            conf = scores[idx]
            label = f"{COCO_CLASSES[cls_id]} ({conf:.2f})"
            color = COLORS[cls_id]
            cv2.rectangle(draw_target, (x, y), (x2, y2), color, 2)
            cv2.putText(draw_target, label, (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detect(self, frame):
        self._ensure_session()
        blob, scale, orig_shape = self._preprocess(frame)
        outputs = self._session.infer(feeds=blob, mode="static")
        self._postprocess(outputs, scale, orig_shape, draw_target=frame)
        return frame


# ═══════════════════════════════════════════════════════════
#  PC 端控制器
# ═══════════════════════════════════════════════════════════
class PCController:
    """连接小车, 接收视频/传感器, 运行 YOLO, 提供 Web 界面"""

    def __init__(self, car_host, control_port, video_port, detector,
                 width=640, height=480):
        self.car_host = car_host
        self.control_port = control_port
        self.video_url = f"http://{car_host}:{video_port}/mjpg"
        self.detector = detector
        self.width = width
        self.height = height

        # ── 小车控制状态 ──
        self.speed = 0
        self.status = 'stop'
        self.running = True

        # ── TCP 控制连接 ──
        self.sock = None
        self.control_connected = False

        # ── 传感器数据 (从小车推送) ──
        self.sensor_lock = threading.Lock()
        self.distance = -1.0
        self.grayscale = [0, 0, 0]
        self.pan_angle = 0
        self.tilt_angle = 0
        self.dir_angle = 0

        # ── 视频帧缓存 ──
        self.raw_frame = None       # 原始帧 (JPEG bytes)
        self.yolo_frame = None      # YOLO 标注帧 (JPEG bytes)
        self.frame_lock = threading.Lock()

        # ── FPS 统计 ──
        self.raw_fps = 0.0
        self.yolo_fps = 0.0

    # ── TCP 控制连接 ─────────────────────────────────────
    def connect_control(self):
        """连接到小车的 TCP 控制端口"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.car_host, self.control_port))
            self.control_connected = True
            print(f"[TCP] 控制连接成功: {self.car_host}:{self.control_port}")
            # 启动传感器监听线程
            t = threading.Thread(target=self._sensor_listener, daemon=True)
            t.start()
            return True
        except Exception as e:
            print(f"[TCP] 控制连接失败: {e}")
            self.sock = None
            self.control_connected = False
            return False

    def _sensor_listener(self):
        """持续接收小车推送的传感器数据"""
        buf = b''
        while self.running and self.sock:
            try:
                data = self.sock.recv(1024)
                if not data:
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    self._parse_sensor(line.decode('utf-8', errors='ignore'))
            except socket.timeout:
                continue
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            except Exception:
                break

    def _parse_sensor(self, line):
        """解析小车推送的传感器行, 格式: @sensor:dist|gs0,gs1,gs2|pan|tilt|dir"""
        if not line.startswith('@sensor:'):
            return
        parts = line[len('@sensor:'):].split('|')
        if len(parts) < 5:
            return
        with self.sensor_lock:
            try:
                self.distance = round(float(parts[0]), 1)
            except ValueError:
                pass
            try:
                self.grayscale = [round(float(v)) for v in parts[1].split(',')]
            except ValueError:
                pass
            try:
                self.pan_angle = int(float(parts[2]))
            except (ValueError, IndexError):
                pass
            try:
                self.tilt_angle = int(float(parts[3]))
            except (ValueError, IndexError):
                pass
            try:
                self.dir_angle = int(float(parts[4]))
            except (ValueError, IndexError):
                pass

    def send_command(self, cmd):
        """发送控制指令到小车"""
        if not self.control_connected:
            if not self.connect_control():
                return
        try:
            self.sock.sendall((cmd + '\n').encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[TCP] 发送失败: {e}, 尝试重连...")
            self.control_connected = False
            self.sock = None

    # ── 按键处理 ─────────────────────────────────────────
    def process_key(self, key_char):
        """处理按键事件, 更新本地状态并发送指令到小车"""
        k = key_char.lower()

        if k == 'f':
            self.status = 'stop'
            self.send_command('stop')

        elif k == 'w':
            if self.speed == 0:
                self.speed = 10
            # 限制高速前进时直接切换的最大速度
            if self.status != 'forward' and self.speed > 60:
                self.speed = 60
            self.status = 'forward'
            self.send_command(f'forward {self.speed}')

        elif k == 's':
            if self.speed == 0:
                self.speed = 10
            if self.status != 'backward' and self.speed > 60:
                self.speed = 60
            self.status = 'backward'
            self.send_command(f'backward {self.speed}')

        elif k == 'a':
            if self.speed == 0:
                self.speed = 10
            self.status = 'turn left'
            self.send_command('left')

        elif k == 'd':
            if self.speed == 0:
                self.speed = 10
            self.status = 'turn right'
            self.send_command('right')

        elif k == 'o':  # 加速
            if self.speed <= 90:
                self.speed += 10
            if self.status in ('forward', 'backward'):
                self.send_command(f'{self.status.split()[0]} {self.speed}')

        elif k == 'p':  # 减速
            if self.speed >= 10:
                self.speed -= 10
            if self.speed == 0:
                self.status = 'stop'
                self.send_command('stop')
            elif self.status in ('forward', 'backward'):
                self.send_command(f'{self.status.split()[0]} {self.speed}')

        elif k == 'i':  # 云台上仰
            with self.sensor_lock:
                a = max(-35, min(65, self.tilt_angle - 5))
                self.tilt_angle = a
            self.send_command(f'tilt {a}')

        elif k == 'k':  # 云台下俯
            with self.sensor_lock:
                a = max(-35, min(65, self.tilt_angle + 5))
                self.tilt_angle = a
            self.send_command(f'tilt {a}')

        elif k == 'j':  # 云台左转
            with self.sensor_lock:
                a = max(-90, min(90, self.pan_angle - 5))
                self.pan_angle = a
            self.send_command(f'pan {a}')

        elif k == 'l':  # 云台右转
            with self.sensor_lock:
                a = max(-90, min(90, self.pan_angle + 5))
                self.pan_angle = a
            self.send_command(f'pan {a}')

        elif k == 'r':  # 云台复位
            with self.sensor_lock:
                self.pan_angle = 0
                self.tilt_angle = 0
            self.send_command('cam_reset')

        elif k == 'h':  # 鸣笛
            self.send_command('horn')

    # ── 视频管线 ─────────────────────────────────────────
    def video_pipeline(self):
        """主循环: 从小车拉取视频 → YOLO 检测 → 缓存帧"""
        print(f"[VIDEO] 连接小车视频流: {self.video_url}")

        # 先确保控制连接建立
        if not self.control_connected:
            self.connect_control()

        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print("[ERROR] 无法打开小车视频流")
            return

        # 设置 MJPG 解码格式
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except Exception:
            pass

        # FPS 统计
        raw_fc = 0
        yolo_fc = 0
        fps_timer = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 视频流断开, 1s 后重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.video_url)
                continue

            # ── 1. 缓存原始帧 ──
            raw_fc += 1
            _, raw_jpeg = cv2.imencode('.jpg', frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self.frame_lock:
                self.raw_frame = raw_jpeg.tobytes()

            # ── 2. YOLO 检测 ──
            yolo_fc += 1
            try:
                annotated = self.detector.detect(frame.copy())
                # 叠加 HUD 信息
                annotated = self._draw_hud(annotated)
                _, yolo_jpeg = cv2.imencode('.jpg', annotated,
                                             [cv2.IMWRITE_JPEG_QUALITY, 70])
                with self.frame_lock:
                    self.yolo_frame = yolo_jpeg.tobytes()
            except Exception as e:
                print(f"[YOLO ERROR] {e}")
                with self.frame_lock:
                    self.yolo_frame = raw_jpeg.tobytes()

            # ── 3. FPS 统计 (每 30 帧) ──
            if raw_fc % 30 == 0:
                elapsed = time.time() - fps_timer
                if elapsed > 0:
                    self.raw_fps = raw_fc / elapsed
                    self.yolo_fps = yolo_fc / elapsed
                # 每 30 帧输出一次状态
                elapsed = time.time() - fps_timer
                if elapsed > 0:
                    print(f"  RAW: {self.raw_fps:.1f} FPS | "
                          f"YOLO: {self.yolo_fps:.1f} FPS | "
                          f"Dist: {self.distance} cm", end='\r')

            # 控制帧率
            time.sleep(0.01)

        cap.release()
        self.cleanup()

    def _draw_hud(self, frame):
        """在帧上绘制 HUD 信息 (传感器数据)"""
        with self.sensor_lock:
            dist = self.distance
            gs = self.grayscale
            pa = self.pan_angle
            ta = self.tilt_angle
            da = self.dir_angle

        lines = [
            f"Status: {self.status}   Speed: {self.speed}",
            f"Ultrasonic: {dist} cm",
            f"Grayscale: {gs[0]:.0f} / {gs[1]:.0f} / {gs[2]:.0f}",
            f"Pan: {pa}  Tilt: {ta}  Dir: {da}",
            f"YOLO FPS: {self.yolo_fps:.1f}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    # ── 帧获取接口 (给 Flask) ──
    def get_raw_frame(self):
        with self.frame_lock:
            return self.raw_frame

    def get_yolo_frame(self):
        with self.frame_lock:
            return self.yolo_frame

    # ── 清理 ──
    def cleanup(self):
        self.running = False
        self.send_command('stop')
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        print("\n[OK] PC 控制器已清理")


# ═══════════════════════════════════════════════════════════
#  Flask Web 服务器
# ═══════════════════════════════════════════════════════════
controller = None
app = Flask(__name__)


# ── 前端 HTML (两个视频窗口 + 传感器数据 + 按键控制) ────
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Picar-X 远程控制 + YOLO</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#ccc;font-family:'Microsoft YaHei',monospace;min-height:100vh}

/* ── 头部 ── */
.header{text-align:center;padding:10px 0 4px}
.header h1{font-size:1.2em;color:#0f0;letter-spacing:3px;text-shadow:0 0 8px rgba(0,255,0,.3)}
.header .sub{color:#555;font-size:0.75em}

/* ── 视频区域 ── */
.video-row{display:flex;flex-wrap:wrap;justify-content:center;gap:6px;padding:4px 8px}
.video-box{flex:1 1 45%;min-width:300px;max-width:640px;background:#111;border:1px solid #333;border-radius:6px;overflow:hidden}
.video-box .label{background:#1a1a1a;padding:4px 10px;font-size:0.75em;color:#888;border-bottom:1px solid #333}
.video-box .label .tag{color:#0f0;font-weight:bold}
.video-box .label .tag.yolo{color:#fa0}
.video-box img{width:100%;display:block}

/* ── 传感器面板 ── */
.panel{max-width:640px;margin:6px auto;border:1px solid #333;border-radius:6px;overflow:hidden}
.panel .title{background:#1a1a1a;padding:4px 10px;font-size:0.75em;color:#888;border-bottom:1px solid #333}
.panel .row{display:flex;justify-content:space-between;padding:3px 10px;font-size:0.85em}
.panel .row:nth-child(odd){background:#141414}
.panel .lbl{color:#888}
.panel .val{color:#0f0;font-weight:bold}
.panel .val.warn{color:#fa0}

/* ── 控制按钮 ── */
.controls{max-width:640px;margin:6px auto;-webkit-tap-highlight-color:transparent;user-select:none}
.ctrl-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin:4px 0}
.ctrl-grid button{padding:14px 0;font-size:1.1em;background:#1a1a1a;color:#0f0;
       border:1px solid #444;border-radius:8px;cursor:pointer;transition:all .1s}
.ctrl-grid button:active{background:#0a0;color:#000;border-color:#0f0;transform:scale(.95)}
.ctrl-grid .empty{background:transparent;border:1px solid transparent;pointer-events:none}
.ctrl-row{display:flex;gap:5px;margin:4px 0;flex-wrap:wrap;justify-content:center}
.ctrl-row button{flex:1;min-width:60px;padding:10px 8px;font-size:0.85em;background:#1a1a1a;
       color:#0f0;border:1px solid #444;border-radius:8px;cursor:pointer;transition:all .1s}
.ctrl-row button:active{background:#0a0;color:#000;border-color:#0f0}
.ctrl-row .warn{color:#fa0}
.ctrl-row .warn:active{background:#a50;color:#000;border-color:#fa0}

/* ── 快捷键提示 ── */
.hint{max-width:640px;margin:6px auto;color:#555;font-size:0.7em;text-align:center;line-height:1.8}
.hint kbd{background:#222;color:#0f0;padding:1px 6px;border-radius:3px;border:1px solid #444;font-size:0.9em}

/* ── 状态连接指示器 ── */
.conn-indicator{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.conn-ok{background:#0f0;box-shadow:0 0 6px #0f0}
.conn-err{background:#f00;box-shadow:0 0 6px #f00}

@media(max-width:640px){
  .video-box{flex:1 1 100%;max-width:100%}
  .ctrl-grid button{padding:18px 0;font-size:1.3em}
  .ctrl-row button{padding:14px 8px;font-size:1em}
}
</style>
</head>
<body>

<div class="header">
  <h1>▣ PICAR-X 远程控制</h1>
  <div class="sub">YOLO 实时识别 · 双路视频</div>
</div>

<!-- ── 双视频窗口 ── -->
<div class="video-row">
  <div class="video-box">
    <div class="label"><span class="tag">● 原画</span>  RAW · <span id="raw-fps">0</span> FPS</div>
    <img id="raw-video" src="/raw_mjpg" alt="原始视频流">
  </div>
  <div class="video-box">
    <div class="label"><span class="tag yolo">● YOLO</span>  DETECT · <span id="yolo-fps">0</span> FPS</div>
    <img id="yolo-video" src="/yolo_mjpg" alt="YOLO 识别结果">
  </div>
</div>

<!-- ── 传感器数据 ── -->
<div class="panel">
  <div class="title">📡 传感器数据</div>
  <div class="row">
    <span class="lbl">连接状态</span>
    <span class="val" id="conn-status"><span class="conn-indicator conn-err"></span>未连接</span>
  </div>
  <div class="row">
    <span class="lbl">运行状态</span>
    <span class="val" id="st">停止</span>
  </div>
  <div class="row">
    <span class="lbl">速度</span>
    <span class="val" id="sp">0</span>
  </div>
  <div class="row">
    <span class="lbl">超声波</span>
    <span class="val" id="dist">-- cm</span>
  </div>
  <div class="row">
    <span class="lbl">灰度 (L/M/R)</span>
    <span class="val" id="gs">-- / -- / --</span>
  </div>
  <div class="row">
    <span class="lbl">云台 Pan</span>
    <span class="val" id="pan">0°</span>
  </div>
  <div class="row">
    <span class="lbl">云台 Tilt</span>
    <span class="val" id="tilt">0°</span>
  </div>
  <div class="row">
    <span class="lbl">转向舵机</span>
    <span class="val" id="dir">0°</span>
  </div>
</div>

<!-- ── 控制按钮 ── -->
<div class="controls">
  <!-- 方向键 -->
  <div class="ctrl-grid">
    <div class="empty"></div>
    <button id="bw" title="前进 W">▲<br><small>W</small></button>
    <div class="empty"></div>
    <button id="ba" title="左转 A">◀<br><small>A</small></button>
    <button id="bf" title="停止 F">■<br><small>F</small></button>
    <button id="bd" title="右转 D">▶<br><small>D</small></button>
    <div class="empty"></div>
    <button id="bs" title="后退 S">▼<br><small>S</small></button>
    <div class="empty"></div>
  </div>

  <!-- 速度 + 功能 -->
  <div class="ctrl-row">
    <button id="bo">➕ 加速 <small>O</small></button>
    <button id="bp">➖ 减速 <small>P</small></button>
    <button class="warn" id="bh">🔊 鸣笛 <small>H</small></button>
  </div>

  <!-- 云台控制 -->
  <div class="ctrl-row">
    <button id="bi">⬆ 上仰 <small>I</small></button>
    <button id="bk">⬇ 下俯 <small>K</small></button>
    <button id="bj">⬅ 左转 <small>J</small></button>
    <button id="bl">➡ 右转 <small>L</small></button>
    <button id="br">⟲ 复位 <small>R</small></button>
  </div>
</div>

<!-- ── 快捷键提示 ── -->
<div class="hint">
  <kbd>W</kbd>前进 <kbd>S</kbd>后退 <kbd>A</kbd>左转 <kbd>D</kbd>右转
  <kbd>F</kbd>停止 &nbsp;|&nbsp;
  <kbd>O</kbd>加速 <kbd>P</kbd>减速 <kbd>H</kbd>鸣笛 &nbsp;|&nbsp;
  <kbd>I</kbd><kbd>K</kbd>云台俯仰 <kbd>J</kbd><kbd>L</kbd>云台水平 <kbd>R</kbd>复位
</div>

<script>
// ── 按键映射 ──
const keyMap = {
  bw:'w', bs:'s', ba:'a', bd:'d', bf:'f',
  bo:'o', bp:'p', bh:'h',
  bi:'i', bk:'k', bj:'j', bl:'l', br:'r'
};

// ── 发送控制指令 ──
function send(k) {
  fetch('/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: k})
  })
  .then(r => r.json())
  .then(d => {
    document.getElementById('st').textContent = d.status;
    document.getElementById('sp').textContent = d.speed;
  })
  .catch(() => {});
}

// ── 键盘事件 ──
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if ('wasdfopthijklr'.includes(k)) {
    e.preventDefault();
    send(k);
  }
});

// ── 按钮事件 ──
Object.entries(keyMap).forEach(([id, k]) => {
  const btn = document.getElementById(id);
  if (!btn) return;
  const fn = e => { e.preventDefault(); send(k); };
  btn.addEventListener('mousedown', fn);
  btn.addEventListener('touchstart', fn);
});

// ── 轮询状态 (传感器 + FPS) ──
function pollStatus() {
  fetch('/status')
    .then(r => r.json())
    .then(d => {
      document.getElementById('dist').textContent = d.distance + ' cm';
      document.getElementById('gs').textContent = d.gs.join(' / ');
      document.getElementById('pan').textContent  = d.pan + '°';
      document.getElementById('tilt').textContent = d.tilt + '°';
      document.getElementById('dir').textContent  = d.dir + '°';
      document.getElementById('raw-fps').textContent = d.raw_fps.toFixed(1);
      document.getElementById('yolo-fps').textContent = d.yolo_fps.toFixed(1);
      const connEl = document.getElementById('conn-status');
      if (d.connected) {
        connEl.innerHTML = '<span class="conn-indicator conn-ok"></span>已连接';
      } else {
        connEl.innerHTML = '<span class="conn-indicator conn-err"></span>未连接';
      }
    })
    .catch(() => {});
  setTimeout(pollStatus, 300);
}
pollStatus();
</script>
</body>
</html>
'''


# ── Flask 路由 ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/raw_mjpg')
def raw_mjpg():
    """原始视频流 (从小车转发)"""
    def generate():
        while controller and controller.running:
            frame = controller.get_raw_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/yolo_mjpg')
def yolo_mjpg():
    """YOLO 标注视频流"""
    def generate():
        while controller and controller.running:
            frame = controller.get_yolo_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control', methods=['POST'])
def control():
    """接收前端控制指令, 转发到小车"""
    data = request.get_json()
    key = data.get('key', '').lower()
    if key and controller:
        controller.process_key(key)
    return {'status': controller.status if controller else 'unknown',
            'speed': controller.speed if controller else 0}


@app.route('/status')
def get_status():
    """返回传感器 + 连接状态"""
    if not controller:
        return {'connected': False, 'distance': 0, 'gs': [0,0,0],
                'pan': 0, 'tilt': 0, 'dir': 0,
                'raw_fps': 0, 'yolo_fps': 0, 'status': 'unknown', 'speed': 0}
    with controller.sensor_lock:
        distance = controller.distance
        gs = controller.grayscale[:]
        pan = controller.pan_angle
        tilt = controller.tilt_angle
        dir_ = controller.dir_angle
    return {
        'connected': controller.control_connected,
        'distance': distance,
        'gs': gs,
        'pan': pan,
        'tilt': tilt,
        'dir': dir_,
        'status': controller.status,
        'speed': controller.speed,
        'raw_fps': round(controller.raw_fps, 1),
        'yolo_fps': round(controller.yolo_fps, 1),
    }


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════
def build_detector(args):
    """根据参数创建 Detector 实例"""
    if args.npu:
        print(f"[NPU] 加载 OM 模型: {args.om_model}")
        return AscendNPUDetector(
            om_path=args.om_model,
            conf_threshold=args.confidence,
            nms_threshold=args.nms,
        )
    else:
        print(f"[YOLO] 加载模型: {args.model} (device={args.device})")
        return UltralyticsDetector(
            model_path=args.model,
            conf=args.confidence,
            iou=args.nms,
        )


def main():
    global controller

    parser = argparse.ArgumentParser(
        description='Picar-X 电脑端远程控制器 (YOLO + 双路视频)')
    parser.add_argument('--car-host', type=str, default='10.129.137成都.156',
                        help='小车 IP 地址')
    parser.add_argument('--control-port', type=int, default=8888,
                        help='小车 TCP 控制端口')
    parser.add_argument('--video-port', type=int, default=9000,
                        help='小车 MJPG 视频端口')
    parser.add_argument('--serve-port', type=int, default=5000,
                        help='本机 Web 服务端口')

    # YOLO 参数
    parser.add_argument('--model', type=str, default='yolov10n.pt',
                        help='Ultralytics 模型路径')
    parser.add_argument('--device', type=str, default='cpu',
                        help='推理设备: cpu, cuda, mps')
    parser.add_argument('--npu', action='store_true',
                        help='使用 Ascend NPU 推理')
    parser.add_argument('--om-model', type=str, default='yolov10n.om',
                        help='NPU OM 模型路径')
    parser.add_argument('--confidence', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--nms', type=float, default=0.45,
                        help='NMS IoU 阈值')

    # 视频参数
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)

    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Picar-X 电脑端远程控制器")
    print(f"{'='*55}")

    # ── 创建检测器 ──
    print("[INFO] 初始化 YOLO 检测器...")
    detector = build_detector(args)

    # ── 创建控制器 ──
    controller = PCController(
        car_host=args.car_host,
        control_port=args.control_port,
        video_port=args.video_port,
        detector=detector,
        width=args.width,
        height=args.height,
    )

    # ── 启动视频管线线程 ──
    video_thread = threading.Thread(target=controller.video_pipeline,
                                     daemon=True)
    video_thread.start()

    # 等待连接建立
    time.sleep(1)

    print(f"\n{'='*55}")
    print(f"  Web 界面:      http://localhost:{args.serve_port}/")
    print(f"  原画流:        http://localhost:{args.serve_port}/raw_mjpg")
    print(f"  YOLO 流:       http://localhost:{args.serve_port}/yolo_mjpg")
    print(f"  小车地址:      {args.car_host}:{args.control_port}")
    mode = "Ascend NPU" if args.npu else f"Ultralytics ({args.device})"
    print(f"  推理模式:      {mode}")
    print(f"{'='*55}\n")

    # ── 启动 Flask ──
    try:
        app.run(host='0.0.0.0', port=args.serve_port,
                threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断")
    finally:
        if controller:
            controller.cleanup()


if __name__ == '__main__':
    main()
