#!/usr/bin/env python3
"""
本机远程控制 Picar-X — 手动控制 + 自动循迹 + 红绿灯识别

三种模式:
  手动模式: 键盘 WASD 控制小车
  自动循迹模式: 灰度传感器循迹 + YOLO 红绿灯识别，停止线前自动响应

架构:
  浏览器 ──HTTP──> 本机 (YOLO + 控制逻辑) ──TCP──> 小车 (树莓派)
  浏览器 <──MJPG── 本机 (标注视频)          <──TCP── 小车 (传感器)

用法:
    python local_control.py --host <小车IP>
"""

import cv2
import numpy as np
import socket
import time
import threading
import argparse

from flask import Flask, Response, request, render_template_string

app = Flask(__name__)
controller = None


# ---------------------------------------------------------------------------
# HSV 颜色范围 — 红绿灯颜色分析 (与 traffic_light_control.py 一致)
# ---------------------------------------------------------------------------

RED_LOWER1 = (0, 60, 120)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (165, 60, 120)
RED_UPPER2 = (180, 255, 255)
YELLOW_LOWER = (15, 60, 120)
YELLOW_UPPER = (35, 255, 255)
GREEN_LOWER = (40, 60, 80)
GREEN_UPPER = (85, 255, 255)
MIN_PIXEL_THRESHOLD = 15


# ---------------------------------------------------------------------------
# Detector — Ultralytics YOLO (GPU/CPU)
# ---------------------------------------------------------------------------

class Detector:
    def detect(self, frame):
        raise NotImplementedError


class UltralyticsDetector(Detector):
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0].plot()


# ---------------------------------------------------------------------------
# Web 控制面板
# ---------------------------------------------------------------------------

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Picar-X 控制 (手动/自动循迹)</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#ccc;font-family:monospace;
     display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:0 8px}
h1{margin:10px 0 4px;font-size:1.1em;color:#0f0;letter-spacing:2px}
#video{max-width:100%;max-height:45vh;border:2px solid #222;border-radius:4px}

.panel{width:100%;max-width:640px;margin:4px 0}
.panel .row{display:flex;justify-content:space-between;padding:3px 6px;font-size:0.9em}
.panel .row:nth-child(odd){background:#141414}
.panel .lbl{color:#888}
.panel .val{color:#0f0;font-weight:bold}

/* 模式按钮 */
.mode-bar{display:flex;gap:8px;margin:6px 0;width:100%;max-width:640px}
.mode-btn{flex:1;padding:10px;font-size:1em;font-family:monospace;font-weight:bold;
          border:2px solid #444;border-radius:8px;cursor:pointer;transition:all .2s;
          -webkit-tap-highlight-color:transparent;user-select:none}
.mode-btn.manual{background:#1a1a1a;color:#0f0}
.mode-btn.manual.active{background:#0a0;color:#000;border-color:#0f0}
.mode-btn.auto{background:#1a1a1a;color:#fa0}
.mode-btn.auto.active{background:#a50;color:#000;border-color:#fa0}

/* 红绿灯指示器 */
.tl-indicator{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:4px}
.tl-indicator.red{background:#f44;box-shadow:0 0 8px #f44}
.tl-indicator.yellow{background:#ff4;box-shadow:0 0 8px #ff4}
.tl-indicator.green{background:#4f4;box-shadow:0 0 8px #4f4}
.tl-indicator.unknown{background:#666}

.grid{display:grid;grid-template-columns:60px 60px 60px;gap:6px;margin:6px 0}
.grid button,.row-btn button{padding:12px 0;font-size:1.2em;background:#1a1a1a;color:#0f0;
       border:1px solid #444;border-radius:6px;cursor:pointer;
       -webkit-tap-highlight-color:transparent;user-select:none}
.grid button:active,.row-btn button:active{background:#0a0;color:#000;border-color:#0f0}
.grid button:disabled,.row-btn button:disabled{opacity:0.3;pointer-events:none}
.row-btn{display:flex;gap:6px;margin:4px 0;flex-wrap:wrap;justify-content:center}
.row-btn button{padding:10px 14px;font-size:0.9em}
.row-btn .warn{color:#fa0}
.hint{color:#555;font-size:0.7em;margin:8px 0;text-align:center;line-height:1.6}
.hint kbd{background:#222;color:#0f0;padding:1px 5px;border-radius:3px;border:1px solid #444}
</style>
</head>
<body>
<h1>▣ PICAR-X 远程控制</h1>
<img id="video" src="/mjpg" alt="视频流">

<!-- 模式切换 -->
<div class="mode-bar">
  <button id="mode-manual" class="mode-btn manual active">◉ 手动</button>
  <button id="mode-auto" class="mode-btn auto">◎ 自动循迹</button>
</div>

<div class="panel">
  <div class="row"><span class="lbl">模式</span><span class="val" id="mode-display">手动</span></div>
  <div class="row"><span class="lbl">红绿灯</span><span class="val" id="tl-display">⚫ --</span></div>
  <div class="row"><span class="lbl">小车状态</span><span class="val" id="st">stop</span></div>
  <div class="row"><span class="lbl">速度</span><span class="val" id="sp">0</span></div>
  <div class="row"><span class="lbl">超声波</span><span class="val" id="dist">-- cm</span></div>
  <div class="row"><span class="lbl">灰度</span><span class="val" id="gs">-- / -- / --</span></div>
  <div class="row"><span class="lbl">云台</span><span class="val" id="cam">Pan 0&deg; Tilt 0&deg;</span></div>
  <div class="row"><span class="lbl">方向舵机</span><span class="val" id="dir">0&deg;</span></div>
</div>

<div class="grid">
  <div></div><button id="bw">W</button><div></div>
  <button id="ba">A</button><button id="bf">F</button><button id="bd">D</button>
  <div></div><button id="bs">S</button><div></div>
</div>

<div class="row-btn">
  <button id="bo">O +spd</button>
  <button id="bp">P -spd</button>
  <button id="bt">T 拍照</button>
  <button class="warn" id="bh">H 鸣笛</button>
</div>

<div class="row-btn">
  <button id="bi">I 仰</button>
  <button id="bk">K 俯</button>
  <button id="bj">J 左</button>
  <button id="bl">L 右</button>
  <button id="br">R 复位</button>
</div>

<div class="hint">
手动: <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> 移动 <kbd>F</kbd> 停止 &nbsp;|&nbsp;
<kbd>M</kbd> 切换模式 &nbsp;|&nbsp;
<kbd>O</kbd><kbd>P</kbd> 加减速 <kbd>H</kbd> 鸣笛 &nbsp;|&nbsp;
<kbd>I</kbd><kbd>K</kbd><kbd>J</kbd><kbd>L</kbd> 云台 <kbd>R</kbd> 复位
</div>

<script>
let isAuto = false;

// 发送控制指令
function send(k) {
  fetch('/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: k})
  })
  .then(r => r.json())
  .then(d => updateUI(d))
  .catch(() => {});
}

// 切换模式
function setMode(auto) {
  isAuto = auto;
  document.getElementById('mode-manual').className = 'mode-btn manual' + (auto ? '' : ' active');
  document.getElementById('mode-auto').className  = 'mode-btn auto'    + (auto ? ' active' : '');
  document.getElementById('mode-display').textContent = auto ? '自动循迹' : '手动';
  // 通知后端
  fetch('/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: auto ? 'auto_on' : 'auto_off'})
  }).then(r => r.json()).then(d => updateUI(d)).catch(() => {});
}

document.getElementById('mode-manual').addEventListener('click', () => setMode(false));
document.getElementById('mode-auto').addEventListener('click',  () => setMode(true));

// 键盘事件
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'm') { e.preventDefault(); setMode(!isAuto); return; }
  if ('wasdfopthijklr'.includes(k)) { e.preventDefault(); send(k); }
});

// 按钮事件
const keyMap = {
  bw:'w', bs:'s', ba:'a', bd:'d', bf:'f',
  bo:'o', bp:'p', bt:'t', bh:'h',
  bi:'i', bk:'k', bj:'j', bl:'l', br:'r'
};
Object.entries(keyMap).forEach(([id, k]) => {
  const btn = document.getElementById(id);
  if (!btn) return;
  const fn = e => { e.preventDefault(); send(k); };
  btn.addEventListener('mousedown', fn);
  btn.addEventListener('touchstart', fn);
});

// UI 更新
function updateUI(d) {
  document.getElementById('st').textContent = d.status;
  document.getElementById('sp').textContent = d.speed;
  if (d.tl_color) {
    let emoji = '⚫';
    if (d.tl_color === 'red') emoji = '🔴';
    else if (d.tl_color === 'yellow') emoji = '🟡';
    else if (d.tl_color === 'green') emoji = '🟢';
    const tlText = d.tl_color === 'unknown' ? '⚫ 无灯' : emoji + ' ' + d.tl_color;
    document.getElementById('tl-display').textContent = tlText + ' (' + d.tl_conf.toFixed(2) + ')';
  }
}

// 轮询状态
function pollStatus() {
  fetch('/status').then(r => r.json()).then(d => {
    document.getElementById('dist').textContent = d.distance + ' cm';
    document.getElementById('gs').textContent = d.gs.join(' / ');
    document.getElementById('cam').textContent = 'Pan ' + d.pan + '° Tilt ' + d.tilt + '°';
    document.getElementById('dir').textContent = d.dir + '°';
    document.getElementById('mode-display').textContent = d.auto_mode ? '自动循迹' : '手动';
    updateUI(d);
  }).catch(() => {});
  setTimeout(pollStatus, 300);
}
pollStatus();
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# 颜色分析工具函数
# ---------------------------------------------------------------------------

def _analyze_tl_roi(roi):
    """对红绿灯 ROI 做 HSV 二值化，返回 (dominant_color, counts_dict)."""
    if roi.size == 0:
        return 'unknown', {'red': 0, 'yellow': 0, 'green': 0}

    h, w = roi.shape[:2]
    if max(h, w) > 100:
        scale = 100.0 / max(h, w)
        roi = cv2.resize(roi, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    counts = {
        'red': cv2.countNonZero(mask_red),
        'yellow': cv2.countNonZero(mask_yellow),
        'green': cv2.countNonZero(mask_green),
    }

    dominant = max(counts, key=counts.get)
    if counts[dominant] < MIN_PIXEL_THRESHOLD:
        return 'unknown', counts
    return dominant, counts


# ---------------------------------------------------------------------------
# 远程控制器
# ---------------------------------------------------------------------------

class RemoteController:
    def __init__(self, picar_host, control_port, video_port, detector):
        self.detector = detector
        self.picar_host = picar_host
        self.control_port = control_port
        self.video_url = f"http://{picar_host}:{video_port}/mjpg"

        # ── 基础控制状态 ──
        self.speed = 0
        self.status = 'stop'
        self.running = True
        self.sock = None

        # ── 传感器数据 ──
        self._sensor_lock = threading.Lock()
        self.ultrasonic_distance = -1
        self.grayscale_values = [0, 0, 0]
        self.pan_angle = 0
        self.tilt_angle = 0
        self.dir_angle = 0

        # ── 视频帧 ──
        self.latest_frame = None
        self._frame_lock = threading.Lock()

        # ── 自动循迹模式 ──
        self.auto_mode = False
        self.auto_state = 'tracking'    # tracking | stopped | passing
        self.auto_state_start = 0.0
        self.auto_pass_speed = 50       # 通过路口时的速度

        # 灰度传感器阈值 (低于此值 = 在线)
        self.gs_threshold = 800

        # 停止线防抖计数器
        self._stop_line_count = 0

        # 红绿灯状态 (由 YOLO 分析更新)
        self._tl_color = 'unknown'
        self._tl_conf = 0.0

    # -- TCP 控制连接 ---------------------------------------------------------

    def connect_control(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.picar_host, self.control_port))
            print(f"[控制] 已连接至小车 {self.picar_host}:{self.control_port}")
            threading.Thread(target=self._sensor_listener, daemon=True).start()
            return True
        except Exception as e:
            print(f"[控制] 连接失败: {e}")
            self.sock = None
            return False

    def _sensor_listener(self):
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
        if not line.startswith('@sensor:'):
            return
        parts = line[len('@sensor:'):].split('|')
        if len(parts) < 5:
            return
        with self._sensor_lock:
            try:
                self.ultrasonic_distance = float(parts[0])
            except ValueError:
                pass
            try:
                self.grayscale_values = [float(v) for v in parts[1].split(',')]
            except ValueError:
                pass
            for idx, key in enumerate(['pan_angle', 'tilt_angle', 'dir_angle'], 2):
                try:
                    setattr(self, key, int(float(parts[idx])))
                except (ValueError, IndexError):
                    pass

    def send_command(self, cmd):
        if self.sock is None:
            if not self.connect_control():
                return
        try:
            self.sock.sendall((cmd + '\n').encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[控制] 发送失败，正在重连: {e}")
            self.sock = None

    def _send_state(self):
        """发送完整状态包: state <direction> <speed> <pan> <tilt> <dir_angle>"""
        self.send_command(
            f'state {self.status} {self.speed} '
            f'{self.pan_angle} {self.tilt_angle} {self.dir_angle}'
        )

    # -- 按键处理 ------------------------------------------------------------

    def process_key(self, key_char):
        k = key_char.lower()

        # 模式切换
        if key_char == 'auto_on':
            self.auto_mode = True
            self.auto_state = 'tracking'
            print("[模式] 切换到自动循迹模式")
            return
        elif key_char == 'auto_off':
            self.auto_mode = False
            self.status = 'stop'
            self.speed = 0
            self._send_state()
            print("[模式] 切换到手动模式")
            return
        elif k == 'm':
            # M 键也切换模式
            if not self.auto_mode:
                self.auto_mode = True
                self.auto_state = 'tracking'
                print("[模式] M键: 切换到自动循迹")
            else:
                self.auto_mode = False
                self.status = 'stop'
                self.speed = 0
                self._send_state()
                print("[模式] M键: 切换到手动")
            return

        # ── 手动控制 (自动模式下也响应，回到手动) ──
        if self.auto_mode and k in ('w', 'a', 's', 'd', 'f'):
            self.auto_mode = False
            self.status = 'stop'
            self.speed = 0
            print("[模式] 手动按键 → 切回手动模式")

        if k == 'f':
            self.status = 'stop'
            self.speed = 0

        elif k == 'w':
            if self.speed == 0:
                self.speed = 50
            self.status = 'forward'
            self.dir_angle = 0

        elif k == 's':
            if self.speed == 0:
                self.speed = 50
            self.status = 'backward'
            self.dir_angle = 0

        elif k == 'a':
            if self.speed == 0:
                self.speed = 50
            self.status = 'forward'
            self.dir_angle = -20

        elif k == 'd':
            if self.speed == 0:
                self.speed = 50
            self.status = 'forward'
            self.dir_angle = 20

        elif k == 'o':
            if self.speed <= 90:
                self.speed += 10

        elif k == 'p':
            if self.speed >= 10:
                self.speed -= 10
            if self.speed == 0:
                self.status = 'stop'

        elif k == 'i':
            with self._sensor_lock:
                self.tilt_angle = max(-35, min(65, self.tilt_angle - 5))

        elif k == 'k':
            with self._sensor_lock:
                self.tilt_angle = max(-35, min(65, self.tilt_angle + 5))

        elif k == 'j':
            with self._sensor_lock:
                self.pan_angle = max(-90, min(90, self.pan_angle - 5))

        elif k == 'l':
            with self._sensor_lock:
                self.pan_angle = max(-90, min(90, self.pan_angle + 5))

        elif k == 'r':
            with self._sensor_lock:
                self.pan_angle = 0
                self.tilt_angle = 0
                self.dir_angle = 0

        elif k == 'h':
            self.send_command('horn')
            return

        elif k == 't':
            self.send_command('photo')
            return

        self._send_state()

    # -- 红绿灯检测 (YOLO + HSV) ---------------------------------------------

    def _detect_traffic_light(self, frame):
        """在帧上检测红绿灯，返回 (color, confidence)，并在帧上画出."""
        results = self.detector.model(
            frame, classes=[9], conf=0.25, verbose=False)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._tl_color = 'unknown'
            self._tl_conf = 0.0
            return self._tl_color, self._tl_conf

        # 取置信度最高的红绿灯
        best_idx = int(boxes.conf.argmax())
        conf = float(boxes.conf[best_idx])
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])

        roi = frame[y1:y2, x1:x2]
        color, counts = _analyze_tl_roi(roi)

        # 在帧上绘制红绿灯检测信息
        color_bgr = {'red': (0, 0, 255), 'yellow': (0, 255, 255),
                     'green': (0, 255, 0), 'unknown': (128, 128, 128)}
        c = color_bgr.get(color, (128, 128, 128))
        label = f"TL: {color.upper()} ({conf:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), c, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cnt_txt = f"R:{counts['red']} Y:{counts['yellow']} G:{counts['green']}"
        cv2.putText(frame, cnt_txt, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        self._tl_color = color
        self._tl_conf = conf
        return color, conf

    # -- 灰度传感器循迹逻辑 (参考 6.line_tracking.py) -------------------------

    @staticmethod
    def _get_line_status_from_raw(gs_values, threshold):
        """将原始灰度 ADC 值转为线状态 (0=在线, 1=背景).

        返回: 'stop' | 'forward' | 'left' | 'right' | 'unknown'
        逻辑: 与 6.line_tracking.py 中 get_status() 一致
        """
        # 转换为布尔: 0 = 在线(黑/低反光), 1 = 背景(白/高反光)
        st = [0 if v < threshold else 1 for v in gs_values]

        if st == [0, 0, 0]:
            return 'stop'        # 停止线 (三路全黑)
        elif st[1] == 1:
            return 'forward'     # 中间在背景上，继续直行
        elif st[0] == 1:
            return 'right'       # 左在背景上 → 偏左 → 右转纠正
        elif st[2] == 1:
            return 'left'        # 右在背景上 → 偏右 → 左转纠正
        return 'unknown'

    # -- 自动循迹控制 --------------------------------------------------------

    def _line_follow(self, line_status):
        """根据线状态发送循迹指令."""
        if line_status == 'forward':
            self.status = 'forward'
            self.speed = 50
            self.dir_angle = 0
        elif line_status == 'left':
            self.status = 'forward'
            self.speed = 50
            self.dir_angle = 20
        elif line_status == 'right':
            self.status = 'forward'
            self.speed = 50
            self.dir_angle = -20
        elif line_status == 'stop':
            self.status = 'stop'
            self.speed = 0
        else:
            # 丢失线 — 直行低速尝试找回
            self.status = 'forward'
            self.speed = 30
            self.dir_angle = 0

        self._send_state()

    def _auto_control(self, frame):
        """自动循迹主逻辑：循迹 → 停止线 → 红绿灯判断 → 通过 → 恢复."""
        # 读取最新灰度值
        with self._sensor_lock:
            gs = list(self.grayscale_values)

        line_status = self._get_line_status_from_raw(gs, self.gs_threshold)
        now = time.time()

        # ════════════════════════════════════════════════════════════
        # 状态: tracking — 正在循迹
        # ════════════════════════════════════════════════════════════
        if self.auto_state == 'tracking':
            # 检测到停止线 (防抖: 连续多次才确认)
            if line_status == 'stop':
                self._stop_line_count += 1
                if self._stop_line_count >= 3:  # 连续3帧确认
                    print("[自动] ⛔ 停止线检测，停车等待红绿灯")
                    self.auto_state = 'stopped'
                    self.auto_state_start = now
                    self._stop_line_count = 0
                    self.status = 'stop'
                    self.speed = 0
                    self._send_state()
            else:
                self._stop_line_count = 0
                self._line_follow(line_status)

        # ════════════════════════════════════════════════════════════
        # 状态: stopped — 停止线等待红绿灯
        # ════════════════════════════════════════════════════════════
        elif self.auto_state == 'stopped':
            # 持续检测红绿灯
            color, conf = self._detect_traffic_light(frame)

            if color == 'green':
                print(f"[自动] 🟢 绿灯 (conf={conf:.2f}) → 速度50 通过")
                self.auto_state = 'passing'
                self.auto_state_start = now
                self.auto_pass_speed = 50
                self.status = 'forward'
                self.speed = 50
                self.dir_angle = 0
                self._send_state()

            elif color == 'yellow':
                print(f"[自动] 🟡 黄灯 (conf={conf:.2f}) → 鸣笛 + 速度10")
                self.send_command('horn')
                self.auto_state = 'passing'
                self.auto_state_start = now
                self.auto_pass_speed = 10
                self.status = 'forward'
                self.speed = 10
                self.dir_angle = 0
                self._send_state()

            # 红灯/无灯 → 保持停车

        # ════════════════════════════════════════════════════════════
        # 状态: passing — 正在通过路口
        # ════════════════════════════════════════════════════════════
        elif self.auto_state == 'passing':
            elapsed = now - self.auto_state_start
            if elapsed >= 3.0:
                print("[自动] ✅ 路口通过，恢复循迹")
                self.auto_state = 'tracking'
                self.auto_pass_speed = 50
                # 立即执行一次循迹
                with self._sensor_lock:
                    gs_now = list(self.grayscale_values)
                ls_now = self._get_line_status_from_raw(gs_now, self.gs_threshold)
                self._line_follow(ls_now)
            # 否则保持当前速度 (已由状态设置)

    # -- 视频流水线 -----------------------------------------------------------

    def draw_hud(self, frame, fps):
        with self._sensor_lock:
            dist = self.ultrasonic_distance
            gs = self.grayscale_values
            pa = self.pan_angle
            ta = self.tilt_angle
            da = self.dir_angle

        mode_str = 'AUTO' if self.auto_mode else 'MANUAL'
        tl_str = self._tl_color.upper()
        auto_state_str = {
            'tracking': 'TRACK',
            'stopped': 'WAIT_TL',
            'passing': 'PASS',
        }.get(self.auto_state, self.auto_state)

        lines = [
            f"Mode: {mode_str}" + (f" [{auto_state_str}]" if self.auto_mode else ""),
            f"Status: {self.status.upper():8s}  Speed: {self.speed:3d}  FPS: {fps:.1f}",
            f"TL: {tl_str:7s}  Dist: {dist} cm",
            f"G: {gs[0]:.0f} {gs[1]:.0f} {gs[2]:.0f}  Pan:{pa} Tilt:{ta} Dir:{da}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def video_loop(self):
        self.connect_control()

        print(f"[视频] 连接小车摄像头: {self.video_url}")
        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print("错误: 无法打开视频流")
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except Exception:
            pass

        frame_count = 0
        fps_timer = time.time()
        fps = 0.0

        print("[视频] 视频流已连接，开始 YOLO 推理...")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[视频] 流中断，正在重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.video_url)
                continue

            # ── 常规 YOLO 检测 (全类别) ──
            annotated = self.detector.detect(frame)

            # ── 自动循迹控制 (每 2 帧执行一次) ──
            if self.auto_mode and frame_count % 2 == 0:
                self._auto_control(annotated)

            # ── 红绿灯检测 (始终运行，用于显示) ──
            if frame_count % 3 == 0:
                self._detect_traffic_light(annotated)

            self.draw_hud(annotated, fps)

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_timer
                fps = 15.0 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._frame_lock:
                self.latest_frame = jpeg.tobytes()

        cap.release()
        self.cleanup()

    def get_frame(self):
        with self._frame_lock:
            return self.latest_frame

    def cleanup(self):
        self.running = False
        self.send_command('stop')
        if self.sock:
            self.sock.close()
            self.sock = None
        print("[清理] 资源已释放")


# ---------------------------------------------------------------------------
# Flask 路由
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/mjpg')
def mjpg():
    def generate():
        while True:
            frame = controller.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    key = data.get('key', '').lower()
    if key and controller:
        controller.process_key(key)
    return {
        'status': controller.status if controller else 'unknown',
        'speed': controller.speed if controller else 0,
        'auto_mode': controller.auto_mode if controller else False,
        'tl_color': controller._tl_color if controller else 'unknown',
        'tl_conf': controller._tl_conf if controller else 0.0,
    }


@app.route('/status')
def get_status():
    if not controller:
        return {'distance': -1, 'gs': [0, 0, 0], 'pan': 0, 'tilt': 0, 'dir': 0,
                'status': 'unknown', 'speed': 0, 'auto_mode': False,
                'tl_color': 'unknown', 'tl_conf': 0.0}
    with controller._sensor_lock:
        distance = controller.ultrasonic_distance
        gs = controller.grayscale_values[:]
        pan = controller.pan_angle
        tilt = controller.tilt_angle
        dir_ = controller.dir_angle
    return {
        'distance': distance,
        'gs': gs,
        'pan': pan,
        'tilt': tilt,
        'dir': dir_,
        'status': controller.status,
        'speed': controller.speed,
        'auto_mode': controller.auto_mode,
        'tl_color': controller._tl_color,
        'tl_conf': controller._tl_conf,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    global controller

    parser = argparse.ArgumentParser(
        description='Picar-X 本机远程控制 (手动/自动循迹 + 红绿灯)')
    parser.add_argument('--host', type=str, default='10.129.137.156',
                        help='小车(树莓派) IP 地址')
    parser.add_argument('--control-port', type=int, default=8888,
                        help='小车 TCP 控制端口')
    parser.add_argument('--video-port', type=int, default=9000,
                        help='小车摄像头视频流端口')
    parser.add_argument('--serve-port', type=int, default=9001,
                        help='本机 Web 面板端口')
    parser.add_argument('--model', type=str, default='../环境准备/yolo26n.pt',
                        help='YOLO .pt 模型文件路径')
    parser.add_argument('--gs-threshold', type=int, default=800,
                        help='灰度传感器阈值 (低于此值=在线)，默认800')
    args = parser.parse_args()

    print(f"[YOLO] 加载模型: {args.model}")
    detector = UltralyticsDetector(model_path=args.model)
    print("[YOLO] 模型加载完成")

    controller = RemoteController(
        picar_host=args.host,
        control_port=args.control_port,
        video_port=args.video_port,
        detector=detector,
    )
    controller.gs_threshold = args.gs_threshold

    video_thread = threading.Thread(target=controller.video_loop, daemon=True)
    video_thread.start()

    print(f"\n{'='*50}")
    print(f"  Web 面板:     http://localhost:{args.serve_port}/")
    print(f"  目标小车:     {args.host}:{args.control_port}")
    print(f"  灰度阈值:     {args.gs_threshold}")
    print(f"{'='*50}\n")
    print("  [M] 切换手动/自动循迹")
    print("  自动模式: 循迹 → 停止线 → 红绿灯判断 → 通过 → 恢复\n")

    try:
        app.run(host='0.0.0.0', port=args.serve_port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print('\n[退出] 收到中断信号')
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()
