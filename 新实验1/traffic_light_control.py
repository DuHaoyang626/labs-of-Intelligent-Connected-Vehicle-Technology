#!/usr/bin/env python3
"""
红绿灯识别与自动控制程序 — 本机 GPU/CPU YOLO + HSV 颜色分析

基于 YOLO 检测红绿灯位置，通过 HSV 颜色空间二值化判断灯色，
自动控制小车：红灯停、绿灯行(50)、黄灯行(10)+鸣笛。

只识别红绿灯 (COCO class 9)，取消其他物品识别。

架构:
  浏览器 ──HTTP──> 本机 (YOLO+HSV) ──TCP──> 小车 (树莓派)
  浏览器 <──MJPG── 本机 (标注视频)  <──TCP── 小车 (传感器)

用法:
    source ../本机远程控制/venv/Scripts/activate
    python traffic_light_control.py --host <小车IP>

依赖:
    ultralytics opencv-python numpy flask  (见 ../本机远程控制/requirements.txt)
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
# HSV 颜色范围 — 针对红绿灯发光特征调优
# ---------------------------------------------------------------------------

# 红色（HSV 中红色分布在 0° 两侧）
RED_LOWER1 = (0, 60, 120)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (165, 60, 120)
RED_UPPER2 = (180, 255, 255)

# 黄色
YELLOW_LOWER = (15, 60, 120)
YELLOW_UPPER = (35, 255, 255)

# 绿色
GREEN_LOWER = (40, 60, 80)
GREEN_UPPER = (85, 255, 255)

# 当 roi 中某颜色像素数低于此值，认为该灯未亮起
MIN_PIXEL_THRESHOLD = 15


# ---------------------------------------------------------------------------
# 红绿灯检测器 — YOLO 定位 + HSV 颜色分析
# ---------------------------------------------------------------------------

class TrafficLightDetector:
    """YOLO 检测红绿灯(仅 class 9) + HSV 二值化判断颜色状态."""

    def __init__(self, model_path, conf_threshold=0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # 线程安全的状态共享
        self._lock = threading.Lock()
        self._color = 'unknown'       # 'red' | 'yellow' | 'green' | 'unknown'
        self._confidence = 0.0
        self._pixel_counts = {'red': 0, 'yellow': 0, 'green': 0}

    # -- 公开接口 ------------------------------------------------------------

    def get_state(self):
        """返回 (color, confidence)."""
        with self._lock:
            return self._color, self._confidence

    def get_pixel_counts(self):
        """调试用: 返回各颜色像素计数."""
        with self._lock:
            return dict(self._pixel_counts)

    # -- YOLO 检测 + HSV 分析 ------------------------------------------------

    def detect(self, frame):
        """对帧执行检测，返回标注后的帧."""
        results = self.model(
            frame,
            classes=[9],              # COCO class 9 = traffic light
            conf=self.conf_threshold,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        # 默认状态
        color = 'unknown'
        conf = 0.0
        bbox = None
        counts = {'red': 0, 'yellow': 0, 'green': 0}

        if boxes is not None and len(boxes) > 0:
            # -- 取置信度最高的红绿灯 --
            best_idx = int(boxes.conf.argmax())
            conf = float(boxes.conf[best_idx])
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
            bbox = (x1, y1, x2, y2)

            # 截取 ROI 并做颜色分析
            roi = frame[y1:y2, x1:x2]
            color, counts = self._analyze_roi(roi)

            # -- 画面绘制 --
            self._draw_detection(frame, bbox, color, conf, counts)

        # -- 更新共享状态 --
        with self._lock:
            self._color = color
            self._confidence = conf
            self._pixel_counts = counts

        # 顶部大状态条
        self._draw_status_bar(frame, color)

        return frame

    # -- 颜色分析核心 --------------------------------------------------------

    @staticmethod
    def _analyze_roi(roi):
        """对红绿灯 ROI 做 HSV 二值化，返回 dominant color 和像素计数."""
        if roi.size == 0:
            return 'unknown', {'red': 0, 'yellow': 0, 'green': 0}

        h, w = roi.shape[:2]

        # 缩小 ROI 提高速度
        if max(h, w) > 100:
            scale = 100.0 / max(h, w)
            roi = cv2.resize(roi, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # --- 三色掩码 ---
        # 红色（两区间合并）
        mask_r1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask_r2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        # 黄色
        mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

        # 绿色
        mask_green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

        # 形态学开运算 — 去除孤立噪点
        kernel = np.ones((3, 3), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

        r_px = cv2.countNonZero(mask_red)
        y_px = cv2.countNonZero(mask_yellow)
        g_px = cv2.countNonZero(mask_green)

        counts = {'red': r_px, 'yellow': y_px, 'green': g_px}

        # 取像素数最多的颜色
        dominant = max(counts, key=counts.get)
        max_val = counts[dominant]

        # 低于阈值 = 无灯亮起
        if max_val < MIN_PIXEL_THRESHOLD:
            return 'unknown', counts

        return dominant, counts

    # -- 画面绘制辅助 --------------------------------------------------------

    @staticmethod
    def _draw_detection(frame, bbox, color, conf, counts):
        """在检测框周围绘制颜色标签和像素计数."""
        x1, y1, x2, y2 = bbox

        # 根据颜色选边框色
        color_bgr = {
            'red':    (0, 0, 255),
            'yellow': (0, 255, 255),
            'green':  (0, 255, 0),
            'unknown': (128, 128, 128),
        }.get(color, (128, 128, 128))

        label_map = {
            'red':    f'RED ({conf:.2f})',
            'yellow': f'YELLOW ({conf:.2f})',
            'green':  f'GREEN ({conf:.2f})',
            'unknown': f'UNKNOWN ({conf:.2f})',
        }
        label = label_map.get(color, f'UNKNOWN ({conf:.2f})')

        # 框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)

        # 标签背景 + 文字
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 10, y1), color_bgr, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # 像素计数（框下方）
        cnt_text = f"R:{counts['red']} Y:{counts['yellow']} G:{counts['green']}"
        cv2.putText(frame, cnt_text, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)

    @staticmethod
    def _draw_status_bar(frame, color):
        """画面顶部状态条 — 大号醒目."""
        h, w = frame.shape[:2]
        bar_h = 40

        config = {
            'red':    ((0, 0, 200), 'RED — STOP', (255, 255, 255)),
            'yellow': ((0, 200, 200), 'YELLOW — SLOW + HORN', (0, 0, 0)),
            'green':  ((0, 200, 0), 'GREEN — GO (50)', (0, 0, 0)),
            'unknown':((60, 60, 60), 'NO LIGHT — STOP', (255, 255, 255)),
        }
        bg, text, tc = config.get(color, config['unknown'])

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), bg, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tx = (w - tw) // 2
        ty = (bar_h + th) // 2
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, tc, 2)


# ---------------------------------------------------------------------------
# Web 控制面板
# ---------------------------------------------------------------------------

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Picar-X 红绿灯自动控制</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#ccc;font-family:monospace;
     display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:0 8px}
h1{margin:10px 0 4px;font-size:1.1em;color:#0f0;letter-spacing:2px}
#video{max-width:100%;max-height:50vh;border:2px solid #222;border-radius:4px}
.panel{width:100%;max-width:640px;margin:4px 0}
.panel .row{display:flex;justify-content:space-between;padding:3px 6px;font-size:0.9em}
.panel .row:nth-child(odd){background:#141414}
.panel .lbl{color:#888}
.panel .val{color:#0f0;font-weight:bold}
.tl-box{padding:2px 8px;border-radius:4px;font-weight:bold}
.tl-box.red{background:#400;color:#f44}
.tl-box.yellow{background:#440;color:#ff4}
.tl-box.green{background:#040;color:#4f4}
.tl-box.unknown{background:#222;color:#888}
.grid{display:grid;grid-template-columns:60px 60px 60px;gap:6px;margin:6px 0}
.grid button,.row-btn button{padding:12px 0;font-size:1.2em;background:#1a1a1a;color:#0f0;
       border:1px solid #444;border-radius:6px;cursor:pointer;
       -webkit-tap-highlight-color:transparent;user-select:none}
.grid button:active,.row-btn button:active{background:#0a0;color:#000;border-color:#0f0}
.row-btn{display:flex;gap:6px;margin:4px 0;flex-wrap:wrap;justify-content:center}
.row-btn button{padding:10px 14px;font-size:0.9em}
.row-btn .warn{color:#fa0}
.hint{color:#555;font-size:0.7em;margin:8px 0;text-align:center;line-height:1.6}
.hint kbd{background:#222;color:#0f0;padding:1px 5px;border-radius:3px;border:1px solid #444}
</style>
</head>
<body>
<h1>PICAR-X 红绿灯自动控制</h1>
<img id="video" src="/mjpg" alt="视频流">

<div class="panel">
  <div class="row"><span class="lbl">红绿灯</span><span class="val" id="tl">--</span></div>
  <div class="row"><span class="lbl">置信度</span><span class="val" id="tlconf">--</span></div>
  <div class="row"><span class="lbl">小车状态</span><span class="val" id="st">stop</span></div>
  <div class="row"><span class="lbl">速度</span><span class="val" id="sp">0</span></div>
  <div class="row"><span class="lbl">距离</span><span class="val" id="dist">-- cm</span></div>
  <div class="row"><span class="lbl">灰度</span><span class="val" id="gs">-- / -- / --</span></div>
</div>

<div class="hint">
自动模式 &nbsp;|&nbsp;
<kbd>Q</kbd> 切模式 &nbsp; <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> 手动 &nbsp; <kbd>F</kbd> 停止
</div>

<script>
let autoMode = true;

function send(k) {
  fetch('/control', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({key:k, auto:autoMode})})
  .then(r=>r.json()).then(d=>{
    document.getElementById('st').textContent = d.status;
    document.getElementById('sp').textContent = d.speed;
  });
}

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'q') {
    autoMode = !autoMode;
    fetch('/control', {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({key:'q', auto:autoMode})});
    return;
  }
  if ('wasdf'.includes(k)) { e.preventDefault(); send(k); }
});

function pollStatus() {
  fetch('/status').then(r=>r.json()).then(d=>{
    const tlEl = document.getElementById('tl');
    const color = d.tl_color;
    tlEl.textContent = color === 'red' ? '🔴 红灯' :
                       color === 'yellow' ? '🟡 黄灯' :
                       color === 'green' ? '🟢 绿灯' : '⚫ 无灯';
    tlEl.className = 'val tl-box ' + color;
    document.getElementById('tlconf').textContent = d.tl_conf.toFixed(2);
    document.getElementById('st').textContent = d.status + (d.auto ? ' (自动)' : ' (手动)');
    document.getElementById('sp').textContent = d.speed;
    document.getElementById('dist').textContent = d.distance + ' cm';
    document.getElementById('gs').textContent = d.gs.join(' / ');
  });
  setTimeout(pollStatus, 300);
}
pollStatus();
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------

class TrafficLightController:
    """红绿灯自动/手动控制器."""

    def __init__(self, picar_host, control_port, video_port, detector):
        self.detector = detector
        self.picar_host = picar_host
        self.control_port = control_port
        self.video_url = f"http://{picar_host}:{video_port}/mjpg"

        self.speed = 0
        self.status = 'stop'
        self.running = True
        self.sock = None
        self.auto_mode = True

        # 传感器数据
        self._sensor_lock = threading.Lock()
        self.ultrasonic_distance = -1
        self.grayscale_values = [0, 0, 0]
        self.pan_angle = 0
        self.tilt_angle = 0
        self.dir_angle = 0

        # 视频帧
        self.latest_frame = None
        self._frame_lock = threading.Lock()

        # 红绿灯状态跟踪（避免重复打印/鸣笛）
        self._prev_color = 'unknown'

    # -- TCP 连接与通信 -------------------------------------------------------

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

    # -- 手动控制（键盘） ----------------------------------------------------

    def _send_state(self):
        """发送完整状态包: state <direction> <speed> <pan> <tilt> <dir_angle>"""
        self.send_command(
            f'state {self.status} {self.speed} '
            f'{self.pan_angle} {self.tilt_angle} {self.dir_angle}'
        )

    def process_key(self, key_char):
        if key_char == 'q':
            return  # Q 只用于切换模式，不执行动作
        k = key_char.lower()
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
            self.dir_angle = -30
        elif k == 'd':
            if self.speed == 0:
                self.speed = 50
            self.status = 'forward'
            self.dir_angle = 30
        self._send_state()

    # -- 红绿灯自动控制核心逻辑 ----------------------------------------------

    def _auto_control(self):
        """根据红绿灯颜色执行一次控制决策."""
        color, conf = self.detector.get_state()

        if color == 'red':
            if self._prev_color != 'red':
                print(f"[红绿灯] 🔴 红灯 → STOP")
            self._ensure_action('stop', 0)

        elif color == 'green':
            if self._prev_color != 'green':
                print(f"[红绿灯] 🟢 绿灯 → 速度 50")
            self._ensure_action('forward', 50)

        elif color == 'yellow':
            if self._prev_color != 'yellow':
                print(f"[红绿灯] 🟡 黄灯 → 速度 10 + HORN")
                self.send_command('horn')   # 仅状态切换时鸣笛
            self._ensure_action('forward', 10)

        else:  # unknown
            if self._prev_color != 'unknown':
                print(f"[红绿灯] ⚫ 未识别 → STOP")
            self._ensure_action('stop', 0)

        self._prev_color = color

    def _ensure_action(self, action, speed):
        """避免重复发送相同指令. 通过完整状态包发送, dir_angle=0 保持直行."""
        if self.status != action or self.speed != speed:
            self.status = action
            self.speed = speed
            self.dir_angle = 0  # 自动驾驶直行
            self._send_state()

    # -- 视频流水线 -----------------------------------------------------------

    def draw_hud(self, frame, fps):
        with self._sensor_lock:
            dist = self.ultrasonic_distance
            gs = self.grayscale_values

        mode_str = 'AUTO' if self.auto_mode else 'MANUAL'
        lines = [
            f"Mode: {mode_str}  |  {self.status.upper():8s}  Speed: {self.speed}",
            f"Dist: {dist} cm  |  G: {gs[0]:.0f} / {gs[1]:.0f} / {gs[2]:.0f}  |  FPS: {fps:.1f}",
        ]
        h_img = frame.shape[0]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, h_img - 40 + i * 22),
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
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except Exception:
            pass

        frame_count = 0
        fps_timer = time.time()
        fps = 0.0

        print("[视频] 已连接，开始红绿灯检测与自动控制...")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[视频] 流中断，准备重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.video_url)
                continue

            # YOLO 检测 + HSV 颜色分析
            annotated = self.detector.detect(frame)
            self.draw_hud(annotated, fps)

            # 自动控制 — 每 3 帧执行一次
            if self.auto_mode and frame_count % 3 == 0:
                self._auto_control()

            # FPS 统计
            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_timer
                fps = 15.0 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            # JPEG 编码
            _, jpeg = cv2.imencode('.jpg', annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
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
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    key = data.get('key', '').lower()
    controller.auto_mode = data.get('auto', True)
    if key:
        controller.process_key(key)
    return {
        'status': controller.status,
        'speed': controller.speed,
        'auto': controller.auto_mode,
    }


@app.route('/status')
def get_status():
    with controller._sensor_lock:
        distance = controller.ultrasonic_distance
        gs = controller.grayscale_values[:]
    tl_color, tl_conf = controller.detector.get_state()
    return {
        'distance': distance,
        'gs': gs,
        'status': controller.status,
        'speed': controller.speed,
        'auto': controller.auto_mode,
        'tl_color': tl_color,
        'tl_conf': tl_conf,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    global controller

    parser = argparse.ArgumentParser(
        description='Picar-X 红绿灯自动控制 (YOLO + HSV)')
    parser.add_argument('--host', type=str, default='10.129.137.156',
                        help='小车(树莓派) IP 地址')
    parser.add_argument('--control-port', type=int, default=8888)
    parser.add_argument('--video-port', type=int, default=9000)
    parser.add_argument('--serve-port', type=int, default=9002,
                        help='本机 Web 面板端口')
    parser.add_argument('--model', type=str,
                        default='../环境准备/yolo26n.pt',
                        help='YOLO .pt 模型文件路径')
    parser.add_argument('--confidence', type=float, default=0.25,
                        help='YOLO 置信度阈值')
    args = parser.parse_args()

    print(f"[YOLO] 加载模型: {args.model}")
    detector = TrafficLightDetector(
        model_path=args.model,
        conf_threshold=args.confidence,
    )
    print("[YOLO] 模型加载完成")

    controller = TrafficLightController(
        picar_host=args.host,
        control_port=args.control_port,
        video_port=args.video_port,
        detector=detector,
    )

    video_thread = threading.Thread(target=controller.video_loop, daemon=True)
    video_thread.start()

    print(f"\n{'='*50}")
    print(f"  Web 面板:      http://localhost:{args.serve_port}/")
    print(f"  视频流:        http://localhost:{args.serve_port}/mjpg")
    print(f"  目标小车:      {args.host}:{args.control_port}")
    print(f"{'='*50}\n")
    print("  红绿灯自动控制已启动")
    print("  🔴 红灯      → 停车")
    print("  🟡 黄灯      → 速度 10 + 鸣笛")
    print("  🟢 绿灯      → 速度 50 通过")
    print("  ⚫ 未识别    → 停车\n")
    print("  [Q] 切换自动/手动模式")
    print("  [WASD] 手动方向  [F] 手动停止\n")

    try:
        app.run(host='0.0.0.0', port=args.serve_port,
                threaded=True, debug=False)
    except KeyboardInterrupt:
        print('\n[退出] 收到中断信号')
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()
