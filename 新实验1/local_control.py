#!/usr/bin/env python3
"""
本机远程控制 Picar-X — 使用本机 GPU/CPU 运行 YOLO 检测

基于实验二 remote_control.py 适配，将控制器从香橙派(昇腾NPU)切换到本机(Ultralytics YOLO)。

架构:
  浏览器 ──HTTP──> 本机 (GPU/CPU YOLO) ──TCP──> 小车 (树莓派)
  浏览器 <──MJPG── 本机 (标注后视频)    <──TCP── 小车 (传感器数据)

用法:
    # 激活 venv 后运行
    python local_control.py --host <小车IP地址>

    # 示例（小车 IP 为 192.168.1.100）
    python local_control.py --host 192.168.1.100 --model ../环境准备/yolo26n.pt

依赖:
    flask opencv-python numpy ultralytics  (见 requirements.txt)
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
# Detector — Ultralytics YOLO (GPU/CPU)
# ---------------------------------------------------------------------------

class Detector:
    """接口: 输入原始 BGR 帧，返回标注后的 BGR 帧."""
    def detect(self, frame):
        raise NotImplementedError


class UltralyticsDetector(Detector):
    """本机 GPU/CPU 推理: ultralytics YOLO (PyTorch)."""
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
<title>Picar-X YOLO Remote (本机)</title>
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
<h1>PICAR-X YOLO REMOTE (本机)</h1>
<img id="video" src="/mjpg" alt="标注视频流">

<div class="panel">
  <div class="row"><span class="lbl">状态</span><span class="val" id="st">stop</span></div>
  <div class="row"><span class="lbl">速度</span><span class="val" id="sp">0</span></div>
  <div class="row"><span class="lbl">超声波</span><span class="val" id="dist">-- cm</span></div>
  <div class="row"><span class="lbl">灰度传感器</span><span class="val" id="gs">-- / -- / --</span></div>
  <div class="row"><span class="lbl">云台水平</span><span class="val" id="pan">0&deg;</span></div>
  <div class="row"><span class="lbl">云台垂直</span><span class="val" id="tilt">0&deg;</span></div>
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
<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> 移动 &nbsp;
<kbd>F</kbd> 停止 &nbsp;
<kbd>O</kbd><kbd>P</kbd> 加减速 &nbsp;
<kbd>H</kbd> 鸣笛 &nbsp;
<kbd>T</kbd> 拍照 &nbsp;
<kbd>I</kbd><kbd>K</kbd><kbd>J</kbd><kbd>L</kbd> 云台 &nbsp;
<kbd>R</kbd> 云台复位
</div>

<script>
const keyMap = {
  bw:'w',bs:'s',ba:'a',bd:'d',bf:'f',
  bo:'o',bp:'p',bt:'t',bh:'h',
  bi:'i',bk:'k',bj:'j',bl:'l',br:'r'
};

function send(k) {
  fetch('/control', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({key:k})})
  .then(r=>r.json()).then(d=>{
    document.getElementById('st').textContent = d.status;
    document.getElementById('sp').textContent = d.speed;
  });
}

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if ('wasdfopthijklr'.includes(k)) { e.preventDefault(); send(k); }
});

Object.entries(keyMap).forEach(([id,k]) => {
  const btn = document.getElementById(id);
  const fn = e => { e.preventDefault(); send(k); };
  btn.addEventListener('mousedown', fn);
  btn.addEventListener('touchstart', fn);
});

function pollStatus() {
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('dist').textContent = d.distance + ' cm';
    document.getElementById('gs').textContent = d.gs.join(' / ');
    document.getElementById('pan').textContent  = d.pan + '°';
    document.getElementById('tilt').textContent = d.tilt + '°';
    document.getElementById('dir').textContent  = d.dir + '°';
  });
  setTimeout(pollStatus, 300);
}
pollStatus();
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# 远程控制器
# ---------------------------------------------------------------------------

class RemoteController:
    def __init__(self, picar_host, control_port, video_port, detector):
        self.detector = detector
        self.picar_host = picar_host
        self.control_port = control_port
        self.video_url = f"http://{picar_host}:{video_port}/mjpg"

        self.speed = 0
        self.status = 'stop'
        self.running = True
        self.sock = None

        self.sensor_lock = threading.Lock()
        self.ultrasonic_distance = -1
        self.grayscale_values = [0, 0, 0]
        self.pan_angle = 0
        self.tilt_angle = 0
        self.dir_angle = 0

        self.latest_frame = None
        self.frame_lock = threading.Lock()

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
        with self.sensor_lock:
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

    def process_key(self, key_char):
        k = key_char.lower()

        if k == 'f':
            self.status = 'stop'
            self.send_command('stop')

        elif k == 'w':
            if self.speed == 0:
                self.speed = 10
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

        elif k == 'o':
            if self.speed <= 90:
                self.speed += 10
            if self.status in ('forward', 'backward'):
                self.send_command(f'{self.status} {self.speed}')

        elif k == 'p':
            if self.speed >= 10:
                self.speed -= 10
            if self.speed == 0:
                self.status = 'stop'
                self.send_command('stop')
            elif self.status in ('forward', 'backward'):
                self.send_command(f'{self.status} {self.speed}')

        elif k == 'i':
            with self.sensor_lock:
                a = max(-35, min(65, self.tilt_angle - 5))
                self.tilt_angle = a
            self.send_command(f'tilt {a}')

        elif k == 'k':
            with self.sensor_lock:
                a = max(-35, min(65, self.tilt_angle + 5))
                self.tilt_angle = a
            self.send_command(f'tilt {a}')

        elif k == 'j':
            with self.sensor_lock:
                a = max(-90, min(90, self.pan_angle - 5))
                self.pan_angle = a
            self.send_command(f'pan {a}')

        elif k == 'l':
            with self.sensor_lock:
                a = max(-90, min(90, self.pan_angle + 5))
                self.pan_angle = a
            self.send_command(f'pan {a}')

        elif k == 'r':
            with self.sensor_lock:
                self.pan_angle = 0
                self.tilt_angle = 0
            self.send_command('cam_reset')

        elif k == 'h':
            self.send_command('horn')

        elif k == 't':
            self.send_command('photo')

    # -- 视频流水线 -----------------------------------------------------------

    def draw_hud(self, frame, fps):
        with self.sensor_lock:
            dist = self.ultrasonic_distance
            gs = self.grayscale_values
            pa = self.pan_angle
            ta = self.tilt_angle
            da = self.dir_angle

        lines = [
            f"Status: {self.status}   Speed: {self.speed}",
            f"Ultrasonic: {dist} cm",
            f"Grayscale: {gs[0]:.0f} / {gs[1]:.0f} / {gs[2]:.0f}",
            f"Pan:{pa} Tilt:{ta} Dir:{da}",
            f"FPS: {fps:.1f}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
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

            annotated = self.detector.detect(frame)
            self.draw_hud(annotated, fps)

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_timer
                fps = 15.0 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.frame_lock:
                self.latest_frame = jpeg.tobytes()

        cap.release()
        self.cleanup()

    def get_frame(self):
        with self.frame_lock:
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
    if key:
        controller.process_key(key)
    return {'status': controller.status, 'speed': controller.speed}


@app.route('/status')
def get_status():
    with controller.sensor_lock:
        distance = controller.ultrasonic_distance
        gs = controller.grayscale_values[:]
        pan = controller.pan_angle
        tilt = controller.tilt_angle
        dir_ = controller.dir_angle
    return {'distance': distance, 'gs': gs, 'pan': pan, 'tilt': tilt, 'dir': dir_}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    global controller

    parser = argparse.ArgumentParser(
        description='Picar-X 本机远程控制 (Ultralytics YOLO)')
    parser.add_argument('--host', type=str, default='10.129.137.156',
                        help='小车(树莓派) IP 地址')
    parser.add_argument('--control-port', type=int, default=8888,
                        help='小车 TCP 控制端口 (默认: 8888)')
    parser.add_argument('--video-port', type=int, default=9000,
                        help='小车摄像头视频流端口 (默认: 9000)')
    parser.add_argument('--serve-port', type=int, default=9001,
                        help='本机 Web 面板端口 (默认: 9001)')
    parser.add_argument('--model', type=str, default='../环境准备/yolo26n.pt',
                        help='YOLO .pt 模型文件路径')
    parser.add_argument('--device', type=str, default='',
                        help='推理设备, 如 "cuda:0" / "cpu" (留空自动选择)')
    args = parser.parse_args()

    # 构建检测器
    print(f"[YOLO] 加载模型: {args.model}")
    detector = UltralyticsDetector(model_path=args.model)
    print("[YOLO] 模型加载完成")

    controller = RemoteController(
        picar_host=args.host,
        control_port=args.control_port,
        video_port=args.video_port,
        detector=detector,
    )

    video_thread = threading.Thread(target=controller.video_loop, daemon=True)
    video_thread.start()

    print(f"\n{'='*50}")
    print(f"  Web 面板:     http://localhost:{args.serve_port}/")
    print(f"  控制地址:     本机局域网IP:{args.serve_port}/")
    print(f"  标注视频流:   http://localhost:{args.serve_port}/mjpg")
    print(f"  目标小车:     {args.host}:{args.control_port}")
    print(f"  推理模式:     Ultralytics YOLO (本机 GPU/CPU)")
    print(f"{'='*50}\n")
    print("提示: 浏览器打开 Web 面板后，点击页面或按任意键激活键盘控制")
    print("      键盘: WASD 移动 | F 停止 | O/P 加减速 | H 鸣笛 | I/K/J/L 云台\n")

    try:
        app.run(host='0.0.0.0', port=args.serve_port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print('\n[退出] 收到中断信号')
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()
