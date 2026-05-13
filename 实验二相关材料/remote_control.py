#!/usr/bin/env python3
"""
Remote control relay for Picar-X with NPU-accelerated YOLO detection.

Architecture:
  browser ──HTTP──> aipro (Orange Pi AIpro, Ascend NPU) ──TCP──> picar (Pi)
  browser <──MJPG── aipro (YOLO on NPU)                 <──TCP── picar (sensor)

Uses Ascend NPU (ais_bench / InferSession) for YOLOv10 inference.
Falls back to ultralytics CPU inference when --npu is not specified.

Usage (NPU mode):
    python remote_control.py --host 10.129.45.139 --npu --om-model yolov10n.om

Usage (CPU/GPU mode, fallback):
    python remote_control.py --host 10.129.45.139
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
# COCO 80 class names (for NPU mode, since we bypass ultralytics)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Detector abstraction — NPU mode or ultralytics fallback
# ---------------------------------------------------------------------------

class Detector:
    """Interface: returns annotated frame given a raw BGR frame."""
    def detect(self, frame):
        raise NotImplementedError


class UltralyticsDetector(Detector):
    """CPU/GPU fallback: ultralytics YOLO."""
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0].plot()


class AscendNPUDetector(Detector):
    """Ascend NPU detection matching the official ais_bench tutorial.

    - Preprocess: square canvas → blobFromImage(1/255, 640, swapRB)
    - Infer: InferSession.infer(feeds=blob, mode="static")
    - Postprocess: transpose → loop over rows → minMaxLoc → NMS → draw on frame

    Thread-safe: InferSession created lazily on first detect() call so that the
    ACL context is bound to the inference thread.
    """

    def __init__(self, om_path, conf_threshold=0.25, nms_threshold=0.45):
        self.om_path = om_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._session = None
        self._init_lock = threading.Lock()

    # -- Thread-safe lazy init -----------------------------------------------

    def _ensure_session(self):
        if self._session is not None:
            return
        with self._init_lock:
            if self._session is not None:
                return
            from ais_bench.infer.interface import InferSession
            self._session = InferSession(device_id=0, model_path=self.om_path)
            print(f"[NPU] InferSession ready (thread: {threading.current_thread().name})")

    # -- Preprocess ----------------------------------------------------------

    @staticmethod
    def _preprocess(frame):
        """Square canvas → blobFromImage → NCHW blob.

        Returns:
            blob (np.ndarray): shape (1, 3, 640, 640)
            scale (float):     length / 640, to map coords back to original
            orig (tuple):      (height, width)
        """
        h, w = frame.shape[:2]
        length = max(h, w)
        canvas = np.zeros((length, length, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        scale = length / 640.0
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255, (640, 640), swapRB=True)
        return blob, scale, (h, w)

    # -- Postprocess ---------------------------------------------------------

    def _postprocess(self, outputs, scale, orig_shape, draw_target):
        """Identical post-processing to the official ais_bench tutorial."""
        # InferSession returns a list of output tensors.
        # YOLO OM output shape: [1, 84, 8400] (NCHW).
        # Transpose → [1, 8400, 84] so dim-1 is detections and dim-2 is (cx,cy,w,h,cls...).
        preds = np.array([cv2.transpose(outputs[0][0])])
        rows = preds.shape[1]  # 8400

        boxes, scores, class_ids = [], [], []

        for i in range(rows):
            classes_scores = preds[0][i][4:]
            # Find best class & score (exactly like the tutorial)
            minScore, maxScore, minClassLoc, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
            if maxScore < self.conf_threshold:
                continue

            # Box in [cx - w/2, cy - h/2, w, h] format (same as tutorial)
            box = [
                preds[0][i][0] - 0.5 * preds[0][i][2],
                preds[0][i][1] - 0.5 * preds[0][i][3],
                preds[0][i][2],
                preds[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(int(maxClassIndex))

        if not boxes:
            return

        # NMS (5-param call, same as tutorial)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold, 0.5)

        # Draw on the target image directly (same as tutorial)
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

    # -- Public interface ----------------------------------------------------

    def detect(self, frame):
        """NPU inference → post-process → draw boxes directly on frame."""
        self._ensure_session()
        blob, scale, orig_shape = self._preprocess(frame)
        outputs = self._session.infer(feeds=blob, mode="static")
        self._postprocess(outputs, scale, orig_shape, draw_target=frame)
        return frame


# ---------------------------------------------------------------------------
# Web control panel
# ---------------------------------------------------------------------------

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Picar-X YOLO Remote (NPU)</title>
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
<h1>PICAR-X YOLO REMOTE</h1>
<img id="video" src="/mjpg" alt="annotated stream">

<div class="panel">
  <div class="row"><span class="lbl">Status</span><span class="val" id="st">stop</span></div>
  <div class="row"><span class="lbl">Speed</span><span class="val" id="sp">0</span></div>
  <div class="row"><span class="lbl">Ultrasonic</span><span class="val" id="dist">-- cm</span></div>
  <div class="row"><span class="lbl">Grayscale</span><span class="val" id="gs">-- / -- / --</span></div>
  <div class="row"><span class="lbl">Servo Pan</span><span class="val" id="pan">0&deg;</span></div>
  <div class="row"><span class="lbl">Servo Tilt</span><span class="val" id="tilt">0&deg;</span></div>
  <div class="row"><span class="lbl">Servo Dir</span><span class="val" id="dir">0&deg;</span></div>
</div>

<div class="grid">
  <div></div><button id="bw">W</button><div></div>
  <button id="ba">A</button><button id="bf">F</button><button id="bd">D</button>
  <div></div><button id="bs">S</button><div></div>
</div>

<div class="row-btn">
  <button id="bo">O +spd</button>
  <button id="bp">P -spd</button>
  <button id="bt">T photo</button>
  <button class="warn" id="bh">H horn</button>
</div>

<div class="row-btn">
  <button id="bi">I tilt up</button>
  <button id="bk">K tilt dn</button>
  <button id="bj">J pan &larr;</button>
  <button id="bl">L pan &rarr;</button>
  <button id="br">R reset</button>
</div>

<div class="hint">
<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move &nbsp;
<kbd>F</kbd> stop &nbsp;
<kbd>O</kbd><kbd>P</kbd> speed &nbsp;
<kbd>H</kbd> horn &nbsp;
<kbd>T</kbd> photo &nbsp;
<kbd>I</kbd><kbd>K</kbd><kbd>J</kbd><kbd>L</kbd> camera &nbsp;
<kbd>R</kbd> cam reset
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
# Remote controller
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

    # -- TCP control ---------------------------------------------------------

    def connect_control(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.picar_host, self.control_port))
            print(f"Control connected to {self.picar_host}:{self.control_port}")
            threading.Thread(target=self._sensor_listener, daemon=True).start()
            return True
        except Exception as e:
            print(f"Control connection failed: {e}")
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
            print(f"Send failed, reconnecting: {e}")
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

    # -- Video pipeline ------------------------------------------------------

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

        print(f"Connecting to picar video stream: {self.video_url}")
        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print("Failed to open video stream")
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except Exception:
            pass

        frame_count = 0
        fps_timer = time.time()
        fps = 0.0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Stream lost, reconnecting...")
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
        print("Cleanup done.")


# ---------------------------------------------------------------------------
# Flask routes
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
# Entry point
# ---------------------------------------------------------------------------

def build_detector(args):
    """Return a Detector instance based on CLI args."""
    if args.npu:
        print(f"[NPU mode] Loading OM model: {args.om_model}")
        detector = AscendNPUDetector(
            om_path=args.om_model,
            conf_threshold=args.confidence,
            nms_threshold=args.nms,
        )
        print("[NPU mode] InferSession ready.")
        return detector
    else:
        print(f"[CPU mode] Loading YOLO model: {args.model}")
        detector = UltralyticsDetector(model_path=args.model)
        print("[CPU mode] YOLO model ready.")
        return detector


def main():
    global controller

    parser = argparse.ArgumentParser(description='Picar-X Remote Control with YOLO on NPU')
    parser.add_argument('--host', type=str, default='10.129.45.139',
                        help='Picar-X IP address (default: 10.129.45.139)')
    parser.add_argument('--control-port', type=int, default=8888)
    parser.add_argument('--video-port', type=int, default=9000)
    parser.add_argument('--serve-port', type=int, default=9001,
                        help='Port for web interface and annotated MJPG stream')

    # Model selection
    parser.add_argument('--npu', action='store_true',
                        help='Use Ascend NPU inference (requires --om-model)')
    parser.add_argument('--om-model', type=str, default='yolov10n.om',
                        help='Path to OM model for NPU inference')
    parser.add_argument('--model', type=str, default='yolov10n.pt',
                        help='Ultralytics model path (fallback when --npu is off)')
    parser.add_argument('--confidence', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--nms', type=float, default=0.45,
                        help='NMS IoU threshold (default: 0.45)')
    args = parser.parse_args()

    print("Building detector ...")
    detector = build_detector(args)

    controller = RemoteController(
        picar_host=args.host,
        control_port=args.control_port,
        video_port=args.video_port,
        detector=detector,
    )

    video_thread = threading.Thread(target=controller.video_loop, daemon=True)
    video_thread.start()

    print(f"\n{'='*50}")
    print(f"Web panel:     http://<this_ip>:{args.serve_port}/")
    print(f"Annotated MJPG: http://<this_ip>:{args.serve_port}/mjpg")
    print(f"Target picar:  {args.host}:{args.control_port}")
    mode = "Ascend NPU" if args.npu else "CPU (ultralytics)"
    print(f"Inference:     {mode}")
    print(f"{'='*50}\n")

    try:
        app.run(host='0.0.0.0', port=args.serve_port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print('\nInterrupted')
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()
