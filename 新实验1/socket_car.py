#!/usr/bin/env python3
"""
Picar-X socket control demo.
Receives control commands via TCP, controls the car,
streams camera video via HTTP (port 9000),
and sends sensor data back to the client.
"""

from robot_hat.utils import reset_mcu
from robot_hat import Music
from picarx import Picarx
from vilib import Vilib
from time import sleep, time, strftime, localtime
import socket
import threading
import os

user = os.getlogin()
user_home = os.path.expanduser(f'~{user}')

reset_mcu()
sleep(0.2)

px = Picarx()
music = Music()

speed = 0
status = 'stop'
running = True

cam_pan_angle = 0
cam_tilt_angle = 0

SOUND_HORN = f"{user_home}/picar-x/sounds/car-double-horn.wav"


def take_photo():
    _time = strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))
    name = 'photo_%s' % _time
    path = f"{user_home}/Pictures/picar-x/"
    Vilib.take_photo(name, path)
    print('\nphoto save as %s%s.jpg' % (path, name))


def handle_command(cmd):
    """Parse and execute a command. Returns True to continue, False to stop.

    协议格式:
      state <direction> <speed> <pan> <tilt> <dir_angle>  — 完整状态包
         direction: stop | forward | backward
         speed:     0~100
         pan:       -90~90
         tilt:      -35~65
         dir_angle: -30~30 (方向舵机角度, 0=直行)
      horn                                                   — 鸣笛
      photo                                                  — 拍照
      quit                                                   — 退出
    """
    global speed, status, cam_pan_angle, cam_tilt_angle

    cmd = cmd.strip().lower()
    if not cmd or cmd.startswith('@'):
        return True

    parts = cmd.split()
    action = parts[0]

    # ── 状态包：包含速度、方向、所有舵机角度 ──
    if action == 'state':
        if len(parts) < 6:
            return True
        direction = parts[1]
        spd = int(parts[2])
        pan = int(parts[3])
        tilt = int(parts[4])
        dir_angle = int(parts[5])

        # 限幅
        spd = max(0, min(100, spd))
        pan = max(-90, min(90, pan))
        tilt = max(-35, min(65, tilt))
        dir_angle = max(-30, min(30, dir_angle))

        # 应用舵机角度
        px.set_cam_pan_angle(pan)
        px.set_cam_tilt_angle(tilt)
        px.set_dir_servo_angle(dir_angle)
        cam_pan_angle = pan
        cam_tilt_angle = tilt

        # 应用运动
        if direction == 'stop' or spd == 0:
            px.stop()
            status = 'stop'
            speed = 0
        elif direction == 'forward':
            px.forward(spd)
            status = 'forward'
            speed = spd
        elif direction == 'backward':
            px.backward(spd)
            status = 'backward'
            speed = spd

    elif action == 'horn':
        music.sound_play_threading(SOUND_HORN)

    elif action == 'photo':
        take_photo()

    elif action == 'quit':
        return False

    return True


def read_sensors():
    """Read all sensors and return a formatted string."""
    try:
        dist = round(px.get_distance(), 1)
    except Exception:
        dist = -1
    try:
        gs = px.get_grayscale_data()
        gs_str = f"{gs[0]:.0f},{gs[1]:.0f},{gs[2]:.0f}"
    except Exception:
        gs_str = "0,0,0"

    return f"@sensor:{dist}|{gs_str}|{cam_pan_angle}|{cam_tilt_angle}|{px.dir_current_angle}\n"


def handle_client(conn, addr):
    """Handle one TCP client connection with bidirectional communication."""
    global running
    print(f"Client connected: {addr}")
    conn.settimeout(0.2)
    buffer = ''
    last_sensor_time = 0

    while running:
        try:
            data = conn.recv(1024)
            if not data:
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if not handle_command(line):
                    running = False
                    break
        except socket.timeout:
            pass
        except Exception as e:
            if running:
                print(f"recv error: {e}")
            break

        now = time()
        if now - last_sensor_time >= 0.2:
            try:
                conn.sendall(read_sensors().encode('utf-8'))
            except Exception:
                break
            last_sensor_time = now

    conn.close()
    print(f"Client disconnected: {addr}")


def tcp_server(host='0.0.0.0', port=8888):
    """TCP server that accepts control connections."""
    global running

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(1.0)

    print(f"TCP control server listening on {host}:{port}")

    while running:
        try:
            conn, addr = server.accept()
            handle_client(conn, addr)
        except socket.timeout:
            continue
        except Exception as e:
            if running:
                print(f"Server error: {e}")
                sleep(1)

    server.close()
    print("TCP server stopped")


def main():
    global running

    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=True, web=True)
    sleep(2)

    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n{'='*50}")
    print(f"Picar-X Socket Control Demo")
    print(f"Video stream: http://{local_ip}:9000/mjpg")
    print(f"Control port: {local_ip}:8888")
    print(f"{'='*50}\n")

    server_thread = threading.Thread(target=tcp_server, args=('0.0.0.0', 8888), daemon=True)
    server_thread.start()

    try:
        while running:
            sleep(0.1)
    except KeyboardInterrupt:
        print('\nquit ...')
        running = False
    finally:
        px.stop()
        Vilib.camera_close()
        print("cleanup done")


if __name__ == "__main__":
    main()
