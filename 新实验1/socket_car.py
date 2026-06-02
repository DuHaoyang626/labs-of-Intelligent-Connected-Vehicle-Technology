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


def move(operate, spd):
    if operate == 'stop':
        px.stop()
    elif operate == 'forward':
        px.set_dir_servo_angle(0)
        px.forward(spd)
    elif operate == 'backward':
        px.set_dir_servo_angle(0)
        px.backward(spd)
    elif operate == 'turn left':
        px.set_dir_servo_angle(-30)
        px.forward(spd)
    elif operate == 'turn right':
        px.set_dir_servo_angle(30)
        px.forward(spd)


def handle_command(cmd):
    """Parse and execute a command. Returns True to continue, False to stop."""
    global speed, status, cam_pan_angle, cam_tilt_angle

    cmd = cmd.strip().lower()
    if not cmd or cmd.startswith('@'):
        return True

    parts = cmd.split()
    action = parts[0]

    if action == 'forward':
        spd = int(parts[1]) if len(parts) > 1 else speed
        if spd > 60:
            spd = 60
        speed = spd
        status = 'forward'
        move(status, speed)

    elif action == 'backward':
        spd = int(parts[1]) if len(parts) > 1 else speed
        if spd > 60:
            spd = 60
        speed = spd
        status = 'backward'
        move(status, speed)

    elif action in ('left', 'turn_left'):
        status = 'turn left'
        move(status, speed)

    elif action in ('right', 'turn_right'):
        status = 'turn right'
        move(status, speed)

    elif action == 'stop':
        status = 'stop'
        move(status, 0)

    elif action == 'speed':
        spd = int(parts[1])
        speed = max(0, min(100, spd))

    elif action == 'pan':
        angle = int(parts[1])
        cam_pan_angle = max(-90, min(90, angle))
        px.set_cam_pan_angle(cam_pan_angle)

    elif action == 'tilt':
        angle = int(parts[1])
        cam_tilt_angle = max(-35, min(65, angle))
        px.set_cam_tilt_angle(cam_tilt_angle)

    elif action == 'cam_reset':
        cam_pan_angle = 0
        cam_tilt_angle = 0
        px.set_cam_pan_angle(0)
        px.set_cam_tilt_angle(0)

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
