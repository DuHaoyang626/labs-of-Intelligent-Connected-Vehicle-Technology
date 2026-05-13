"""
融合控制：
- 按下 f 后开始循迹（3 路灰度）+ 超声波跟车启停/调速
- 摄像头持续工作：
  - 循环检测 6 种颜色，取面积最大的色块颜色
  - 人脸检测：近距离停车；中距离限速；人脸消失恢复
- 停止线（3 探测器均为黑线）策略：
  - 最大色块为红：停车等待离开停止线
  - 最大色块为绿：直接通过
  - 其他：鸣笛一次，低速通过 3 秒，后恢复速度

依赖 API 参考：
  - 5.minecart_plus.py（循迹）
  - follow.py（超声波跟车速度逻辑）
  - 7.display.py（摄像头、颜色、人脸检测）
  - 3.tts_example.py（喇叭）
"""

import threading
import time
import readchar
from picarx import Picarx
from vilib import Vilib
from robot_hat import Music


# ========================= 可调参数 =========================
# 跟车（超声波）速度逻辑，阈值单位 cm
FAR_DISTANCE = 65      # > FAR_DISTANCE 以最大功率前进
MID_DISTANCE = 15      # FAR..MID 线性减速
STOP_DISTANCE = 10     # MID..STOP 停车；< STOP 根据距离比例倒退

MAX_POWER = 50         # 正常的最大驱动功率（0..100）
LF_OFFSET = 20         # 循迹转向偏置角

# 人脸阈值（基于 7.display.py 中的 human_w/h 尺寸像素）
FACE_NEAR_W = 90
FACE_NEAR_H = 90
FACE_SLOW_W = 40
FACE_SLOW_H = 40
FACE_SLOW_CAP = 18     # 人脸存在且不近时的限速（上限功率）

# 停止线通过策略
PASS_SLOW_SPEED = 15   # 低速通过停止线的功率
PASS_SLOW_SECS = 3.0   # 低速通过持续时长
STOP_LINE_COOLDOWN = 1.0  # 在一次停止线判决后，至少等待该时间再触发新判决

# 摄像头颜色循环检测节奏
COLOR_LIST = ['red','green']
COLOR_SWITCH_INTERVAL = 0.25  # s，每种颜色停留时间

# 蜂鸣器音量与路径（沿用示例相对路径）
HORN_VOLUME = 80
HORN_SOUND_PATH = '../sounds/car-double-horn.wav'


# ========================= 全局状态 =========================
running_flag = False  # f 切换启停

largest_color = None
largest_color_area = 0
largest_color_size = (0, 0)  # (w, h)

face_present = False
face_size = (0, 0)           # (w, h)

stop_line_debounce_until = 0.0
slow_override_until = 0.0


def grayscale_state(px: Picarx):
    """
    将 3 探头灰度数据映射为状态：'forward' | 'left' | 'right' | 'stop'
    同时返回三个探头的状态位列表（[L, M, R]），便于调试打印。
    逻辑与 5.minecart_plus.py 保持一致：
    - px.get_line_status(val_list) 返回 [bool,bool,bool]，1=线，0=底色
    - [1,1,1] => stop（停止线）
    - 中间探头=1 => forward
    - 左探头=1 => right（线在左，向右修正）
    - 右探头=1 => left（线在右，向左修正）
    返回：(gm_state: str, bits: list[int])
    """
    val_list = px.get_grayscale_data()
    bits = px.get_line_status(val_list)
    if bits == [1, 1, 1]:
        return 'stop', bits
    if bits[1] == 1:
        return 'forward', bits
    if bits[0] == 1:
        return 'right', bits
    if bits[2] == 1:
        return 'left', bits
    # 兜底：保持直行
    return 'forward', bits


def compute_follow_speed(distance_cm: float) -> tuple[str, int]:
    """仿照 follow.py 的速度曲线。
    返回 (mode, power): mode in {'forward','backward','stop'}，power=0..MAX_POWER
    """
    if distance_cm is None:
        return ('stop', 0)

    d = max(0.0, float(distance_cm))
    if d > FAR_DISTANCE:
        return ('forward', MAX_POWER)
    if d > MID_DISTANCE:
        ratio = (d - MID_DISTANCE) / (FAR_DISTANCE - MID_DISTANCE)  # 0..1
        return ('forward', max(0, min(int(MAX_POWER * ratio), MAX_POWER)))
    if d >= STOP_DISTANCE:
        return ('stop', 0)
    # d < STOP_DISTANCE：按比例倒退
    ratio = (STOP_DISTANCE - d) / STOP_DISTANCE  # 0..1
    return ('backward', max(0, min(int(MAX_POWER * ratio), MAX_POWER)))


def camera_worker():
    """摄像头线程：
    - 开启摄像头显示，打开人脸检测
    - 轮询 6 种颜色，选择面积最大的色块
    将结果写入全局 largest_color/area/size、face_present/face_size
    """
    global largest_color, largest_color_area, largest_color_size
    global face_present, face_size

    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=True, web=True)
    Vilib.face_detect_switch(True)

    time.sleep(1.5)  # 启动缓冲

    best_color = None
    best_area = 0
    best_size = (0, 0)
    idx = 0

    while True:
        # 人脸信息更新（不需要切换）
        human_n = Vilib.detect_obj_parameter.get('human_n', 0)
        if human_n and human_n > 0:
            w = int(Vilib.detect_obj_parameter.get('human_w', 0) or 0)
            h = int(Vilib.detect_obj_parameter.get('human_h', 0) or 0)
            face_present = True
            face_size = (w, h)
        else:
            face_present = False
            face_size = (0, 0)

        # 切换当前检测颜色
        color_name = COLOR_LIST[idx % len(COLOR_LIST)]
        Vilib.color_detect(color_name)
        time.sleep(COLOR_SWITCH_INTERVAL)

        # 读取该颜色的块信息
        c_n = Vilib.detect_obj_parameter.get('color_n', 0)
        if c_n and c_n > 0:
            w = int(Vilib.detect_obj_parameter.get('color_w', 0) or 0)
            h = int(Vilib.detect_obj_parameter.get('color_h', 0) or 0)
            area = w * h
            if area > best_area:
                best_area = area
                best_color = color_name
                best_size = (w, h)

        # 每轮全部颜色扫描结束后提交一次结果
        if (idx + 1) % len(COLOR_LIST) == 0:
            largest_color = best_color
            largest_color_area = best_area
            largest_color_size = best_size
            # 为下一轮重置
            best_color, best_area, best_size = None, 0, (0, 0)

        idx += 1


def keyboard_worker():
    """键盘线程：按 f 切换启停；Ctrl+C 由主线程捕获或直接退出。"""
    global running_flag
    while True:
        key = readchar.readkey()
        if not key:
            continue
        key = key.lower()
        if key == 'f':
            running_flag = not running_flag
            print(f"\n[KEY] f -> running = {running_flag}")


def horn_beep(music: Music):
    try:
        music.sound_play_threading(HORN_SOUND_PATH)
    except Exception as e:
        print(f"[WARN] horn play failed: {e}")


def main():
    global slow_override_until, stop_line_debounce_until

    px = Picarx()
    music = Music()
    try:
        music.music_set_volume(HORN_VOLUME)
    except Exception:
        pass

    # 摄像头与键盘线程
    t_cam = threading.Thread(target=camera_worker, daemon=True)
    t_cam.start()
    t_kbd = threading.Thread(target=keyboard_worker, daemon=True)
    t_kbd.start()


    last_debug = 0.0
    stop_line_active = False  # 是否刚离开停止线
    stop_line_leave_time = 0.0  # 离开停止线的时间戳
    stop_line_window = 200.0      # 离开后持续识别的窗口秒数

    while True:
        now = time.time()

        # 读取感应器
        gm_state, gm_bits = grayscale_state(px)
        dist = px.ultrasonic.read()
        dist = None if dist is None else round(float(dist), 2)

        # 循迹转向：不管前进/后退都打方向
        if gm_state == 'left':
            px.set_dir_servo_angle(LF_OFFSET)
        elif gm_state == 'right':
            px.set_dir_servo_angle(-LF_OFFSET)
        else:
            px.set_dir_servo_angle(0)

        # 停止线逻辑判断（带去抖）
        if gm_state == 'stop':
            # 停止线状态，进入识别窗口
            if not stop_line_active:
                stop_line_active = True
                stop_line_leave_time = 0.0
            # 停止线判决
            if now >= stop_line_debounce_until:
                col = largest_color
                if col == 'red':
                    px.stop()
                elif col == 'green':
                    # 直接放行，不改速度，但给一个轻微延迟避免抖动
                    stop_line_debounce_until = now + STOP_LINE_COOLDOWN
                else:
                    # 其他颜色或未识别：鸣笛 + 低速通过 3s
                    horn_beep(music)
                    slow_override_until = now + PASS_SLOW_SECS
                    stop_line_debounce_until = now + STOP_LINE_COOLDOWN
            # 停止线期间直接跳过后续逻辑
            time.sleep(0.02)
            continue
        else:
            # 离开停止线，进入2秒识别窗口
            if stop_line_active:
                stop_line_active = False
                stop_line_leave_time = now

        # 离开停止线后2秒内持续识别颜色
        in_window = (now - stop_line_leave_time) < stop_line_window if stop_line_leave_time > 0 else False
        if in_window:
            col = largest_color
            if col == 'red':
                px.stop()
                time.sleep(0.02)
                continue
            elif col == 'green':
                pass  # 直接前进，后续速度逻辑不变
            else:
                # 其他颜色或未识别：鸣笛 + 低速通过 3s
                horn_beep(music)
                slow_override_until = now + PASS_SLOW_SECS
                # 只触发一次，避免多次鸣笛
                stop_line_leave_time = 0.0

        # 未按 f 开始则保持静止（摄像头继续）
        if not running_flag:
            px.stop()
            time.sleep(0.02)
            continue

        # 超声波跟车速度（forward/back/stop）
        mode, power = compute_follow_speed(dist)

        # 人脸干预：近 -> 停；中 -> 限速（仅前进方向）
        if face_present:
            fw, fh = face_size
            if fw >= FACE_NEAR_W or fh >= FACE_NEAR_H:
                mode, power = ('stop', 0)
            elif fw >= FACE_SLOW_W or fh >= FACE_SLOW_H:
                if mode == 'forward':
                    power = min(power, FACE_SLOW_CAP)

        # 停止线慢行覆盖（只影响前进模式）
        if now < slow_override_until and mode == 'forward':
            power = min(power, PASS_SLOW_SPEED)

        # 执行动作
        if mode == 'forward':
            px.forward(power)
        elif mode == 'backward':
            px.backward(power)
        else:
            px.stop()

        # 简要调试输出（每 0.5s）
        if now - last_debug > 0.5:
            last_debug = now
            print(
                f"state={gm_state:>7} gm={gm_bits} dist={dist!s:>5}cm mode={mode:>8} p={power:>2} "
                f"color={largest_color or '-':>6} face={face_present} size={face_size} run={running_flag}"
            )

        time.sleep(0.02)

    # 结束循环

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        try:
            Vilib.camera_close()
        except Exception:
            pass
        try:
            Picarx().stop()
        except Exception:
            pass