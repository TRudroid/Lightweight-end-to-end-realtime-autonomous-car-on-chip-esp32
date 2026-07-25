"""
PC-side TinyNav autopilot: UDP JPEG stream from ESP32-S3 → 20-frame CNN → motor commands.
"""

import os
import socket
import struct
import time

import cv2
import numpy as np

from tinynav_config import (
    LISTEN_PORT,
    CONTROL_PORT,
    WINDOW_SIZE,
    IMG_SIZE,
    BEST_MODEL_FILE,
    TFLITE_FILE,
    ROI_CROP_TOP,
    STOP,
    STRAIGHT,
    LEFT,
    RIGHT,
)

BASE_SPEED = 80
STEER_DEADZONE = 0.05

use_tflite = False
interpreter = None
model = None
input_details = output_details = None
input_scale = output_scale = 1.0
input_zero_point = output_zero_point = 0

if os.path.exists(TFLITE_FILE):
    try:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        interpreter = tflite.Interpreter(model_path=TFLITE_FILE)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_scale, input_zero_point = input_details[0]["quantization"]
        output_scale, output_zero_point = output_details[0]["quantization"]
        if input_scale == 0.0:
            input_scale = 1.0
        if output_scale == 0.0:
            output_scale = 1.0
        use_tflite = True
        print("Loaded TFLite model:", TFLITE_FILE)
    except Exception as e:
        print("TFLite load failed, falling back to Keras:", e)

if not use_tflite:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    def custom_loss(y_true, y_pred):
        ps = tf.tanh(y_pred[:, 0:1])
        pt = tf.sigmoid(y_pred[:, 1:2])
        return tf.reduce_mean(tf.square(y_true[:, 0:1] - ps)) + tf.reduce_mean(
            tf.square(y_true[:, 1:2] - pt)
        )

    def steer_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, 0:1] - tf.tanh(y_pred[:, 0:1])))

    def throt_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, 1:2] - tf.sigmoid(y_pred[:, 1:2])))

    if not os.path.exists(BEST_MODEL_FILE):
        raise FileNotFoundError(f"Missing {BEST_MODEL_FILE} / {TFLITE_FILE}")
    model = tf.keras.models.load_model(
        BEST_MODEL_FILE,
        custom_objects={
            "custom_loss": custom_loss,
            "steer_mae": steer_mae,
            "throt_mae": throt_mae,
        },
    )
    print("Loaded Keras model:", BEST_MODEL_FILE)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("0.0.0.0", LISTEN_PORT))
sock.settimeout(0.01)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

frame_queue = []
buffer = bytearray()
end_marker = b"\xff\xd9"
esp32_ip = None
auto_pilot = False
last_frame_time = time.time()
last_command_time = 0.0
command_interval = 0.05
last_sent_speed = 0
last_heartbeat_time = 0.0
steering = throttle = 0.0
status_msg = "Waiting for camera stream..."
current_frame = np.zeros((240, 320, 3), dtype=np.uint8)

print("\n--- TinyNav PC Autopilot ---")
print("  Space : toggle autopilot")
print("  S     : emergency stop")
print("  +/-   : BASE_SPEED")
print("  Q     : quit")
print("----------------------------\n")


def predict(stacked_uint8):
    x = np.expand_dims(stacked_uint8.astype(np.float32) / 255.0, axis=0)
    if use_tflite:
        if input_details[0]["dtype"] == np.int8:
            xq = (x / input_scale + input_zero_point).astype(np.int8)
        else:
            xq = x
        interpreter.set_tensor(input_details[0]["index"], xq)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details[0]["index"])[0]
        if output_details[0]["dtype"] == np.int8:
            raw = (raw.astype(np.float32) - output_zero_point) * output_scale
        steer_raw, throt_raw = float(raw[0]), float(raw[1])
    else:
        raw = model.predict(x, verbose=0)[0]
        steer_raw, throt_raw = float(raw[0]), float(raw[1])
    return float(np.tanh(steer_raw)), float(1.0 / (1.0 + np.exp(-throt_raw)))


def command_from_prediction(steer, throt):
    if throt < 0.05:
        return STOP, 0, 0, "STOP (low throttle)"
    steer_val = abs(steer)
    speed = int(BASE_SPEED * (0.55 + 0.45 * throt))
    speed = min(140, speed)
    if steer_val > STEER_DEADZONE:
        direction = LEFT if steer < 0 else RIGHT
        speed = min(145, int(speed + steer_val * 35))
        alpha = int(steer_val * speed)
        name = "LEFT" if direction == LEFT else "RIGHT"
        return direction, speed, alpha, f"{name} s={steer:+.2f} thr={throt:.2f}"
    return STRAIGHT, speed, 0, f"STRAIGHT s={steer:+.2f} thr={throt:.2f}"


while True:
    now = time.time()
    if now - last_frame_time > 1.0 and now - last_heartbeat_time > 1.0:
        try:
            sock.sendto(b"HB", ("255.255.255.255", CONTROL_PORT))
            last_heartbeat_time = now
        except Exception:
            pass

    try:
        data, addr = sock.recvfrom(65535)
        if data in (b"HB", b"HB\n"):
            continue
        if esp32_ip != addr[0]:
            esp32_ip = addr[0]
            print(f"ESP32 connected: {esp32_ip}")

        frame = None
        if data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"):
            buffer.clear()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            buffer.extend(data)
            start = buffer.find(b"\xff\xd8")
            if start == -1:
                if len(buffer) > 65535:
                    buffer.clear()
                continue
            buffer = buffer[start:]
            end = buffer.find(end_marker)
            if end == -1:
                continue
            end += 2
            frame = cv2.imdecode(np.frombuffer(buffer[:end], np.uint8), cv2.IMREAD_COLOR)
            buffer = buffer[end:]

        if frame is None:
            frame = current_frame
        if frame is not None:
            current_frame = frame
            last_frame_time = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            crop_top = int(h * ROI_CROP_TOP)
            roi = gray[crop_top:, :]
            resized = cv2.resize(roi, IMG_SIZE)

            frame_queue.append(resized)
            if len(frame_queue) > WINDOW_SIZE:
                frame_queue.pop(0)

            if len(frame_queue) == WINDOW_SIZE:
                stacked = np.stack(frame_queue, axis=-1)
                steering, throttle = predict(stacked)
                target_dir, target_speed, target_alpha, status_msg = command_from_prediction(
                    steering, throttle
                )
                now = time.time()
                if auto_pilot and esp32_ip and (now - last_command_time > command_interval):
                    if last_sent_speed == 0 and target_dir == STRAIGHT and target_speed < 110:
                        sock.sendto(f"{target_dir} 130 {target_alpha}".encode(), (esp32_ip, CONTROL_PORT))
                        time.sleep(0.12)
                    sock.sendto(
                        f"{target_dir} {target_speed} {target_alpha}".encode(),
                        (esp32_ip, CONTROL_PORT),
                    )
                    last_command_time = now
                    last_sent_speed = target_speed
            else:
                status_msg = f"Buffering {len(frame_queue)}/{WINDOW_SIZE}"
                steering = throttle = 0.0

    except socket.timeout:
        if time.time() - last_frame_time > 2.0:
            buffer.clear()
            frame_queue.clear()

    display = current_frame.copy()
    h, w, _ = display.shape
    connected = time.time() - last_frame_time < 1.5
    if not connected:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(display, "CONNECTION LOST", (w // 2 - 90, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    crop_top = int(h * ROI_CROP_TOP)
    cv2.line(display, (0, crop_top), (w, crop_top), (0, 255, 255), 1)
    pilot = "AUTO-PILOT" if auto_pilot else "MANUAL"
    color = (0, 255, 0) if auto_pilot else (0, 255, 255)
    cv2.putText(display, pilot, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(
        display,
        f"Steer:{steering:+.2f} Throt:{throttle:.2f} BASE:{BASE_SPEED}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(display, status_msg, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.imshow("TinyNav PC Autopilot", display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q")):
        if esp32_ip:
            sock.sendto(b"0 0 0", (esp32_ip, CONTROL_PORT))
        break
    if key == ord(" "):
        auto_pilot = not auto_pilot
        print("Autopilot:", "ON" if auto_pilot else "OFF")
        if not auto_pilot and esp32_ip:
            sock.sendto(b"0 0 0", (esp32_ip, CONTROL_PORT))
            last_sent_speed = 0
    if key in (ord("s"), ord("S")):
        auto_pilot = False
        if esp32_ip:
            sock.sendto(b"0 0 0", (esp32_ip, CONTROL_PORT))
            last_sent_speed = 0
        print("EMERGENCY STOP")
    if key in (ord("+"), ord("=")):
        BASE_SPEED = min(220, BASE_SPEED + 10)
        print("BASE_SPEED", BASE_SPEED)
    if key in (ord("-"), ord("_")):
        BASE_SPEED = max(40, BASE_SPEED - 10)
        print("BASE_SPEED", BASE_SPEED)

sock.close()
cv2.destroyAllWindows()
