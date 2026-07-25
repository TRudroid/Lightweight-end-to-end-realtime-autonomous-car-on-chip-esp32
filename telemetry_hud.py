"""
Onboard telemetry HUD for ESP32 TinyNav (TM packets).
"""

import socket
import struct
import time

import cv2
import numpy as np

from tinynav_config import LISTEN_PORT, CONTROL_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("0.0.0.0", LISTEN_PORT))
sock.settimeout(0.02)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

print("\n--- TinyNav Onboard Telemetry HUD ---")
print("  A / Space : enable onboard autopilot")
print("  M / S     : emergency stop / manual")
print("  +/-       : BASE_SPEED")
print("  Q         : quit")
print("-------------------------------------\n")

esp32_ip = None
last_heartbeat_time = 0.0
last_frame_time = time.time()
steering = throttle = 0.0
inf_time = 0
auto_mode = True
current_frame = np.zeros((128, 128), dtype=np.uint8)
HEADER_SIZE = 21

while True:
    now = time.time()
    if now - last_frame_time > 1.0 and now - last_heartbeat_time > 1.0:
        try:
            sock.sendto(b"HB", ("255.255.255.255", CONTROL_PORT))
            if esp32_ip:
                sock.sendto(b"HB", (esp32_ip, CONTROL_PORT))
            last_heartbeat_time = now
        except Exception:
            pass

    try:
        data, addr = sock.recvfrom(65535)
        if data in (b"HB", b"HB\n"):
            continue
        if esp32_ip != addr[0]:
            esp32_ip = addr[0]
            print(f"Telemetry from {esp32_ip}")

        if len(data) >= HEADER_SIZE and data.startswith(b"TM"):
            steer, throt, _obst, inf_t, mode, img_size = struct.unpack("<fffIBH", data[2:21])
            steering, throttle, inf_time = steer, throt, inf_t
            auto_mode = bool(mode)
            last_frame_time = time.time()
            img_bytes = data[21 : 21 + img_size]
            if len(img_bytes) == 16384:
                current_frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape((128, 128))
    except socket.timeout:
        pass

    display = cv2.cvtColor(
        cv2.resize(current_frame, (384, 384), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    h, w, _ = display.shape
    if time.time() - last_frame_time >= 6.0:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(display, "TELEMETRY DISCONNECTED", (40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    mode_str = "ONBOARD AUTOPILOT" if auto_mode else "MANUAL"
    mode_color = (0, 255, 0) if auto_mode else (0, 255, 255)
    cv2.putText(display, f"Mode: {mode_str}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
    cv2.putText(
        display,
        f"Steer:{steering:+.2f}  Throt:{throttle:.2f}  Inf:{inf_time}ms",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    if esp32_ip:
        cv2.putText(display, f"ESP32: {esp32_ip}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    center_x = w // 2
    pointer_x = int(center_x + steering * (w // 2 - 20))
    cv2.line(display, (center_x, h - 40), (pointer_x, h - 15), (0, 255, 0), 2)
    cv2.imshow("TinyNav Onboard HUD", display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q")):
        break
    if key in (ord("a"), ord("A"), ord(" ")) and esp32_ip:
        sock.sendto(b"A", (esp32_ip, CONTROL_PORT))
    if key in (ord("m"), ord("M"), ord("s"), ord("S")) and esp32_ip:
        sock.sendto(b"M", (esp32_ip, CONTROL_PORT))
    if key in (ord("+"), ord("=")) and esp32_ip:
        sock.sendto(b"+", (esp32_ip, CONTROL_PORT))
    if key in (ord("-"), ord("_")) and esp32_ip:
        sock.sendto(b"-", (esp32_ip, CONTROL_PORT))

sock.close()
cv2.destroyAllWindows()
