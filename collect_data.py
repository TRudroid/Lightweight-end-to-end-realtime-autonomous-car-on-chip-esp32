import socket
import cv2
import numpy as np
import os
import time
import struct

# TinyNav data collection for ESP32-S3 + OV2640 (labels used by 20-frame preprocess).
# --- CẤU HÌNH UDP ---
LISTEN_PORT = 3000        # Port nhận ảnh từ ESP32-S3
CONTROL_PORT = 3001       # Port gửi lệnh điều khiển đến ESP32-S3
DATASET_DIR = "dataset"   # Thư mục lưu dữ liệu (giữ nguyên khi rebuild pipeline)

# Đảm bảo thư mục lưu dataset tồn tại
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
    print(f"Đã tạo thư mục lưu dữ liệu: {DATASET_DIR}")

# Thiết lập Socket UDP (Kích hoạt Broadcast để dò tìm xe tự động)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("0.0.0.0", LISTEN_PORT))
sock.settimeout(0.01) # Timeout ngắn để giải phóng luồng kiểm tra bàn phím liên tục

print(f"Đang chạy script nhận ảnh trên Port {LISTEN_PORT}...")
print("Hãy đảm bảo Laptop và ESP32-S3 kết nối cùng mạng WiFi/Hotspot.")

# Trạng thái điều khiển xe
# Định nghĩa các trạng thái chuyển động (khớp với firmware nhúng)
STOP = 0
STRAIGHT = 1
SLOW = 2
LEFT = 3
RIGHT = 4
TURN_LEFT = 5
TURN_RIGHT = 6
STANDBY = 7

current_dir = STOP
current_speed = 0
current_alpha = 0

# Tốc độ di chuyển mặc định khi lái tay (sửa cứng về 80 theo yêu cầu)
BASE_SPEED = 80
TURN_ALPHA = 40  # Độ lệch tốc độ giữa 2 bánh khi rẽ (chênh lệch motor)

# Trạng thái ghi dữ liệu (Nhấn 'R' để bật/tắt ghi hình tự động khi xe chạy)
recording_active = False
frame_count = 0

print("\n--- HƯỚNG DẪN ĐIỀU KHIỂN ---")
print("  W / Mũi tên lên   : Đi thẳng (STRAIGHT)")
print("  S / Mũi tên xuống : Dừng lại (STOP)")
print("  A / Mũi tên trái  : Rẽ trái (LEFT)")            
print("  D / Mũi tên phải  : Rẽ phải (RIGHT)")
print("  Space (Phím cách) : Phanh khẩn cấp (STOP)")
print("  R                 : Bật/Tắt chế độ tự động lưu ảnh khi xe chạy (Auto-record)")
print("  Q                 : Thoát chương trình")
print("---------------------------\n")

buffer = bytearray()
end_marker = b"\xff\xd9" # Marker kết thúc của ảnh JPEG
esp32_ip = None

# Khởi tạo màn hình chờ OpenCV mặc định để cv2.waitKey hoạt động lập tức
current_frame = np.zeros((240, 320, 3), dtype=np.uint8)

last_frame_time = time.time()
last_timeout_print = 0
last_heartbeat_time = 0

while True:
    # Send broadcast heartbeat to register Laptop IP on ESP32-S3 if disconnected
    now = time.time()
    if now - last_frame_time > 1.0 and now - last_heartbeat_time > 1.0:
        try:
            sock.sendto(b"HB", ("255.255.255.255", 3001))
            if esp32_ip:
                sock.sendto(b"HB", (esp32_ip, 3001))
            last_heartbeat_time = now
        except Exception:
            pass

    try:
        # Nhận dữ liệu từ socket
        data, addr = sock.recvfrom(65535)
        
        # Nếu là gói tin Heartbeat (HB) thì bỏ qua
        if data == b"HB" or data == b"HB\n":
            continue
            
        # Lưu địa chỉ IP của ESP32 tự động để phản hồi lệnh
        if esp32_ip is None or esp32_ip != addr[0]:
            esp32_ip = addr[0]
            print(f"-> Đã phát hiện kết nối từ ESP32-S3 tại IP: {esp32_ip}")

        # TH1: Gói tin Telemetry nhúng trực tiếp từ esp32_firmware_onboard ("TM" header)
        if len(data) >= 21 and data.startswith(b"TM"):
            img_size = struct.unpack("<H", data[19:21])[0]
            img_bytes = data[21:21 + img_size]
            if len(img_bytes) == 16384:
                gray_128 = np.frombuffer(img_bytes, dtype=np.uint8).reshape((128, 128))
                current_frame = cv2.cvtColor(gray_128, cv2.COLOR_GRAY2BGR)
                last_frame_time = time.time()

                if recording_active and current_speed > 0:
                    timestamp = int(time.time() * 1000)
                    filename = f"frame_{timestamp}_{current_dir}_{current_speed}_{current_alpha}.jpg"
                    filepath = os.path.join(DATASET_DIR, filename)
                    cv2.imwrite(filepath, gray_128)
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Đã lưu {frame_count} ảnh vào dataset...")
            continue

        # TH2: Ghép mảnh dữ liệu ảnh JPEG từ esp32_firmware gốc
        buffer.extend(data)
        
        # Đồng bộ hóa bộ đệm: Tìm marker bắt đầu của JPEG (\xff\xd8)
        start_pos = buffer.find(b"\xff\xd8")
        if start_pos != -1:
            buffer = buffer[start_pos:] # Bỏ các byte rác trước đó
        else:
            if len(buffer) > 65535:
                buffer.clear()
            continue
        
        # Tìm điểm kết thúc của ảnh JPEG trong bộ đệm
        pos = buffer.find(end_marker)
        if pos != -1:
            pos += 2
            img_bytes = buffer[:pos]
            buffer = buffer[pos:]  # Giữ phần còn lại cho frame tiếp theo
            
            # Giải mã ảnh JPEG sang ma trận OpenCV
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                current_frame = frame
                last_frame_time = time.time() # Cập nhật thời gian nhận ảnh thành công
                
                # --- XỬ LÝ LƯU TRỮ DATASET ---
                # Chỉ lưu khi đang bật chế độ ghi hình (recording_active) và xe đang di chuyển (speed > 0)
                if recording_active and current_speed > 0:
                    timestamp = int(time.time() * 1000)
                    filename = f"frame_{timestamp}_{current_dir}_{current_speed}_{current_alpha}.jpg"
                    filepath = os.path.join(DATASET_DIR, filename)
                    
                    # Cắt bỏ 40% phần trên (hậu cảnh nhiễu) và lưu ảnh xám 128x128 để huấn luyện
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray_frame.shape
                    crop_top = int(h * 0.4)
                    roi = gray_frame[crop_top:, :]
                    resized = cv2.resize(roi, (128, 128))
                    cv2.imwrite(filepath, resized)
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Đã lưu {frame_count} ảnh vào dataset...")

    except socket.timeout:
        now = time.time()
        # Chỉ cảnh báo nếu quá 6 giây không nhận được gì và cách lần in trước ít nhất 5 giây
        if now - last_frame_time > 6.0 and now - last_timeout_print > 5.0:
            print("Đang đợi ảnh từ ESP32-S3... (Timeout socket)")
            last_timeout_print = now
            buffer.clear()
            
    # --- HIỂN THỊ HÌNH ẢNH LÊN MÀN HÌNH ---
    # Sao chép frame hiện tại để vẽ text trạng thái (tránh vẽ đè trực tiếp làm hỏng dataset)
    display_frame = current_frame.copy()
    h, w, _ = display_frame.shape
    crop_top = int(h * 0.4)
    
    # Vẽ đường cắt ROI và vùng AI sẽ nhìn thấy
    cv2.line(display_frame, (0, crop_top), (w, crop_top), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "AI VIEW ZONE (CROP)", (w - 130, crop_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    
    status_text = f"Control: Dir={current_dir}, Speed={current_speed}, Alpha={current_alpha}"
    rec_text = "RECORDING (Auto)" if (recording_active and current_speed > 0) else "REC OFF"
    rec_color = (0, 0, 255) if (recording_active and current_speed > 0) else (0, 255, 0)
    
    cv2.putText(display_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Record: {rec_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rec_color, 1)
    cv2.putText(display_frame, f"Total saved: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if esp32_ip is None:
        cv2.putText(display_frame, "STATUS: WAITING ESP32 CONNECTION...", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(display_frame, "PLEASE PLUG IN ESP32 OR CHECK WIFI", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    else:
        cv2.putText(display_frame, f"STATUS: CONNECTED ({esp32_ip})", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
    cv2.imshow("ESP32-S3 Camera Feed (Collect Data)", display_frame)
        
    # --- ĐỌC PHÍM ĐIỀU KHIỂN & GỬI LỆNH QUA UDP ---
    key = cv2.waitKey(1) & 0xFF
    
    # Thoát chương trình
    if key == ord('q') or key == ord('Q'):
        # Dừng xe trước khi thoát
        if esp32_ip:
            cmd = f"{STOP} 0 0"
            sock.sendto(cmd.encode(), (esp32_ip, CONTROL_PORT))
        break
        
    # Bật/Tắt chế độ lưu ảnh
    elif key == ord('r') or key == ord('R'):
        recording_active = not recording_active
        print(f"==> Chế độ tự động lưu ảnh (Auto-record): {'BẬT' if recording_active else 'TẮT'}")
        
    # Xử lý các phím di chuyển
    cmd_changed = False
    
    # 1. Đi thẳng (W)
    if key == ord('w') or key == ord('W') or key == 82: # 82 là mã phím mũi tên lên
        current_dir = STRAIGHT
        current_speed = BASE_SPEED
        current_alpha = 0
        cmd_changed = True
        
    # 2. Rẽ trái (A)
    elif key == ord('a') or key == ord('A') or key == 81:
        current_dir = LEFT
        current_speed = max(120, BASE_SPEED + 35)  # Đảm bảo bánh ngoài đủ lực xoay xe ở tốc độ thấp
        current_alpha = current_speed              # Dừng hẳn bánh trong để cua xoáy trục mượt mà
        cmd_changed = True
        
    # 3. Rẽ phải (D)
    elif key == ord('d') or key == ord('D') or key == 83:
        current_dir = RIGHT
        current_speed = max(120, BASE_SPEED + 35)  # Đảm bảo bánh ngoài đủ lực xoay xe ở tốc độ thấp
        current_alpha = current_speed              # Dừng hẳn bánh trong để cua xoáy trục mượt mà
        cmd_changed = True
        
    # 4. Phanh / Dừng xe (S hoặc Space)
    elif key == ord('s') or key == ord('S') or key == ord(' ') or key == 84:
        current_dir = STOP
        current_speed = 0
        current_alpha = 0
        cmd_changed = True

    # Gửi lệnh điều khiển sang ESP32-S3
    if cmd_changed and esp32_ip:
        # Tự động kích đề (Kick-start) nếu xe chuyển từ đứng yên sang chạy thẳng ở tốc độ thấp
        # Gửi xung lực mạnh 130 trong 120ms để thắng ma sát tĩnh, sau đó trả về tốc độ thật
        if current_dir == STRAIGHT and current_speed > 0 and current_speed < 110:
            kick_cmd = f"{current_dir} 130 {current_alpha}"
            sock.sendto(kick_cmd.encode(), (esp32_ip, CONTROL_PORT))
            time.sleep(0.12)
            
        cmd = f"{current_dir} {current_speed} {current_alpha}"
        sock.sendto(cmd.encode(), (esp32_ip, CONTROL_PORT))
        print(f"Gửi lệnh -> Hướng: {current_dir}, Tốc độ: {current_speed}, Lệch: {current_alpha}")

# Đóng các cửa sổ và giải phóng socket
sock.close()
cv2.destroyAllWindows()
print(f"Đã đóng chương trình. Tổng số ảnh thu thập được: {frame_count}")
