# Xe tự hành bám lane trên ESP32-S3 (end-to-end TinyML)

Đồ án xây dựng xe hai bánh dẫn động vi sai, tự điều khiển theo làn đường xám bằng mạng CNN nhẹ chạy trực tiếp trên ESP32-S3 (hoặc suy luận trên máy tính qua UDP). Cảm biến chính là camera OV2640; dữ liệu huấn luyện lấy từ ảnh xám đã thu bằng điều khiển tay.

Phương pháp tham khảo công thức TinyML của TinyNav (cửa sổ 20 khung hình xếp theo kênh, CNN 2D lượng tử hóa INT8, hai đầu ra steering / throttle) và chỉnh lại cho bài toán bám lane bằng camera thay vì cảm biến độ sâu ToF.

**Tác giả:** Nhóm AIP491_G19

---

## 1. Mục tiêu

- Thu thập dữ liệu lái thật trên đường thử (lane xám, viền trắng, nền xanh).
- Huấn luyện mô hình CNN nhẹ (~26k tham số) để dự đoán góc lái và ga liên tục.
- Triển khai hai chế độ chạy:
  - **Chế độ A – Stream:** ESP32 gửi JPEG/UDP, máy tính suy luận và gửi lệnh động cơ.
  - **Chế độ B – Onboard:** ESP32 chạy TFLite INT8 trên chip (input 24x24x20, gần budget latency của bài báo gốc).
- Giữ được bộ dữ liệu camera đã thu; pipeline preprocess / train / quantize tái tạo được từ repo.

---

## 2. Phần cứng

| Thành phần | Ghi chú |
| --- | --- |
| MCU | ESP32-S3 (bắt buộc có **PSRAM**, khuyến nghị board dạng ESP32-S3-EYE / Dev Module) |
| Camera | OV2640, khung QVGA 320x240, lấy ROI phần dưới rồi resize xám |
| Động cơ | 2 động cơ DC + driver L298N (hoặc tương đương), dẫn động vi sai |
| Nguồn | Nên tách nguồn logic (3.3 V / 5 V board) và nguồn động cơ |

### Sơ đồ chân động cơ (mặc định sketch ESP32-S3-EYE)

| ESP32-S3 | Driver động cơ |
| --- | --- |
| GPIO 42 | PWM motor A |
| GPIO 41 | PWM motor B |

Nếu xe quay ngược hướng mong muốn, đổi hai chân PWM hoặc bật cờ đảo steering trong firmware onboard.

### Wi-Fi

Trong các file `.ino`, điền SSID / mật khẩu mạng thật trước khi nạp:

```cpp
const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";
```

PC và ESP32 phải cùng subnet để UDP hoạt động (cổng mặc định: PC `3000`, ESP `3001`).

---

## 3. Cấu trúc thư mục

```
dataset/                      # Ảnh thu thập (không đẩy Git; giữ local / dataset.rar)
tinynav_config.py             # Cấu hình chung: WINDOW=20, size 24x24, đường dẫn
notebooks/
  design_model.py             # Kiến trúc CNN
  preprocess_dataset.py       # Ghép cửa sổ 20 frame vào preprocessed_data.npz
  train_model.py              # Huấn luyện + early stopping
  quantize_model.py           # INT8 TFLite + model_data.h
  augment_dataset.py          # Tiện ích tăng cường dữ liệu (nếu dùng)
esp32_firmware/               # Chế độ A: stream camera + nhận lệnh UDP
esp32_firmware_onboard/       # Chế độ B: TFLite trên chip
libraries/EloquentTinyML/     # Thư viện Arduino kèm theo cho chế độ onboard
collect_data.py               # Lái tay + ghi ảnh cá nhân
clean_dataset.py              # Lọc JPEG hỏng
autonomous_drive.py           # Autopilot phía PC (chế độ A)
telemetry_hud.py              # HUD xem telemetry khi chạy onboard
train_colab.py                # Script train trên Colab / GPU
Colab_Training_TinyNav.ipynb
models/                       # Checkpoint Keras + TFLite + header C
requirements.txt
```

---

## 4. Môi trường phần mềm

### Python (PC)

```bash
pip install -r requirements.txt
```

Các thư viện chính: TensorFlow / Keras, OpenCV, NumPy, scikit-learn.

### Arduino IDE (ESP32)

1. Cài board package **esp32** (khuyến nghị nhánh 2.0.x ổn định với camera).
2. Chọn board **ESP32S3 Dev Module**.
3. Bật **PSRAM** (OPI / Enabled tùy board).
4. Copy thư mục `libraries/EloquentTinyML` vào `Documents/Arduino/libraries/` nếu chưa có.
5. Mở sketch tương ứng (`esp32_firmware` hoặc `esp32_firmware_onboard`), sửa Wi-Fi, rồi Upload.

---

## 5. Định dạng dữ liệu

Mỗi frame khi thu có tên:

```text
frame_{timestamp_ms}_{direction}_{speed}_{alpha}.jpg
```

- `direction`: 0 dừng, 1 thẳng, 3 trái, 4 phải (theo quy ước trong `tinynav_config.py`).
- `speed`, `alpha`: PWM và độ lệch bánh dùng để suy ra nhãn steering / throttle liên tục khi preprocess.
- Ảnh lưu xám; khi train được crop ROI (bỏ ~40% phần trên) rồi resize về **24x24** và xếp **20** frame liên tiếp thành một mẫu.

`dataset/` không đưa lên Git vì dung lượng lớn. Dùng bản local hoặc giải nén `dataset.rar` nếu có.

---

## 6. Pipeline huấn luyện

Thực hiện tuần tự từ thư mục gốc project:

```bash
# 1) Lọc ảnh hỏng (nếu cần)
python clean_dataset.py

# 2) Tạo tensor cửa sổ 20 frame
python notebooks/preprocess_dataset.py

# 3) Tạo model baseline
python notebooks/design_model.py

# 4) Train (CPU/GPU local)
python notebooks/train_model.py

# 5) Lượng tử hóa INT8 + sinh header cho firmware
python notebooks/quantize_model.py
```

Hoặc đẩy `dataset.zip` / dùng `train_colab.py` và notebook Colab khi cần GPU.

Sau bước quantize, file `models/model_data.h` được copy sang:

- `esp32_firmware_onboard/model_data.h`
- `esp32_firmware/model_data.h` (dự phòng)

---

## 7. Cách chạy xe

### 7.1. Thu thêm dữ liệu

1. Nạp `esp32_firmware/esp32_firmware.ino`.
2. Trên PC:

```bash
python collect_data.py
```

3. Phím: `W/A/S/D` điều khiển, `R` bật/tắt ghi, `Q` thoát.  
   Lái chậm, giữ xe trong lane xám; thu nhiều đoạn cua trái/phải và thẳng.

### 7.2. Chế độ A – Autopilot trên PC (dễ chỉnh, latency thấp hơn nếu PC mạnh)

1. Nạp `esp32_firmware`.
2. Chạy:

```bash
python autonomous_drive.py
```

PC nhận khung hình, chạy CNN, gửi steering/throttle về ESP.

### 7.3. Chế độ B – Onboard trên ESP32-S3

1. Đảm bảo đã có `esp32_firmware_onboard/model_data.h` mới nhất (sau `quantize_model.py`).
2. Nạp `esp32_firmware_onboard/esp32_firmware_onboard.ino`.
3. Mở Serial Monitor **115200**.
4. Đặt xe trên lane xám, cấp nguồn. Xe đứng yên đến khi có kết quả suy luận đầu tiên, rồi mới chạy.
5. Theo dõi dòng dạng `inf=XXms steer=... throt=...` và trạng thái `AI-DRIVE` / `STOP(no-grey)`.

Tùy chọn xem HUD:

```bash
python telemetry_hud.py
```

---

## 8. Mô hình và suy luận

- Input: `(24, 24, 20)` — 20 khung xám liên tiếp.
- Backbone: chuỗi Conv2D stride 2, ~26k tham số.
- Output: steering ∈ [-1, 1], throttle ∈ [0, 1] (sau hậu xử lý / ánh xạ PWM).
- Onboard: INT8 TFLite qua EloquentTinyML; kích thước tensor nhỏ để tiệm cận latency thực tế trên MCU.
- An toàn cơ bản trên firmware onboard: dừng nếu mất mắt đường xám quá ngưỡng thời gian; không chạy lệnh AI quá cũ.

Ánh xạ động cơ: dẫn động vi sai ở bánh trong chậm theo `|steering|`; có xung kick-start khi xuất phát từ đứng yên để vượt ma sát tĩnh.

---

## 9. Gợi ý chỉnh thực nghiệm

- Đèn phòng ổn định; tránh chói mạnh làm mất vùng xám.
- Nếu xe lệch một phía: kiểm tra đầu dây PWM, hoặc `INVERT_STEERING` trong sketch onboard.
- PWM quá thấp -> không khởi động; quá cao -> khó giữ lane. Chỉnh `BASE_SPEED` / `MIN_DRIVE_PWM` / `MAX_DRIVE_PWM`.
- Onboard chậm bất thường: xác nhận input đang là 24x24 (không còn model 128x128 cũ), PSRAM bật, và đã nạp `model_data.h` mới.

---

