"""
Shared TinyNav configuration for ESP32-S3 + OV2640 camera adaptation.

Paper reference:
  https://arxiv.org/abs/2603.11071
  https://github.com/regularpooria/tinynav

Official TinyNav uses ToF depth 24x24x20 on ESP32-P4.
This project keeps the same TinyML recipe (stacked temporal channels,
compact 2D CNN, INT8 TFLite) but adapts input to grayscale camera frames
already collected in dataset/.
"""

# Temporal stack (paper: 20 frames ≈ 1s history)
WINDOW_SIZE = 20

# Match paper spatial size (ToF was 24x24; we downsample camera gray to 24x24).
# 128x128x20 ≈ 28× more pixels → ~seconds/inference on ESP; 24x24 targets ~30ms.
IMG_HEIGHT = 24
IMG_WIDTH = 24
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, WINDOW_SIZE)

# ROI: drop top fraction of QVGA frame before resize (background clutter)
ROI_CROP_TOP = 0.40

# Session split gap (ms) when grouping continuous driving runs
MAX_GAP_MS = 2000

# Motor / label mapping
MAX_PWM = 255.0
STOP, STRAIGHT, SLOW, LEFT, RIGHT = 0, 1, 2, 3, 4

# UDP ports (PC <-> ESP32)
LISTEN_PORT = 3000
CONTROL_PORT = 3001

# Paths
DATASET_DIR = "dataset"
PREPROCESSED_FILE = "dataset/preprocessed_data.npz"
MODELS_DIR = "models"
BASE_MODEL_FILE = "models/model_baseline.keras"
BEST_MODEL_FILE = "models/best_model.keras"
TFLITE_FILE = "models/tiny_nav_quantized.tflite"
MODEL_HEADER_FILE = "models/model_data.h"
FIRMWARE_HEADER_STREAM = "esp32_firmware/model_data.h"
FIRMWARE_HEADER_ONBOARD = "esp32_firmware_onboard/model_data.h"
