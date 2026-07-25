"""
TinyNav Colab / GPU training script (20-frame stack, dual steer/throttle).
Upload dataset.zip or place jpgs under dataset/, then run end-to-end.
"""

import os
import re
import sys
import zipfile

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WINDOW_SIZE = 20
IMG_SIZE = (24, 24)  # paper spatial size — required for ~30ms onboard
BATCH_SIZE = 32
EPOCHS = 500
EARLY_STOP_PATIENCE = 40
EARLY_STOP_MIN_DELTA = 1e-5
REDUCE_LR_PATIENCE = 12
MAX_GAP_MS = 2000

print("TensorFlow", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

if os.path.exists("dataset.zip"):
    with zipfile.ZipFile("dataset.zip", "r") as zf:
        zf.extractall("dataset")
elif os.path.exists("dataset.rar"):
    os.system('unrar x -o+ "dataset.rar" "dataset/"')


def parse_filename(filepath):
    name = os.path.basename(filepath)
    m = re.match(r"frame_(\d+)(?:_aug_\d+)?_(\d+)_(\d+)_(\d+)\.jpg", name)
    if not m:
        return None
    return {
        "path": filepath,
        "timestamp": int(m.group(1)),
        "direction": int(m.group(2)),
        "speed": int(m.group(3)),
        "alpha": int(m.group(4)),
    }


def labels_from_meta(meta):
    d, speed, alpha = meta["direction"], meta["speed"], meta["alpha"]
    if speed <= 0 or d == 0:
        return 0.0, 0.0
    mag = float(np.clip(alpha / max(speed, 1), 0.0, 1.0))
    if d == 3:
        steer = -mag
    elif d == 4:
        steer = mag
    else:
        steer = 0.0
    return steer, float(np.clip(speed / 255.0, 0.0, 1.0))


def build_model(input_shape=(24, 24, 20)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(8, 5, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(24, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    steer = layers.Dense(1)(x)
    throt = layers.Dense(1)(x)
    outputs = layers.Concatenate(name="output")([steer, throt])
    return models.Model(inputs, outputs, name="TinyNav_CNN")


def custom_loss(y_true, y_pred):
    ps = tf.tanh(y_pred[:, 0:1])
    pt = tf.sigmoid(y_pred[:, 1:2])
    return tf.reduce_mean(tf.square(y_true[:, 0:1] - ps)) + tf.reduce_mean(
        tf.square(y_true[:, 1:2] - pt)
    )


img_files = []
for root, _, names in os.walk("dataset"):
    for n in names:
        if n.endswith(".jpg"):
            img_files.append(os.path.join(root, n))
print(f"Images: {len(img_files)}")
if not img_files:
    raise SystemExit("No dataset images found.")

meta = [parse_filename(p) for p in img_files]
meta = [m for m in meta if m]
meta.sort(key=lambda x: x["timestamp"])

sessions, cur = [], [meta[0]]
for i in range(1, len(meta)):
    if meta[i]["timestamp"] - meta[i - 1]["timestamp"] > MAX_GAP_MS:
        sessions.append(cur)
        cur = [meta[i]]
    else:
        cur.append(meta[i])
sessions.append(cur)

X, Ys, Yt = [], [], []
for sess in sessions:
    if len(sess) < WINDOW_SIZE:
        continue
    frames = []
    for item in sess:
        img = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(IMG_SIZE, dtype=np.uint8)
        else:
            img = cv2.resize(img, IMG_SIZE)
        frames.append(img)
    for i in range(WINDOW_SIZE - 1, len(sess)):
        X.append(np.stack(frames[i - WINDOW_SIZE + 1 : i + 1], axis=-1))
        s, t = labels_from_meta(sess[i])
        Ys.append(s)
        Yt.append(t)

X = np.array(X, dtype=np.uint8)
Ys = np.array(Ys, dtype=np.float32)
Yt = np.array(Yt, dtype=np.float32)
print(f"Windows: {len(X)} shape={X.shape}")

x_tv, x_te, ys_tv, ys_te, yt_tv, yt_te = train_test_split(
    X, Ys, Yt, test_size=0.15, random_state=42
)
x_tr, x_va, ys_tr, ys_va, yt_tr, yt_va = train_test_split(
    x_tv, ys_tv, yt_tv, test_size=0.25 / 0.85, random_state=42
)

model = build_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=custom_loss)

y_tr = np.column_stack([ys_tr, yt_tr])
y_va = np.column_stack([ys_va, yt_va])

# light online aug via tf.data
def gen(xs, ys):
    idx = np.arange(len(xs))
    while True:
        np.random.shuffle(idx)
        for i in idx:
            x = xs[i].astype(np.float32) / 255.0
            y = ys[i].copy()
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1)
                y[0] = -y[0]
            yield x, y


ds_tr = tf.data.Dataset.from_generator(
    lambda: gen(x_tr, y_tr),
    output_signature=(
        tf.TensorSpec(shape=(24, 24, 20), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
    ),
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint("models/best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
]

print(f"Training up to {EPOCHS} epochs (early_stop patience={EARLY_STOP_PATIENCE})")
history = model.fit(
    x_tr.astype(np.float32) / 255.0,
    y_tr,
    validation_data=(x_va.astype(np.float32) / 255.0, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)
model.save("models/best_model.keras")
print(f"Finished after {len(history.history['loss'])}/{EPOCHS} epochs. Best weights saved.")
print("Download models/best_model.keras — then run notebooks/quantize_model.py locally.")
