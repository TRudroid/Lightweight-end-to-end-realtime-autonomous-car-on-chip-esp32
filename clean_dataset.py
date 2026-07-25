import os
import shutil
import cv2
import sys
from PIL import Image

# Reconfigure stdout to use UTF-8 if available to support accents
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DATASET_DIR = "dataset"
ERROR_DIR = "loi"

def clean_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Thu muc {DATASET_DIR} khong ton tai!")
        return

    # Tao thu muc "loi" neu chua co
    os.makedirs(ERROR_DIR, exist_ok=True)

    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]
    total = len(files)
    print(f"Tim thay tong cong {total} anh trong thu muc {DATASET_DIR}.")
    
    moved_corrupt = 0
    moved_empty = 0
    valid_count = 0
    
    for filename in files:
        filepath = os.path.join(DATASET_DIR, filename)
        error_path = os.path.join(ERROR_DIR, filename)
        
        # 1. Kiem tra tinh toan ven bang PIL (phat hien loi truyen dan, hut byte)
        is_error = False
        try:
            with Image.open(filepath) as img:
                img.verify()
        except Exception:
            print(f"Phat hien anh hong truyen dan -> Di chuyen: {filename}")
            shutil.move(filepath, error_path)
            moved_corrupt += 1
            is_error = True
            
        if is_error:
            continue
            
        # 2. Kiem tra doc bang OpenCV
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            print(f"Phat hien anh OpenCV khong doc duoc -> Di chuyen: {filename}")
            shutil.move(filepath, error_path)
            moved_empty += 1
            continue
            
        valid_count += 1

    print("\n=== KET QUA PHAN LOAI DATASET ===")
    print(f"Tong so anh ban dau: {total}")
    print(f"Da chuyen sang thu muc '{ERROR_DIR}' (loi truyen dan): {moved_corrupt}")
    print(f"Da chuyen sang thu muc '{ERROR_DIR}' (OpenCV khong doc duoc): {moved_empty}")
    print(f"So anh HOP LE con lai trong '{DATASET_DIR}': {valid_count}")

if __name__ == "__main__":
    clean_dataset()
