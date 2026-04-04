import os
import shutil
import random
from pathlib import Path

# ========== 設定 ==========
RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/split"
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
# TEST_RATIO = 0.15（剩下的全部）

random.seed(42)  # 固定隨機種子，每次結果一樣

# ========== 開始切分 ==========
classes = os.listdir(RAW_DIR)
print(f"找到 {len(classes)} 個類別：{classes}\n")

total_train, total_val, total_test = 0, 0, 0

for cls in classes:
    cls_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    for split_name, split_images in splits.items():
        dst_dir = os.path.join(OUTPUT_DIR, split_name, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(dst_dir, img)
            shutil.copy(src, dst)

    print(f"{cls:8s} → train: {len(splits['train']):4d} | val: {len(splits['val']):4d} | test: {len(splits['test']):4d}")
    total_train += len(splits['train'])
    total_val   += len(splits['val'])
    total_test  += len(splits['test'])

print(f"\n{'合計':8s} → train: {total_train:4d} | val: {total_val:4d} | test: {total_test:4d}")
print(f"\n✅ 切分完成！資料儲存於 data/split/")