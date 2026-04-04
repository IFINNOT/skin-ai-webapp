import os, shutil, pandas as pd

df = pd.read_csv("data/HAM10000_metadata.csv")
os.makedirs("data/raw", exist_ok=True)

for _, row in df.iterrows():
    label = row["dx"]                          # 類別名稱
    src = f"data/images/{row['image_id']}.jpg" # 原始圖片路徑
    dst_dir = f"data/raw/{label}"
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src, dst_dir)

print("✅ 圖片分類完成")