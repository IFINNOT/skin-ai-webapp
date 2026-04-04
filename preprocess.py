import os
import shutil
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import pandas as pd

# ========== 設定路徑 ==========
RAW_DATA_DIR = "data/raw"        # 原始圖片資料夾
PROCESSED_DIR = "data/processed" # 處理後輸出資料夾

# ========== 前處理步驟 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),        # 統一 resize 成 224x224
    transforms.ToTensor(),                # 轉成 Tensor（值域 0~1）
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       # ImageNet 標準均值
        std=[0.229, 0.224, 0.225]         # ImageNet 標準標準差
    )
])

# ========== 載入資料集 ==========
dataset = ImageFolder(root=RAW_DATA_DIR, transform=transform)

print(f"總圖片數量：{len(dataset)}")
print(f"類別列表：{dataset.classes}")

# ========== 切分 train / val / test ==========
total = len(dataset)
train_size = int(total * 0.7)
val_size   = int(total * 0.15)
test_size  = total - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

print(f"\n資料集分割結果：")
print(f"  Train : {len(train_set)} 筆")
print(f"  Val   : {len(val_set)} 筆")
print(f"  Test  : {len(test_set)} 筆")

# ========== DataLoader ==========
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

# ========== 統計各類別數量並畫圖 ==========
labels = [dataset.targets[i] for i in range(len(dataset))]
class_names = dataset.classes
counts = [labels.count(i) for i in range(len(class_names))]

plt.figure(figsize=(10, 5))
bars = plt.bar(class_names, counts, color='steelblue')
plt.title("Distribution of the number of images by category")
plt.xlabel("Dermatoscopic Lesion Type")
plt.ylabel("Count")
plt.xticks(rotation=20)
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(count), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("data/class_distribution.png")
plt.show()
print("\n✅ 類別分布圖已儲存至 data/class_distribution.png")