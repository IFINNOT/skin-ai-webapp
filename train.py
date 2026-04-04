import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ========== 設定 ==========
SPLIT_DIR  = "data/split"
MODEL_DIR  = "model"
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"使用裝置：{DEVICE}\n")

# ========== 前處理 ==========
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),        # 隨機左右翻轉（Data Augmentation）
    transforms.RandomRotation(15),            # 隨機旋轉 ±15 度
    transforms.ColorJitter(brightness=0.2),   # 隨機亮度調整
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== 載入資料 ==========
train_set = datasets.ImageFolder(f"{SPLIT_DIR}/train", transform=train_transform)
val_set   = datasets.ImageFolder(f"{SPLIT_DIR}/val",   transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

print(f"類別：{train_set.classes}")
print(f"Train: {len(train_set)} 筆 | Val: {len(val_set)} 筆\n")

# ========== 模型（ResNet-18 遷移學習）==========
model = models.resnet18(weights="IMAGENET1K_V1")

# 凍結前面的層
for param in model.parameters():
    param.requires_grad = True

# 只訓練最後的分類層
model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
model = model.to(DEVICE)

# ========== 訓練設定 ==========
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(train_set.classes)),
    y=train_set.targets
)
weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam([
    {"params": list(model.parameters())[:-2], "lr": 1e-5},
    {"params": model.fc.parameters(),         "lr": 1e-4},
])

# ========== 訓練紀錄 ==========
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0

# ========== 訓練迴圈 ==========
for epoch in range(EPOCHS):

    # --- Train ---
    model.train()
    train_loss, train_correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss    += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    # --- 計算平均 ---
    t_loss = train_loss / len(train_set)
    v_loss = val_loss   / len(val_set)
    t_acc  = train_correct / len(train_set) * 100
    v_acc  = val_correct   / len(val_set)   * 100

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
          f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.1f}%  |  "
          f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.1f}%")

    # --- 儲存最佳模型 ---
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pth")
        print(f"  ✅ 最佳模型已儲存（Val Acc: {v_acc:.1f}%）")

# ========== 儲存訓練紀錄 CSV ==========
with open(f"{MODEL_DIR}/history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
    for i in range(EPOCHS):
        writer.writerow([i+1,
                         history["train_loss"][i],
                         history["val_loss"][i],
                         history["train_acc"][i],
                         history["val_acc"][i]])

# ========== 畫訓練曲線 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"],   label="Val Loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history["train_acc"], label="Train Acc")
ax2.plot(history["val_acc"],   label="Val Acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/training_curve.png")
plt.show()

print(f"\n🏆 訓練完成！最佳 Val Acc：{best_val_acc:.1f}%")
print(f"模型儲存於 model/best_model.pth")
print(f"訓練曲線儲存於 model/training_curve.png")