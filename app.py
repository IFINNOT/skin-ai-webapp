from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mysql.connector
import io
import os
from datetime import datetime

app = Flask(__name__)

# ========== 設定 ==========
MODEL_PATH = "model/best_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_INFO  = {
    "akiec": "光化性角化病",
    "bcc":   "基底細胞癌",
    "bkl":   "良性角化病",
    "df":    "皮膚纖維瘤",
    "mel":   "黑色素瘤",
    "nv":    "黑色素細胞痣",
    "vasc":  "血管病變"
}

# ========== 載入模型 ==========
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ 模型載入成功")

# ========== 圖片前處理 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== MySQL 連線 ==========
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="YWxsZW4=",
        database="skin_ai"
    )

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            filename    VARCHAR(255),
            prediction  VARCHAR(50),
            confidence  FLOAT,
            created_at  DATETIME
        )
    """)
    db.commit()
    cursor.close()
    db.close()

# ========== 路由 ==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "沒有收到圖片"}), 400

    file = request.files["image"]
    filename = file.filename

    # 讀取圖片
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 模型推論
    with torch.no_grad():
        output = model(img_tensor)
        prob   = torch.nn.functional.softmax(output[0], dim=0)

    confidence = prob.max().item()
    class_idx  = prob.argmax().item()
    class_en   = CLASS_NAMES[class_idx]
    class_zh   = CLASS_INFO[class_en]

    # 所有類別的信心度
    all_probs = {CLASS_INFO[CLASS_NAMES[i]]: round(prob[i].item() * 100, 1)
                 for i in range(len(CLASS_NAMES))}

    # 寫入 MySQL
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO predictions (filename, prediction, confidence, created_at) VALUES (%s, %s, %s, %s)",
            (filename, class_zh, round(confidence * 100, 1), datetime.now())
        )
        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        print(f"MySQL 寫入失敗：{e}")

    return jsonify({
        "prediction": class_zh,
        "confidence": round(confidence * 100, 1),
        "all_probs":  all_probs
    })

@app.route("/history")
def history():
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 20")
        records = cursor.fetchall()
        cursor.close()
        db.close()
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True)