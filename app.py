from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sqlite3
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
_model = None

def get_model():
    global _model
    if _model is None:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(CLASS_NAMES))
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        _model = m
        print("✅ 模型載入成功")
    return _model

# ========== 圖片前處理 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== SQLite 連線 ==========
DB_PATH = "predictions.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT,
            prediction  TEXT,
            confidence  REAL,
            created_at  TEXT
        )
    """)
    db.commit()
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
        output = get_model()(img_tensor)
        prob   = torch.nn.functional.softmax(output[0], dim=0)

    confidence = prob.max().item()
    class_idx  = prob.argmax().item()
    class_en   = CLASS_NAMES[class_idx]
    class_zh   = CLASS_INFO[class_en]

    # 所有類別的信心度
    all_probs = {CLASS_INFO[CLASS_NAMES[i]]: round(prob[i].item() * 100, 1)
                 for i in range(len(CLASS_NAMES))}

    # 寫入 SQLite
    try:
        db = get_db()
        db.execute(
            "INSERT INTO predictions (filename, prediction, confidence, created_at) VALUES (?, ?, ?, ?)",
            (filename, class_zh, round(confidence * 100, 1), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        db.commit()
        db.close()
    except Exception as e:
        print(f"SQLite 寫入失敗：{e}")

    return jsonify({
        "prediction": class_zh,
        "confidence": round(confidence * 100, 1),
        "all_probs":  all_probs
    })

@app.route("/history")
def history():
    try:
        db = get_db()
        cursor = db.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 20")
        rows = cursor.fetchall()
        db.close()
        records = [dict(row) for row in rows]
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True)