# skin-ai-webapp
# 🩺 AI 皮膚病灶辨識 Web App

上傳皮膚圖片，AI 自動辨識 7 種常見病灶類型並顯示信心度。  
使用 ResNet-18 遷移學習訓練，整合 Flask 後端與 SQLite 資料庫，提供完整的 Web 使用介面。

> 🔗 Live Demo：(https://skin-ai-webapp.onrender.com)  
> 📹 Demo 影片：（錄製完成後補上連結）

---

## ✨ 功能

- 上傳皮膚圖片，即時取得 AI 辨識結果
- 顯示 7 種病灶類別與各類別信心度
- 歷史查詢紀錄（儲存至 SQLite）
- 支援 RWD，手機版面正常顯示

---

## 🛠️ 技術架構

| 層面 | 技術 |
|------|------|
| AI 模型 | PyTorch / ResNet-18 Transfer Learning |
| 後端 | Python Flask |
| 前端 | HTML / CSS / JavaScript |
| 資料庫 | SQLite |
| 部署 | Render |

---

## 📁 專案結構
```
skin-ai-webapp/
├── app.py              # Flask 主程式
├── model/
│   └── best_model.pth  # 訓練完成的模型
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── data/               # 資料集（不上傳至 GitHub）
├── train.py            # 模型訓練腳本
├── requirements.txt
└── README.md
```
---

## 🚀 本機安裝與執行
```bash
# 1. Clone 專案
git clone https://github.com/你的帳號/skin-ai-webapp.git
cd skin-ai-webapp

# 2. 建立虛擬環境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# 3. 安裝套件
pip install -r requirements.txt

# 4. 啟動伺服器
python app.py
```

瀏覽器開啟 `http://localhost:5000`

---

## 📊 資料集

- **來源**：[HAM10000 - Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)
- **內容**：10,000 張皮膚病灶圖片，共 7 種類別
- **類別**：黑色素瘤、黑色素細胞痣、基底細胞癌、光化性角化病、良性角化病、皮膚纖維瘤、血管病變

---

## 📸 截圖

（Week 3 完成後補上）

---

