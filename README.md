# 人臉手勢介紹展示系統

## 建立環境與執行

本專題使用 Python 3.11。為使用課程指定的 `mp.solutions` API，套件版本已固定在相容組合。

目前專案路徑含有中文，MediaPipe 的 Windows 原生元件可能無法由中文路徑載入模型。因此這台電腦請使用 ASCII 路徑下的虛擬環境啟動：

```powershell
& "C:\Users\user\.codex\visualizations\2026\10\20\01a159d1-a890-7063-a3db-c217892d3327\smart_order_venv\Scripts\python.exe" "C:\Users\user\OneDrive\Desktop\AI感知期中\smart_ordering_system.py"
```

若把專案移到純英文路徑，可在專案資料夾以一般 `.venv` 執行以下建立環境步驟。

Windows PowerShell 指令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python smart_ordering_system.py
```

如果 PowerShell 不允許啟用腳本，也可以直接執行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe smart_ordering_system.py
```

按 `s` 將目前狀態寫入 `content_records.csv`；手勢切換內容時也會自動記錄。按 `q` 離開。

## 功能與測試參數

程式會同時開啟攝影機狀態視窗與獨立的「介紹預覽」視窗。比出 1–5 並維持一秒後，介紹預覽會顯示對應的彩色待確認卡片；點頭確認後才顯示介紹，搖頭取消會停止目前介紹並清空選擇。取消後須重新比出內容並點頭確認才會開始介紹。之後可將 `make_content_preview` 的占位畫面改為影片影格。

- MediaPipe Face Detection 標註人臉框並顯示目前畫面人臉數；Face Mesh 以嘴唇關鍵點判斷張嘴，張嘴認證後即可操作。
- 觀眾在虛擬線右側張嘴認證成功時，立即累計進入一次；已認證觀眾由右往左穩定穿越虛擬線時，累計離開一次並清除認證。左往右穿越僅更新位置，不重複增加進入數。
- MediaPipe Hands 偵測手部關鍵點並估算伸出手指數；維持 1–5 根手指一秒後，對應介紹卡進入待確認狀態。
- 點頭會確認並切換介紹，搖頭會取消；確認或取消後放下手，再比手勢才能選下一項。確認與取消都會自動寫入紀錄。
- `smart_ordering_system.py` 開頭的設定區可調整 `FACE_CONFIDENCE`、`HAND_CONFIDENCE`、`MOUTH_OPEN_THRESHOLD`、`FLOW_LINE_RATIO`、`FLOW_LINE_BUFFER`、`FLOW_SIDE_CONFIRM_FRAMES`、`TRACK_MAX_DISTANCE`、`GESTURE_HOLD_SECONDS`、`HEAD_GESTURE_WINDOW_SECONDS`、`HEAD_GESTURE_THRESHOLD` 與 `CAMERA_INDEX`。提高 `FLOW_SIDE_CONFIRM_FRAMES` 可降低短暫抖動誤計，但會讓進出計數稍慢；調整頭部動作閾值可改變點頭／搖頭靈敏度。

## 注意事項

客流追蹤採用相鄰畫面的人臉中心距離配對，適合課堂展示；多人交錯或快速遮擋時，可能把追蹤身分配錯。若手掌側轉、手指互相遮擋，或頭部動作幅度太小，辨識結果可能不準。之後可將 `make_content_preview` 的彩色占位卡替換成影片播放區。
