"""人臉手勢介紹展示系統：MediaPipe 人臉、客流與手勢內容選擇。

執行：.venv\\Scripts\\python.exe smart_ordering_system.py
按 s 儲存一筆 CSV 紀錄；按 q 離開。

調整參數：可從下方設定區修改人臉信心門檻、虛擬線位置、追蹤距離、
手勢確認秒數與攝影機編號。張嘴認證成功會累計進入；已認證觀眾由畫面右側
穿越虛擬線往左離開時，累計離開。若攝影機視角不同，可調整 FLOW_LINE_RATIO。
"""

from __future__ import annotations

import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    print("缺少必要套件。請先啟用虛擬環境並安裝：")
    print("  .venv\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    print(f"詳細資訊：{exc}")
    raise SystemExit(1)


# ========================= 可調整參數 =========================
CAMERA_INDEX = 0
FACE_CONFIDENCE = 0.55       # 人臉偵測信心門檻；提高可減少誤判，但可能漏掉遠處人臉
HAND_CONFIDENCE = 0.55       # 手部偵測信心門檻
FLOW_LINE_RATIO = 0.50       # 虛擬線水平位置：畫面寬度比例（0.5 為中央）
FLOW_LINE_BUFFER = 20        # 虛擬線兩側緩衝像素，避免站在邊界時反覆計數
TRACK_MAX_DISTANCE = 90      # 前後影格人臉中心最大配對距離（像素）
TRACK_MAX_MISSED = 12        # 人臉暫時消失幾個影格後移除追蹤
FLOW_SIDE_CONFIRM_FRAMES = 6 # 進入另一側須連續維持的影格數，避免抖動誤計
GESTURE_HOLD_SECONDS = 1.0  # 手勢持續時間，穩定後切換到對應介紹卡
INTRO_DURATION_SECONDS = 3.0  # 點頭確認後全畫面播放介紹的秒數
HEAD_GESTURE_WINDOW_SECONDS = 1.1  # 點頭／搖頭動作判定時間窗
HEAD_GESTURE_THRESHOLD = 0.12     # 鼻尖位移門檻；越低越靈敏，也越容易誤觸
MOUTH_OPEN_THRESHOLD = 0.12       # 嘴巴張開比例門檻；降低可提高靈敏度但較易誤認
CSV_PATH = Path(__file__).with_name("content_records.csv")
WINDOW_NAME = "Gesture Intro Showcase | s:Save  q:Quit"

# 五種內容卡使用不同底色；之後可把各編號對應到實際影片檔。
CONTENT_COLORS = {
    1: (239, 83, 80),    # 紅
    2: (66, 165, 245),   # 藍
    3: (102, 187, 106),  # 綠
    4: (255, 167, 38),   # 橘
    5: (171, 71, 188),   # 紫
}
CONTENT_IMAGE_DIR = Path(__file__).with_name("images")
CONTENT_IMAGE_PATHS = {
    content_id: CONTENT_IMAGE_DIR / f"intro_{content_id}.png"
    for content_id in CONTENT_COLORS
}
CONTENT_IMAGE_CACHE: dict[int, Optional[np.ndarray]] = {}
CONTENT_IMAGE_RETRY_AT: dict[int, float] = {}


@dataclass
class FaceTrack:
    """追蹤一張臉，記錄中心點、最近一次觀測影格與是否已計數。"""

    track_id: int
    center: tuple[int, int]
    side: str
    missed: int = 0
    is_customer: bool = False
    mouth_open: bool = False
    candidate_side: Optional[str] = None
    side_stable_frames: int = 0


def get_face_centers(face_results, width: int, height: int):
    """將 MediaPipe 相對座標轉換成像素座標與人臉框。"""
    centers = []
    boxes = []
    if not face_results.detections:
        return centers, boxes
    for detection in face_results.detections:
        box = detection.location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * width))
        y1 = max(0, int(box.ymin * height))
        x2 = min(width - 1, int((box.xmin + box.width) * width))
        y2 = min(height - 1, int((box.ymin + box.height) * height))
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        boxes.append((x1, y1, x2, y2))
    return centers, boxes


def count_raised_fingers(hand_landmarks, handedness_label: str, mirrored: bool = True) -> int:
    """以指尖與關節相對位置估算伸出的手指數（0 至 5）。

    食指至小指：指尖 y 座標比 PIP 關節小（畫面座標往上）視為伸直。
    拇指：依左右手及鏡像畫面比較 x 座標。手掌角度太側時，拇指判斷可能不準。
    """
    lm = hand_landmarks.landmark
    count = 0
    # 四指：指尖索引 8/12/16/20，PIP 關節索引 6/10/14/18
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if lm[tip].y < lm[pip].y:
            count += 1

    # MediaPipe handedness 是以未鏡像影像為基準；我們將 webcam 畫面鏡像顯示，
    # 所以左右手在畫面中的方向需要反轉。拇指伸直時，指尖通常比 MCP 更外側。
    thumb_is_rightward = (handedness_label == "Left") if mirrored else (handedness_label == "Right")
    thumb_extended = lm[4].x > lm[2].x if thumb_is_rightward else lm[4].x < lm[2].x
    if thumb_extended:
        count += 1
    return max(0, min(5, count))


def get_mouth_open_ratio(face_landmarks) -> float:
    """用 Face Mesh 內唇上下距離除以嘴角寬度估算張嘴程度。"""
    points = face_landmarks.landmark
    vertical_gap = abs(points[13].y - points[14].y)
    mouth_width = max(abs(points[78].x - points[308].x), 1e-6)
    return vertical_gap / mouth_width


def load_content_image(content_id: int) -> Optional[np.ndarray]:
    """載入介紹圖片；讀取失敗時隔一秒重試，不永久快取失敗狀態。"""
    cached = CONTENT_IMAGE_CACHE.get(content_id)
    if cached is not None:
        return cached
    now = time.monotonic()
    if now < CONTENT_IMAGE_RETRY_AT.get(content_id, 0.0):
        return None
    CONTENT_IMAGE_RETRY_AT[content_id] = now + 1.0
    try:
        encoded = np.frombuffer(CONTENT_IMAGE_PATHS[content_id].read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is not None:
            CONTENT_IMAGE_CACHE[content_id] = image
            return image
    except (OSError, cv2.error):
        pass
    return None


def get_face_motion(face_landmarks) -> tuple[float, float]:
    """取鼻尖相對雙眼中心的位置，並以眼距正規化頭部位移。"""
    points = face_landmarks.landmark
    left_eye, right_eye, nose = points[33], points[263], points[1]
    eye_span = max(abs(right_eye.x - left_eye.x), 1e-6)
    eye_center_x = (left_eye.x + right_eye.x) / 2
    eye_center_y = (left_eye.y + right_eye.y) / 2
    return (nose.x - eye_center_x) / eye_span, (nose.y - eye_center_y) / eye_span


def _returned_to_start(values: list[float], threshold: float) -> bool:
    """判斷頭部是否先離開原位，再回到起始位置。"""
    if len(values) < 8:
        return False
    edge_count = max(2, len(values) // 6)
    start = float(np.mean(values[:edge_count]))
    end = float(np.mean(values[-edge_count:]))
    middle = values[edge_count:-edge_count]
    if not middle or abs(start - end) > threshold * 0.55:
        return False
    return max(abs(value - start) for value in middle) >= threshold


def detect_head_gesture(samples) -> Optional[str]:
    """依鼻尖移動方向判斷點頭（nod）或搖頭（shake）。"""
    if len(samples) < 8:
        return None
    x_values = [sample[1] for sample in samples]
    y_values = [sample[2] for sample in samples]
    nod = _returned_to_start(y_values, HEAD_GESTURE_THRESHOLD)
    shake = _returned_to_start(x_values, HEAD_GESTURE_THRESHOLD)
    if nod and shake:
        nod_range = max(y_values) - min(y_values)
        shake_range = max(x_values) - min(x_values)
        return "nod" if nod_range >= shake_range else "shake"
    if nod:
        return "nod"
    if shake:
        return "shake"
    return None


def ensure_csv_header(path: Path) -> None:
    """確保 CSV 存在且有欄位標題。"""
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(
                ["時間", "事件", "目前人臉數", "累計進入", "累計離開", "手指數", "內容編號"]
            )


def draw_traditional_chinese_hud(frame, lines: list[str]):
    """OpenCV Hershey 字型不支援中文，使用 Windows 微軟正黑體畫出繁中狀態。"""
    candidates = [Path(r"C:\Windows\Fonts\msjh.ttc"), Path(r"C:\Windows\Fonts\mingliu.ttc")]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        # 找不到中文字型時，以 OpenCV 顯示英文備援文字，避免主程式中斷。
        fallback = ["Faces", "Entered / Exited", "Customers", "Fingers",
                    "Pending intro", "Status", "Current intro", "Gesture countdown"]
        for row, label in enumerate(fallback):
            cv2.putText(frame, f"{label}: {lines[row]}", (22, 70 + row * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 22)
    for row, text in enumerate(lines):
        draw.text((22, 50 + row * 28), text, font=font, fill=(255, 255, 255),
                  stroke_width=1, stroke_fill=(25, 25, 25))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_traditional_chinese_title(frame, text: str):
    """以支援繁中的字型繪製黃色標題，避免 cv2.putText 出現問號方塊。"""
    candidates = [Path(r"C:\Windows\Fonts\msjh.ttc"), Path(r"C:\Windows\Fonts\mingliu.ttc")]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        cv2.putText(frame, "Face Recognition Order System", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
        return frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 27)
    bounds = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    text_width = bounds[2] - bounds[0]
    x = max(10, (frame.shape[1] - text_width) // 2)
    draw.text((x, 7), text, font=font, fill=(255, 215, 0),
              stroke_width=1, stroke_fill=(40, 40, 40))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def make_content_preview(content_id: Optional[int], pending: bool = False,
                         width: int = 640, height: int = 360):
    """建立介紹預覽；有對應圖片時等比例完整顯示，否則使用彩色占位卡。"""
    # 待確認階段只顯示占位卡，避免在點頭確認前先曝光照片。
    if content_id in CONTENT_IMAGE_PATHS and not pending:
        content_image = load_content_image(content_id)
        if content_image is not None:
            image_height, image_width = content_image.shape[:2]
            scale = min(width / image_width, height / image_height)
            resized_width = max(1, int(image_width * scale))
            resized_height = max(1, int(image_height * scale))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            resized = cv2.resize(content_image, (resized_width, resized_height),
                                 interpolation=interpolation)
            canvas = np.full((height, width, 3), (18, 18, 18), dtype=np.uint8)
            x = (width - resized_width) // 2
            y = (height - resized_height) // 2
            canvas[y:y + resized_height, x:x + resized_width] = resized
            return canvas

    if content_id in CONTENT_COLORS:
        color_rgb = CONTENT_COLORS[content_id]
        canvas = np.full((height, width, 3), color_rgb[::-1], dtype=np.uint8)
        cv2.rectangle(canvas, (0, height - 105), (width, height), (28, 28, 28), -1)
        title = f"{content_id} 號介紹" + ("（待確認）" if pending else "")
        note = "點頭確認／搖頭取消" if pending else "影片待補"
    else:
        canvas = np.full((height, width, 3), (48, 48, 48), dtype=np.uint8)
        cv2.rectangle(canvas, (0, height - 105), (width, height), (28, 28, 28), -1)
        title = "介紹預覽"
        note = "比出 1～5 選擇介紹"

    cv2.rectangle(canvas, (8, 8), (width - 9, height - 9), (255, 255, 255), 2)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_path = next((p for p in (Path(r"C:\Windows\Fonts\msjh.ttc"),
                                  Path(r"C:\Windows\Fonts\mingliu.ttc")) if p.exists()), None)
    if font_path:
        title_font = ImageFont.truetype(str(font_path), 30 if pending else 36)
        note_font = ImageFont.truetype(str(font_path), 24)
        draw.text((24, height - 92), title, font=title_font,
                  fill=(255, 255, 255), stroke_width=1, stroke_fill=(45, 45, 45))
        draw.text((24, height - 43), note, font=note_font,
                  fill=(255, 255, 255), stroke_width=1, stroke_fill=(45, 45, 45))
    else:
        cv2.putText(canvas, title, (24, height - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        fallback_note = ("NOD=OK / SHAKE=CANCEL" if pending
                         else "VIDEO TBD" if content_id else "SHOW 1-5")
        cv2.putText(canvas, fallback_note,
                    (24, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def save_record(path: Path, event: str, face_count: int, entered: int,
                exited: int, fingers: Optional[int], content_id: Optional[int]) -> None:
    """將內容選擇或手動儲存事件附加到 CSV。"""
    ensure_csv_header(path)
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([
            datetime.now().astimezone().isoformat(timespec="seconds"), event,
            face_count, entered, exited,
            "" if fingers is None else fingers,
            "" if content_id is None else content_id,
        ])


def main() -> None:
    if sys.version_info < (3, 9):
        print("建議使用 Python 3.9 以上版本（目前：" + sys.version.split()[0] + "）")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"無法開啟攝影機（索引 {CAMERA_INDEX}）。請檢查攝影機連線或調整 CAMERA_INDEX。")
        return
    # 先把五張介紹照片讀入記憶體，避免點頭後才讀檔造成畫面延遲；失敗時會持續重試。
    for content_id in CONTENT_IMAGE_PATHS:
        if load_content_image(content_id) is None:
            print(f"介紹圖片 {CONTENT_IMAGE_PATHS[content_id].name} 尚未載入，程式執行時會自動重試。")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    mp_face = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    face_tracks: dict[int, FaceTrack] = {}
    next_track_id = 0
    entered = 0
    exited = 0
    current_fingers: Optional[int] = None
    selected_content: Optional[int] = None
    candidate_content: Optional[int] = None
    candidate_since = 0.0
    candidate_confirmed = False
    pending_content: Optional[int] = None
    pending_since = 0.0
    intro_ends_at: Optional[float] = None
    intro_waiting_for_image = False
    head_motion_samples = deque(maxlen=90)
    gesture_latched_until_release = False
    last_saved_event = "尚未儲存"
    frame_index = 0

    try:
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=FACE_CONFIDENCE) as face_detector, \
             mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4,
                                   refine_landmarks=False,
                                   min_detection_confidence=FACE_CONFIDENCE,
                                   min_tracking_confidence=HAND_CONFIDENCE) as face_mesh, \
             mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                            min_detection_confidence=HAND_CONFIDENCE,
                            min_tracking_confidence=HAND_CONFIDENCE) as hand_detector:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("讀取攝影機影像失敗，程式即將結束。")
                    break
                frame_index += 1
                # 鏡像操作較符合面對鏡子的互動體驗；所有模組共用同一影像。
                frame = cv2.flip(frame, 1)
                camera_only_frame = frame.copy()
                height, width = frame.shape[:2]
                line_x = int(width * FLOW_LINE_RATIO)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                face_results = face_detector.process(rgb)
                face_mesh_results = face_mesh.process(rgb)
                hand_results = hand_detector.process(rgb)
                rgb.flags.writeable = True

                centers, boxes = get_face_centers(face_results, width, height)

                # 人臉框與中心點配對追蹤：採最近距離的一對一簡易追蹤，適合課堂展示。
                for track in face_tracks.values():
                    track.missed += 1
                unmatched_tracks = set(face_tracks.keys())
                for center in centers:
                    best_id = None
                    best_distance = TRACK_MAX_DISTANCE
                    for track_id in unmatched_tracks:
                        old = face_tracks[track_id].center
                        distance = float(np.hypot(center[0] - old[0], center[1] - old[1]))
                        if distance < best_distance:
                            best_id, best_distance = track_id, distance
                    # 在線附近留出緩衝帶，降低人臉在虛擬線邊緣晃動造成的重複計數。
                    side = ("left" if center[0] < line_x - FLOW_LINE_BUFFER
                            else "right" if center[0] > line_x + FLOW_LINE_BUFFER else None)
                    if best_id is None:
                        initial_side = side or ("left" if center[0] < line_x else "right")
                        face_tracks[next_track_id] = FaceTrack(next_track_id, center, initial_side)
                        next_track_id += 1
                    else:
                        track = face_tracks[best_id]
                        # 緩衝帶內不更新狀態；穿越到另一側且連續穩定數影格後才計數。
                        if side is None:
                            track.candidate_side = None
                            track.side_stable_frames = 0
                        elif side == track.side:
                            track.candidate_side = None
                            track.side_stable_frames = 0
                        else:
                            if track.candidate_side == side:
                                track.side_stable_frames += 1
                            else:
                                track.candidate_side = side
                                track.side_stable_frames = 1
                            if track.side_stable_frames >= FLOW_SIDE_CONFIRM_FRAMES:
                                # 左往右只更新位置；進入數在張嘴認證成功時增加。
                                # 右往左視為離場，且只有已認證觀眾才計入離開數。
                                if track.side == "right" and side == "left" and track.is_customer:
                                    exited += 1
                                    track.is_customer = False
                                    track.mouth_open = False
                                    last_saved_event = "認證觀眾離開：累計離開 +1"
                                track.side = side
                                track.candidate_side = None
                                track.side_stable_frames = 0
                        track.center = center
                        track.missed = 0
                        unmatched_tracks.discard(best_id)
                # 追蹤中斷時清除側向候選，避免不連續的影格湊成一次穿越。
                for track_id in unmatched_tracks:
                    face_tracks[track_id].candidate_side = None
                    face_tracks[track_id].side_stable_frames = 0
                face_tracks = {
                    tid: track for tid, track in face_tracks.items()
                    if track.missed <= TRACK_MAX_MISSED
                }

                # 只有在虛擬線右側張嘴認證成功時，才立即累計一次進入。
                # 離場時會清除認證；若再次進入並認證，可作為新的一次進場計數。
                for track in face_tracks.values():
                    if track.missed == 0:
                        track.mouth_open = False
                matched_mesh_tracks = set()
                if face_mesh_results.multi_face_landmarks:
                    for landmarks in face_mesh_results.multi_face_landmarks:
                        xs = [point.x for point in landmarks.landmark]
                        ys = [point.y for point in landmarks.landmark]
                        mesh_center = (int((min(xs) + max(xs)) * width / 2),
                                       int((min(ys) + max(ys)) * height / 2))
                        best_track = None
                        best_distance = TRACK_MAX_DISTANCE * 1.5
                        for track_id, track in face_tracks.items():
                            if track.missed != 0 or track_id in matched_mesh_tracks:
                                continue
                            distance = float(np.hypot(
                                mesh_center[0] - track.center[0], mesh_center[1] - track.center[1]
                            ))
                            if distance < best_distance:
                                best_track = track_id
                                best_distance = distance
                        if best_track is not None:
                            track = face_tracks[best_track]
                            track.mouth_open = get_mouth_open_ratio(landmarks) >= MOUTH_OPEN_THRESHOLD
                            if track.mouth_open and not track.is_customer and track.side == "right":
                                track.is_customer = True
                                entered += 1
                                last_saved_event = "觀眾認證成功：累計進入 +1"
                            matched_mesh_tracks.add(best_track)
                qualified_customers = sum(track.is_customer for track in face_tracks.values())

                # 手勢選內容：有效觀眾維持 1～5 手勢一秒後，等待點頭或搖頭。
                current_fingers = None
                handedness_for_text = ""
                if hand_results.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        hand_label = "Right"
                        if hand_results.multi_handedness and i < len(hand_results.multi_handedness):
                            hand_label = hand_results.multi_handedness[i].classification[0].label
                        if current_fingers is None:
                            current_fingers = count_raised_fingers(hand_landmarks, hand_label)
                            handedness_for_text = hand_label
                now = time.monotonic()
                if intro_waiting_for_image and selected_content is not None:
                    if load_content_image(selected_content) is not None:
                        intro_waiting_for_image = False
                        intro_ends_at = now + INTRO_DURATION_SECONDS
                        last_saved_event = (
                            f"{selected_content} 號照片已載入，播放 {INTRO_DURATION_SECONDS:.0f} 秒"
                        )
                if intro_ends_at is not None and now >= intro_ends_at:
                    finished_content = selected_content
                    selected_content = None
                    intro_ends_at = None
                    intro_waiting_for_image = False
                    last_saved_event = (
                        f"{finished_content} 號介紹播放完成（{INTRO_DURATION_SECONDS:.0f} 秒）"
                    )
                if current_fingers not in CONTENT_COLORS:
                    gesture_latched_until_release = False
                    candidate_content = None
                    candidate_since = now
                    candidate_confirmed = False
                elif (qualified_customers > 0 and pending_content is None
                      and intro_ends_at is None
                      and not intro_waiting_for_image
                      and not gesture_latched_until_release):
                    if candidate_content != current_fingers:
                        candidate_content = current_fingers
                        candidate_since = now
                        candidate_confirmed = False
                    elif now - candidate_since >= GESTURE_HOLD_SECONDS and not candidate_confirmed:
                        pending_content = candidate_content
                        pending_since = now
                        head_motion_samples.clear()
                        candidate_confirmed = True
                        gesture_latched_until_release = True
                        last_saved_event = f"待確認 {pending_content} 號介紹"
                else:
                    candidate_content = None
                    candidate_since = now
                    candidate_confirmed = False

                # 預覽待確認內容時，用點頭確認、搖頭取消。
                head_gesture = None
                if pending_content is not None and face_mesh_results.multi_face_landmarks:
                    nose_x, nose_y = get_face_motion(face_mesh_results.multi_face_landmarks[0])
                    head_motion_samples.append((now, nose_x, nose_y))
                    while (head_motion_samples
                           and now - head_motion_samples[0][0] > HEAD_GESTURE_WINDOW_SECONDS):
                        head_motion_samples.popleft()
                    if now - pending_since >= 0.25:
                        head_gesture = detect_head_gesture(head_motion_samples)

                if pending_content is not None and head_gesture == "nod":
                    selected_content = pending_content
                    intro_waiting_for_image = load_content_image(selected_content) is None
                    intro_ends_at = (None if intro_waiting_for_image
                                     else now + INTRO_DURATION_SECONDS)
                    save_record(CSV_PATH, "點頭確認介紹", len(centers), entered, exited,
                                current_fingers, selected_content)
                    last_saved_event = (
                        f"點頭確認：正在載入 {selected_content} 號照片"
                        if intro_waiting_for_image
                        else f"點頭確認：播放 {selected_content} 號介紹 {INTRO_DURATION_SECONDS:.0f} 秒"
                    )
                    pending_content = None
                    head_motion_samples.clear()
                elif pending_content is not None and head_gesture == "shake":
                    cancelled_content = pending_content
                    save_record(CSV_PATH, "搖頭取消介紹", len(centers), entered, exited,
                                current_fingers, cancelled_content)
                    # 搖頭會停止目前介紹並清空選擇；必須重新比手勢，點頭確認後才顯示新介紹。
                    selected_content = None
                    intro_waiting_for_image = False
                    intro_ends_at = None
                    last_saved_event = f"搖頭取消：{cancelled_content} 號介紹，等待重新選擇並點頭"
                    pending_content = None
                    head_motion_samples.clear()

                if qualified_customers == 0 and pending_content is not None:
                    cancelled_content = pending_content
                    pending_content = None
                    head_motion_samples.clear()
                    last_saved_event = f"觀眾離開，取消待確認的 {cancelled_content} 號介紹"

                # 辨識與操作狀態統一顯示在介紹預覽主畫面，不覆蓋右下角的純鏡頭影像。
                lines = [
                    f"目前人臉數：{len(centers)}",
                    f"累計進入：{entered}　累計離開：{exited}",
                    f"已認證觀眾：{qualified_customers}（張嘴後可選介紹）",
                    f"識別手指：{current_fingers if current_fingers is not None else '-'} 根（{handedness_for_text}）",
                    f"待確認介紹：{pending_content if pending_content is not None else '無'}",
                    f"紀錄狀態：{last_saved_event}",
                    f"目前介紹：{selected_content if selected_content is not None else '等待比出 1-5'}",
                ]
                preview_content = pending_content if pending_content is not None else selected_content
                if candidate_content is not None and not candidate_confirmed:
                    remaining = max(0.0, GESTURE_HOLD_SECONDS - (now - candidate_since))
                    lines.append(f"手勢選擇倒數：{remaining:.1f} 秒")

                # 介紹預覽作為主畫面；縮小辨識鏡頭後置於右下角。
                ui_width = min(max(width, 960), 1280)
                ui_height = int(ui_width * 9 / 16)
                combined_frame = make_content_preview(
                    preview_content,
                    pending=pending_content is not None,
                    width=ui_width,
                    height=ui_height,
                )
                if intro_waiting_for_image:
                    combined_frame = np.full((ui_height, ui_width, 3), (18, 18, 18),
                                             dtype=np.uint8)
                    combined_frame = draw_traditional_chinese_title(
                        combined_frame, f"{selected_content} 號照片載入中"
                    )
                elif intro_ends_at is None:
                    # 介紹播放期間使用純全畫面；播放結束後恢復狀態文字與右下角鏡頭。
                    combined_frame = draw_traditional_chinese_title(
                        combined_frame, "人臉手勢介紹展示"
                    )
                    panel_bottom = min(ui_height - 110, 50 + 28 * len(lines) + 12)
                    panel = combined_frame.copy()
                    cv2.rectangle(panel, (10, 42), (ui_width - 10, panel_bottom), (20, 20, 20), -1)
                    cv2.addWeighted(panel, 0.72, combined_frame, 0.28, 0, combined_frame)
                    combined_frame = draw_traditional_chinese_hud(combined_frame, lines)

                    # 右下角顯示鏡頭與必要的辨識標註，不放置整塊狀態文字面板。
                    inset_width = int(ui_width * 0.32)
                    inset_height = max(1, int(height * inset_width / width))
                    inset_x = ui_width - inset_width - 20
                    inset_y = ui_height - inset_height - 20
                    annotated_camera = camera_only_frame.copy()
                    for detection, (x1, y1, x2, y2) in zip(
                            face_results.detections or [], boxes):
                        score = detection.score[0] if detection.score else 0.0
                        cv2.rectangle(annotated_camera, (x1, y1), (x2, y2),
                                      (50, 220, 80), 2)
                        cv2.putText(annotated_camera, f"Face {score:.2f}",
                                    (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (50, 220, 80), 2, cv2.LINE_AA)
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                annotated_camera, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )
                    small_camera = cv2.resize(annotated_camera, (inset_width, inset_height),
                                               interpolation=cv2.INTER_AREA)
                    small_line_x = int(line_x * inset_width / width)
                    cv2.line(small_camera, (small_line_x, 0),
                             (small_line_x, inset_height - 1), (0, 210, 255), 2)
                    cv2.rectangle(combined_frame,
                                  (inset_x - 3, inset_y - 3),
                                  (inset_x + inset_width + 2, inset_y + inset_height + 2),
                                  (255, 255, 255), 3)
                    combined_frame[inset_y:inset_y + inset_height,
                                   inset_x:inset_x + inset_width] = small_camera
                cv2.imshow(WINDOW_NAME, combined_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_record(CSV_PATH, "手動儲存", len(centers), entered, exited,
                                current_fingers, selected_content)
                    last_saved_event = f"已儲存至 {CSV_PATH.name}"
                elif key == ord("q"):
                    break
    except KeyboardInterrupt:
        print("收到中斷訊號，正在安全關閉攝影機。")
    except Exception as exc:
        print(f"執行期間發生錯誤：{exc}")
        print("請確認攝影機可用，且 MediaPipe、OpenCV、NumPy 已安裝完成。")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
