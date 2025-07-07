import cv2
import numpy as np
import math
import json
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# --- グローバル変数と設定 ---
CONFIG_FILE = 'config.json'
FONT_PATH = 'NotoSansJP-VariableFont_wght.ttf'
CONFIG = {}
# ★★★ 変更点: 形状関連の変数を削除 ★★★
# FOCAL_LENGTH と KNOWN_WIDTHS_MM は計算の基準としてのみ使用
FOCAL_LENGTH = 1200
KNOWN_WIDTHS_MM = {'apple': 90.0}


def draw_text_japanese(frame, text, position, font_size, bg_color, text_color=(255, 255, 255)):
    """日本語テキスト（背景付き）を描画する"""
    if not os.path.exists(FONT_PATH):
        cv2.putText(frame, "Font file not found.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    font = ImageFont.truetype(FONT_PATH, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    text_bbox = draw.textbbox(position, text, font=font)
    bg_position = (text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5)
    draw.rectangle(bg_position, fill=bg_color)
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def load_settings():
    """設定ファイルを読み込む"""
    global CONFIG
    if not os.path.exists(CONFIG_FILE):
        print(f"エラー: 設定ファイル '{CONFIG_FILE}' が見つかりません。")
        print("先に `configure_sorter.py` を実行して、設定ファイルを作成してください。")
        return False
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            CONFIG = json.load(f)
        print("✅ 設定ファイルを読み込みました。")
        return True
    except Exception as e:
        print(f"エラー: 設定ファイルの読み込みに失敗しました。 - {e}")
        return False


def main_detector():
    """メインの検出処理"""
    model = YOLO('yolov9c.pt')

    try:
        apple_class_id = [k for k, v in model.names.items() if v == 'apple'][0]
    except IndexError:
        print("エラー: YOLOモデルのクラス名に 'apple' が見つかりませんでした。")
        return

    # 設定値を変数に格納
    DIAMETER_THRESHOLDS = {"min": CONFIG['apple_min_diameter_mm'], "max": CONFIG['apple_max_diameter_mm']}
    AREA_THRESHOLDS = {"min": CONFIG['apple_min_area_mm2'], "max": CONFIG['apple_max_area_mm2']}
    HSV_RANGES = [(np.array(r['lower']), np.array(r['upper'])) for r in CONFIG['apple_hsv_ranges']]

    print("\nカメラを起動します...'q'キーで終了します。")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, device='0', classes=[apple_class_id], verbose=False)

        for result in results:
            for box in result.boxes:
                if float(box.conf[0]) < 0.6: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: continue

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                final_mask = np.zeros(hsv_roi.shape[:2], dtype="uint8")
                for lower, upper in HSV_RANGES:
                    final_mask = cv2.bitwise_or(final_mask, cv2.inRange(hsv_roi, lower, upper))

                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    # 色が範囲外の場合
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    frame = draw_text_japanese(frame, "NG (色が範囲外)", (x1, y1 - 30), 22, (0, 0, 255))
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                area_pixels = cv2.contourArea(largest_contour)
                if area_pixels < 200: continue

                # --- 計算 ---
                pixel_width = x2 - x1
                if pixel_width <= 0: continue
                mm_per_pixel = KNOWN_WIDTHS_MM['apple'] / pixel_width
                diameter_mm = (math.sqrt(area_pixels / math.pi) * 2) * mm_per_pixel
                area_mm2 = area_pixels * (mm_per_pixel ** 2)

                # --- 判定 ---
                # ★★★ 変更点: 形状判定を削除し、面積判定を追加 ★★★
                is_ok = True
                reason = ""
                if not (DIAMETER_THRESHOLDS['min'] <= diameter_mm <= DIAMETER_THRESHOLDS['max']):
                    is_ok = False
                    reason = f"NG(直径): {diameter_mm:.1f}mm"
                elif not (AREA_THRESHOLDS['min'] <= area_mm2 <= AREA_THRESHOLDS['max']):
                    is_ok = False
                    reason = f"NG(面積): {area_mm2:.1f}mm2"

                # --- 描画 ---
                if is_ok:
                    color = (0, 255, 0)
                    text = f"OK D:{diameter_mm:.1f} A:{area_mm2:.1f}"
                else:
                    color = (0, 0, 255)
                    text = reason

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame = draw_text_japanese(frame, text, (x1, y1 - 30), 22, color)

        cv2.imshow("Apple Sorter (Size/Color Only)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("果物品質管理AI 匠ソーター - リアルタイム検出")
    if load_settings():
        main_detector()