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
FOCAL_LENGTH = 1200
KNOWN_WIDTHS_MM = {'apple': 90.0}
DEFECT_MODEL_PATH = r"C:\Users\ok230075\PycharmProjects\pythonProject38\best.pt"


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
    fruit_model = YOLO('yolov9c.pt')
    try:
        defect_model = YOLO(DEFECT_MODEL_PATH)
        print(f"✅ 傷検出モデル '{DEFECT_MODEL_PATH}' を読み込みました。")
    except Exception as e:
        print(f"エラー: 傷検出モデルの読み込みに失敗しました: {e}")
        return

    try:
        apple_class_id = [k for k, v in fruit_model.names.items() if v == 'apple'][0]
    except IndexError:
        print("エラー: YOLOモデルのクラス名に 'apple' が見つかりませんでした。")
        return

    DIAMETER_THRESHOLDS = {"min": CONFIG['apple_min_diameter_mm'], "max": CONFIG['apple_max_diameter_mm']}
    AREA_THRESHOLDS = {"min": CONFIG['apple_min_area_mm2'], "max": CONFIG['apple_max_area_mm2']}
    HSV_RANGES = [(np.array(r['lower']), np.array(r['upper'])) for r in CONFIG['apple_hsv_ranges']]
    defect_confidence_threshold = 0.68

    print("\nカメラを起動します...'q'キーで終了します。")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        fruit_results = fruit_model.predict(frame, device='0', classes=[apple_class_id], verbose=False)

        for result in fruit_results:
            for box in result.boxes:
                if float(box.conf[0]) < 0.6: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: continue

                # --- 1. 全ての情報を収集 ---
                defect_results = defect_model.predict(roi, verbose=False)

                # ★★★ 変更点: テンソルをNumPy配列に変換してから型変換 ★★★
                detected_defects = [d.xyxy[0].cpu().numpy().astype(int) for d in defect_results[0].boxes if
                                    d.conf[0] > defect_confidence_threshold]
                has_defect = bool(detected_defects)

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                final_mask = np.zeros(hsv_roi.shape[:2], dtype="uint8")
                for lower, upper in HSV_RANGES:
                    final_mask = cv2.bitwise_or(final_mask, cv2.inRange(hsv_roi, lower, upper))

                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color_ok = bool(contours)

                diameter_mm = 0
                area_mm2 = 0
                size_ok = False
                if color_ok:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area_pixels = cv2.contourArea(largest_contour)
                    if area_pixels >= 200:
                        pixel_width = x2 - x1
                        if pixel_width > 0:
                            mm_per_pixel = KNOWN_WIDTHS_MM['apple'] / pixel_width
                            diameter_mm = (math.sqrt(area_pixels / math.pi) * 2) * mm_per_pixel
                            area_mm2 = area_pixels * (mm_per_pixel ** 2)

                            if (DIAMETER_THRESHOLDS['min'] <= diameter_mm <= DIAMETER_THRESHOLDS['max']) and \
                                    (AREA_THRESHOLDS['min'] <= area_mm2 <= AREA_THRESHOLDS['max']):
                                size_ok = True

                # --- 2. 収集した情報から最終判定 ---
                final_text = ""
                final_color = (0, 0, 0)
                if has_defect:
                    final_color = (0, 165, 255);
                    final_text = "NG (傷あり)"
                elif not color_ok:
                    final_color = (0, 0, 255);
                    final_text = "NG (色が範囲外)"
                elif not size_ok:
                    final_color = (0, 0, 255);
                    final_text = f"NG(サイズ) D:{diameter_mm:.1f} A:{area_mm2:.1f}"
                else:
                    final_color = (0, 255, 0);
                    final_text = f"OK D:{diameter_mm:.1f} A:{area_mm2:.1f}"

                # --- 3. 最終結果と傷のバウンディングボックスを描画 ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), final_color, 2)
                frame = draw_text_japanese(frame, final_text, (x1, y1 - 30), 22, final_color)

                if has_defect:
                    for dx1, dy1, dx2, dy2 in detected_defects:
                        cv2.rectangle(frame, (x1 + dx1, y1 + dy1), (x1 + dx2, y1 + dy2), (0, 165, 255), 2)

        cv2.imshow("Apple Sorter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("果物品質管理AI 匠ソーター - リアルタイム検出")
    if load_settings():
        main_detector()