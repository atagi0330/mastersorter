import cv2
import numpy as np
import math
import json
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# --- 基本設定 ---
CONFIG_FILE = 'config.json'
FOCAL_LENGTH = 1200
KNOWN_WIDTHS_MM = {'apple': 90.0}
FONT_PATH = 'NotoSansJP-VariableFont_wght.ttf'


def save_config(data):
    """設定データをJSONファイルに保存する"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"✅ 設定を {CONFIG_FILE} に保存しました。")
    except IOError as e:
        print(f"エラー: {CONFIG_FILE} への書き込みに失敗しました。 - {e}")


def draw_text_japanese(frame, text, position, font_size, color):
    """日本語テキストを描画するヘルパー関数"""
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        print(f"警告: フォントファイル '{FONT_PATH}' が見つかりません。英語で表示します。")
        cv2.putText(frame, "Font file not found.", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def run_calibration(model):
    """ご指定の計算方法で、お手本のリンゴの直径・面積・色を登録する"""
    print("\n--- 基準リンゴの自動キャリブレーション ---")

    try:
        apple_class_id = [k for k, v in model.names.items() if v == 'apple'][0]
    except IndexError:
        print("エラー: YOLOモデルのクラス名に 'apple' が見つかりませんでした。")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    calibrated_data = {}
    for sample_type in ['sample1', 'sample2']:  # 'min', 'max'ではなく、単に1つ目、2つ目として登録
        prompt = f"'{sample_type}'のリンゴを写し 's'キーで撮影 (qで中断)"
        while True:
            ret, frame = cap.read()
            if not ret: continue

            frame = draw_text_japanese(frame, prompt, (20, 20), 30, (0, 255, 255))
            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                results = model(frame, device='0', classes=[apple_class_id], verbose=False)
                best_box = None
                if results[0].boxes:
                    best_box = max(results[0].boxes, key=lambda box: box.conf[0])

                if best_box is None or best_box.conf[0] < 0.5:
                    print("リンゴが明確に検出できませんでした。再試行してください。")
                    continue

                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: continue

                # マスクと輪郭の抽出
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv_roi, np.array([0, 100, 50]), np.array([15, 255, 255]))
                mask2 = cv2.inRange(hsv_roi, np.array([165, 100, 50]), np.array([180, 255, 255]))
                final_mask = cv2.bitwise_or(mask1, mask2)

                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    print("輪郭を抽出できませんでした。")
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                area_pixels = cv2.contourArea(largest_contour)
                if area_pixels <= 100: continue

                # 計算ロジック
                known_width_mm = KNOWN_WIDTHS_MM['apple']
                pixel_width = x2 - x1
                if pixel_width <= 0: continue

                mm_per_pixel = known_width_mm / pixel_width
                diameter_mm = (math.sqrt(area_pixels / math.pi) * 2) * mm_per_pixel
                area_mm2 = area_pixels * (mm_per_pixel ** 2)

                h, s, v = cv2.split(hsv_roi)
                h_vals, s_vals, v_vals = h[final_mask > 0], s[final_mask > 0], v[final_mask > 0]

                calibrated_data[sample_type] = {
                    'diameter_mm': diameter_mm,
                    'area_mm2': area_mm2,
                    'hsv_lower': [int(np.percentile(h_vals, 5)), int(np.percentile(s_vals, 5)),
                                  int(np.percentile(v_vals, 5))],
                    'hsv_upper': [int(np.percentile(h_vals, 95)), int(np.percentile(s_vals, 95)),
                                  int(np.percentile(v_vals, 95))]
                }

                print(f"✅ '{sample_type}'基準登録 (直径: {diameter_mm:.1f} mm, 面積: {area_mm2:.1f} mm2)")
                break

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if 'sample1' in calibrated_data and 'sample2' in calibrated_data:
        # ★★★ ここからが修正点 ★★★
        # 登録された2つのサンプルの値を比較し、小さい方をmin、大きい方をmaxに自動で割り当てる
        final_config = {
            "apple_min_diameter_mm": min(calibrated_data['sample1']['diameter_mm'],
                                         calibrated_data['sample2']['diameter_mm']),
            "apple_max_diameter_mm": max(calibrated_data['sample1']['diameter_mm'],
                                         calibrated_data['sample2']['diameter_mm']),

            "apple_min_area_mm2": min(calibrated_data['sample1']['area_mm2'], calibrated_data['sample2']['area_mm2']),
            "apple_max_area_mm2": max(calibrated_data['sample1']['area_mm2'], calibrated_data['sample2']['area_mm2']),

            "apple_hsv_ranges": [
                {
                    # HSVの各値(H, S, V)についても、小さい方の値をlowerに、大きい方の値をupperに設定する
                    'lower': [
                        min(calibrated_data['sample1']['hsv_lower'][i], calibrated_data['sample2']['hsv_lower'][i]) for
                        i in range(3)],
                    'upper': [
                        max(calibrated_data['sample1']['hsv_upper'][i], calibrated_data['sample2']['hsv_upper'][i]) for
                        i in range(3)]
                }
            ]
        }
        # ★★★ 修正ここまで ★★★
        save_config(final_config)
    else:
        print("キャリブレーションが完了しませんでした。")


if __name__ == "__main__":
    model = YOLO('yolov9c.pt')
    run_calibration(model)