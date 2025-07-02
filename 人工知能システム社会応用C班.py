import cv2
import numpy as np
import math
from ultralytics import YOLO

# --- 1. モデルと各種設定を準備 ---
model = YOLO('yolov9c.pt')

# カメラの焦点距離を設定
FOCAL_LENGTH = 1200  # <--- 事前に計算した値に書き換えてください

# 検出する果物の平均的な幅（mm単位）
KNOWN_WIDTHS_MM = {
    'apple': 80.0,
    'banana': 40.0,
    'orange': 75.0
}
target_classes = ['apple', 'banana', 'orange']
color_ranges = {
    'red': ([0, 120, 70], [10, 255, 255]), 'red2': ([170, 120, 70], [180, 255, 255]),
    'yellow': ([20, 100, 100], [35, 255, 255]), 'orange': ([10, 100, 100], [20, 255, 255])
}
masks_to_combine = {'red': ['red', 'red2']}
class_to_color = {'apple': 'red', 'banana': 'yellow', 'orange': 'orange'}

# --- 準備ここまで ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    exit()

print("計測を開始します。'q'キーで終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 2. YOLOv9による物体検出 ---
    results = model(frame, device='0', verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in target_classes:
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # --- 3. 距離と大きさの基準値を計算 ---
                    distance_mm = 0
                    mm_per_pixel = 0
                    if class_name in KNOWN_WIDTHS_MM:
                        known_width_mm = KNOWN_WIDTHS_MM[class_name]
                        pixel_width = x2 - x1
                        if pixel_width > 0:
                            distance_mm = (known_width_mm * FOCAL_LENGTH) / pixel_width
                            mm_per_pixel = known_width_mm / pixel_width

                    # --- 4. OpenCVによる輪郭ベースの正確な特徴量計算 ---
                    fruit_roi = frame[y1:y2, x1:x2]
                    if fruit_roi.size == 0: continue
                    hsv_roi = cv2.cvtColor(fruit_roi, cv2.COLOR_BGR2HSV)

                    color_name = class_to_color.get(class_name)
                    if color_name:
                        # (マスク作成ロジック)
                        if color_name in masks_to_combine:
                            mask1 = cv2.inRange(hsv_roi, np.array(color_ranges['red'][0]),
                                                np.array(color_ranges['red'][1]))
                            mask2 = cv2.inRange(hsv_roi, np.array(color_ranges['red2'][0]),
                                                np.array(color_ranges['red2'][1]))
                            final_mask = cv2.bitwise_or(mask1, mask2)
                        else:
                            final_mask = cv2.inRange(hsv_roi, np.array(color_ranges[color_name][0]),
                                                     np.array(color_ranges[color_name][1]))

                        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            area_pixels = cv2.contourArea(largest_contour)
                            if area_pixels > 100:

                                # --- ここからが変更点 ---
                                # 実際の面積(mm^2)を計算
                                area_mm2 = area_pixels * (mm_per_pixel ** 2)

                                # 面積(ピクセル)から直径(mm)を計算
                                if area_pixels > 0:
                                    radius_from_area_pixels = math.sqrt(area_pixels / 3.14)
                                    diameter_pixels = radius_from_area_pixels * 2
                                    diameter_mm = diameter_pixels * mm_per_pixel
                                else:
                                    diameter_mm = 0
                                # --- 変更点ここまで ---

                                # --- 5. 結果の描画 ---
                                (x_circle_roi, y_circle_roi), _ = cv2.minEnclosingCircle(largest_contour)
                                center_in_frame = (int(x_circle_roi + x1), int(y_circle_roi + y1))

                                contour_in_frame = largest_contour + (x1, y1)
                                cv2.drawContours(frame, [contour_in_frame], -1, (0, 255, 0), 3)
                                cv2.circle(frame, center_in_frame, 7, (0, 0, 255), -1)

                                # 情報テキストを描画
                                text_y_pos = center_in_frame[1] - 30
                                cv2.putText(frame, f"{class_name.capitalize()}", (center_in_frame[0] - 40, text_y_pos),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                cv2.putText(frame, f"Distance: {distance_mm:.1f} mm",
                                            (center_in_frame[0] - 60, text_y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 255), 2)
                                cv2.putText(frame, f"Size: {area_mm2:.1f} mm2",
                                            (center_in_frame[0] - 60, text_y_pos + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 255), 2)
                                cv2.putText(frame, f"Diameter: {diameter_mm:.1f} mm",
                                            (center_in_frame[0] - 60, text_y_pos + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 255), 2)

    cv2.imshow("YOLOv9 + Diameter from Area", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
