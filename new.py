import cv2
import time
import os
from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "number_recogintion.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Model file not found!\n"
        f"   Expected path: {MODEL_PATH}\n"
        f"   Place 'V1_number_ercgonition.pt' in the same folder as this script."
    )

model = YOLO(MODEL_PATH)
print(f"[OK] Model loaded: {MODEL_PATH}")

# =========================
# WORD → NUMBER MAP
# =========================
word_to_num = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
}

CONF_THRESHOLD = 0.30


INFER_SIZE = 320

# CAMERA INDEX
# 0 = default built-in webcam
# 1, 2 ... = external USB cameras
CAMERA_INDEX = 0

# CAMERA RESOLUTION
CAMERA_W = 640
CAMERA_H = 480

# ============================================================

# =========================
# COLORS & STYLE
# =========================
BOX_COLOR  = (0, 220, 120)   # green bounding box
LABEL_BG   = (0, 180, 90)    # label background
LABEL_TEXT = (255, 255, 255) # label text color
PANEL_BG   = (15, 15, 25)    # bottom HUD background (dark navy)
SUM_COLOR  = (0, 200, 255)   # cyan sum number
WHITE      = (255, 255, 255)
GRAY       = (160, 160, 160)

# =========================
# HELPER FUNCTIONS
# =========================
def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, r=8):
    """Draw a rounded-corner bounding box."""
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0, 0, 90, color, thickness)


def draw_label(img, text, x, y):
    """Draw a filled label badge above the bounding box."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    cv2.rectangle(img,
                  (x - pad,      y - th - pad * 2),
                  (x + tw + pad, y + pad),
                  LABEL_BG, -1)
    cv2.putText(img, text, (x, y - pad),
                font, font_scale, LABEL_TEXT, thickness, cv2.LINE_AA)


def draw_hud(img, numbers, total, fps, frame_w, frame_h):
    """Draw the bottom info panel (HUD)."""
    panel_h = 90
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (0, frame_h - panel_h),
                  (frame_w, frame_h),
                  PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # FPS — top right of panel
    cv2.putText(img, f"FPS: {fps:.0f}",
                (frame_w - 110, frame_h - panel_h + 25),
                font, 0.55, GRAY, 1, cv2.LINE_AA)

    # Detected numbers — bottom left
    nums_str = "  +  ".join(str(n) for n in numbers) if numbers else "-"
    cv2.putText(img, "DETECTED",
                (20, frame_h - panel_h + 28),
                font, 0.50, GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, nums_str,
                (20, frame_h - panel_h + 58),
                font, 0.85, WHITE, 2, cv2.LINE_AA)

    # Sum — bottom right (large cyan)
    sum_val = str(total)
    cv2.putText(img, "SUM",
                (frame_w - 140, frame_h - panel_h + 28),
                font, 0.50, GRAY, 1, cv2.LINE_AA)
    (vw, _), _ = cv2.getTextSize(sum_val, font, 1.6, 3)
    cv2.putText(img, sum_val,
                (frame_w - 20 - vw, frame_h - 14),
                font, 1.6, SUM_COLOR, 3, cv2.LINE_AA)

    # Divider line
    cv2.line(img,
             (0, frame_h - panel_h),
             (frame_w, frame_h - panel_h),
             (50, 50, 70), 1)


# =========================
# START CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

prev_time = time.time()
print("[OK] Webcam started — press Q to quit")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera read failed")
        break

    frame_h, frame_w = frame.shape[:2]

    # Scale factors: model runs on small frame, boxes drawn on full frame
    scale_x = frame_w / INFER_SIZE
    scale_y = frame_h / INFER_SIZE

    small_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))

    # -------------------------------------------------------
    # Run YOLO inference
    # -------------------------------------------------------
    results = model.predict(
        small_frame,
        imgsz=INFER_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    numbers = []

    # -------------------------------------------------------
    # Process detections
    # -------------------------------------------------------
    for result in results:
        for box in result.boxes:
            cls        = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = model.names[cls].lower()

            if class_name in word_to_num:
                numbers.append(word_to_num[class_name])

            # Scale coordinates back to full-resolution frame
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_x);  y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x);  y2 = int(y2 * scale_y)

            draw_rounded_rect(frame, x1, y1, x2, y2, BOX_COLOR, 2)
            draw_label(frame, f"{class_name}  {conf:.0%}", x1, y1)

    # -------------------------------------------------------
    # HUD
    # -------------------------------------------------------
    total = sum(numbers)

    now       = time.time()
    fps       = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    draw_hud(frame, numbers, total, fps, frame_w, frame_h)

    print(f"Detected: {numbers}  |  Sum = {total}  |  FPS: {fps:.1f}")

    # -------------------------------------------------------
    # Display
    # -------------------------------------------------------
    cv2.imshow("Number Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
print("[OK] Camera closed.")