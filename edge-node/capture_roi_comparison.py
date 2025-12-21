import cv2
import numpy as np

# ==== SETTINGS ====
ROI_QUALITY = 80       # High quality region (ROI)
BG_QUALITY = 20        # Low quality background
ROI_BOX = (100, 100, 300, 300)  # (x, y, w, h) - adjust if needed
# ===================

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print(" Cannot capture frame from camera")
    exit()

original = frame.copy()

# ---- CREATE ROI MASK ----
x, y, w, h = ROI_BOX
mask = np.zeros(frame.shape[:2], dtype="uint8")
mask[y:y+h, x:x+w] = 255  # ROI = white

# ---- COMPRESS ROI (HIGH QUALITY) ----
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), ROI_QUALITY]
_, roi_encoded = cv2.imencode(".jpg", frame, encode_param)
roi_high = cv2.imdecode(roi_encoded, cv2.IMREAD_COLOR)

# ---- COMPRESS BACKGROUND (LOW QUALITY) ----
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), BG_QUALITY]
_, bg_encoded = cv2.imencode(".jpg", frame, encode_param)
bg_low = cv2.imdecode(bg_encoded, cv2.IMREAD_COLOR)

# ---- MERGE RESULT ----
final = bg_low.copy()
final[mask == 255] = roi_high[mask == 255]

# ---- SAVE FILES ----
cv2.imwrite("original.jpg", original)
cv2.imwrite("roi_high_quality.jpg", roi_high)
cv2.imwrite("background_low_quality.jpg", bg_low)
cv2.imwrite("comparison_side_by_side.jpg",
            np.hstack([original, final]))

print(" Images saved:")
print(" - original.jpg")
print(" - roi_high_quality.jpg")
print(" - background_low_quality.jpg")
print(" - comparison_side_by_side.jpg")

cap.release()
