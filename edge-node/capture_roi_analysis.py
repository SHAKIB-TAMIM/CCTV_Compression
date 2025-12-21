import cv2
import numpy as np

# ==== SETTINGS ====
ROI_QUALITY = 80
BG_QUALITY = 20
ROI_BOX = (100, 100, 300, 300)  # (x, y, w, h)
# ===================

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Cannot capture frame from camera")
    exit()

original = frame.copy()

# ====== ROI Mask ======
x, y, w, h = ROI_BOX
mask = np.zeros(frame.shape[:2], dtype="uint8")
mask[y:y+h, x:x+w] = 255  # ROI region

# ====== Compress ROI (High Quality) ======
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), ROI_QUALITY]
_, roi_encoded = cv2.imencode(".jpg", frame, encode_param)
roi_high = cv2.imdecode(roi_encoded, cv2.IMREAD_COLOR)

# ====== Compress Background (Low Quality) ======
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), BG_QUALITY]
_, bg_encoded = cv2.imencode(".jpg", frame, encode_param)
bg_low = cv2.imdecode(bg_encoded, cv2.IMREAD_COLOR)

# ====== Merge ROI + Background ======
final = bg_low.copy()
final[mask == 255] = roi_high[mask == 255]

# ====== Draw ROI Bounding Box ======
final_with_box = final.copy()
cv2.rectangle(final_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)

# ====== Artifact Heatmap (Difference Map) ======
diff = cv2.absdiff(original, final)  
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Apply color heatmap (JET colormap)
heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

# Blend heatmap with original frame (for visualization)
heatmap_overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

# ====== Save Files ======
cv2.imwrite("original.jpg", original)
cv2.imwrite("roi_high_quality.jpg", roi_high)
cv2.imwrite("background_low_quality.jpg", bg_low)
cv2.imwrite("compressed_final.jpg", final)
cv2.imwrite("compressed_with_box.jpg", final_with_box)
cv2.imwrite("artifact_heatmap.png", heatmap)
cv2.imwrite("artifact_overlay.png", heatmap_overlay)

print(" Images saved:")
print(" - original.jpg")
print(" - roi_high_quality.jpg")
print(" - background_low_quality.jpg")
print(" - compressed_final.jpg")
print(" - compressed_with_box.jpg   (ROI bounding box)")
print(" - artifact_heatmap.png      (heatmap only)")
print(" - artifact_overlay.png      (heatmap blended with original)\n")

cap.release()
