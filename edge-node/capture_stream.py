import cv2
import time
import base64
import socketio
from io import BytesIO
from PIL import Image
import numpy as np
from detector import Detector
import argparse

# Default server URL (change if server remote)
DEFAULT_SERVER = "http://192.168.50.60:3001"

NAMESPACE = "/stream"
CAM_INDEX = 0
DETECT_EVERY_N = 3
BG_SCALE = 0.5
BG_QUALITY = 20
ROI_QUALITY = 90
TARGET_FPS = 12

sio = socketio.Client()

def jpeg_encode_b64(bgr_img, quality=80):
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buff = BytesIO()
    pil.save(buff, format='JPEG', quality=int(quality))
    return base64.b64encode(buff.getvalue()).decode('ascii')

control = {
    "BG_SCALE": BG_SCALE,
    "BG_QUALITY": BG_QUALITY,
    "ROI_QUALITY": ROI_QUALITY,
    "DETECT_EVERY_N": DETECT_EVERY_N
}

@sio.event(namespace=NAMESPACE)
def connect():
    print("Connected to server")

@sio.on('control', namespace=NAMESPACE)
def on_control(msg):
    try:
        if 'bg_scale' in msg:
            control['BG_SCALE'] = float(msg['bg_scale'])
        if 'bg_quality' in msg:
            control['BG_QUALITY'] = int(msg['bg_quality'])
        if 'roi_quality' in msg:
            control['ROI_QUALITY'] = int(msg['roi_quality'])
        if 'detect_every_n' in msg:
            control['DETECT_EVERY_N'] = int(msg['detect_every_n'])
        print("Control updated:", control)
    except Exception as e:
        print("Control parse error:", e)

def main(server_url):
    sio.connect(server_url, namespaces=[NAMESPACE])
    detector = Detector(model_type='face')
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_id = 0
    last_send = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame_id += 1
            h, w = frame.shape[:2]

            bg_scale = float(control['BG_SCALE'])
            bg_w = max(1, int(w * bg_scale))
            bg_h = max(1, int(h * bg_scale))
            bg = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
            bg_b64 = jpeg_encode_b64(bg, quality=int(control['BG_QUALITY']))

            rois = []
            if frame_id % int(control['DETECT_EVERY_N']) == 0:
                bboxes = detector.detect(frame)
                for (x,y,ww,hh) in bboxes:
                    x1, y1 = max(0,int(x)), max(0,int(y))
                    x2, y2 = min(w, int(x+ww)), min(h, int(y+hh))
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    roi_b64 = jpeg_encode_b64(crop, quality=int(control['ROI_QUALITY']))
                    rois.append({"bbox":[x1,y1,x2,y2], "data": roi_b64})

            msg = {
                "frame_id": frame_id,
                "timestamp": time.time(),
                "orig_w": w,
                "orig_h": h,
                "bg_scale": bg_scale,
                "bg_data": bg_b64,
                "rois": rois
            }

            sio.emit('frame', msg, namespace=NAMESPACE)

            elapsed = time.time() - last_send
            sleep_for = max(0, (1.0/float(TARGET_FPS)) - elapsed)
            time.sleep(sleep_for)
            last_send = time.time()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        sio.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', help='server URL', default=DEFAULT_SERVER)
    args = parser.parse_args()
    main(args.server)
