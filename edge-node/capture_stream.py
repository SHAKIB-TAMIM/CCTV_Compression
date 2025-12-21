#!/usr/bin/env python3
"""
capture_stream.py - edge node sending ROI-aware compressed frames
Features:
 - ROI-aware JPEG encoding (background downscale + high-quality ROI crops)
 - Motion detection (Otsu / adaptive) + motion percent
 - Face detection via Detector (user-provided)
 - POST /motion and POST /sample support for evaluation pipeline
 - Remote control via Socket.IO (bg_scale/bg_quality/roi_quality/detect_every_n/save_sample/experiment_id)
 - Graceful handling and helpful debug prints
"""
import cv2
import time
import base64
import socketio
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import argparse
import sys
import os
from detector import Detector  # keep your detector implementation in detector.py

DEFAULT_SERVER = "http://127.0.0.1:5000"
NAMESPACE = "/stream"
CAM_INDEX = 0
TARGET_FPS = 12

# Socket.IO client with auto-reconnect
sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=1)

# default control params (will be updated by server via 'control' messages)
control = {
    "BG_SCALE": 0.5,
    "BG_QUALITY": 20,
    "ROI_QUALITY": 90,
    "DETECT_EVERY_N": 3,
    "SAVE_SAMPLE": False,
    "EXPERIMENT_ID": ""
}

def jpeg_encode_b64(bgr_img, quality=80):
    """Encode BGR numpy image to base64 JPEG string (safe)."""
    if bgr_img is None:
        raise ValueError("Empty image (None) passed to jpeg_encode_b64")
    if isinstance(bgr_img, np.ndarray) and bgr_img.size == 0:
        raise ValueError("Empty numpy array passed to jpeg_encode_b64")
    try:
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        buff = BytesIO()
        pil.save(buff, format='JPEG', quality=int(quality))
        return base64.b64encode(buff.getvalue()).decode('ascii')
    except Exception as e:
        print("JPEG encode error:", e)
        return ""

# ---------------- Socket.IO callbacks ----------------
@sio.event(namespace=NAMESPACE)
def connect():
    print("[socketio] Connected to server")

@sio.event(namespace=NAMESPACE)
def disconnect():
    print("[socketio] Disconnected from server")

@sio.on('control', namespace=NAMESPACE)
def on_control(msg):
    """Update control parameters from server messages.
    Accepts: bg_scale, bg_quality, roi_quality, detect_every_n, save_sample, experiment_id
    """
    try:
        mapping = {
            'bg_scale': 'BG_SCALE',
            'bg_quality': 'BG_QUALITY',
            'roi_quality': 'ROI_QUALITY',
            'detect_every_n': 'DETECT_EVERY_N',
            'save_sample': 'SAVE_SAMPLE',
            'experiment_id': 'EXPERIMENT_ID'
        }
        changed = False
        if not isinstance(msg, dict):
            print("[control] ignored non-dict message:", msg)
            return
        for k, dest in mapping.items():
            if k in msg:
                try:
                    if isinstance(control[dest], bool):
                        control[dest] = bool(msg[k])
                    else:
                        control[dest] = type(control[dest])(msg[k])
                    changed = True
                except Exception:
                    # best-effort parsing
                    try:
                        control[dest] = int(msg[k])
                        changed = True
                    except Exception:
                        control[dest] = msg[k]
                        changed = True
        if changed:
            print("[control] updated:", control)
    except Exception as e:
        print("[control] parse error:", e)

# ---------------- Motion detection & helpers ----------------
def detect_motion(prev_gray, curr_gray, min_area=500):
    """Frame-diff motion detection. Returns motion_percent, list of ROIs, thresh image."""
    frame_delta = cv2.absdiff(prev_gray, curr_gray)
    # adaptive thresholding via Otsu
    try:
        _, thresh = cv2.threshold(frame_delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rois.append([int(x), int(y), int(x + w), int(y + h)])
    motion_percent = 0.0
    try:
        motion_percent = 100.0 * (np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1]))
    except Exception:
        motion_percent = 0.0
    return motion_percent, rois, thresh

def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2

def post_motion(server_base, experiment_id, frame_id, motion_percent, rois):
    """Fire-and-forget POST to /motion (short timeout)."""
    url = server_base.rstrip('/') + "/motion"
    payload = {
        "experiment_id": experiment_id,
        "frame_id": frame_id,
        "motion_percent": round(float(motion_percent), 4),
        "rois": [{"bbox": r} for r in rois]
    }
    try:
        requests.post(url, json=payload, timeout=2)
    except Exception:
        # non-fatal — server may be unreachable during tests
        pass

def post_sample_to_server(server_base, experiment_id, frame_id, orig_img, recon_img, rois, meta):
    """POST sample to server (/sample). Encodes images as base64 JPEGs."""
    url = server_base.rstrip('/') + "/sample"
    try:
        orig_b64 = jpeg_encode_b64(orig_img, quality=95)
        recon_b64 = jpeg_encode_b64(recon_img, quality=90)
        payload = {
            "experiment_id": experiment_id,
            "frame_id": frame_id,
            "timestamp": time.time(),
            "orig_b64": orig_b64,
            "recon_b64": recon_b64,
            "rois": [{"bbox": r} for r in rois],
            "meta": meta
        }
        requests.post(url, json=payload, timeout=5)
        print("[sample] posted to server:", experiment_id, frame_id)
    except Exception as e:
        print("[sample] post error:", e)

def reconstruct_background_with_rois(frame, rois, bg_scale):
    """Simulate reconstruction: downscale background then paste high-quality ROIs back."""
    h, w = frame.shape[:2]
    bg_w = max(1, int(w * float(bg_scale)))
    bg_h = max(1, int(h * float(bg_scale)))
    try:
        bg_small = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
        bg_up = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        bg_up = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    recon = bg_up.copy()
    for r in rois:
        x1, y1, x2, y2 = [int(x) for x in r]
        try:
            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            recon[y1:y2, x1:x2] = cv2.resize(crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        except Exception:
            continue
    return recon

# ---------------- Main loop ----------------
def main(server_url, cam_index, target_fps):
    # connect socketio (non-blocking)
    try:
        sio.connect(server_url, namespaces=[NAMESPACE])
    except Exception as e:
        print("[socketio] connect error (continuing):", e)

    detector = Detector(model_type='face')  # your Detector implementation
    cap = cv2.VideoCapture(cam_index)

    # Helpful debug: print which device path we're trying (if available)
    try:
        print(f"[camera] attempting index {cam_index}")
    except Exception:
        pass

    # If camera can't open, try to attempt a couple fallbacks (0 and 1) before giving up
    if not cap.isOpened():
        print("[camera] cannot open camera index", cam_index, "- trying fallback indices 0 and 1")
        for idx in (0, 1):
            try:
                cap.release()
            except:
                pass
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print("[camera] opened fallback index", idx)
                break
        else:
            print("[camera] failed to open any fallback device. Exiting.")
            return

    prev_gray = None
    frame_id = 0
    last_send = time.time()

    try:
        while True:
            ret, frame = cap.read()
            print("FRAME:", ret, frame.shape if ret else None)
            if not ret or frame is None:
                # short wait, avoid busy spin
                time.sleep(0.05)
                continue
            vis_frame = frame.copy()

            frame_id += 1
            h, w = frame.shape[:2]

            # background downscale
            bg_w = max(1, int(w * float(control['BG_SCALE'])))
            bg_h = max(1, int(h * float(control['BG_SCALE'])))
            try:
                bg = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
            except Exception:
                bg = frame.copy()

            bg_b64 = jpeg_encode_b64(bg, quality=int(control['BG_QUALITY'])) if bg is not None else ""
            # Encode visualization frame (ROI + motion overlay)
            vis_b64 = jpeg_encode_b64(vis_frame, quality=60)


            motion_percent = 0.0
            rois = []

            # Run detection every N frames
            if frame_id % max(1, int(control['DETECT_EVERY_N'])) == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception:
                    gray = None

                if prev_gray is not None and gray is not None:
                    try:
                        motion_percent, rois, thresh = detect_motion(prev_gray, gray)
                    except Exception:
                        motion_percent = 0.0
                        rois = []

                # Face detection adds ROIs (best-effort)
                try:
                    faces = detector.detect(frame)
                    for f in faces:
                        if len(f) == 4:
                            x, y, ww, hh = f
                            rois.append([int(x), int(y), int(x + ww), int(y + hh)])
                        elif len(f) >= 4:
                            rois.append([int(f[0]), int(f[1]), int(f[2]), int(f[3])])
                except Exception:
                    pass

                # Clamp/filter ROIs to frame bounds
                filtered = []
                for r in rois:
                    try:
                        x1, y1, x2, y2 = clamp_bbox(r[0], r[1], r[2], r[3], w, h)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        if (x2 - x1) < 4 or (y2 - y1) < 4:
                            continue
                        filtered.append([x1, y1, x2, y2])
                    except Exception:
                        continue
                rois = filtered

                # ============================
                # ROI VISUALIZATION OVERLAY
                # ============================
                vis_frame = frame.copy()

                for r in rois:
                    x1, y1, x2, y2 = r
                    cv2.rectangle(
                        vis_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),  # green box
                        2
                    )

                # Show motion percentage on frame
                cv2.putText(
                    vis_frame,
                    f"Motion: {motion_percent:.2f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


                # Send motion report (non-blocking, short timeout)
                try:
                    post_motion(server_url, control.get('EXPERIMENT_ID', ''), frame_id, motion_percent, rois)
                except Exception:
                    pass

            # update previous gray for next diff
            try:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                prev_gray = None

            # encode ROI crops
            roi_msgs = []
            for roi in rois:
                x1, y1, x2, y2 = roi
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                try:
                    roi_b64 = jpeg_encode_b64(crop, quality=int(control['ROI_QUALITY']))
                    if roi_b64:
                        roi_msgs.append({"bbox": [x1, y1, x2, y2], "data": roi_b64})
                except Exception:
                    continue

            # Build message and emit to server namespace
            msg = {
                "frame_id": frame_id,
                "timestamp": time.time(),
                "orig_w": w,
                "orig_h": h,
                "bg_data": bg_b64,
                "vis_frame": vis_b64,
                "rois": roi_msgs
            }
            try:
                sio.emit('frame', msg, namespace=NAMESPACE)
            except Exception:
                # not fatal — server may be offline
                pass

            # If SAVE_SAMPLE requested: reconstruct and POST /sample, then reset flag
            if control.get('SAVE_SAMPLE', False):
                try:
                    recon = reconstruct_background_with_rois(frame, rois, float(control['BG_SCALE']))
                    meta = {
                        "bg_scale": control['BG_SCALE'],
                        "bg_quality": control['BG_QUALITY'],
                        "roi_quality": control['ROI_QUALITY'],
                        "motion_percent": motion_percent,
                        "num_rois": len(rois)
                    }
                    post_sample_to_server(server_url, control.get('EXPERIMENT_ID', ''), frame_id, frame, recon, rois, meta)
                except Exception as e:
                    print("[sample] reconstruction error:", e)
                finally:
                    # make sure we clear the flag so evaluate.py doesn't wait forever
                    control['SAVE_SAMPLE'] = False
            # Optional local preview (for debugging / viva)

            # cv2.imshow("Edge Node - Live ROI Stream", vis_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # FPS limiter
            elapsed = time.time() - last_send
            sleep_time = max(0, (1.0 / float(target_fps)) - elapsed)
            time.sleep(sleep_time)
            last_send = time.time()

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print("[main] runtime error:", e)
    finally:
        try:
            cap.release()
        except:
            pass
        try:
            sio.disconnect()
        except:
            pass
        print("Exiting capture_stream.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge node: capture and stream ROI-aware compressed frames")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="Socket.IO server URL")
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index (default 0)")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="Target FPS")
    args = parser.parse_args()

    main(args.server, args.cam, args.fps)
