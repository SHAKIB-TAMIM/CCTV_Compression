#!/usr/bin/env python3
"""
evaluate.py (upgraded)
- Requests edge node to save orig/recon sample pairs using /control (save_sample)
- Polls server /samples/<experiment_id> for saved artifacts
- Downloads artifacts and computes:
    - global PSNR, SSIM
    - per-ROI PSNR, SSIM
    - background PSNR, SSIM
- Writes detailed rows to results.csv and saves images + heatmaps per experiment
- Generates summary plots
"""

import requests
import os
import time
import csv
import io
import base64
import json
from datetime import datetime
import numpy as np
import cv2
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import matplotlib.pyplot as plt
import pandas as pd

# ----------------- Configuration -----------------
SERVER = "http://127.0.0.1:5000"
CONTROL_URL = SERVER + "/control"
METRICS_URL = SERVER + "/metrics"
SAMPLES_URL = SERVER + "/samples"        # list
SAMPLE_GET = SERVER + "/samples/{}"     # GET /samples/<id>

OUTPUT_CSV = "results.csv"
EXPERIMENTS = [
    {"bg_quality": 10, "roi_quality": 70},
    {"bg_quality": 30, "roi_quality": 80},
    {"bg_quality": 50, "roi_quality": 90},
]

# timeouts and waits
SAMPLE_POLL_INTERVAL = 1.0   # seconds
SAMPLE_POLL_TIMEOUT = 25.0   # seconds to wait for sample

# image save path
EXPERIMENTS_DIR = "experiments_eval"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# ----------------- Helpers -----------------
def now_str():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def post_control(bg_quality, roi_quality, experiment_id):
    payload = {
        "bg_quality": bg_quality,
        "roi_quality": roi_quality,
        "experiment_id": experiment_id,
        "save_sample": True
    }
    try:
        r = requests.post(CONTROL_URL, json=payload, timeout=5)
        return r.ok
    except Exception as e:
        print("post_control error:", e)
        return False

def list_samples():
    try:
        r = requests.get(SAMPLES_URL, timeout=5)
        if r.ok:
            return r.json().get("experiments", [])
    except Exception as e:
        # print("list_samples error:", e)
        return []
    return []

def fetch_sample_meta(experiment_id):
    try:
        r = requests.get(SAMPLE_GET.format(experiment_id), timeout=10)
        if r.ok:
            return r.json()
    except Exception as e:
        # print("fetch_sample_meta error:", e)
        return None
    return None

def download_dataurl(dataurl, out_path):
    """Given data:image/...;base64,<b64> or plain base64, save to out_path"""
    if dataurl is None:
        return False
    if isinstance(dataurl, bool):
        return False
    if dataurl.startswith("data:"):
        # format like data:image/jpeg;base64,<b64>
        parts = dataurl.split(",", 1)
        if len(parts) != 2:
            return False
        b64 = parts[1]
    else:
        b64 = dataurl
    try:
        b = base64.b64decode(b64)
        with open(out_path, "wb") as f:
            f.write(b)
        return True
    except Exception as e:
        print("download_dataurl error:", e)
        return False

def compute_psnr(a, b):
    # expect uint8 arrays same shape
    try:
        return float(sk_psnr(a, b, data_range=255))
    except Exception:
        # fallback manual
        mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        if mse == 0:
            return float("inf")
        return 10.0 * np.log10((255.0 ** 2) / mse)

def compute_ssim(a_gray, b_gray, mask=None):
    # sk_ssim supports mask parameter in recent versions; try, otherwise fallback to whole-image SSIM
    try:
        if mask is not None:
            # mask should be boolean where True means evaluate that region
            return float(sk_ssim(a_gray, b_gray, data_range=255, gaussian_weights=True, use_sample_covariance=False, mask=mask))
        else:
            return float(sk_ssim(a_gray, b_gray, data_range=255))
    except TypeError:
        # older skimage may not accept mask; fallback to full-image SSIM
        try:
            return float(sk_ssim(a_gray, b_gray, data_range=255))
        except Exception as e:
            print("SSIM compute error:", e)
            return None
    except Exception as e:
        print("SSIM compute error:", e)
        return None

def mask_background_metrics(orig, recon, rois):
    """
    Compute PSNR and SSIM for background (pixels outside all rois).
    orig, recon are BGR uint8 arrays.
    rois: list of [x1,y1,x2,y2]
    """
    h, w = orig.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    for r in rois:
        x1,y1,x2,y2 = [int(v) for v in r]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        mask[y1:y2, x1:x2] = False

    if not np.any(mask):
        return None, None

    # PSNR: compute mse only over mask pixels (all channels)
    a = orig.copy()
    b = recon.copy()
    # flatten masked pixels across channels
    a_masked = a[mask]
    b_masked = b[mask]
    if a_masked.size == 0 or b_masked.size == 0:
        return None, None

    mse = np.mean((a_masked.astype(np.float32) - b_masked.astype(np.float32)) ** 2)
    psnr_val = float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)

    # ssim: convert to gray and create mask for ssim (boolean mask)
    try:
        a_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
        # skimage expects mask where True indicates pixel to include
        ssim_val = compute_ssim(a_gray, b_gray, mask=mask)
    except Exception:
        ssim_val = None

    return psnr_val, ssim_val

def per_roi_metrics(orig, recon, rois):
    """
    For each ROI compute PSNR and SSIM. Returns list of dicts.
    """
    out = []
    h, w = orig.shape[:2]
    try:
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        recon_gray = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
    except Exception:
        orig_gray = None
        recon_gray = None

    for r in rois:
        x1,y1,x2,y2 = [int(v) for v in r]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        o_crop = orig[y1:y2, x1:x2]
        r_crop = recon[y1:y2, x1:x2]
        if o_crop is None or r_crop is None or o_crop.size == 0 or r_crop.size == 0:
            continue
        ps = compute_psnr(o_crop, r_crop)
        ss = None
        if orig_gray is not None:
            try:
                o_g = orig_gray[y1:y2, x1:x2]
                r_g = recon_gray[y1:y2, x1:x2]
                ss = compute_ssim(o_g, r_g, mask=None)
            except Exception:
                ss = None
        out.append({"bbox": [x1,y1,x2,y2], "psnr": ps, "ssim": ss})
    return out

def make_diff_heatmap(orig, recon, out_path):
    """Make absolute difference heatmap and save."""
    try:
        diff = cv2.absdiff(orig, recon)
        # convert to grayscale diff
        dgray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # normalize
        nm = cv2.normalize(dgray, None, 0, 255, cv2.NORM_MINMAX)
        col = cv2.applyColorMap(nm.astype('uint8'), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.6, col, 0.4, 0)
        cv2.imwrite(out_path, overlay)
        return True
    except Exception as e:
        print("heatmap error:", e)
        return False

def overlay_rois(orig, rois, out_path):
    img = orig.copy()
    for r in rois:
        x1,y1,x2,y2 = [int(v) for v in r]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(out_path, img)
    return True

# ----------------- Main experiment loop -----------------
def save_results_row(csv_path, row_dict, fieldnames):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

def fetch_latest_metrics():
    try:
        r = requests.get(METRICS_URL, timeout=5)
        if r.ok:
            arr = r.json()
            if isinstance(arr, list) and len(arr) > 0:
                return arr[-1]
            elif isinstance(arr, dict):
                return arr
    except Exception:
        pass
    return {}

def wait_for_sample(experiment_id, timeout=SAMPLE_POLL_TIMEOUT):
    """Poll server /samples/<experiment_id> until orig.jpg & recon.jpg are present."""
    start = time.time()
    while time.time() - start < timeout:
        meta = fetch_sample_meta(experiment_id)
        if meta:
            # meta includes keys like 'orig.jpg' and 'recon.jpg' when files present
            has_orig = 'orig.jpg' in meta
            has_recon = 'recon.jpg' in meta
            # rois.json and meta.json may also exist
            if has_orig and has_recon:
                return meta
        time.sleep(SAMPLE_POLL_INTERVAL)
    return None

def download_and_save_artifacts(experiment_id, meta):
    """Save orig.jpg, recon.jpg, rois.json, meta.json locally under experiments_eval/<experiment_id>/"""
    d = os.path.join(EXPERIMENTS_DIR, experiment_id)
    os.makedirs(d, exist_ok=True)
    saved = {}
    for fn, val in meta.items():
        # val might be data URL for images or parsed JSON for .json files
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            out_path = os.path.join(d, fn)
            ok = download_dataurl(val, out_path)
            saved[fn] = out_path if ok else None
        elif fn.endswith('.json'):
            out_path = os.path.join(d, fn)
            try:
                with open(out_path, "w") as f:
                    json.dump(val, f, indent=2)
                saved[fn] = out_path
            except Exception as e:
                saved[fn] = None
        else:
            # skip other entries
            pass
    return d, saved

def run_experiment(bg_quality, roi_quality):
    name = f"BG{bg_quality}_ROI{roi_quality}"
    ts = now_str()
    experiment_id = f"{name}_{ts}"
    print("\n=== Running experiment:", experiment_id, "===")

    # send control to request SAVE_SAMPLE with experiment_id
    ok = post_control(bg_quality, roi_quality, experiment_id)
    if not ok:
        print("Failed to send control, aborting experiment.")
        return None

    print("Requested sample. Waiting for server to receive artifacts...")
    meta = wait_for_sample(experiment_id, timeout=SAMPLE_POLL_TIMEOUT)
    if meta is None:
        print("Timeout waiting for sample for experiment", experiment_id)
        return None

    # download and save
    saved_dir, saved = download_and_save_artifacts(experiment_id, meta)
    orig_local = saved.get('orig.jpg') or saved.get('orig.jpeg') or saved.get('orig.png')
    recon_local = saved.get('recon.jpg') or saved.get('recon.jpeg') or saved.get('recon.png')
    rois_local = saved.get('rois.json') or saved.get('meta.json')

    if not orig_local or not recon_local:
        print("Artifacts missing:", saved)
        return None

    # load images
    orig = cv2.imread(orig_local)
    recon = cv2.imread(recon_local)
    if orig is None or recon is None:
        print("Failed to read orig/recon images.")
        return None

    # ensure same size for metrics: resize recon to orig
    try:
        if (orig.shape[0] != recon.shape[0]) or (orig.shape[1] != recon.shape[1]):
            recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
    except Exception:
        pass

    # parse rois
    rois = []
    if rois_local and os.path.exists(rois_local):
        try:
            with open(rois_local) as f:
                j = json.load(f)
                # server saves 'rois.json' as list of {"bbox":[..]}
                if isinstance(j, list):
                    for item in j:
                        bbox = item.get('bbox') if isinstance(item, dict) else None
                        if bbox:
                            rois.append(bbox)
                elif isinstance(j, dict):
                    # try meta.json structure
                    if 'rois' in j and isinstance(j['rois'], list):
                        for item in j['rois']:
                            if isinstance(item, dict) and 'bbox' in item:
                                rois.append(item['bbox'])
        except Exception as e:
            print("Failed to parse rois.json:", e)
            rois = []
    else:
        rois = []

    # compute global metrics
    global_psnr = compute_psnr(orig, recon)
    try:
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        recon_gray = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
        global_ssim = compute_ssim(orig_gray, recon_gray, mask=None)
    except Exception:
        global_ssim = None

    # per-roi metrics
    roi_results = per_roi_metrics(orig, recon, rois)
    roi_psnrs = [r['psnr'] for r in roi_results if r.get('psnr') is not None]
    roi_ssims = [r['ssim'] for r in roi_results if r.get('ssim') is not None]

    roi_psnr_mean = float(np.mean(roi_psnrs)) if roi_psnrs else None
    roi_psnr_min = float(np.min(roi_psnrs)) if roi_psnrs else None
    roi_ssim_mean = float(np.mean(roi_ssims)) if roi_ssims else None

    # background metrics
    bg_psnr, bg_ssim = mask_background_metrics(orig, recon, rois)

    # save visual artifacts
    heatmap_path = os.path.join(saved_dir, "diff_heatmap.jpg")
    overlay_path = os.path.join(saved_dir, "rois_overlay.jpg")
    _ = make_diff_heatmap(orig, recon, heatmap_path)
    _ = overlay_rois(orig, rois, overlay_path)

    # fetch latest server metrics for context
    latest_metrics = fetch_latest_metrics()
    latest_kbps = latest_metrics.get("total_kbps") if latest_metrics else None
    latest_clients = latest_metrics.get("total_clients") if latest_metrics else None

    # Build result dict
    row = {
        "experiment": experiment_id,
        "timestamp": datetime.utcnow().isoformat(),
        "bg_quality": bg_quality,
        "roi_quality": roi_quality,
        "clients": latest_clients,
        "kbps": latest_kbps,
        "global_psnr": global_psnr,
        "global_ssim": global_ssim if global_ssim is not None else "",
        "roi_psnr_mean": roi_psnr_mean if roi_psnr_mean is not None else "",
        "roi_psnr_min": roi_psnr_min if roi_psnr_min is not None else "",
        "roi_ssim_mean": roi_ssim_mean if roi_ssim_mean is not None else "",
        "bg_psnr": bg_psnr if bg_psnr is not None else "",
        "bg_ssim": bg_ssim if bg_ssim is not None else "",
        "saved_dir": saved_dir
    }

    # append per-ROI details to a small JSON inside saved_dir
    try:
        with open(os.path.join(saved_dir, "metrics_details.json"), "w") as f:
            json.dump({
                "global": {"psnr": global_psnr, "ssim": global_ssim},
                "roi_results": roi_results,
                "background": {"psnr": bg_psnr, "ssim": bg_ssim},
                "server_metrics": latest_metrics
            }, f, indent=2)
    except Exception:
        pass

    print("Experiment result:", row)
    return row

def generate_summary_plots(csv_path="results.csv"):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("No results to plot.")
            return
        # ensure numeric
        for col in ["kbps", "global_psnr", "roi_psnr_mean", "bg_psnr"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # plot kbps and psnr
        plt.figure(figsize=(10,6))
        ax = plt.gca()
        if "kbps" in df.columns:
            df.plot(x='experiment', y='kbps', kind='bar', ax=ax, color='tab:blue', label='Avg Kbps')
            ax.set_ylabel('Avg Kbps', color='tab:blue')
        ax2 = ax.twinx()
        if "global_psnr" in df.columns:
            df.plot(x='experiment', y='global_psnr', kind='line', marker='o', ax=ax2, color='tab:orange', label='Global PSNR')
            ax2.set_ylabel('PSNR', color='tab:orange')
        plt.title("Bandwidth vs PSNR per experiment")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("summary_bandwidth_psnr.png")
        plt.close()
        print("Saved summary_bandwidth_psnr.png")
    except Exception as e:
        print("generate_summary_plots error:", e)

# ----------------- Script Entry -----------------
def main():
    print("=== Artifact-backed Evaluator ===")
    duration = int(input("Duration per test (seconds, used only for initial wait): ") or 10)
    interval = int(input("Metrics fetch interval (seconds): ") or 2)
    print("Waiting short period to stabilize metrics...")
    time.sleep(2)

    fieldnames = [
        "experiment","timestamp","bg_quality","roi_quality","clients","kbps",
        "global_psnr","global_ssim","roi_psnr_mean","roi_psnr_min","roi_ssim_mean",
        "bg_psnr","bg_ssim","saved_dir"
    ]

    all_rows = []
    # run each experiment
    for exp in EXPERIMENTS:
        # send control but also let evaluator wait a bit for metrics to stabilize
        row = run_experiment(exp["bg_quality"], exp["roi_quality"])
        if row:
            save_results_row(OUTPUT_CSV, row, fieldnames)
            all_rows.append(row)
        else:
            print("Experiment failed or timed out for preset:", exp)

    # generate plots
    generate_summary_plots(OUTPUT_CSV)
    print("All done. Results saved to", OUTPUT_CSV)
    print("Per-experiment artifacts in", EXPERIMENTS_DIR)

if __name__ == "__main__":
    main()
