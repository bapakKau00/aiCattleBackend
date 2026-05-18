"""
Cattle Weight Estimation API
=============================
POST /predict      — upload files directly (multipart/form-data)
POST /predict_url  — send Firebase Storage URLs (application/json)
GET  /health       — health check

Run:
  pip install -r requirements.txt
  python app.py
"""

import os, json, tempfile, traceback
import numpy as np
import cv2
import torch
import requests as http_requests
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_yolo_model    = None
_sam_predictor = None
_rf_model      = None
_rf_scaler     = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO('yolo11n.pt')
    return _yolo_model

def get_sam():
    global _sam_predictor
    if _sam_predictor is None:
        import urllib.request
        from segment_anything import sam_model_registry, SamPredictor
        ckpt = 'sam_vit_b_01ec64.pth'
        if not os.path.exists(ckpt):
            print("Downloading SAM checkpoint...")
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                ckpt
            )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading SAM on {device}...")
        sam = sam_model_registry['vit_b'](checkpoint=ckpt)
        sam.to(device)
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor

def get_rf():
    global _rf_model, _rf_scaler
    if _rf_model is None:
        import joblib
        _rf_model  = joblib.load('rf_model_v3.joblib')
        _rf_scaler = joblib.load('rf_scaler_v3.joblib')
    return _rf_model, _rf_scaler


# ── Helper functions ──────────────────────────────────────────────────────────

def snap_to_pointcloud(u, v, conf, pts_3d, fx, fy, ppx, ppy,
                        conf_thresh=0.3, search_k=50,
                        z_min=None, z_max=None):
    if conf < conf_thresh:
        return None
    working = pts_3d
    if z_min is not None and z_max is not None:
        mask    = (pts_3d[:, 2] >= z_min) & (pts_3d[:, 2] <= z_max)
        working = pts_3d[mask]
        if len(working) < 3:
            return None
    z_vals  = working[:, 2]
    u_proj  = working[:, 0] * fx / z_vals + ppx
    v_proj  = working[:, 1] * fy / z_vals + ppy
    dists   = np.sqrt((u_proj - u)**2 + (v_proj - v)**2)
    k       = min(search_k, len(dists) - 1)
    nearest = np.argpartition(dists, k)[:k]
    best    = nearest[np.argmin(dists[nearest])]
    return working[best]

def find_by_class_id(keypoints, target_id):
    for i, kp in enumerate(keypoints):
        if kp['class_id'] == target_id:
            return i
    return None

def measure_3d(idx_a, idx_b, kps_3d):
    pt_a = kps_3d.get(idx_a)
    pt_b = kps_3d.get(idx_b)
    if pt_a is not None and pt_b is not None:
        return float(np.linalg.norm(pt_a - pt_b))
    return None

def download_url(url, dest_path):
    """Download file from URL and save to dest_path."""
    r = http_requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(r.content)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(rgb_path, depth_path, meta_path):

    # ── Load ──────────────────────────────────────────────────────────────────
    rgb_image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    raw_depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
    with open(meta_path) as f:
        meta = json.load(f)

    h, w = rgb_image.shape[:2]
    print(f"RGB: {w}×{h}  Depth: {raw_depth.shape}  Device: {meta.get('deviceModel','?')}")

    # ── YOLO detection ────────────────────────────────────────────────────────
    AI_W, AI_H = 1024, 768
    rgb_small  = cv2.resize(rgb_image, (AI_W, AI_H))
    scale_bx   = w / AI_W
    scale_by   = h / AI_H

    yolo    = get_yolo()
    results = yolo(rgb_small)
    bbox    = None
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 19:
                bs   = box.xyxy[0].cpu().numpy()
                bbox = bs * np.array([scale_bx, scale_by, scale_bx, scale_by])
                break
    if bbox is None:
        bbox = np.array([0, 0, w, h], dtype=np.float32)
        print("⚠️  No cattle detected — using full image")
    else:
        print(f"✅ Cattle detected")

    # ── SAM segmentation ─────────────────────────────────────────────────────
    predictor  = get_sam()
    predictor.set_image(rgb_small)
    bbox_small = np.array([
        bbox[0]/scale_bx, bbox[1]/scale_by,
        bbox[2]/scale_bx, bbox[3]/scale_by
    ])
    masks, scores, _ = predictor.predict(
        box=bbox_small[None, :], multimask_output=True
    )
    best_idx   = 0
    best_score = 0
    for i, (mask, score) in enumerate(zip(masks, scores)):
        cov = np.sum(mask) / mask.size * 100
        if 8 < cov < 60 and score > best_score:
            best_score = score
            best_idx   = i
    cow_mask = cv2.resize(
        masks[best_idx].astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    print(f"✅ SAM mask: coverage={np.sum(cow_mask)/cow_mask.size*100:.1f}%")

    # ── Depth refinement ─────────────────────────────────────────────────────
    depth_unit = meta.get('depthUnit', 'mm')
    if depth_unit == 'm':
        raw_depth = raw_depth * 1000.0

    raw_depth[raw_depth < 300]  = 0
    raw_depth[raw_depth > 5000] = 0

    depth_up = cv2.resize(raw_depth.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_LINEAR)
    depth_up[cow_mask == 0] = 0

    try:
        rgb_bgr_f = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).astype(np.float32)
        refined   = cv2.ximgproc.jointBilateralFilter(
            joint=rgb_bgr_f, src=depth_up, d=15, sigmaColor=50, sigmaSpace=50
        )
        print("✅ Joint bilateral filter applied")
    except Exception:
        refined = depth_up.copy()
        print("⚡ Bilateral filter skipped")

    refined[cow_mask == 0] = 0

    # ── Intrinsics ────────────────────────────────────────────────────────────
    calib_w = meta.get('calibResoX') or meta.get('calibWidth')
    calib_h = meta.get('calibResoY') or meta.get('calibHeight')
    scale_x = w / calib_w
    scale_y = h / calib_h
    fx  = meta['fx']  * scale_x
    fy  = meta['fy']  * scale_y
    ppx = meta['ppx'] * scale_x
    ppy = meta['ppy'] * scale_y
    print(f"Intrinsics scaled: fx={fx:.1f} fy={fy:.1f} ppx={ppx:.1f} ppy={ppy:.1f}")

    # ── Point cloud ───────────────────────────────────────────────────────────
    depth_m    = refined / 1000.0
    valid      = (depth_m > 0) & (cow_mask > 0)
    rows, cols = np.where(valid)
    Z   = depth_m[rows, cols]
    X   = (cols - ppx) * Z / fx
    Y   = (rows - ppy) * Z / fy
    pts = np.stack([X, Y, Z], axis=1)

    z_med = np.median(pts[:, 2])
    z_std = np.std(pts[:, 2])
    pts   = pts[np.abs(pts[:, 2] - z_med) < 1.5 * z_std]
    print(f"✅ Point cloud: {len(pts):,} points")

    # ── Roboflow keypoints ────────────────────────────────────────────────────
    from inference_sdk import InferenceHTTPClient
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="DE2rY8nY3DDIGntoV9pu"
    )
    tmp_img = '/tmp/temp_cattle.jpg'
    cv2.imwrite(tmp_img, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    rf_result = CLIENT.infer(tmp_img, model_id='side-image-key-point-pre-train-km5pt/3')
    print(f"✅ Roboflow: {len(rf_result['predictions'])} detections")

    best_pred    = rf_result['predictions'][0]
    best_overlap = -1
    for pred in rf_result['predictions']:
        overlap = sum(
            1 for kp in pred['keypoints']
            if 0 <= int(kp['y']) < h and 0 <= int(kp['x']) < w
            and cow_mask[int(kp['y']), int(kp['x'])] > 0
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_pred    = pred

    keypoints = best_pred['keypoints']
    num_kps   = len(keypoints)
    kps_2d    = np.zeros((num_kps, 2), dtype=np.float32)
    kps_conf  = np.zeros(num_kps, dtype=np.float32)
    for i, kp in enumerate(keypoints):
        kps_2d[i, 0] = kp['x']
        kps_2d[i, 1] = kp['y']
        kps_conf[i]  = kp['confidence']

    # ── Z constraint ─────────────────────────────────────────────────────────
    kps_3d_pass1 = {
        i: snap_to_pointcloud(kps_2d[i,0], kps_2d[i,1], kps_conf[i],
                              pts, fx, fy, ppx, ppy)
        for i in range(num_kps)
    }
    anchor_z = []
    for cid in [1, 9, 2, 8, 13, 3]:
        idx = find_by_class_id(keypoints, cid)
        if idx is not None and kps_3d_pass1.get(idx) is not None:
            u_i = int(round(kps_2d[idx][0]))
            v_i = int(round(kps_2d[idx][1]))
            if 0 <= v_i < h and 0 <= u_i < w and cow_mask[v_i, u_i] > 0:
                anchor_z.append(kps_3d_pass1[idx][2])

    if anchor_z:
        target_z   = np.median(anchor_z)
        z_min_snap = target_z - 0.5
        z_max_snap = target_z + 0.5
    else:
        z_min_snap = pts[:, 2].min()
        z_max_snap = pts[:, 2].max()

    # ── Snap keypoints ────────────────────────────────────────────────────────
    kps_3d = {
        i: snap_to_pointcloud(kps_2d[i,0], kps_2d[i,1], kps_conf[i],
                              pts, fx, fy, ppx, ppy,
                              z_min=z_min_snap, z_max=z_max_snap)
        for i in range(num_kps)
    }

    # ── Measurements ──────────────────────────────────────────────────────────
    IDX_PT_SHOULDER = find_by_class_id(keypoints, 2)
    IDX_PIN_BONE    = find_by_class_id(keypoints, 7)
    IDX_WITHERS     = find_by_class_id(keypoints, 1)
    IDX_ELBOW       = find_by_class_id(keypoints, 9)
    IDX_TOS_RIB     = find_by_class_id(keypoints, 3)
    IDX_BOB_RIB     = find_by_class_id(keypoints, 8)
    IDX_HOOF_EDGE   = find_by_class_id(keypoints, 12)

    body_length_m    = measure_3d(IDX_PT_SHOULDER, IDX_PIN_BONE,  kps_3d)
    heart_girth_half = measure_3d(IDX_WITHERS,     IDX_ELBOW,     kps_3d)
    tinggi_belakang  = measure_3d(IDX_PIN_BONE,    IDX_HOOF_EDGE, kps_3d)
    belly_height_m   = measure_3d(IDX_TOS_RIB,     IDX_BOB_RIB,   kps_3d)
    heart_girth_m    = (heart_girth_half * 2) if heart_girth_half else None

    body_length_cm     = body_length_m   * 100 if body_length_m   else None
    heart_girth_cm     = heart_girth_m   * 100 if heart_girth_m   else None
    tinggi_belakang_cm = tinggi_belakang * 100 if tinggi_belakang else None
    belly_height_cm    = belly_height_m  * 100 if belly_height_m  else None

    print(f"Body length:    {body_length_cm:.1f} cm"    if body_length_cm    else "Body length:    ❌")
    print(f"Heart girth:    {heart_girth_cm:.1f} cm"    if heart_girth_cm    else "Heart girth:    ❌")
    print(f"Tinggi belakang:{tinggi_belakang_cm:.1f} cm" if tinggi_belakang_cm else "Tinggi belakang:❌")
    print(f"Belly height:   {belly_height_cm:.1f} cm"   if belly_height_cm   else "Belly height:   ❌")

    # ── Schaeffer weight ──────────────────────────────────────────────────────
    schaeffer_weight = None
    if body_length_cm and heart_girth_cm:
        schaeffer_weight = (heart_girth_cm**2 * body_length_cm) / 10841
        print(f"Schaeffer: {schaeffer_weight:.1f} kg")

    # ── RF prediction ─────────────────────────────────────────────────────────
    rf_weight = None
    try:
        rf_model, rf_scaler = get_rf()
        if body_length_m and belly_height_m and tinggi_belakang:
            X_input  = np.array([[
                body_length_m,
                belly_height_m,
                tinggi_belakang,
                belly_height_m ** 2
            ]])
            X_scaled = rf_scaler.transform(X_input)
            rf_weight = float(rf_model.predict(X_scaled)[0])
            print(f"RF model: {rf_weight:.1f} kg")
    except Exception as e:
        print(f"RF prediction error: {e}")

    final_weight = rf_weight or schaeffer_weight

    return {
        "status":  "success",
        "device":  meta.get('deviceModel', 'Unknown'),
        "measurements": {
            "body_length_cm":      round(body_length_cm,     1) if body_length_cm     else None,
            "heart_girth_cm":      round(heart_girth_cm,     1) if heart_girth_cm     else None,
            "tinggi_belakang_cm":  round(tinggi_belakang_cm, 1) if tinggi_belakang_cm else None,
            "belly_height_cm":     round(belly_height_cm,    1) if belly_height_cm    else None,
        },
        "weight": {
            "rf_kg":        round(rf_weight,        1) if rf_weight        else None,
            "schaeffer_kg": round(schaeffer_weight, 1) if schaeffer_weight else None,
            "final_kg":     round(final_weight,     1) if final_weight     else None,
        },
        "depth_quality": meta.get('depthQuality', 'Unknown'),
        "avg_depth_mm":  meta.get('avgDepthMm', 0),
    }


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Cattle Weight API running"})


# ── Route 1: File upload ──────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Upload files directly as multipart/form-data.
    Fields: color, depth, metadata
    """
    try:
        if 'color'    not in request.files: return jsonify({"error": "Missing color"}),    400
        if 'depth'    not in request.files: return jsonify({"error": "Missing depth"}),    400
        if 'metadata' not in request.files: return jsonify({"error": "Missing metadata"}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path   = os.path.join(tmpdir, 'color.jpg')
            depth_path = os.path.join(tmpdir, 'depth.npy')
            meta_path  = os.path.join(tmpdir, 'metadata.json')

            request.files['color'].save(rgb_path)
            request.files['depth'].save(depth_path)
            request.files['metadata'].save(meta_path)

            result = run_pipeline(rgb_path, depth_path, meta_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e),
                        "trace": traceback.format_exc()}), 500


# ── Route 2: Firebase Storage URLs ───────────────────────────────────────────
@app.route('/predict_url', methods=['POST'])
def predict_url():
    """
    Send Firebase Storage download URLs as JSON.
    Body: { "color_url": "...", "depth_url": "...", "metadata_url": "..." }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Send JSON body with color_url, depth_url, metadata_url"}), 400

        color_url    = data.get('color_url')
        depth_url    = data.get('depth_url')
        metadata_url = data.get('metadata_url')

        if not all([color_url, depth_url, metadata_url]):
            return jsonify({"error": "Missing color_url, depth_url or metadata_url"}), 400

        print(f"Downloading files from Firebase...")

        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path   = os.path.join(tmpdir, 'color.jpg')
            depth_path = os.path.join(tmpdir, 'depth.npy')
            meta_path  = os.path.join(tmpdir, 'metadata.json')

            # Download all 3 files
            download_url(color_url,    rgb_path)
            print(f"✅ color.jpg downloaded")
            download_url(depth_url,    depth_path)
            print(f"✅ depth.npy downloaded")
            download_url(metadata_url, meta_path)
            print(f"✅ metadata.json downloaded")

            result = run_pipeline(rgb_path, depth_path, meta_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e),
                        "trace": traceback.format_exc()}), 500


if __name__ == '__main__':
    print("🐄 Cattle Weight API")
    print("   POST /predict      — multipart file upload")
    print("   POST /predict_url  — Firebase Storage URLs (JSON)")
    print("   GET  /health       — health check")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False)
