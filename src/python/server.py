import threading
import numpy as np
import torch
from flask import Flask, request, send_file
from flask_cors import CORS
from line_extract import predict_line, compute_frangi_response, apply_frangi_percentile, refine_scribble
import cv2
import io

app = Flask(__name__)
CORS(app)

global img_u8, scr_u8, frangi_u8, refined_scr_u8, progress_data, cancel_flag
img_u8 = None
scr_u8 = None
frangi_u8 = None
refined_scr_u8 = None
progress_data = {"it": 0, "iters": 0, "loss": 0}
cancel_flag = threading.Event()

@app.route("/api/cancel_prediction", methods=["POST"])
def api_cancel_prediction():
    global cancel_flag
    cancel_flag.set()
    return {"message": "生成をキャンセルしました．"}

@app.route("/api/progress", methods=["GET"])
def api_progress_bar():
    global progress_data
    if progress_data is None:
        return {"error": "プログレスデータがありません．"}, 400
    return progress_data

# 画像を最大サイズにリサイズ
def _resize_to_max(img, max_size: int, interpolation):
    if img is None:
        raise ValueError("画像のデコードに失敗しました．対応していない形式、または空のファイルの可能性があります．")
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= int(max_size):
        return img
    scale = float(max_size) / float(s)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

# 画像を読み込んでnumpy配列に変換
def decode_image(file_storage, max_size: int, interpolation=cv2.INTER_AREA):
    if file_storage is None:
        return None
    data = file_storage.read()
    if not data:
        return None
    # デコード
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    return _resize_to_max(arr, int(max_size), interpolation=interpolation)

# Frangi応答
@app.route("/api/compute_frangi", methods=["POST"])
def api_compute_frangi():
    global img_u8, frangi_u8

    image = request.files["image"]
    use_clahe = request.form.get("use_clahe", "false") == "true"
    clahe_clip = float(request.form.get("clahe_clip", 2.0))
    clahe_grid = int(request.form.get("clahe_grid", 8))
    max_size = int(request.form.get("max_size", 2000))

    if image is None:
        return {"error": "画像がありません．"}, 400

    img_u8 = decode_image(image, max_size, interpolation=cv2.INTER_AREA)
    if img_u8 is None:
        return {"error": "画像のデコードに失敗しました．対応していない形式、または空のファイルの可能性があります．"}, 400
    # Frangi応答を計算
    frangi_u8 = compute_frangi_response(img_u8, use_clahe, clahe_clip, clahe_grid, max_size)

    _, buf = cv2.imencode(".png", frangi_u8)
    return send_file(io.BytesIO(buf), mimetype="image/png")

# Frangi応答のパーセンタイル
@app.route("/api/apply_frangi_percentile", methods=["POST"])
def api_apply_frangi_percentile():
    percentile = float(request.form.get("percentile", 99.0))

    if frangi_u8 is None:
        return {"error": "Frangi応答がありません．"}, 400
    # パーセンタイルを適用
    refined_u8 = apply_frangi_percentile(frangi_u8, percentile, img_u8=img_u8)

    _, buf = cv2.imencode(".png", np.clip(refined_u8, 0, 255).astype(np.uint8))
    return send_file(io.BytesIO(buf), mimetype="image/png")

# スクリブルの線画化
@app.route("/api/refine_scribble", methods=["POST"])
def api_refine_scribble():
    global img_u8, scr_u8, refined_scr_u8

    image = request.files["image"]
    scribble = request.files["scribble"]
    use_clahe = request.form.get("use_clahe", "false").lower() == "true"
    clahe_clip = float(request.form.get("clahe_clip", 2.0))
    clahe_grid = int(request.form.get("clahe_grid", 8))
    max_size = int(request.form.get("max_size", 2000))

    if image is None:
        return {"error": "画像がありません．"}, 400
    if scribble is None:
        return {"error": "スクリブルがありません．"}, 400

    img_u8 = decode_image(image, max_size, interpolation=cv2.INTER_AREA)
    if img_u8 is None:
        return {"error": "画像のデコードに失敗しました．対応していない形式、または空のファイルの可能性があります．"}, 400

    scr_u8 = decode_image(scribble, max_size, interpolation=cv2.INTER_NEAREST)
    if scr_u8 is None:
        return {"error": "スクリブル画像のデコードに失敗しました．"}, 400
    # スクリブルの線画化
    try:
        refined_scr_u8 = refine_scribble(
            img_u8,
            scr_u8,
            use_clahe,
            clahe_clip,
            clahe_grid,
            max_size,
        )
    except Exception as e:
        return {"error": f"スクリブルの線画化中にエラーが発生しました: {e}"}, 500

    _, buf = cv2.imencode(".png", refined_scr_u8)
    return send_file(io.BytesIO(buf), mimetype="image/png")

# 全体線画の生成
@app.route("/api/predict_line", methods=["POST"])
def api_predict_line():
    global cancel_flag, progress_data
    lr = float(request.form.get("lr", 1e-3))
    iters = int(request.form.get("iters", 1000))
    max_size = int(request.form.get("max_size", 2000))

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    if scr_u8 is None:
        return {"error": "スクリブルがありません．"}, 400
    elif refined_scr_u8 is None:
        return {"error": "スクリブルの線画化がされていません．"}, 400
    elif img_u8 is None:
        return {"error": "画像がありません．"}, 400

    cancel_flag.clear()
    progress_data = {"it": 0, "iters": 0, "loss": 0}

    def progress_bar(it, iters, loss):
        global progress_data
        progress_data["it"] = it
        progress_data["iters"] = iters
        progress_data["loss"] = loss

    # 全体線画の生成
    try:
        predicted_line = predict_line(img_u8, scr_u8, refined_scr_u8, lr, iters, device, progress_bar=progress_bar, max_size=max_size, cancel_flag=cancel_flag)
    except Exception as e:
        return {"error": f"線画の生成中にエラーが発生しました: {e}"}, 500

    _, buf = cv2.imencode(".png", predicted_line)
    return send_file(io.BytesIO(buf), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)