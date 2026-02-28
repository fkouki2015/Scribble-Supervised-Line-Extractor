import os
import shutil
import numpy as np
import torch
from flask import Flask, request, send_file
from flask_cors import CORS
from line_extract import predict_line, compute_frangi_response, apply_frangi_percentile, refine_scribble
import cv2
import io

app = Flask(__name__)
CORS(app)

global img_path, scr_path, frangi_path, refined_scr_path, line_path
img_path: str = "temp/img.png"
scr_path: str = "temp/scr.png"
frangi_path: str = "temp/frangi.png"
refined_scr_path: str = "temp/scr_refined.png"
line_path: str = "temp/line.png"

# 一時ファイル
os.makedirs("temp", exist_ok=True)

# 画像を最大サイズにリサイズ
def _resize_to_max(img, max_size: int, interpolation):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= int(max_size):
        return img
    scale = float(max_size) / float(s)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

# アップロードされた画像を保存
def save_image_upload(file_storage, destination: str, max_size: int, interpolation=cv2.INTER_AREA):
    data = file_storage.read()
    # numpy配列に変換
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return {"error": "アップロードされた画像の読み込みに失敗しました．"}
    arr = _resize_to_max(arr, int(max_size), interpolation=interpolation)
    cv2.imwrite(destination, arr)
    return arr

# Frangi応答
@app.route("/api/compute_frangi", methods=["POST"])
def api_compute_frangi():
    image = request.files["image"]
    use_clahe = request.form.get("use_clahe", "false") == "true"
    clahe_clip = float(request.form.get("clahe_clip", 2.0))
    clahe_grid = int(request.form.get("clahe_grid", 8))
    max_size = int(request.form.get("max_size", 2000))

    if image == None:
        return {"error": "画像がありません．"}

    save_image_upload(image, img_path, max_size, interpolation=cv2.INTER_AREA)
    # Frangi応答を計算
    response_bgr = compute_frangi_response(img_path, use_clahe, clahe_clip, clahe_grid, max_size)

    cv2.imwrite(frangi_path, response_bgr)
    _, buffer = cv2.imencode(".png", response_bgr)
    return send_file(io.BytesIO(buffer), mimetype="image/png")

# Frangi応答のパーセンタイル
@app.route("/api/apply_frangi_percentile", methods=["POST"])
def api_apply_frangi_percentile():
    percentile = float(request.form.get("percentile", 99.0))

    if not os.path.exists(frangi_path):
        return {"error": "Frangi応答がありません．"}
    # パーセンタイルを適用
    refined_bgr = apply_frangi_percentile(frangi_path, percentile, img_path=img_path)

    # cv2.imwrite(refined_scr_path, refined_bgr)
    # 連続でリクエストが来た際にファイルが上書きされて破損するのを防ぐためメモリ上で返す
    _, buffer = cv2.imencode(".png", refined_bgr)
    return send_file(io.BytesIO(buffer), mimetype="image/png")

# スクリブルの線画化
@app.route("/api/refine_scribble", methods=["POST"])
def api_refine_scribble():
    image = request.files["image"]
    scribble = request.files["scribble"]
    use_clahe = request.form.get("use_clahe", "false").lower() == "true"
    clahe_clip = float(request.form.get("clahe_clip", 2.0))
    clahe_grid = int(request.form.get("clahe_grid", 8))
    max_size = int(request.form.get("max_size", 2000))

    if image == None:
        return {"error": "画像がありません．"}
    if scribble == None:
        return {"error": "スクリブルがありません．"}

    save_image_upload(image, img_path, max_size, interpolation=cv2.INTER_AREA)
    save_image_upload(scribble, scr_path, max_size, interpolation=cv2.INTER_NEAREST)
    # スクリブルの線画化
    try:
        refined_bgr = refine_scribble(
            img_path,
            scr_path,
            use_clahe,
            clahe_clip,
            clahe_grid,
            max_size,
        )
    except Exception as e:
        return {"error": f"スクリブルの線画化中にエラーが発生しました: {e}"}

    cv2.imwrite(refined_scr_path, refined_bgr)
    return send_file(refined_scr_path, mimetype="image/png")

# 全体線画の生成
@app.route("/api/predict_line", methods=["POST"])
def api_predict_line():
    lr = float(request.form.get("lr", 1e-3))
    iters = int(request.form.get("iters", 1000))
    max_size = int(request.form.get("max_size", 2000))

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    if not os.path.exists(img_path):
        return {"error": "画像がありません．"}
    if not os.path.exists(scr_path):
        return {"error": "スクリブルがありません．"}
    if not os.path.exists(refined_scr_path):
        return {"error": "スクリブルの線画化がされていません．"}

    # 全体線画の生成
    try:
        predicted_line = predict_line(img_path, scr_path, refined_scr_path, lr, iters, device, max_size=max_size)
    except Exception as e:
        return {"error": f"線画の生成中にエラーが発生しました: {e}"}

    cv2.imwrite(line_path, predicted_line)

    if os.path.exists(line_path):
        return send_file(line_path, mimetype="image/png")
    else:
        return {"error": "線画の生成に失敗しました．"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)