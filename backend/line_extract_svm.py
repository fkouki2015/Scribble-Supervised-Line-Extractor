import argparse
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Utilities (from line_extract.py)
# -----------------------------
def load_image(path, color_mode=cv2.IMREAD_COLOR):
    img = cv2.imread(path, color_mode)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img

def resize_to_max(img, max_size=1024):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_size:
        return img, 1.0
    scale = max_size / s
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale

def compute_edge_map(gray_img):
    g = cv2.GaussianBlur(gray_img, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    return mag

def make_scribble_label_mask(scr_bgr):
    b, g, r = cv2.split(scr_bgr)
    line = (r > 250) & (g > 250) & (b > 250)
    bg = (r > 250) & (g < 5) & (b < 5)
    mask = np.full(scr_bgr.shape[:2], -1, dtype=np.int8)
    mask[line] = 1
    mask[bg] = 0
    return mask

def _xmeans(X, max_k=6):
    from sklearn.cluster import KMeans

    def bic(km, X):
        n, d = X.shape
        k = km.n_clusters
        lbl = km.labels_
        var = sum(
            np.sum((X[lbl == i] - km.cluster_centers_[i]) ** 2)
            for i in range(k)
        ) / max(n - k, 1)
        var = max(var, 1e-6)
        log_lik = sum(
            len(X[lbl == i]) * (
                np.log(len(X[lbl == i]) / n + 1e-9)
                - 0.5 * d * np.log(2 * np.pi * var)
                - 0.5 * np.sum((X[lbl == i] - km.cluster_centers_[i]) ** 2) / var
            )
            for i in range(k)
        )
        n_params = k * d + k - 1
        return -2 * log_lik + n_params * np.log(n)

    X = X.astype(np.float32)
    best_bic = np.inf
    best_km = None
    for k in range(2, min(max_k + 1, len(X))):
        try:
            km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(X)
            b = bic(km, X)
            if b < best_bic:
                best_bic = b
                best_km = km
        except Exception:
            break
    if best_km is None:
        best_km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(X)
    return best_km.labels_, best_km.cluster_centers_

def extract_line_pixels_in_scribble(img_bgr, pos_scr, max_k=6):
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_norm = lab_img / 255.0

    refined = np.zeros_like(pos_scr, dtype=bool)
    scr_u8 = pos_scr.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(scr_u8, connectivity=8)
    for lab in range(1, num_labels):
        region_mask = labels == lab
        
        feats = lab_norm[region_mask]
        if len(feats) < 2:
            continue

        cluster_labels, centers = _xmeans(feats, max_k=max_k)
        darkest_cluster = int(np.argmin(centers[:, 0]))
        line_pixels = cluster_labels == darkest_cluster

        coords = np.argwhere(region_mask)
        line_in_region = np.zeros_like(region_mask)
        line_in_region[coords[line_pixels, 0], coords[line_pixels, 1]] = True
        refined[line_in_region] = True
    return refined

def apply_clahe_bgr(image, clip_limit=2.0, tile_grid=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def binarize_and_thin(mask01):
    try:
        th = (mask01 > 0).astype(np.uint8) * 255
        thin = cv2.ximgproc.thinning(th, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thin
    except Exception:
        return mask01

# -----------------------------
# SVM Extraction
# -----------------------------
def extract_features(img_bgr, gray_img):
    """
    ピクセルごとにSVMに学習させるための特徴量を抽出する
    """
    H, W = gray_img.shape
    
    # 1. 色情報 (Lab)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    
    # 2. 局所的なエッジ強度と方向 (Hessian または Sobel/Scharr)
    # 線画は局所的なリッジ（谷）構造を持つため、Hessianの固有値等が有効だが
    # ここでは計算を軽くするために複数のスケールでの平滑化画像との差分やSobelを用いる
    g1 = cv2.GaussianBlur(gray_img, (3, 3), 0).astype(np.float32) / 255.0
    g2 = cv2.GaussianBlur(gray_img, (7, 7), 0).astype(np.float32) / 255.0
    g3 = cv2.GaussianBlur(gray_img, (15, 15), 0).astype(np.float32) / 255.0
    
    gray_f = gray_img.astype(np.float32) / 255.0
    
    # DoG (Difference of Gaussians) 的な特徴（周辺より暗いか？）
    dog1 = gray_f - g1
    dog2 = gray_f - g2
    dog3 = gray_f - g3
    
    # エッジ強度
    edge = compute_edge_map(gray_img)

    # 結合して (H, W, D) にする
    # lab(3), diffs(3), edge(1) -> 7次元特徴
    features = np.stack([
        lab[..., 0], lab[..., 1], lab[..., 2],
        dog1, dog2, dog3,
        edge
    ], axis=-1)
    
    return features

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="images/img2.jpeg")
    ap.add_argument("--scribble", default="images/scr2.jpg")
    ap.add_argument("--out", default="line_on_white_svm.png")
    ap.add_argument("--out_mask", default="mask_svm.png")
    ap.add_argument("--out_alpha", default="line_alpha_svm.png")
    ap.add_argument("--max_size", type=int, default=5000)
    ap.add_argument("--use_clahe", action="store_true")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--bg_edge_thr", type=float, default=0.10)
    ap.add_argument("--svm_c", type=float, default=1.0, help="SVM Regularization parameter")
    ap.add_argument("--subsample", type=int, default=10000, help="Max samples per class for SVM training")
    args = ap.parse_args()

    img_bgr = load_image(args.img)
    orig_image = img_bgr.copy()
    orig_h, orig_w = img_bgr.shape[:2]
    scr_bgr = load_image(args.scribble)

    # resize
    img_bgr, _ = resize_to_max(img_bgr, args.max_size)
    scr_bgr = cv2.resize(scr_bgr, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if args.use_clahe:
        img_bgr = apply_clahe_bgr(img_bgr, clip_limit=args.clahe_clip, tile_grid=args.clahe_grid)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Define scribble mask: 1=line, 0=bg, -1=unknown
    scr_mask = make_scribble_label_mask(scr_bgr)
    pos_scr = scr_mask == 1
    neg_scr = scr_mask == 0

    # Extract refined line pixels via x-means
    print("[INFO] Applying x-means to refine scribble...")
    pos_scr_refined = extract_line_pixels_in_scribble(img_bgr, pos_scr)

    # Auto background sampling
    edge = compute_edge_map(img_gray)
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(pos_scr_refined.astype(np.uint8), kernel, iterations=2)
    ring = (dil == 1) & (~pos_scr_refined)
    bg_auto = ring & (edge < args.bg_edge_thr)
    bg = neg_scr | bg_auto

    # ----------- SVM Training Setup -----------
    print("[INFO] Extracting features for SVM...")
    features = extract_features(img_bgr, img_gray)
    H, W, D = features.shape

    # Collect training samples
    # Foreground (1)
    pos_feats = features[pos_scr_refined]
    # Background (0)
    neg_feats = features[bg]

    print(f"[INFO] Initial samples -> Pos: {len(pos_feats)}, Neg: {len(neg_feats)}")

    # Subsample to keep training fast
    if len(pos_feats) > args.subsample:
        idx = np.random.choice(len(pos_feats), args.subsample, replace=False)
        pos_feats = pos_feats[idx]
    if len(neg_feats) > args.subsample:
        idx = np.random.choice(len(neg_feats), args.subsample, replace=False)
        neg_feats = neg_feats[idx]

    X_train = np.vstack([pos_feats, neg_feats])
    y_train = np.hstack([np.ones(len(pos_feats)), np.zeros(len(neg_feats))])

    print(f"[INFO] Training SVM with {len(X_train)} samples...")
    # Using RBF kernel with class weights balanced
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=args.svm_c, class_weight='balanced', probability=True))
    clf.fit(X_train, y_train)

    # ----------- Inference Setup -----------
    print("[INFO] Predicting full image with SVM...")
    # Reshape features to (N, D) for prediction
    X_all = features.reshape(-1, D)
    
    # Predict probabilities (memory efficient chunking might be needed for huge images)
    chunk_size = 500000
    probs = np.zeros(len(X_all), dtype=np.float32)
    for i in range(0, len(X_all), chunk_size):
        chunk = X_all[i : i+chunk_size]
        # proba returns [P(class 0), P(class 1)]
        probs[i : i+chunk_size] = clf.predict_proba(chunk)[:, 1]

    prob_map = probs.reshape(H, W)

    # Save probability map
    prob_u8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite("prob_svm.png", prob_u8)

    # Threshold Otsu
    _, bin01 = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Post-process
    # Noise removal
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    cleaned = np.zeros_like(bin01)
    area_thr = 5
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_thr:
            cleaned[labels == i] = 255

    thinned = binarize_and_thin(cleaned)

    if thinned.shape[:2] != (orig_h, orig_w):
        thinned = cv2.resize(thinned, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Apply mask to orig image
    out_img = np.full_like(orig_image, 255, dtype=np.uint8)
    keep = thinned > 0
    out_img[keep] = orig_image[keep]

    cv2.imwrite(args.out, out_img)
    print(f"[OK] Saved: {args.out}")
    
    if args.out_mask:
        cv2.imwrite(args.out_mask, thinned)
        print(f"[OK] Saved mask: {args.out_mask}")

    if args.out_alpha:
        prob_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        alpha = (prob_resized * 255.0).astype(np.float32)
        orig_bgr = cv2.resize(orig_image, (orig_w, orig_h)) if orig_image.shape[:2] != (orig_h, orig_w) else orig_image
        bgra = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha.astype(np.uint8)
        cv2.imwrite(args.out_alpha, bgra)
        print(f"[OK] Saved alpha: {args.out_alpha}")

if __name__ == "__main__":
    main()
