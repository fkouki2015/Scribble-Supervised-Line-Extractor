import argparse
import os
import cv2
import numpy as np
import maxflow

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
# Graph Cut
# -----------------------------
def perform_graph_cut(img_gray, pos_mask, bg_mask, lambda_weight=50.0, sigma=15.0):
    """
    Perform max-flow/min-cut graph cut.
    Returns boolean mask of the line drawing.
    """
    H, W = img_gray.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((H, W))

    INF = 1e8

    # Terminal capacities
    source_cap = np.zeros((H, W), dtype=np.float32)
    sink_cap = np.zeros((H, W), dtype=np.float32)

    # Source = Line (Foreground), Sink = Background
    source_cap[pos_mask] = INF
    sink_cap[bg_mask] = INF

    # Small prior for unseeded pixels: darker is more likely line, brighter is background
    img_f = img_gray.astype(np.float32) / 255.0
    unseeded = ~(pos_mask | bg_mask)
    
    # These priors are relatively small so they don't override pairwise smoothness, 
    # but help guide the result in ambiguous regions.
    source_cap[unseeded] = 1.0 * (1.0 - img_f[unseeded])
    sink_cap[unseeded] = 1.0 * img_f[unseeded]

    g.add_grid_tedges(nodeids, source_cap, sink_cap)

    # Pairwise capacities
    img_gray_f = img_gray.astype(np.float32)
    beta = 1.0 / (2.0 * (sigma ** 2))

    # Horizontal edges (right)
    structure_x = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
    wh = np.zeros((H, W), dtype=np.float32)
    diff_h = img_gray_f[:, :-1] - img_gray_f[:, 1:]
    wh[:, :-1] = lambda_weight * np.exp(-beta * (diff_h ** 2))
    g.add_grid_edges(nodeids, wh, structure=structure_x, symmetric=True)

    # Vertical edges (down)
    structure_y = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0]])
    wv = np.zeros((H, W), dtype=np.float32)
    diff_v = img_gray_f[:-1, :] - img_gray_f[1:, :]
    wv[:-1, :] = lambda_weight * np.exp(-beta * (diff_v ** 2))
    g.add_grid_edges(nodeids, wv, structure=structure_y, symmetric=True)

    # Maxflow execution
    print("[INFO] Computing Graph Cut...")
    g.maxflow()
    
    # g.get_grid_segments returns False for Source segment and True for Sink segment
    # Let's invert it so True = Line (Source)
    segments = g.get_grid_segments(nodeids)
    line_mask = ~segments
    
    return line_mask


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="images/illust.jpg")
    ap.add_argument("--scribble", default="images/scr7.jpg")
    ap.add_argument("--out", default="line_on_white_gc.png")
    ap.add_argument("--out_mask", default="mask_gc.png")
    ap.add_argument("--out_alpha", default="line_alpha_gc.png", help="抽出された線画をアルファとして持つ画像")
    ap.add_argument("--max_size", type=int, default=5000)
    ap.add_argument("--use_clahe", action="store_true")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--bg_edge_thr", type=float, default=0.10, help="Lower -> stricter auto background sampling.")
    ap.add_argument("--gc_lambda", type=float, default=50.0, help="Graph cut smoothness weight")
    ap.add_argument("--gc_sigma", type=float, default=15.0, help="Graph cut intensity sigma")
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
    pos_scr_refined = pos_scr

    # Auto background sampling
    edge = compute_edge_map(img_gray)
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(pos_scr_refined.astype(np.uint8), kernel, iterations=2)
    ring = (dil == 1) & (~pos_scr_refined)
    bg_auto = ring & (edge < args.bg_edge_thr)
    bg = neg_scr | bg_auto
    cv2.imwrite("bg_auto.png", bg_auto.astype(np.uint8) * 255)

    # Run Graph Cut
    line_mask = perform_graph_cut(img_gray, pos_scr_refined, bg, 
                                  lambda_weight=args.gc_lambda, sigma=args.gc_sigma)

    # Post-process
    bin01 = (line_mask.astype(np.uint8) * 255)

    # Noise removal (connected components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    cleaned = np.zeros_like(bin01)
    # Area matching logic from line_extract.py 
    area_thr = 5 # default min_area
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_thr:
            cleaned[labels == i] = 255

    thinned = binarize_and_thin(cleaned)

    if thinned.shape[:2] != (orig_h, orig_w):
        thinned = cv2.resize(thinned, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Apply mask to original image over a white background
    out_img = np.full_like(orig_image, 255, dtype=np.uint8)
    keep = thinned > 0
    out_img[keep] = orig_image[keep]

    cv2.imwrite(args.out, out_img)
    print(f"[OK] Saved: {args.out}")
    
    if args.out_mask:
        cv2.imwrite(args.out_mask, thinned)
        print(f"[OK] Saved mask: {args.out_mask}")

    if args.out_alpha:
        # Create alpha channel where lines are opaque and background is transparent
        # Invert thinned to give 255 to lines? `thinned` is 255 for lines.
        alpha = (thinned / 255.0).astype(np.float32) * 255.0
        # Wait, if we use mask as alpha, keeping black lines on transparent BG:
        # We probably want the line drawing colors from the original image.
        orig_bgr = cv2.resize(orig_image, (orig_w, orig_h)) if orig_image.shape[:2] != (orig_h, orig_w) else orig_image
        bgra = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha.astype(np.uint8)
        cv2.imwrite(args.out_alpha, bgra)
        print(f"[OK] Saved alpha: {args.out_alpha}")

if __name__ == "__main__":
    main()
