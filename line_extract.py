import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Small U-Net
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        # down blocks
        self.conv1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_ch*2, base_ch*4)
        # up blocks
        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_ch*4, base_ch*2)
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(base_ch*2, base_ch)
        # output
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_pool = self.pool1(x1)
        x2 = self.conv2(x1_pool)
        x2_pool = self.pool2(x2)
        x3 = self.conv3(x2_pool)

        y2 = self.up1(x3)
        if y2.shape[2:] != x2.shape[2:]: # 入力サイズが奇数の場合
            y2 = F.interpolate(y2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.conv4(y2)

        y1 = self.up2(y2)   
        if y1.shape[2:] != x1.shape[2:]:
            y1 = F.interpolate(y1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.conv5(y1)

        return self.out(y1) # logits


# -----------------------------
# Utilities
# -----------------------------
def load_image(path, color_mode=cv2.IMREAD_COLOR):
    '''
    path: path to image
    return: image in BGR format
    '''
    img = cv2.imread(path, color_mode)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img

def resize_to_max(img, max_size=1024):
    '''
    img: image in BGR format
    max_size: maximum size
    return: resized image and scale factor
    '''
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
    '''
    Compute edge map from grayscale image.
    '''
    g = cv2.GaussianBlur(gray_img, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    return mag

def make_scribble_label_mask(scr_bgr):
    """
    Returns mask with:
      1 for line-labeled pixels
     -1 for unknown
    """
    b, g, r = cv2.split(scr_bgr)

    line = (r > 250) & (g > 250) & (b > 250)
    bg = (r >250) & (g < 5) & (b < 5)
    mask = np.full(scr_bgr.shape[:2], -1, dtype=np.int8)
    mask[line] = 1
    mask[bg] = 0
    return mask

import numpy as np
import cv2

def _xmeans(X, max_k=6):
    """
    多次元データに対するX-means（BIC基準でクラスタ数を自動決定）．
    X: float32 の (N, D) numpy array
    return: (cluster_labels, cluster_centers)  centers: (k, D)
    """
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



    # Lab色空間に変換（OpenCV uint8: L,a,b すべて [0,255]）
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # [0,1] に正規化してスケールを揃える
    lab_norm = lab_img / 255.0

    refined = np.zeros_like(pos_scr, dtype=bool)
    scr_u8 = pos_scr.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(scr_u8, connectivity=8)
    for lab in range(1, num_labels):
        region_mask = labels == lab
        region_area = int(region_mask.sum())

        # (N, 3) の Lab 特徴量
        feats = lab_norm[region_mask]  # shape: (N, 3)

        # X-means で Lab 空間をクラスタリング
        cluster_labels, centers = _xmeans(feats, max_k=max_k)
        # L値（第0次元）が最も小さいクラスタ = 最も暗い = 線画
        darkest_cluster = int(np.argmin(centers[:, 0]))
        line_pixels = cluster_labels == darkest_cluster

        # region_mask 内の座標に line_pixels を戻す
        coords = np.argwhere(region_mask)
        line_in_region = np.zeros_like(region_mask)
        line_in_region[coords[line_pixels, 0], coords[line_pixels, 1]] = True

        refined[line_in_region] = True
    return refined



def apply_clahe_bgr(image, clip_limit=2.0, tile_grid=8):
    """
    Contrast normalization on luminance while preserving color.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def normalize_image(img_rgb):
    """
    Simple channel-wise normalization for training stability.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img_rgb - mean) / std


def augment_tensors(x, t, m, e, aug_prob=0.8):
    """
    Lightweight augmentation for single-image training.
    Geometric transforms are applied consistently to x/t/m/e.
    Photometric transforms are applied to x only.
    """
    if float(torch.rand(1).item()) >= float(aug_prob):
        return x, t, m, e

    xb, tb, mb, eb = x, t, m, e

    if float(torch.rand(1).item()) < 0.5:
        xb = torch.flip(xb, dims=[3])
        tb = torch.flip(tb, dims=[3])
        mb = torch.flip(mb, dims=[3])
        eb = torch.flip(eb, dims=[3])
    if float(torch.rand(1).item()) < 0.5:
        xb = torch.flip(xb, dims=[2])
        tb = torch.flip(tb, dims=[2])
        mb = torch.flip(mb, dims=[2])
        eb = torch.flip(eb, dims=[2])

    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        xb = torch.rot90(xb, k=k, dims=[2, 3])
        tb = torch.rot90(tb, k=k, dims=[2, 3])
        mb = torch.rot90(mb, k=k, dims=[2, 3])
        eb = torch.rot90(eb, k=k, dims=[2, 3])

    # Mild photometric jitter on normalized RGB.
    if float(torch.rand(1).item()) < 0.8:
        gain = 0.85 + 0.30 * float(torch.rand(1).item())  # [0.85, 1.15]
        bias = (float(torch.rand(1).item()) - 0.5) * 0.20  # [-0.1, 0.1]
        xb = xb * gain + bias
    if float(torch.rand(1).item()) < 0.5:
        noise_std = 0.01 + 0.03 * float(torch.rand(1).item())  # [0.01, 0.04]
        xb = xb + torch.randn_like(xb) * noise_std

    return xb, tb, mb, eb


def masked_bce_with_logits(logits, targets01, labeled_mask, pos_weight=1.0):
    """
    logits: (1,1,H,W)
    targets01: (1,1,H,W) in {0,1}
    labeled_mask: (1,1,H,W) in {0,1} indicates which pixels are supervised
    """
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets01, reduction="none", pos_weight=pw
    )
    loss = loss * labeled_mask
    denom = labeled_mask.sum().clamp_min(1.0)
    return loss.sum() / denom

def masked_dice_loss(prob, targets01, labeled_mask, smooth=1.0):
    prob = prob * labeled_mask
    tgt = targets01 * labeled_mask
    inter = (prob * tgt).sum()
    denom = prob.sum() + tgt.sum()
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice

def masked_focal_with_logits(logits, targets01, labeled_mask, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets01, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets01 + (1.0 - p) * (1.0 - targets01)
    alpha_t = alpha * targets01 + (1.0 - alpha) * (1.0 - targets01)
    loss = alpha_t * torch.pow((1.0 - pt).clamp(min=1e-6), gamma) * bce
    loss = loss * labeled_mask
    denom = labeled_mask.sum().clamp_min(1.0)
    return loss.sum() / denom

def total_variation(prob):
    # Encourages smoothness; keep small weight to avoid over-smoothing thin lines
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean()
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
    return dx + dy

def edge_alignment_loss(prob, edge):
    """
    Encourage prob to be higher where edges exist.
    edge: (1,1,H,W) in [0,1]
    We use soft penalty: (1-edge)*prob  (prob should be low where no edges)
    """
    return ((1.0 - edge) * prob).mean()

def binarize_and_thin(mask01):
    """
    mask01: uint8 {0,255}
    Try thinning (skeletonization) if OpenCV-contrib is available.
    Fallback: return as-is.
    """
    try:
        # Requires opencv-contrib-python
        th = (mask01 > 0).astype(np.uint8) * 255
        thin = cv2.ximgproc.thinning(th, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thin
    except Exception:
        return mask01

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="images/img2.jpeg")
    ap.add_argument("--scribble", default="images/scr2.jpg")
    ap.add_argument("--out", default="line_on_white.png")
    ap.add_argument("--out_mask", default="mask.png")
    ap.add_argument("--out_alpha", default="line_alpha.png", help="probを透明度として元画像から線画を抽出したPNG（BGRA）")
    ap.add_argument("--out_pos_scr_refined", default="scribble.png", help="Optional path to save an overlay visualizing refined positive scribble pixels.")
    ap.add_argument("--out_prob", default="prob.png", help="Optional path to save the probability map as an 8-bit grayscale image.")
    ap.add_argument("--out_prob_npy", default="", help="Optional path to save the probability map as float32 .npy.")
    ap.add_argument("--max_size", type=int, default=5000)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.65, help="Used when --thr_method=fixed")
    ap.add_argument("--thr_method", choices=["fixed", "otsu", "percentile", "none"], default="otsu")
    ap.add_argument("--thr_percentile", type=float, default=92.0)
    ap.add_argument("--use_clahe", action="store_true")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--use_aug", action="store_true")
    ap.add_argument("--aug_prob", type=float, default=0.8)
    ap.add_argument("--use_bg_scribble", action="store_true", help="Treat black scribble pixels as background labels.")
    ap.add_argument("--bg_scribble_black_thr", type=int, default=10)
    ap.add_argument("--w_bce", type=float, default=1.0)
    ap.add_argument("--w_dice", type=float, default=0.8)
    ap.add_argument("--w_focal", type=float, default=0.6)
    ap.add_argument("--w_tv", type=float, default=0.01)
    ap.add_argument("--w_edge", type=float, default=0.10)
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--pos_weight", type=float, default=0.0, help="<=0 to auto-estimate from labels")
    ap.add_argument("--lr_patience", type=int, default=80)
    ap.add_argument("--lr_factor", type=float, default=0.6)
    ap.add_argument("--early_stop_patience", type=int, default=220)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    ap.add_argument("--min_area", type=int, default=5)
    ap.add_argument("--min_area_ratio", type=float, default=1e-5)
    ap.add_argument("--bg_edge_thr", type=float, default=0.10, help="Lower -> stricter auto background sampling.")
    ap.add_argument("--scribble_line_adaptive_block", type=int, default=31)
    ap.add_argument("--scribble_line_adaptive_c", type=int, default=8)
    ap.add_argument("--scribble_line_edge_thr", type=float, default=0.08)
    ap.add_argument("--open_ksize", type=int, default=0)
    ap.add_argument("--close_ksize", type=int, default=0)
    ap.add_argument("--debug_dir", default="debug", help="デバッグ画像の保存先ディレクトリ（空文字で無効）")
    ap.add_argument("--device", default="cuda:3")
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

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = normalize_image(img_rgb)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Create partial labels: 1=line, 0=bg, -1=unknown
    scr_mask = make_scribble_label_mask(scr_bgr)
    pos_scr = scr_mask == 1 # line scribble
    neg_scr = scr_mask == 0 # bg scribble

    # extract line pixels in scribble
    edge = compute_edge_map(img_gray)
    pos_scr_refined = extract_line_pixels_in_scribble(
        img_bgr,
        pos_scr,
    )
    pos_scr_orig = pos_scr.copy()
    pos_scr = pos_scr_refined
    mask_u8 = (pos_scr.astype(np.uint8) * 255)
    cv2.imwrite(args.out_pos_scr_refined, mask_u8)


    # Auto-generate background samples from positive scribble only.
    # Use outside of dilated scribble as BG, without edge-based filtering.
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(pos_scr.astype(np.uint8), kernel, iterations=2)
    cv2.imwrite("dil.png", dil.astype(np.uint8) * 255)
    ring = (dil == 1) & (~pos_scr)

    # Use a conservative BG: ring pixels with low gradient are likely non-line interior
    bg_auto = ring & (edge < args.bg_edge_thr)
    cv2.imwrite("bg_auto.png", bg_auto.astype(np.uint8) * 255)
    bg = neg_scr | bg_auto

    # Build target + labeled mask
    # target: 1 for line scribble, 0 for bg scribble/bg auto, else unused
    target = np.zeros_like(img_gray, dtype=np.float32)
    labeled = np.zeros_like(img_gray, dtype=np.float32)
    target[pos_scr] = 1.0
    labeled[pos_scr] = 1.0
    target[bg] = 0.0
    labeled[bg] = 1.0

    # Torch tensors
    x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(args.device)  # 1x3xHxW
    t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(args.device)        # 1x1xHxW
    m = torch.from_numpy(labeled).unsqueeze(0).unsqueeze(0).to(args.device)       # 1x1xHxW
    e = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(args.device)          # 1x1xHxW

    model = UNet(in_ch=3, base_ch=args.base_ch).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=200
    )

    # Auto class-balance for sparse lines
    pos_count = float(((t > 0.5) * (m > 0.5)).sum().item())
    neg_count = float(((t <= 0.5) * (m > 0.5)).sum().item())
    pos_weight = (neg_count + 1.0) / (pos_count + 1.0)
    pos_weight = float(np.clip(pos_weight, 1.0, 20.0))

    model.train()
    best_loss = float("inf")
    bad_iters = 0
    for it in range(args.iters):
        opt.zero_grad()

        xb, tb, mb, eb = x, t, m, e
        if args.use_aug:
            xb, tb, mb, eb = augment_tensors(xb, tb, mb, eb, aug_prob=args.aug_prob)

        logits = model(xb)
        prob = torch.sigmoid(logits)

        loss_bce = masked_bce_with_logits(logits, tb, mb, pos_weight=pos_weight)
        loss_dice = masked_dice_loss(prob, tb, mb)
        loss_focal = masked_focal_with_logits(
            logits, tb, mb, alpha=args.focal_alpha, gamma=args.focal_gamma
        )
        loss_tv = total_variation(prob)
        loss_ed = edge_alignment_loss(prob, eb)

        loss = (
            args.w_bce * loss_bce
            + args.w_dice * loss_dice
            + args.w_focal * loss_focal
            + args.w_tv * loss_tv
            + args.w_edge * loss_ed
        )
        loss.backward()
        opt.step()
        scheduler.step(loss.item())

        cur_loss = float(loss.item())
        if cur_loss < (best_loss - args.early_stop_min_delta):
            best_loss = cur_loss
            bad_iters = 0
        else:
            bad_iters += 1
        if bad_iters >= args.early_stop_patience:
            print(f"[INFO] Early stop at iter {it+1} (best_loss={best_loss:.4f})")
            break

        if (it + 1) % 100 == 0 or it == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"iter {it+1:4d}/{args.iters} | "
                f"bce {loss_bce.item():.4f} dice {loss_dice.item():.4f} focal {loss_focal.item():.4f} "
                f"tv {loss_tv.item():.4f} edge {loss_ed.item():.4f} | total {loss.item():.4f} | lr {lr_now:.2e}"
            )
            # 途中経過のprobを保存
            model.eval()
            with torch.no_grad():
                prob_tmp = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            model.train()
            # グレースケールprobを保存
            prob_tmp_u8 = np.clip(prob_tmp * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(f"prob_iter{it+1:04d}.png", prob_tmp_u8)
            # 白背景合成出力（probを透明度として元画像から線画を抽出）
            if args.out_alpha:
                prob_tmp_resized = cv2.resize(prob_tmp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                a = prob_tmp_resized[:, :, np.newaxis].astype(np.float32)
                orig_bgr = orig_image if orig_image.shape[:2] == (orig_h, orig_w) else cv2.resize(orig_image, (orig_w, orig_h))
                white = np.ones_like(orig_bgr, dtype=np.float32) * 255.0
                blended = orig_bgr.astype(np.float32) * a + white * (1.0 - a)
                cv2.imwrite(f"line_alpha_iter{it+1:04d}.png", blended.astype(np.uint8))

    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Optionally save raw probability map (before thresholding/postprocess).
    prob_out = prob
    if prob_out.shape[:2] != (orig_h, orig_w):
        prob_out = cv2.resize(prob_out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    if args.out_prob:
        prob_u8 = np.clip(prob_out * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(args.out_prob, prob_u8)
    if args.out_prob_npy:
        np.save(args.out_prob_npy, prob_out.astype(np.float32))

    # Threshold -> binary
    if args.thr_method == "none":
        bin01 = prob.astype(np.uint8) * 255
        thr_val = 0.0
    elif args.thr_method == "fixed":
        thr_val = float(args.thr)
        bin01 = (prob >= thr_val).astype(np.uint8) * 255
    elif args.thr_method == "otsu":
        prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        _, bin01 = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_val = float(_ / 255.0)
    else:
        p = float(np.clip(args.thr_percentile, 0.0, 100.0))
        thr_val = float(np.percentile(prob, p))
        bin01 = (prob >= thr_val).astype(np.uint8) * 255

    # Post-process: morphology + remove tiny noise, then thin (optional)
    if args.open_ksize > 1:
        k = np.ones((args.open_ksize, args.open_ksize), np.uint8)
        bin01 = cv2.morphologyEx(bin01, cv2.MORPH_OPEN, k, iterations=1)
    if args.close_ksize > 1:
        k = np.ones((args.close_ksize, args.close_ksize), np.uint8)
        bin01 = cv2.morphologyEx(bin01, cv2.MORPH_CLOSE, k, iterations=1)

    # Noise removal
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((bin01 > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(bin01)
    area_thr = max(int(args.min_area), int(args.min_area_ratio * bin01.shape[0] * bin01.shape[1]))
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_thr:
            cleaned[labels == i] = 255

    thinned = binarize_and_thin(cleaned)

    # Restore mask to input resolution if resized for training/inference.
    if thinned.shape[:2] != (orig_h, orig_w):
        thinned = cv2.resize(thinned, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Apply mask to original image over a white background.
    # mask=255 -> keep original pixel, mask=0 -> white.
    out_img = np.full_like(orig_image, 255, dtype=np.uint8)
    keep = thinned > 0
    out_img[keep] = orig_image[keep]

    cv2.imwrite(args.out, out_img)
    if args.out_mask:
        thinned = cv2.bitwise_not(thinned)
        cv2.imwrite(args.out_mask, thinned)

    # probを透明度として白背景に合成（元画像から線画を抽出）
    if args.out_alpha:
        a = prob_out[:, :, np.newaxis].astype(np.float32)
        orig_bgr = cv2.resize(orig_image, (orig_w, orig_h)) if orig_image.shape[:2] != (orig_h, orig_w) else orig_image
        white = np.ones_like(orig_bgr, dtype=np.float32) * 255.0
        blended = orig_bgr.astype(np.float32) * a + white * (1.0 - a)
        cv2.imwrite(args.out_alpha, blended.astype(np.uint8))
        print(f"[OK] Saved alpha: {args.out_alpha}")

    print(f"[INFO] threshold={thr_val:.4f} ({args.thr_method}) | min_area={area_thr}")
    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()
