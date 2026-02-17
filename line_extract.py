import argparse
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
def read_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img

def resize_to_max(img, max_side=1024):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img, 1.0
    scale = max_side / s
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale

def make_scribble_label_mask(scribble_bgr, use_black_bg=False, black_thr=10):
    """
    Returns mask with:
      1 for line-labeled pixels
      0 for bg-labeled pixels
     -1 for unknown
    We treat:
      white-ish pixels as line
      black-ish pixels as background (only when use_black_bg=True)
      others as unknown
    """
    b, g, r = cv2.split(scribble_bgr)

    line = (r > 250) & (g > 250) & (b > 250)
    bg = (r < black_thr) & (g < black_thr) & (b < black_thr)

    mask = np.full(scribble_bgr.shape[:2], -1, dtype=np.int8)
    mask[line] = 1
    if use_black_bg:
        mask[bg] = 0
    return mask

def compute_edge_map(gray):
    # Edge map used to encourage line probability at edges, but not required
    # Normalize to [0,1]
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    return mag

def extract_line_pixels_in_scribble(gray, pos_scr, edge_map, adaptive_block=31, adaptive_c=8, edge_thr=0.08):
    """
    Refine positive scribble labels by keeping only likely line pixels inside scribble.
    This makes thick scribbles robust by suppressing non-line interior pixels.
    """
    block = int(max(3, adaptive_block))
    if block % 2 == 0:
        block += 1

    # Dark-line prior (for typical line-art) from local adaptive threshold.
    dark_line = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, adaptive_c
    ) > 0

    # Edge prior from Canny + Sobel magnitude.
    canny = cv2.Canny(gray, 40, 120) > 0
    edge_like = canny | (edge_map >= float(edge_thr))

    refined = pos_scr & dark_line & edge_like

    # If refinement is too aggressive, fallback to the original positive scribble.
    orig_count = int(pos_scr.sum())
    refined_count = int(refined.sum())
    if orig_count == 0:
        return pos_scr, 1.0
    keep_ratio = refined_count / max(orig_count, 1)
    if refined_count < 16 or keep_ratio < 0.02:
        return pos_scr, 1.0
    return refined, keep_ratio

def apply_clahe_bgr(img_bgr, clip_limit=2.0, tile_grid=8):
    """
    Contrast normalization on luminance while preserving color.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def normalize_image01(img_rgb01):
    """
    Simple channel-wise normalization for training stability.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img_rgb01 - mean) / std

def random_augment(x, t, m, e, p=0.8):
    """
    Lightweight augmentation for single-image optimization.
    Apply same geometric transform to x/t/m/e; photometric only to x.
    """
    if torch.rand(1).item() > p:
        return x, t, m, e

    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
        t = torch.flip(t, dims=[3])
        m = torch.flip(m, dims=[3])
        e = torch.flip(e, dims=[3])
    if torch.rand(1).item() < 0.3:
        x = torch.flip(x, dims=[2])
        t = torch.flip(t, dims=[2])
        m = torch.flip(m, dims=[2])
        e = torch.flip(e, dims=[2])

    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        x = torch.rot90(x, k, dims=[2, 3])
        t = torch.rot90(t, k, dims=[2, 3])
        m = torch.rot90(m, k, dims=[2, 3])
        e = torch.rot90(e, k, dims=[2, 3])

    alpha = 0.9 + 0.2 * torch.rand(1, device=x.device)  # contrast
    beta = -0.06 + 0.12 * torch.rand(1, device=x.device)  # brightness
    x = torch.clamp(x * alpha + beta, -3.0, 3.0)

    if torch.rand(1).item() < 0.4:
        sigma = 0.015
        x = x + sigma * torch.randn_like(x)
        x = torch.clamp(x, -3.0, 3.0)

    return x, t, m, e

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
    ap.add_argument("--img", required=True)
    ap.add_argument("--scribble", required=True)
    ap.add_argument("--out", default="line_on_white.png")
    ap.add_argument("--out_mask", default=None, help="Optional path to save the final binary mask.")
    ap.add_argument("--max_side", type=int, default=5000)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-4)
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    img_bgr = read_bgr(args.img)
    orig_img_bgr = img_bgr.copy()
    orig_h, orig_w = img_bgr.shape[:2]
    scr_bgr = read_bgr(args.scribble)

    # Resize consistently (important for training speed)
    img_bgr, _ = resize_to_max(img_bgr, args.max_side)
    scr_bgr = cv2.resize(scr_bgr, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if args.use_clahe:
        img_bgr = apply_clahe_bgr(img_bgr, clip_limit=args.clahe_clip, tile_grid=args.clahe_grid)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = normalize_image01(img_rgb)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Create partial labels: 1=line, 0=bg, -1=unknown
    scr_mask = make_scribble_label_mask(
        scr_bgr, use_black_bg=args.use_bg_scribble, black_thr=args.bg_scribble_black_thr
    )
    pos_scr = scr_mask == 1
    neg_scr = scr_mask == 0

    # Refine positive scribble so only line pixels inside scribble become positive labels.
    # This makes thick scribbles less noisy for supervision.
    edge = compute_edge_map(gray)
    pos_scr_refined, keep_ratio = extract_line_pixels_in_scribble(
        gray,
        pos_scr,
        edge,
        adaptive_block=args.scribble_line_adaptive_block,
        adaptive_c=args.scribble_line_adaptive_c,
        edge_thr=args.scribble_line_edge_thr,
    )
    pos_scr = pos_scr_refined

    # Auto-generate background samples from a ring around line scribble
    # This stabilizes training hugely without user needing to mark BG.
    # We take pixels near scribble as BG candidates (outside scribble itself).
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(pos_scr.astype(np.uint8), kernel, iterations=2)
    ring = (dil == 1) & (~pos_scr)

    # Use a conservative BG: ring pixels with low gradient are likely non-line interior
    bg_auto = ring & (edge < args.bg_edge_thr)
    bg = neg_scr | bg_auto

    # Build target + labeled mask
    # target: 1 for line scribble, 0 for bg scribble/bg auto, else unused
    target = np.zeros_like(gray, dtype=np.float32)
    labeled = np.zeros_like(gray, dtype=np.float32)
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
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    # Auto class-balance for sparse lines
    if args.pos_weight > 0:
        pos_weight = args.pos_weight
    else:
        pos_count = float(((t > 0.5) * (m > 0.5)).sum().item())
        neg_count = float(((t <= 0.5) * (m > 0.5)).sum().item())
        pos_weight = (neg_count + 1.0) / (pos_count + 1.0)
        pos_weight = float(np.clip(pos_weight, 1.0, 20.0))

    # If too few labeled pixels, warn
    labeled_count = float(m.sum().item())
    if labeled_count < 200:
        print(f"[WARN] Labeled pixels are few: {int(labeled_count)}. Add more scribble for stability.")
    neg_ratio = float(neg_scr.mean())
    if args.use_bg_scribble and neg_ratio > 0.50:
        print(
            f"[WARN] bg_scribble covers {neg_ratio*100:.1f}% pixels. "
            "If this is unintended (e.g., black canvas), disable --use_bg_scribble."
        )
    print(
        f"[INFO] pos_weight={pos_weight:.3f} | pos_scribble={int(pos_scr.sum())} "
        f"| bg_scribble={int(neg_scr.sum())} | bg_edge_thr={args.bg_edge_thr:.3f}"
    )
    print(f"[INFO] scribble_line_keep_ratio={keep_ratio:.3f}")

    model.train()
    best_loss = float("inf")
    bad_iters = 0
    for it in range(args.iters):
        opt.zero_grad()
        if args.use_aug:
            xb, tb, mb, eb = random_augment(x, t, m, e, p=args.aug_prob)
        else:
            xb, tb, mb, eb = x, t, m, e

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

    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

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
    out_img = np.full_like(orig_img_bgr, 255, dtype=np.uint8)
    keep = thinned > 0
    out_img[keep] = orig_img_bgr[keep]

    cv2.imwrite(args.out, out_img)
    if args.out_mask:
        cv2.imwrite(args.out_mask, thinned)
    print(f"[INFO] threshold={thr_val:.4f} ({args.thr_method}) | min_area={area_thr}")
    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()
