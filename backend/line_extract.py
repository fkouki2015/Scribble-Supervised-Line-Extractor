import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
if not hasattr(np, 'warnings'):
    np.warnings = warnings
from skimage.filters import frangi
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"
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
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(base_ch*8, base_ch*16)
        self.pool5 = nn.MaxPool2d(2)
        self.conv5_1 = DoubleConv(base_ch*16, base_ch*16)


        # up blocks
        self.up1 = nn.ConvTranspose2d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(base_ch*16, base_ch*8)
        self.up2 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(base_ch*8, base_ch*4)
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(base_ch*4, base_ch*2)
        self.up4 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(base_ch*2, base_ch)
        # output
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_pool = self.pool1(x1)
        x2 = self.conv2(x1_pool)
        x2_pool = self.pool2(x2)
        x3 = self.conv3(x2_pool)
        x3_pool = self.pool3(x3)
        x4 = self.conv4(x3_pool)
        x4_pool = self.pool4(x4)
        x5 = self.conv5(x4_pool)    
        # x5_pool = self.pool5(x5)
        # x6 = self.conv5_1(x5_pool)
        

        y4 = self.up1(x5)
        if y4.shape[2:] != x4.shape[2:]: # 入力サイズが奇数の場合
            y4 = F.interpolate(y4, size=x4.shape[2:], mode='bilinear', align_corners=False)
        y4 = torch.cat([y4, x4], dim=1)
        y4 = self.conv6(y4)

        y3 = self.up2(y4)
        if y3.shape[2:] != x3.shape[2:]: # 入力サイズが奇数の場合
            y3 = F.interpolate(y3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.conv7(y3)

        y2 = self.up3(y3)
        if y2.shape[2:] != x2.shape[2:]: # 入力サイズが奇数の場合
            y2 = F.interpolate(y2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.conv8(y2)

        y1 = self.up4(y2)   
        if y1.shape[2:] != x1.shape[2:]:
            y1 = F.interpolate(y1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.conv9(y1)

        return self.out(y1) # logits






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

    line = (r < 5) & (g > 250) & (b < 5)
    bg = (r >250) & (g < 5) & (b < 5)
    mask = np.full(scr_bgr.shape[:2], -1, dtype=np.int8)
    mask[line] = 1
    mask[bg] = 0
    return mask





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

    import torchvision.transforms.functional as TF
    # Random Affine (translation & scale)
    if float(torch.rand(1).item()) < 0.7:
        angle = float((torch.rand(1).item() - 0.5) * 30.0) # -15 to +15 deg
        translate = [float((torch.rand(1).item() - 0.5) * 0.1), float((torch.rand(1).item() - 0.5) * 0.1)] # up to 10% translation
        scale = float(0.8 + 0.4 * torch.rand(1).item()) # 0.8x to 1.2x scale
        shear = float((torch.rand(1).item() - 0.5) * 10.0) # -5 to +5 deg shear

        # Apply to all spatial tensors consistently
        # Use bilinear for images to avoid harsh pixelation
        xb = TF.affine(xb, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR)
        # Use nearest neighbor for masks/labels to keep them clean {0,1}
        tb = TF.affine(tb, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST)
        mb = TF.affine(mb, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST)
        eb = TF.affine(eb, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR)

    # Mild photometric jitter on normalized RGB.
    if float(torch.rand(1).item()) < 0.8:
        # Increase gain variance
        gain = 0.70 + 0.60 * float(torch.rand(1).item())  # [0.70, 1.30]
        bias = (float(torch.rand(1).item()) - 0.5) * 0.40  # [-0.2, 0.2]
        xb = xb * gain + bias
    
    # Color jitter (applied to normalized tensors, so it acts like a mild tint/contrast shift)
    if float(torch.rand(1).item()) < 0.5:
        # Random channel scaling
        c_scale = 0.9 + 0.2 * torch.rand(3, 1, 1).to(xb.device)
        xb = xb * c_scale

    if float(torch.rand(1).item()) < 0.5:
        # Increase noise std slightly
        noise_std = 0.01 + 0.05 * float(torch.rand(1).item())  # [0.01, 0.06]
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

def _apply_hysteresis_percentile(response, neg_scr, percentile, low_percentile=None):

    valid = ~neg_scr
    resp_vals = response[valid]
    if resp_vals.size == 0:
        return np.zeros(response.shape, dtype=bool)

    maxv = float(np.max(resp_vals))
    if not np.isfinite(maxv) or maxv <= 1e-8:
        return np.zeros(response.shape, dtype=bool)

    hi_p = float(np.clip(percentile, 0.0, 100.0))
    lo_p = max(0.0, hi_p - 2.0) if low_percentile is None else float(np.clip(low_percentile, 0.0, hi_p))

    thr_hi = float(np.percentile(resp_vals, hi_p))
    thr_lo = float(np.percentile(resp_vals, lo_p))

    eps = 1e-8
    seed = (response >= thr_hi) & (response > eps) & valid
    grow = (response >= thr_lo) & (response > eps) & valid

    if not np.any(seed):
        return grow

    grow_u8 = grow.astype(np.uint8) * 255
    num, labels = cv2.connectedComponents(grow_u8, connectivity=8)
    if num <= 1:
        return seed

    keep = np.zeros(num, dtype=bool)
    seed_labels = labels[seed]
    if seed_labels.size > 0:
        keep[np.unique(seed_labels)] = True

    out = keep[labels]
    out &= valid
    return out


# def _frangi_percentile_mask(img_bgr, neg_scr, percentile=99.0, sigmas=(1, 2, 3), black_ridges=True, low_percentile=None):
#     """Compute Frangi response then apply hysteresis percentile threshold."""
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#     response = frangi(gray, sigmas=sigmas, black_ridges=black_ridges)
#     return _apply_hysteresis_percentile(response, neg_scr, percentile, low_percentile)

def compute_frangi_response(img_path, scr_path, use_clahe, clahe_clip, clahe_grid, max_size):

    img_bgr = cv2.imread(img_path)
    scr_bgr = cv2.imread(scr_path, -1)

    if scr_bgr.ndim == 3 and scr_bgr.shape[2] == 4:
        idx = np.where(scr_bgr[:, :, 3] == 0)
        scr_bgr[idx] = [0, 0, 0, 0]
        scr_bgr = cv2.cvtColor(scr_bgr, cv2.COLOR_BGRA2BGR)

    if use_clahe:
        img_bgr = apply_clahe_bgr(img_bgr, clip_limit=clahe_clip, tile_grid=clahe_grid)


    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    response = frangi(gray, sigmas=(1, 2, 3), black_ridges=True)

    return np.clip(response / (response.max() + 1e-9) * 255, 0, 255).astype(np.uint8)



def _shift_with_valid(img_f32, dy, dx):
    """Shift image by (dy, dx) with valid mask (no wrap-around).

    Returns:
      shifted: float32 array (same shape)
      valid: uint8 {0,1} array indicating pixels with a valid shifted value
    """
    h, w = img_f32.shape
    shifted = np.zeros((h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.uint8)

    if dy >= 0:
        ys_src = slice(0, h - dy)
        ys_dst = slice(dy, h)
    else:
        ys_src = slice(-dy, h)
        ys_dst = slice(0, h + dy)

    if dx >= 0:
        xs_src = slice(0, w - dx)
        xs_dst = slice(dx, w)
    else:
        xs_src = slice(-dx, w)
        xs_dst = slice(0, w + dx)

    shifted[ys_dst, xs_dst] = img_f32[ys_src, xs_src]
    valid[ys_dst, xs_dst] = 1
    return shifted, valid


def _keep_two_sided_contrast(gray_u8, candidate_mask, *, delta=8.0, distances=(1, 2)):
    """Reject edge-like candidates where intensity changes only on one side.

    We keep a candidate pixel if there exists a direction where both sides
    (along that direction) are brighter than the center by at least `delta`.

    Args:
      gray_u8: grayscale uint8 image
      candidate_mask: boolean mask of Frangi candidates
      delta: minimum per-side intensity difference in uint8 units
      distances: sample distances (in pixels) to average on each side
    """
    if candidate_mask is None or not np.any(candidate_mask):
        return candidate_mask

    g = gray_u8.astype(np.float32)
    c = g
    delta_f = float(delta)

    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    keep_any = np.zeros_like(candidate_mask, dtype=bool)

    for dx, dy in directions:
        sum_pos = np.zeros_like(g, dtype=np.float32)
        cnt_pos = np.zeros_like(g, dtype=np.uint8)
        sum_neg = np.zeros_like(g, dtype=np.float32)
        cnt_neg = np.zeros_like(g, dtype=np.uint8)

        for dist in distances:
            sp, vp = _shift_with_valid(g, dy * int(dist), dx * int(dist))
            sn, vn = _shift_with_valid(g, -dy * int(dist), -dx * int(dist))
            sum_pos += sp
            cnt_pos += vp
            sum_neg += sn
            cnt_neg += vn

        valid = (cnt_pos > 0) & (cnt_neg > 0)
        pos_mean = sum_pos / np.maximum(cnt_pos.astype(np.float32), 1.0)
        neg_mean = sum_neg / np.maximum(cnt_neg.astype(np.float32), 1.0)

        # Two-sided contrast for dark ridges: both sides brighter than center.
        cond = valid & ((pos_mean - c) > delta_f) & ((neg_mean - c) > delta_f)
        keep_any |= cond

    return candidate_mask & keep_any



def apply_frangi_percentile(frangi_path, scr_path, percentile, img_path=None, *, two_sided_delta=8.0):

    frangi_bgr = cv2.imread(frangi_path, -1)
    scr_bgr = cv2.imread(scr_path, -1)

    scr = scr_bgr

    if scr_bgr.ndim == 3 and scr_bgr.shape[2] == 4:
        idx = np.where(scr_bgr[:, :, 3] == 0)
        scr_bgr[idx] = [0, 0, 0, 0]
        scr = cv2.cvtColor(scr_bgr, cv2.COLOR_BGRA2BGR)

    neg_scr = (make_scribble_label_mask(scr) == 0)

    response = frangi_bgr.astype(np.float64) / 255.0
    pos_mask = _apply_hysteresis_percentile(response, neg_scr, float(percentile))

    # Filter out edge-like responses: keep only pixels with two-sided intensity change.
    if img_path is not None and os.path.exists(str(img_path)):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is not None:
            gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            pos_mask = _keep_two_sided_contrast(gray_u8, pos_mask, delta=float(two_sided_delta))

    # refined_bgr = (pos_mask.astype(np.uint8) * 255)
    # refined_bgr = cv2.LUT(refined_bgr, 255 - np.arange(256)).astype(np.uint8)
    a = pos_mask[:, :, np.newaxis].astype(np.float32)
    white = np.ones_like(img_bgr, dtype=np.float32) * 255.0
    blended = img_bgr.astype(np.float32) * a + white * (1.0 - a)
    return blended


def predict_line(img_path, scr_path, refined_scr_path, lr, iters, device, max_size=5000):
    img_bgr = cv2.imread(img_path)
    scr_bgr = cv2.imread(scr_path, -1)
    orig_h, orig_w = img_bgr.shape[:2]
    if scr_bgr.shape[2] == 4:
        index = np.where(scr_bgr[:, :, 3] == 0)
        scr_bgr[index] = [0, 0, 0, 0]
        scr_bgr = cv2.cvtColor(scr_bgr, cv2.COLOR_BGRA2BGR)

    cv2.imwrite("debug_scr.png", scr_bgr)
    refined_scr_bgr = cv2.imread(refined_scr_path)
    refined_scr_bgr = cv2.LUT(refined_scr_bgr, 255-np.arange(256)).astype(np.uint8)


    pos_scr = scr_bgr[:, :, 1] == 255
    neg_scr = scr_bgr[:, :, 2] == 255
    refined_pos_scr = refined_scr_bgr[:, :, 0] == 255
    cv2.imwrite("debug_refined_pos_scr.png", refined_pos_scr.astype(np.uint8)*255)
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(refined_pos_scr.astype(np.uint8), kernel, iterations=2)
    cv2.imwrite("dil.png", dil.astype(np.uint8) * 255)
    ring = (dil == 1) & (refined_pos_scr == 0)
    cv2.imwrite("ring.png", ring.astype(np.uint8) * 255)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = normalize_image(img_rgb)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    edge = compute_edge_map(img_gray)
    bg_auto = ring & (edge < 0.1)
    bg = neg_scr | bg_auto
    cv2.imwrite("debug_bg.png", bg.astype(np.uint8)*255)

    target = np.zeros_like(img_gray, dtype=np.float32)
    labeled = np.zeros_like(img_gray, dtype=np.float32)
    target[refined_pos_scr] = 1.0
    target[bg] = 0.0
    labeled[refined_pos_scr] = 1.0
    labeled[bg] = 1.0

    x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)  # 1x3xHxW
    t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device)        # 1x1xHxW
    m = torch.from_numpy(labeled).unsqueeze(0).unsqueeze(0).to(device)       # 1x1xHxW
    e = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(device)

    model = UNet(in_ch=3, base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=200
    )

    pos_count = float(((t > 0.5) * (m > 0.5)).sum().item())
    neg_count = float(((t <= 0.5) * (m > 0.5)).sum().item())
    pos_weight = (neg_count + 1.0) / (pos_count + 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)

    model.train()

    pbar = tqdm(range(iters), desc="UNet Training")
    for it in pbar:
        opt.zero_grad()

        xb, tb, mb, _ = augment_tensors(x, t, m, e, aug_prob=1.0)


        logits = model(xb)
        prob = torch.sigmoid(logits)

        loss_bce = masked_bce_with_logits(logits, tb, mb, pos_weight=pos_weight)
        loss_dice = masked_dice_loss(prob, tb, mb)

        loss = (loss_bce + loss_dice)
        loss.backward()
        opt.step()
        scheduler.step(loss.item())

        pbar.set_postfix({
            "loss_bce": f"{loss_bce.item():.4f}",
            "loss_dice": f"{loss_dice.item():.4f}",
        })

        if (it + 1) % 100 == 0 or it == 0:
            model.eval()
            with torch.no_grad():
                prob_tmp = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            model.train()
            a = prob_tmp[:, :, np.newaxis].astype(np.float32)
            white = np.ones_like(img_bgr, dtype=np.float32) * 255.0
            blended = img_bgr.astype(np.float32) * a + white * (1.0 - a)
            cv2.imwrite(f"debug_out_{it}.png", blended.astype(np.uint8))

    model.eval()
    with torch.no_grad():
        prob_tmp = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    a = prob_tmp[:, :, np.newaxis].astype(np.float32)
    white = np.ones_like(img_bgr, dtype=np.float32) * 255.0
    blended = img_bgr.astype(np.float32) * a + white * (1.0 - a)
    out = blended.astype(np.uint8)
    if out.shape[:2] != (orig_h, orig_w):
        out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("debug_out_resized.png", out)
    return out
