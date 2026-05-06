import argparse
from contextlib import nullcontext
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import warnings
if not hasattr(np, 'warnings'):
    np.warnings = warnings
from skimage.filters import frangi
from tqdm import tqdm
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
        # x4_pool = self.pool4(x4)
        # x5 = self.conv5(x4_pool)    
        # x5_pool = self.pool5(x5)
        # x6 = self.conv5_1(x5_pool)
        

        # y4 = self.up1(x5)
        # if y4.shape[2:] != x4.shape[2:]: # 入力サイズが奇数の場合
        #     y4 = F.interpolate(y4, size=x4.shape[2:], mode='bilinear', align_corners=False)
        # y4 = torch.cat([y4, x4], dim=1)
        # y4 = self.conv6(y4)

        y3 = self.up2(x4)
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




def _ensure_bgr_u8(img_u8):
    if img_u8 is None:
        return None
    if img_u8.ndim == 2:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    if img_u8.ndim == 3 and img_u8.shape[2] == 1:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    if img_u8.ndim == 3 and img_u8.shape[2] == 4:
        return cv2.cvtColor(img_u8, cv2.COLOR_BGRA2BGR)
    return img_u8


# def compute_edge_map(gray_img):
#     '''
#     Compute edge map from grayscale image.
#     '''
#     g = cv2.GaussianBlur(gray_img, (0, 0), 1.0)
#     gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
#     mag = cv2.magnitude(gx, gy)
#     mag = mag / (mag.max() + 1e-6)
#     return mag

def make_scribble_label_mask(scr_u8):
    """
    1: 線画
    0: 背景
    -1: 未定義
    """
    b, g, r = cv2.split(scr_u8)

    line = (r < 5) & (g > 250) & (b < 5)
    bg = (r >250) & (g < 5) & (b < 5)
    mask = np.full(scr_u8.shape[:2], -1, dtype=np.int8)
    mask[line] = 1
    mask[bg] = 0
    return mask



def extract_line_pixels_in_scribble(img_u8, pos_scr, neg_scr=None, sigmas=(1, 2, 3)):
    """
    img_u8: 入力画像(uint8, HxWxC)
    pos_scr: 正のスクリブル(bool, HxW)
    neg_scr: 負のスクリブル(bool, HxW)
    sigmas: Frangiスケール(tuple)
    """
    img_u8 = _ensure_bgr_u8(img_u8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    refined = np.zeros_like(pos_scr, dtype=bool)
    scr_u8 = pos_scr.astype(np.uint8)
    # 正のスクリブルを連結成分に分割
    num_labels, labels = cv2.connectedComponents(scr_u8, connectivity=8)

    for lab_id in range(1, num_labels):
        region_mask = labels == lab_id

        ys, xs = np.where(region_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        patch = gray[y0:y1, x0:x1]
        response_patch = frangi(patch, sigmas=sigmas, black_ridges=True)

        local_mask = region_mask[y0:y1, x0:x1]
        resp_vals = response_patch[local_mask]

        # Frangi応答が小さすぎる場合はスキップ
        if resp_vals.size == 0 or float(resp_vals.max()) < 1e-8:
            continue

        # 正規化
        resp_norm = resp_vals / (float(resp_vals.max()) + 1e-9)
        resp_u8 = np.clip(resp_norm * 255.0, 0, 255).astype(np.uint8)

        # 1色のみの場合
        if len(np.unique(resp_u8)) < 2:
            continue

        # Otsuの二値化
        otsu_thresh, _ = cv2.threshold(resp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = max(float(otsu_thresh) * 0.5, 1.0)
        line_pixels = resp_u8 >= low_thresh

        if int(line_pixels.sum()) == 0:
            continue
        
        # ローカル座標から大域座標へ
        local_coords = np.argwhere(local_mask)
        chosen = local_coords[line_pixels]
        refined[chosen[:, 0] + y0, chosen[:, 1] + x0] = True

    return refined


def refine_scribble(img_u8, scr_u8, use_clahe=False, clahe_clip=2.0, clahe_grid=8, max_size=5000, *, sigmas=(1, 2, 3)):

    img_u8 = _ensure_bgr_u8(img_u8)

    # スクリブルのアルファチャンネルを黒に
    if scr_u8.ndim == 3 and scr_u8.shape[2] == 4:
        idx = np.where(scr_u8[:, :, 3] == 0)
        scr_u8[idx] = [0, 0, 0, 0]
        scr_u8 = cv2.cvtColor(scr_u8, cv2.COLOR_BGRA2BGR)

    # CLAHEの適用
    if use_clahe:
        img_u8 = apply_clahe(img_u8, clip_limit=clahe_clip, tile_grid=clahe_grid)

    scr_mask = make_scribble_label_mask(scr_u8)
    pos_scr = scr_mask == 1
    neg_scr = scr_mask == 0

    refined_pos = extract_line_pixels_in_scribble(img_u8, pos_scr, neg_scr=neg_scr, sigmas=sigmas)

    # 白黒反転
    refined_pos_inv = ~refined_pos
    return refined_pos_inv.astype(np.uint8) * 255





def apply_clahe(image, clip_limit=2.0, tile_grid=8):
    """
    コントラスト正規化
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def normalize_image(img_rgb):
    """
    学習安定化のための正規化
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img_rgb - mean) / std


def augment_tensors(x, t, m, aug_prob=0.8):
    """
    オーグメンテーション
    """

    # 一定確率でスキップ
    if float(torch.rand(1).item()) >= float(aug_prob):
        return x, t, m

    xb, tb, mb = x, t, m

    # 横方向反転
    if float(torch.rand(1).item()) < 0.5:
        xb = torch.flip(xb, dims=[3])
        tb = torch.flip(tb, dims=[3])
        mb = torch.flip(mb, dims=[3])
    # 縦方向反転
    if float(torch.rand(1).item()) < 0.5:
        xb = torch.flip(xb, dims=[2])
        tb = torch.flip(tb, dims=[2])
        mb = torch.flip(mb, dims=[2])

    # 90*k度回転
    k = int(torch.randint(0, 4, (1,)).item()) # 0, 1, 2, 3
    if k > 0:
        xb = torch.rot90(xb, k=k, dims=[2, 3])
        tb = torch.rot90(tb, k=k, dims=[2, 3])
        mb = torch.rot90(mb, k=k, dims=[2, 3])

    # アフィン変換
    if float(torch.rand(1).item()) < 0.7:
        # 連続化
        xb = xb.contiguous()
        tb = tb.contiguous()
        mb = mb.contiguous()

        angle = float((torch.rand(1).item() - 0.5) * 30.0) # -15 ~ 15度
        translate = [float((torch.rand(1).item() - 0.5) * 0.1), float((torch.rand(1).item() - 0.5) * 0.1)] # -10% ~ 10% translation
        scale = float(0.8 + 0.4 * torch.rand(1).item()) # 0.8x ~ 1.2x scale
        shear = float((torch.rand(1).item() - 0.5) * 10.0) # -5 ~ 5 度せん断

        xb = TF.affine(xb, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR)
        tb = TF.affine(tb, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST)
        mb = TF.affine(mb, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST)

    # # 輝度変動
    # if float(torch.rand(1).item()) < 0.8:
    #     gain = 0.70 + 0.60 * float(torch.rand(1).item())  # 0.70 ~ 1.30
    #     bias = (float(torch.rand(1).item()) - 0.5) * 0.40  # -0.2 ~ 0.2
    #     xb = xb * gain + bias
    
    # # 色相変動
    # if float(torch.rand(1).item()) < 0.5:
    #     c_scale = 0.9 + 0.2 * torch.rand(3, 1, 1).to(xb.device)
    #     xb = xb * c_scale

    # # ノイズ
    # if float(torch.rand(1).item()) < 0.5:
    #     noise_std = 0.01 + 0.05 * float(torch.rand(1).item())  # 0.01 ~ 0.06
    #     xb = xb + torch.randn_like(xb) * noise_std

    return xb, tb, mb


def masked_bce_with_logits(logits, targets01, labeled_mask, pos_weight=1.0):
    """
    重み付き二値交差エントロピー損失
    """
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets01, reduction="none", pos_weight=pw
    )
    loss = loss * labeled_mask
    denom = labeled_mask.sum().clamp_min(1.0)
    return loss.sum() / denom

def masked_dice_loss(prob, targets01, labeled_mask, smooth=1.0):
    """
    dice損失：重なり度を最大化する損失
    """
    prob = prob * labeled_mask
    tgt = targets01 * labeled_mask
    inter = (prob * tgt).sum()
    denom = prob.sum() + tgt.sum()
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice

def compute_frangi_response(img_u8, use_clahe, clahe_clip, clahe_grid, max_size):
    img_u8 = _ensure_bgr_u8(img_u8)

    if use_clahe:
        img_u8 = apply_clahe(img_u8, clip_limit=clahe_clip, tile_grid=clahe_grid)


    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    response = frangi(gray, sigmas=(1, 2, 3), black_ridges=True)

    return np.clip(response / (response.max() + 1e-9) * 255, 0, 255).astype(np.uint8)





def apply_frangi_percentile(frangi_u8, percentile, img_u8, *, two_sided_delta=8.0):
    img_u8 = _ensure_bgr_u8(img_u8)

    response = frangi_u8.astype(np.float64) / 255.0
    thr = float(np.percentile(response, float(np.clip(percentile, 0.0, 100.0))))
    pos_mask = response >= thr

    a = pos_mask[:, :, np.newaxis].astype(np.float32)
    white = np.ones_like(img_u8, dtype=np.float32) * 255.0
    blended = img_u8.astype(np.float32) * a + white * (1.0 - a)
    return blended


def predict_line(img_u8, scr_u8, refined_scr_u8, lr, iters, device, progress_bar=None, preview=None, max_size=5000, cancel_flag=None):
    img_u8 = _ensure_bgr_u8(img_u8)
    orig_h, orig_w = img_u8.shape[:2]
    if scr_u8.shape[2] == 4:
        index = np.where(scr_u8[:, :, 3] == 0)
        scr_u8[index] = [0, 0, 0, 0]
        scr_u8 = cv2.cvtColor(scr_u8, cv2.COLOR_BGRA2BGR)

    # cv2.imwrite("debug_scr.png", scr_bgr)
    refined_scr_u8 = cv2.LUT(refined_scr_u8, 255-np.arange(256)).astype(np.uint8)


    pos_scr = scr_u8[:, :, 1] == 255
    neg_scr = scr_u8[:, :, 2] == 255
    refined_pos_scr = refined_scr_u8 == 255
    # cv2.imwrite("debug_refined_pos_scr.png", refined_pos_scr.astype(np.uint8)*255)
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(refined_pos_scr.astype(np.uint8), kernel, iterations=2)
    # cv2.imwrite("dil.png", dil.astype(np.uint8) * 255)
    ring = (dil == 1) & (refined_pos_scr == 0)
    # cv2.imwrite("ring.png", ring.astype(np.uint8) * 255)

    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = normalize_image(img_rgb)
    img_gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    # edge = compute_edge_map(img_gray)
    bg_auto = ring & pos_scr
    bg = neg_scr | bg_auto
    # cv2.imwrite("debug_bg.png", bg.astype(np.uint8)*255)
    
    target = np.zeros_like(img_gray, dtype=np.float32)
    labeled = np.zeros_like(img_gray, dtype=np.float32)
    target[refined_pos_scr] = 1.0
    target[bg] = 0.0
    labeled[refined_pos_scr] = 1.0
    labeled[bg] = 1.0

    x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)  # 1xCxHxW
    t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device)        # 1x1xHxW
    m = torch.from_numpy(labeled).unsqueeze(0).unsqueeze(0).to(device)       # 1x1xHxW
    # e = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(device)

    model = UNet(in_ch=x.shape[1], base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr * 10, total_steps=iters
    )

    pos_count = float(((t > 0.5) * (m > 0.5)).sum().item())
    neg_count = float(((t <= 0.5) * (m > 0.5)).sum().item())
    pos_weight = (neg_count + 1.0) / (pos_count + 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)

    model.train()

    for it in range(iters):
        if cancel_flag is not None and cancel_flag.is_set():
            break

        opt.zero_grad()

        xb, tb, mb = augment_tensors(x, t, m, aug_prob=0.9)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            logits = model(xb)
            prob = torch.sigmoid(logits)

        loss_bce = masked_bce_with_logits(
            logits.float(), tb.float(), mb.float(), pos_weight=pos_weight
        )
        loss_dice = masked_dice_loss(prob.float(), tb.float(), mb.float())

        loss = (loss_bce + loss_dice)
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()
        
        if progress_bar is not None:
            progress_bar(it + 1, iters, loss.item())

        if (it + 1) % 10 == 0 or it == 0:
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    prob_tmp = torch.sigmoid(model(x)).squeeze().float().cpu().numpy()
            model.train()
            a = prob_tmp[:, :, np.newaxis].astype(np.float32)
            white = np.ones_like(img_u8, dtype=np.float32) * 255.0
            blended = img_u8.astype(np.float32) * a + white * (1.0 - a)
            out = blended.astype(np.uint8)
            if out.shape[:2] != (orig_h, orig_w):
                out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if preview is not None:
                preview(out)

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            prob_tmp = torch.sigmoid(model(x)).squeeze().float().cpu().numpy()
    a = prob_tmp[:, :, np.newaxis].astype(np.float32)
    white = np.ones_like(img_u8, dtype=np.float32) * 255.0
    blended = img_u8.astype(np.float32) * a + white * (1.0 - a)
    out = blended.astype(np.uint8)
    if out.shape[:2] != (orig_h, orig_w):
        out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite("debug_out_resized.png", out)
    return out
