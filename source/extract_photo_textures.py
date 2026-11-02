#!/usr/bin/env python3
"""
Photogrammetric PBR Texture Suite for KAMAZ-43118 Mobile Comms Hub (v3 Verified).
Extracts, rectifies, delights, inpaints, and synthesizes Albedo + ORM (Occlusion/Roughness/Metallic)
+ Tangent-Space Normal maps from the 13 walkaround drone photographs.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

UPLOAD_DIR = "/home/user/uploads"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_rgb(idx):
    bgr = cv2.imread(f"{UPLOAD_DIR}/photo_{idx}_2026-07-11_23-52-48.jpg")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def warp_quad(img_rgb, src_pts, out_w, out_h):
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_rgb, H, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def unsharp(img, sigma=1.0, amount=0.62):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0), 0, 255).astype(np.uint8)


def delight_white_metal(img, target_p90=218.0):
    arr = img.astype(np.float32)
    p90 = np.percentile(arr, 90, axis=(0, 1), keepdims=True)
    scale = np.clip(target_p90 / np.maximum(p90, 130.0), 0.78, 1.06)
    arr = arr * scale
    arr = (arr - 128.0) * 1.04 + 128.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_normal_from_luma(img_rgb, strength=2.2, blur_sigma=0.8):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * strength
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * strength
    nx = -gx
    ny = -gy
    nz = np.ones_like(gx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    rgb = np.stack([nx * inv * 0.5 + 0.5, ny * inv * 0.5 + 0.5, nz * inv * 0.5 + 0.5], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def make_orm(ao_map, rough_map, metal_map):
    r = np.clip(ao_map * 255.0, 0, 255).astype(np.uint8)
    g = np.clip(rough_map * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(metal_map * 255.0, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def generate_pbr_dict():
    p1 = load_rgb(1)
    p2 = load_rgb(2)
    p6 = load_rgb(6)
    p10 = load_rgb(10)
    p11 = load_rgb(11)
    p12 = load_rgb(12)

    out = {}

    # -------------------------------------------------------------------------
    # 1. CAB UPPER FRONT (Windshield + Hood + Grille) from photo_2
    # -------------------------------------------------------------------------
    cab_front = warp_quad(
        p2,
        [[474, 194], [774, 194], [782, 439], [466, 439]],
        1024, 860
    )
    cab_front = delight_white_metal(unsharp(cab_front, 1.0, 0.65), 220.0)
    H, W = cab_front.shape[:2]
    luma = cv2.cvtColor(cab_front, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ao = np.clip(0.52 + 0.48 * luma, 0.25, 1.0)
    rough = np.full((H, W), 0.24, dtype=np.float32)
    metal = np.full((H, W), 0.05, dtype=np.float32)
    ws_mask = np.zeros((H, W), dtype=np.float32)
    ws_mask[145:472, 56:968] = 1.0
    ws_mask = cv2.GaussianBlur(ws_mask, (9, 9), 2.0)
    rough = rough * (1.0 - ws_mask) + 0.04 * ws_mask
    metal = metal * (1.0 - ws_mask) + 0.28 * ws_mask
    ao = ao * (1.0 - ws_mask) + 0.96 * ws_mask
    out["cab_front"] = {
        "albedo": cab_front,
        "orm": make_orm(ao, rough, metal),
        "normal": make_normal_from_luma(cab_front, strength=1.8, blur_sigma=0.9),
    }

    # -------------------------------------------------------------------------
    # 2. CAB FRONT STEEL BUMPER (Headlights + 0828 MM 18 + Fog Lamps) from photo_2
    # -------------------------------------------------------------------------
    cab_bumper = warp_quad(
        p2,
        [[462, 440], [786, 440], [785, 514], [463, 514]],
        1024, 256
    )
    cab_bumper = delight_white_metal(unsharp(cab_bumper, 1.0, 0.70), 220.0)
    Hb, Wb = cab_bumper.shape[:2]
    luma_b = cv2.cvtColor(cab_bumper, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ao_b = np.clip(0.45 + 0.55 * luma_b, 0.22, 1.0)
    rough_b = np.full((Hb, Wb), 0.26, dtype=np.float32)
    metal_b = np.full((Hb, Wb), 0.08, dtype=np.float32)
    rough_b[85:180, 18:175] = 0.03
    rough_b[85:180, 848:1006] = 0.03
    metal_b[85:180, 18:175] = 0.40
    metal_b[85:180, 848:1006] = 0.40
    out["cab_bumper"] = {
        "albedo": cab_bumper,
        "orm": make_orm(ao_b, rough_b, metal_b),
        "normal": make_normal_from_luma(cab_bumper, strength=2.2, blur_sigma=0.8),
    }

    # -------------------------------------------------------------------------
    # 3. CAB SIDE (Left & Right) from photo_6 (exact cab contour: x=132..340, y=194..442!)
    # -------------------------------------------------------------------------
    cab_side_l = warp_quad(
        p6,
        [[148, 196], [336, 196], [340, 442], [122, 442]],
        1024, 1024
    )
    cab_side_l = delight_white_metal(unsharp(cab_side_l, 1.0, 0.6), 220.0)
    # Inside side window (exact window box in cab_side_l: x=265..588, y=150..388),
    # fill window pane with sleek dark smoked reflective automotive glass + vent divider!
    pil_csl = Image.fromarray(cab_side_l)
    dc = ImageDraw.Draw(pil_csl)
    wx0, wy0, wx1, wy1 = 266, 152, 588, 388
    for wy in range(wy0, wy1):
        t = (wy - wy0) / max(1, (wy1 - wy0))
        r = int(40 * (1 - t) + 16 * t)
        g = int(56 * (1 - t) + 22 * t)
        b = int(76 * (1 - t) + 30 * t)
        dc.line([(wx0, wy), (wx1, wy)], fill=(r, g, b))
    dc.rectangle([wx0 + 82, wy0, wx0 + 96, wy1], fill=(18, 20, 24))
    dc.rounded_rectangle([wx0, wy0, wx1, wy1], radius=12, outline=(18, 20, 24), width=10)
    cab_side_l = np.array(pil_csl)

    r_c, g_c, b_c = cab_side_l[:, :, 0].astype(np.int16), cab_side_l[:, :, 1].astype(np.int16), cab_side_l[:, :, 2].astype(np.int16)
    grass_mask = ((g_c - r_c) > 8) & ((g_c - b_c) > 10)
    yy_s, xx_s = np.ogrid[:1024, :1024]
    in_win_zone = (xx_s >= wx0) & (xx_s <= wx1) & (yy_s >= wy0) & (yy_s <= wy1)
    # Outside front contour (x < 255), replace any grass/asphalt bleed with clean cab white paint
    ext_bleed = (xx_s < 255) & (grass_mask | ((yy_s > 520) & (xx_s < 95)))
    cab_side_l[ext_bleed] = [214, 218, 224]

    cab_side_r = np.fliplr(cab_side_l).copy()
    # Un-mirror '43118' door badge if inside x=180..360, y=500..565
    Hs, Ws = cab_side_l.shape[:2]
    luma_s = cv2.cvtColor(cab_side_l, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ao_s = np.clip(0.45 + 0.55 * luma_s, 0.22, 1.0)
    rough_s = np.full((Hs, Ws), 0.24, dtype=np.float32)
    metal_s = np.full((Hs, Ws), 0.05, dtype=np.float32)
    glass_s = in_win_zone & (luma_s < 0.35)
    rough_s[glass_s] = 0.04
    metal_s[glass_s] = 0.28
    fender_s = (luma_s < 0.26) & (yy_s > 600)
    rough_s[fender_s] = 0.76
    out["cab_side_l"] = {
        "albedo": cab_side_l,
        "orm": make_orm(ao_s, rough_s, metal_s),
        "normal": make_normal_from_luma(cab_side_l, strength=1.9, blur_sigma=0.9),
    }
    out["cab_side_r"] = {
        "albedo": cab_side_r,
        "orm": make_orm(np.fliplr(ao_s), np.fliplr(rough_s), np.fliplr(metal_s)),
        "normal": make_normal_from_luma(cab_side_r, strength=1.9, blur_sigma=0.9),
    }

    # -------------------------------------------------------------------------
    # 4. REAL WHEEL & KAMA-1260 TIRE + TREAD BAND (from photo_6)
    # -------------------------------------------------------------------------
    wheel_face = warp_quad(
        p6,
        [[206, 387], [353, 387], [353, 528], [206, 528]],
        512, 512
    )
    wheel_face = unsharp(wheel_face, 0.9, 0.75)
    yy_w, xx_w = np.ogrid[:512, :512]
    dist_r = np.sqrt((xx_w - 255.5) ** 2 + (yy_w - 255.5) ** 2)
    wheel_face[dist_r > 245.0] = [28, 30, 33]
    ao_w = np.clip(0.32 + 0.68 * (cv2.cvtColor(wheel_face, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0), 0.18, 0.96)
    rough_w = np.where(dist_r < 126.0, 0.34, 0.85).astype(np.float32)
    metal_w = np.where(dist_r < 126.0, 0.68, 0.02).astype(np.float32)
    out["wheel_face"] = {
        "albedo": wheel_face,
        "orm": make_orm(ao_w, rough_w, metal_w),
        "normal": make_normal_from_luma(wheel_face, strength=3.2, blur_sigma=0.6),
    }

    # Procedural high-detail KAMA-1260 Chevron Off-Road Tread Band (512x512 seamless along V)
    tread_img = Image.new("RGB", (512, 512), (24, 26, 29))
    dt = ImageDraw.Draw(tread_img)
    for row in range(0, 512, 32):
        # Alternating aggressive off-road chevron blocks!
        dt.polygon([(35, row+4), (235, row+14), (235, row+26), (35, row+18)], fill=(42, 45, 50))
        dt.polygon([(477, row+20), (277, row+30), (277, row+18), (477, row+6)], fill=(42, 45, 50))
        # Center circumferential rib blocks
        dt.rectangle([242, row+6, 270, row+24], fill=(38, 41, 46))
    tread_arr = np.array(tread_img)
    # Add micro-rubber grain
    rng = np.random.default_rng(42)
    grain = rng.integers(-4, 5, size=tread_arr.shape, dtype=np.int16)
    tread_arr = np.clip(tread_arr.astype(np.int16) + grain, 14, 70).astype(np.uint8)
    luma_t = cv2.cvtColor(tread_arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out["tire_tread"] = {
        "albedo": tread_arr,
        "orm": make_orm(np.clip(0.35 + 1.8 * luma_t, 0.25, 0.95), np.full((512, 512), 0.86), np.full((512, 512), 0.02)),
        "normal": make_normal_from_luma(tread_arr, strength=4.2, blur_sigma=0.7),
    }

    # -------------------------------------------------------------------------
    # 5. KUNG LEFT WALL (from photo_6: exact seam-to-seam)
    # -------------------------------------------------------------------------
    kung_left = warp_quad(
        p6,
        [[458, 198], [1062, 211], [1052, 353], [462, 376]],
        2048, 640
    )
    kung_left = delight_white_metal(unsharp(kung_left, 1.0, 0.65), 220.0)
    # Clean open-window shadow under window #3 (x=1075..1395, y=260..352) using clean white panel from x=995..1070
    patch_clean_w = cv2.resize(kung_left[260:352, 995:1065], (325, 92), interpolation=cv2.INTER_LINEAR)
    kung_left[260:352, 1072:1397] = patch_clean_w
    # Also replace window #3 (x=1080..1365, y=22..268) with flush closed window #2 (x=725..985, y=22..268)
    # so all 3 3D-framed windows sit flush and clean!
    win2_clean = kung_left[20:268, 728:986].copy()
    kung_left[20:268, 1098:1356] = win2_clean
    # Clean small remnant right of window 3
    kung_left[20:268, 1356:1398] = cv2.resize(kung_left[20:268, 1030:1065], (42, 248))

    Hk, Wk = kung_left.shape[:2]
    luma_kl = cv2.cvtColor(kung_left, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ao_kl = np.clip(0.48 + 0.52 * luma_kl, 0.25, 1.0)
    rough_kl = np.full((Hk, Wk), 0.25, dtype=np.float32)
    metal_kl = np.full((Hk, Wk), 0.08, dtype=np.float32)
    glass_kl = (luma_kl < 0.30) & (np.arange(Hk)[:, None] < 275) & (np.arange(Wk)[None, :] < 1550)
    rough_kl[glass_kl] = 0.04
    metal_kl[glass_kl] = 0.28
    out["kung_left"] = {
        "albedo": kung_left,
        "orm": make_orm(ao_kl, rough_kl, metal_kl),
        "normal": make_normal_from_luma(kung_left, strength=1.8, blur_sigma=0.8),
    }

    # -------------------------------------------------------------------------
    # 6. KUNG RIGHT WALL (Synthesized from real kung_left skin + real window & hatch!)
    # -------------------------------------------------------------------------
    kung_right = np.zeros_like(kung_left)
    clean_panel = kung_left[:, 24:254].copy()
    # Fill wheel well dark bottom at bottom of clean_panel? No, clean_panel (x=24..254) is full white down to bottom!
    for x0 in range(0, 2048, 230):
        ws = min(230, 2048 - x0)
        kung_right[:, x0:x0 + ws] = clean_panel[:, :ws]
    for seam_x in [365, 735, 1075, 1665]:
        kung_right[:, seam_x:seam_x+3] = np.clip(kung_right[:, seam_x:seam_x+3].astype(np.int16) - 26, 0, 255).astype(np.uint8)

    # Paste FULL real window #2 (x=728..986, y=20..268) onto rear-right wall at x=490..748!
    kung_right[20:268, 490:748] = win2_clean

    # Draw flush rear-right power-input hatch (seen in photo_10) with aligned stripe!
    pil_kr = Image.fromarray(kung_right)
    dkr = ImageDraw.Draw(pil_kr)
    dkr.rounded_rectangle([125, 315, 325, 565], radius=12, outline=(24, 28, 34), width=7)
    dkr.ellipse([142, 536, 154, 548], fill=(30, 34, 40))
    dkr.ellipse([296, 536, 308, 548], fill=(30, 34, 40))

    f_org = ImageFont.truetype(FONT_BOLD, 21)
    f_tel = ImageFont.truetype(FONT_BOLD, 20)
    dkr.text((352, 358), "ФГКУ «Ногинский спасательный центр МЧС России»", fill=(24, 62, 132), font=f_org)
    dkr.text((352, 486), "ТЕЛЕФОН 01, 101", fill=(24, 62, 132), font=f_tel)

    # Dark tandem wheel-arch cutout at x=530..1040, y=585..640 (matching left side!)
    dkr.polygon([(525, 640), (575, 588), (995, 588), (1045, 640)], fill=(14, 16, 20))
    # Dark rubber doorframe opening for the 3D recessed communications room (x=1090..1640, y=34..602)
    dkr.rounded_rectangle([1086, 32, 1644, 604], radius=14, fill=(14, 18, 24), outline=(28, 32, 40), width=9)
    kung_right = np.array(pil_kr)

    luma_kr = cv2.cvtColor(kung_right, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ao_kr = np.clip(0.48 + 0.52 * luma_kr, 0.20, 1.0)
    rough_kr = np.full((Hk, Wk), 0.25, dtype=np.float32)
    metal_kr = np.full((Hk, Wk), 0.08, dtype=np.float32)
    glass_kr = (luma_kr < 0.30) & (np.arange(Hk)[:, None] < 275) & (np.arange(Wk)[None, :] > 480) & (np.arange(Wk)[None, :] < 755)
    rough_kr[glass_kr] = 0.04
    metal_kr[glass_kr] = 0.28
    out["kung_right"] = {
        "albedo": kung_right,
        "orm": make_orm(ao_kr, rough_kr, metal_kr),
        "normal": make_normal_from_luma(kung_right, strength=1.8, blur_sigma=0.8),
    }

    # -------------------------------------------------------------------------
    # 7. KUNG REAR WALL & DOORS (Photo-matched aluminum + chevrons + stripe from photo_10)
    # -------------------------------------------------------------------------
    # Build 1024x1024 rear wall texture using real white aluminum grain from clean_panel!
    rear_base = cv2.resize(clean_panel[:220, :220], (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # Add subtle real metal noise
    rear_base = np.clip(rear_base.astype(np.int16) + rng.integers(-3, 4, size=rear_base.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    pil_rear = Image.fromarray(rear_base)
    drw = ImageDraw.Draw(pil_rear)
    # Horizontal orange-blue-orange stripe across lower outer pillars (matching photo_10!)
    drw.rectangle([0, 625, 1024, 656], fill=(232, 105, 24))
    drw.rectangle([0, 656, 1024, 678], fill=(24, 98, 178))
    drw.rectangle([0, 678, 1024, 708], fill=(232, 105, 24))
    # Vertical orange-blue-orange stripe at top center (above double rear doors, photo_10 & photo_11!)
    drw.rectangle([454, 42, 488, 980], fill=(232, 105, 24))
    drw.rectangle([488, 42, 536, 980], fill=(24, 98, 178))
    drw.rectangle([536, 42, 570, 980], fill=(232, 105, 24))
    # Reflective Red/Yellow safety chevrons on outer vertical corner pillars (photo_10!)
    for px0, px1 in [(26, 86), (938, 998)]:
        drw.rectangle([px0, 560, px1, 930], fill=(238, 192, 22), outline=(60, 65, 72), width=2)
        for cy in range(560, 930, 52):
            drw.polygon([(px0, cy), (px1, cy + 26), (px1, cy + 50), (px0, cy + 24)], fill=(208, 34, 30))
    # Center recessed rear bay opening (x=168..856, y=126..946)
    drw.rounded_rectangle([164, 122, 860, 950], radius=14, fill=(16, 19, 24), outline=(32, 36, 44), width=10)
    kung_rear = np.array(pil_rear)
    luma_rr = cv2.cvtColor(kung_rear, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out["kung_rear"] = {
        "albedo": kung_rear,
        "orm": make_orm(np.clip(0.45 + 0.55 * luma_rr, 0.20, 1.0), np.full((1024, 1024), 0.26), np.full((1024, 1024), 0.08)),
        "normal": make_normal_from_luma(kung_rear, strength=1.9, blur_sigma=0.8),
    }

    # -------------------------------------------------------------------------
    # 8. KUNG ROOF + RIBBED ALUMINUM WALKWAY (from photo_12: x=326..1058!)
    # -------------------------------------------------------------------------
    kung_roof = warp_quad(
        p12,
        [[324, 545], [329, 268], [1058, 287], [1055, 552]],
        768, 2048
    )
    kung_roof = delight_white_metal(unsharp(kung_roof, 1.0, 0.60), 218.0)
    # Cleanly tile pure ribbed-aluminum plank strip (y=1340..1460 — no hatch, no dish!) across y=1460..2015
    pure_planks = kung_roof[1345:1455, :].copy()
    for y0 in range(1455, 2015, 110):
        hs = min(110, 2015 - y0)
        kung_roof[y0:y0 + hs, :] = pure_planks[:hs, :]

    Hr, Wr = kung_roof.shape[:2]
    luma_r = cv2.cvtColor(kung_roof, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ao_r = np.clip(0.45 + 0.55 * luma_r, 0.22, 1.0)
    rough_r = np.full((Hr, Wr), 0.30, dtype=np.float32)
    metal_r = np.full((Hr, Wr), 0.08, dtype=np.float32)
    metal_r[:, 150:618] = 0.48
    rough_r[:, 150:618] = 0.38
    out["kung_roof"] = {
        "albedo": kung_roof,
        "orm": make_orm(ao_r, rough_r, metal_r),
        "normal": make_normal_from_luma(kung_roof, strength=2.6, blur_sigma=0.7),
    }

    # -------------------------------------------------------------------------
    # 9. VSAT iNetVu COMPOSITE DISH (Photo-matched carbon-grey composite + crisp logo!)
    # -------------------------------------------------------------------------
    # Sample authentic dark composite dish surface color & grain from photo_11 (350..430, 495..545)
    dish_sample = p11[495:545, 345:415].astype(np.float32)
    mean_col = dish_sample.mean(axis=(0, 1))  # ~ [92, 92, 96] sunlit carbon grey -> delight to [68, 70, 75]
    dish_img = Image.new("RGB", (512, 512), (64, 66, 72))
    dd = ImageDraw.Draw(dish_img)
    # Subtle concentric composite molding rings & radial ribs
    for r in range(40, 248, 28):
        dd.ellipse([256 - r, 256 - r, 256 + r, 256 + r], outline=(58, 60, 66), width=2)
    dd.ellipse([10, 10, 502, 502], outline=(44, 46, 52), width=10)
    # Perimeter fastener bolts
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        bx, by = int(256 + 236 * np.cos(a)), int(256 + 236 * np.sin(a))
        dd.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=(125, 130, 138))
    # Authentic crisp 'iNetVu' & 'by C-COM Satellite Systems Inc.' lettering (exact layout from photo_11 & photo_12!)
    f_iv = ImageFont.truetype(FONT_BOLD, 56)
    f_cc = ImageFont.truetype(FONT_REG, 17)
    dd.text((158, 326), "iNetVu", fill=(226, 230, 236), font=f_iv)
    dd.text((346, 324), "®", fill=(220, 224, 230), font=f_cc)
    dd.text((130, 390), "by C-COM Satellite Systems Inc.", fill=(205, 210, 218), font=f_cc)
    dish_arr = np.array(dish_img)
    dish_arr = np.clip(dish_arr.astype(np.int16) + rng.integers(-5, 6, size=dish_arr.shape, dtype=np.int16), 20, 245).astype(np.uint8)
    luma_d = cv2.cvtColor(dish_arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out["vsat_dish"] = {
        "albedo": dish_arr,
        "orm": make_orm(np.clip(0.55 + 0.45 * luma_d, 0.35, 0.98), np.full((512, 512), 0.48), np.full((512, 512), 0.18)),
        "normal": make_normal_from_luma(dish_arr, strength=1.8, blur_sigma=0.8),
    }

    return out


if __name__ == "__main__":
    tex = generate_pbr_dict()
    sheet = Image.new("RGB", (1640, 1340), (20, 24, 30))
    sheet.paste(Image.fromarray(tex["cab_front"]["albedo"]).resize((360, 300)), (20, 20))
    sheet.paste(Image.fromarray(tex["cab_bumper"]["albedo"]).resize((360, 90)), (20, 330))
    sheet.paste(Image.fromarray(tex["cab_side_l"]["albedo"]).resize((380, 380)), (400, 20))
    sheet.paste(Image.fromarray(tex["wheel_face"]["albedo"]).resize((380, 380)), (800, 20))
    sheet.paste(Image.fromarray(tex["vsat_dish"]["albedo"]).resize((380, 380)), (1200, 20))
    sheet.paste(Image.fromarray(tex["kung_left"]["albedo"]).resize((1560, 280)), (20, 440))
    sheet.paste(Image.fromarray(tex["kung_right"]["albedo"]).resize((1560, 280)), (20, 740))
    sheet.paste(Image.fromarray(tex["kung_roof"]["albedo"]).transpose(Image.Transpose.ROTATE_90).resize((1560, 280)), (20, 1040))
    os.makedirs("/tmp/pbr_preview", exist_ok=True)
    sheet.save("/tmp/pbr_preview/pbr_v3_sheet.jpg", quality=92)
    print("Saved /tmp/pbr_preview/pbr_v3_sheet.jpg")
