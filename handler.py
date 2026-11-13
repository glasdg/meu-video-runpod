#!/usr/bin/env python3
"""
RunPod Serverless GPU Worker - Renderizador de Vídeo Completo (Timeline + Legendas + Efeitos + Áudio)
Recebe o projeto JSON da aba "6. Finalizar Vídeo -> Capcut (Timeline)",
baixa todas as mídias/áudios, renderiza os clipes com movimentos Ken Burns, filtros,
personagens, mixagem de áudio/música de fundo e queima as legendas estilizadas (.ASS/.SRT) via FFmpeg GPU (NVENC).
"""

import os
import json
import uuid
import math
import time
import random
import shutil
import base64
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import runpod

try:
    import numpy as np
    import cv2
    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False

# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def run_cmd(cmd, desc="cmd"):
    print(f"[RunPod FFmpeg] {desc}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err_tail = proc.stderr[-2500:] if proc.stderr else "Unknown error"
        raise RuntimeError(f"Erro em {desc} (code {proc.returncode}):\n{err_tail}")
    return proc


def has_nvenc() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT)
        if "h264_nvenc" not in out:
            return False
        # Teste rápido para confirmar que a GPU / driver NVIDIA está realmente acessível no container
        test = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=128x128:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return test.returncode == 0
    except Exception:
        return False


def download_file(url: str, dest_path: str, timeout: int = 180) -> str:
    if not url:
        raise ValueError("URL vazia para download")
    if url.startswith("data:"):
        header, b64 = url.split(",", 1)
        with open(dest_path, "wb") as f:
            f.write(base64.b64decode(b64))
        return dest_path

    headers = {"User-Agent": "RunPod-VideoRenderer/1.0"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(f"URL retornou página HTML em vez de mídia ({url[:80]})")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(dest_path) == 0:
        raise RuntimeError(f"Arquivo baixado vazio: {url}")
    with open(dest_path, "rb") as f:
        head = f.read(64).lstrip().lower()
        if head.startswith(b"<!doctype") or head.startswith(b"<html"):
            raise RuntimeError(f"Arquivo baixado é HTML inválido ({url[:80]})")
    return dest_path


def get_visual_filter_expr(visual_filter: str) -> str:
    vf = (visual_filter or "none").lower()
    mapping = {
        "cinematic": "eq=contrast=1.10:saturation=1.20:brightness=-0.05",
        "vintage": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,eq=contrast=1.15:brightness=-0.04",
        "bw": "hue=s=0,eq=contrast=1.20",
        "noir": "hue=s=0,eq=contrast=1.50:brightness=-0.10",
        "cyberpunk": "eq=contrast=1.30:saturation=1.50:brightness=0.05,hue=h=-30",
        "neon": "eq=saturation=2.0:contrast=1.40,hue=h=45",
        "pastel": "eq=saturation=0.70:brightness=0.08:contrast=0.90",
        "warm": "colorbalance=rs=0.08:gs=0.03:bs=-0.06,eq=saturation=1.20:contrast=1.10",
        "cool": "colorbalance=rs=-0.06:gs=0.02:bs=0.08,eq=saturation=0.85:contrast=1.10:brightness=0.04",
        "high_contrast": "eq=contrast=1.50:saturation=1.10",
        "dream": "gblur=sigma=1.2,eq=brightness=0.08:saturation=1.20",
        "sepia_old": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,eq=contrast=1.25:brightness=-0.06:saturation=0.80",
        "technicolor": "eq=saturation=2.0:contrast=1.15:brightness=0.03,hue=h=5",
    }
    return mapping.get(vf, "")


def get_overlay_filter_expr(overlay: str, width: int, height: int) -> str:
    ov = (overlay or "none").lower()
    if ov == "vignette":
        return "vignette=PI/4"
    elif ov == "letterbox":
        bar_h = max(20, int(height * 0.12))
        return f"drawbox=x=0:y=0:w={width}:h={bar_h}:color=black:t=fill,drawbox=x=0:y={height - bar_h}:w={width}:h={bar_h}:color=black:t=fill"
    elif ov in ("film_grain", "vhs"):
        return "noise=alls=14:allf=t+u"
    elif ov == "scanlines":
        return "drawgrid=w=0:h=4:t=1:c=black@0.35"
    elif ov == "blood_screen":
        return "vignette=PI/3,colorbalance=rs=0.14:gs=-0.04:bs=-0.04"
    elif ov == "dream":
        return "gblur=sigma=0.8,eq=brightness=0.06"
    return ""


def compute_capcut_transform(
    movement: str,
    camera_effect: str,
    progress: float,
    width: int,
    height: int,
    duration: float,
    rng: random.Random,
):
    """Espelha exatamente getKenBurnsTransform de FinalizarVideo.tsx (com ciclo ping-pong de 8s)."""
    scale = 1.0
    tx = 0.0
    ty = 0.0

    effective_progress = progress
    if duration > 8.0:
        cycle_duration = 8.0
        current_seconds = progress * duration
        cycle_phase = (current_seconds / cycle_duration) % 2.0
        effective_progress = cycle_phase if cycle_phase <= 1.0 else (2.0 - cycle_phase)

    mov = (movement or "none").lower()
    if mov == "zoom_in":
        scale = 1.0 + 0.15 * effective_progress
    elif mov == "zoom_out":
        scale = 1.15 - 0.15 * effective_progress
    elif mov == "pan_left":
        scale = 1.2
        tx = width * 0.05 - width * 0.1 * effective_progress
    elif mov == "pan_right":
        scale = 1.2
        tx = -(width * 0.05) + width * 0.1 * effective_progress
    elif mov == "pan_up":
        scale = 1.2
        ty = height * 0.05 - height * 0.1 * effective_progress
    elif mov == "pan_down":
        scale = 1.2
        ty = -(height * 0.05) + height * 0.1 * effective_progress

    t = progress * duration
    ce = (camera_effect or "none").lower()
    if ce == "handheld":
        tx += math.sin(t * 3.0) * (width * 0.01) + math.cos(t * 4.5) * (width * 0.005)
        ty += math.cos(t * 2.5) * (height * 0.01) + math.sin(t * 5.0) * (height * 0.005)
        scale *= 1.05
    elif ce == "shake":
        tx += (rng.random() - 0.5) * (width * 0.005)
        ty += (rng.random() - 0.5) * (height * 0.005)
        scale *= 1.02
    elif ce == "earthquake":
        tx += (rng.random() - 0.5) * (width * 0.02)
        ty += (rng.random() - 0.5) * (height * 0.02)
        scale *= 1.1
    elif ce == "float":
        tx += math.sin(t * 1.0) * (width * 0.03)
        ty += math.cos(t * 0.8) * (height * 0.02)
        scale *= 1.05
    elif ce == "pulse":
        scale *= 1.0 + math.sin(t * 8.0) * 0.03
    elif ce == "rock":
        tx += math.sin(t * 1.5) * (width * 0.01)
        ty += math.cos(t * 3.0) * (height * 0.005)

    return scale, tx, ty


def build_affine_matrix(
    src_w: int,
    src_h: int,
    width: int,
    height: int,
    scale: float,
    tx: float,
    ty: float,
    custom_transform: dict | None,
):
    img_ratio = src_w / float(max(1, src_h))
    canvas_ratio = width / float(max(1, height))
    if img_ratio > canvas_ratio:
        draw_w = float(width)
        draw_h = float(src_h) * (width / float(max(1, src_w)))
    else:
        draw_h = float(height)
        draw_w = float(src_w) * (height / float(max(1, src_h)))

    t_fit = np.array([
        [draw_w / float(max(1, src_w)), 0.0, -draw_w / 2.0],
        [0.0, draw_h / float(max(1, src_h)), -draw_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    t_kb = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    ct = custom_transform or {}
    scale_ratio = width / 1280.0
    cx = (width / 2.0) + float(ct.get("x", 0.0) or 0.0) * scale_ratio
    cy = (height / 2.0) + float(ct.get("y", 0.0) or 0.0) * scale_ratio
    csx = float(ct.get("scaleX", 1.0) or 1.0)
    csy = float(ct.get("scaleY", 1.0) or 1.0)
    rot_rad = math.radians(float(ct.get("rotation", 0.0) or 0.0))
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)

    t_custom = np.array([
        [csx * cos_r, -csy * sin_r, cx],
        [csx * sin_r, csy * cos_r, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    mat = t_custom @ t_kb @ t_fit
    return mat[:2, :]


def render_clip_with_opencv(
    idx: int,
    clip: dict,
    src_path: str,
    seg_out: str,
    width: int,
    height: int,
    fps: int,
    dur: float,
    vf_post_parts: list[str],
    v_enc_args: list[str],
) -> None:
    """Renderiza clipe usando OpenCV warpAffine sub-pixel (zero tremor) + FFmpeg NVENC/x264."""
    is_video = clip.get("type") == "video"
    movement = clip.get("movement", "none")
    camera_effect = clip.get("cameraEffect", "none")
    custom_transform = clip.get("customTransform")
    speed = float(clip.get("playbackSpeed", 1.0) or 1.0)
    total_frames = max(2, int(round(dur * fps)))

    ff_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if vf_post_parts:
        ff_cmd.extend(["-vf", ",".join(vf_post_parts)])
    ff_cmd.extend([*v_enc_args, "-an", seg_out])

    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rng = random.Random(10007 * (idx + 1))

    try:
        if not is_video:
            img = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"OpenCV não conseguiu abrir imagem: {src_path}")
            src_h, src_w = img.shape[:2]
            for f_idx in range(total_frames):
                progress = f_idx / float(max(1, total_frames - 1))
                scale, tx, ty = compute_capcut_transform(movement, camera_effect, progress, width, height, dur, rng)
                mat = build_affine_matrix(src_w, src_h, width, height, scale, tx, ty, custom_transform)
                frame = cv2.warpAffine(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                proc.stdin.write(frame.tobytes())
        else:
            cap = cv2.VideoCapture(src_path)
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV não conseguiu abrir vídeo: {src_path}")
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            vid_frames = []
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                vid_frames.append(frm)
            cap.release()
            if not vid_frames:
                raise RuntimeError(f"Vídeo sem frames legíveis: {src_path}")
            src_h, src_w = vid_frames[0].shape[:2]
            n_vid = len(vid_frames)
            for f_idx in range(total_frames):
                progress = f_idx / float(max(1, total_frames - 1))
                t_sec = f_idx / float(fps)
                src_frame_idx = int((t_sec * speed) * vid_fps) % n_vid
                scale, tx, ty = compute_capcut_transform(movement, camera_effect, progress, width, height, dur, rng)
                mat = build_affine_matrix(src_w, src_h, width, height, scale, tx, ty, custom_transform)
                frame = cv2.warpAffine(vid_frames[src_frame_idx], mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                proc.stdin.write(frame.tobytes())
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

    _, stderr_bytes = proc.communicate()
    if proc.returncode != 0:
        err_tail = (stderr_bytes or b"").decode("utf-8", errors="ignore")[-2000:]
        raise RuntimeError(f"Erro OpenCV+FFmpeg clipe {idx+1} (code {proc.returncode}):\n{err_tail}")


def _hex_to_bgr(hex_str: str, default=(255, 255, 255)) -> tuple[int, int, int]:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) < 6:
        return default
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)
    except Exception:
        return default


def compute_character_anim(animation: str, t: float, width: int, height: int):
    """Espelha exatamente as 14 animações de personagem de FinalizarVideo.tsx."""
    dx = 0.0
    dy = 0.0
    d_scale = 1.0
    d_rot = 0.0
    d_opacity = 1.0

    # Proporciona deslocamentos em px para 1080x1920 / 4K manterem a mesma amplitude visual da prévia
    px_scale = max(1.0, min(width, height) / 720.0)
    anim = (animation or "none").lower()

    if anim == "floating":
        dy = -10.0 * px_scale * (1.0 - math.cos(t * math.pi))
    elif anim == "jumping":
        dy = -10.0 * px_scale * (1.0 - math.cos(t * math.pi * 2.0))
    elif anim == "breathing":
        d_scale = 1.0 + 0.025 * (1.0 - math.cos((t * math.pi * 2.0) / 3.0))
    elif anim == "sliding":
        phase = (t % 3.0) / 3.0
        if phase < 0.25:
            dx = 4.0 * px_scale * (phase / 0.25)
            dy = dx
            d_rot = 2.0 * (phase / 0.25)
        elif phase < 0.5:
            dx = (4.0 - 8.0 * ((phase - 0.25) / 0.25)) * px_scale
            dy = dx
            d_rot = 2.0 - 4.0 * ((phase - 0.25) / 0.25)
        elif phase < 0.75:
            dx = (-4.0 + 8.0 * ((phase - 0.5) / 0.25)) * px_scale
            dy = -4.0 * px_scale
            d_rot = -2.0 + 4.0 * ((phase - 0.5) / 0.25)
        else:
            dx = (4.0 - 4.0 * ((phase - 0.75) / 0.25)) * px_scale
            dy = (-4.0 + 4.0 * ((phase - 0.75) / 0.25)) * px_scale
            d_rot = 2.0 - 2.0 * ((phase - 0.75) / 0.25)
    elif anim == "wobble":
        d_rot = 4.0 * math.cos((t * math.pi * 2.0) / 1.5)
    elif anim == "zoom_pulse":
        d_scale = 1.0 + 0.025 * (1.0 - math.cos(t * math.pi))
    elif anim == "swinging":
        d_rot = 4.0 * math.cos(t * math.pi)
    elif anim == "shaking":
        phase = (t % 0.2) / 0.2
        if phase < 0.25:
            dx, dy, d_rot = 4.0 * px_scale, 4.0 * px_scale, 2.0
        elif phase < 0.5:
            dx, dy, d_rot = -4.0 * px_scale, -4.0 * px_scale, -2.0
        elif phase < 0.75:
            dx, dy, d_rot = 4.0 * px_scale, -4.0 * px_scale, 2.0
    elif anim == "slide_in":
        if t < 1.0:
            dy = (1.0 - t) * height
            d_opacity = max(0.0, min(1.0, t))
    elif anim == "bobbing":
        dy = -10.0 * px_scale * (1.0 - math.cos((t * math.pi * 2.0) / 1.5))
    elif anim == "zoom_in_slow":
        d_scale = (1.0 + 0.2 * (t / 10.0)) if t < 10.0 else 1.2
    elif anim == "pulsing":
        d_scale = 1.0 + 0.025 * (1.0 - math.cos(t * math.pi * 2.0))
    elif anim == "pop_up":
        if t < 0.075:
            d_scale = max(0.01, (t / 0.075) * 1.1)
            d_opacity = max(0.0, min(1.0, t / 0.075))
        elif t < 0.125:
            d_scale = 1.1 - 0.1 * ((t - 0.075) / 0.05)
    elif anim == "glitch":
        phase = (t % 0.2) / 0.2
        if 0.2 < phase < 0.4:
            dx, dy = -4.0 * px_scale, 2.0 * px_scale
        elif 0.4 <= phase < 0.6:
            dx, dy = 4.0 * px_scale, -2.0 * px_scale
        elif 0.6 <= phase < 0.8:
            dx, dy = -2.0 * px_scale, 4.0 * px_scale
        elif phase >= 0.8:
            dx, dy = 2.0 * px_scale, -4.0 * px_scale

    return dx, dy, d_scale, d_rot, d_opacity


def build_character_bg_and_effect_layers(
    width: int,
    height: int,
    bg_type: str,
    bg_color: str,
    bg_img_path: str | None,
    ef_type: str,
    ef_radius: float,
    ef_color: str,
    ef_opacity: float,
    ef_margin: float,
    ef_border_width: float,
):
    """Pré-calcula (bg_layer BGRA uint8, effect_layer BGRA float32, glass_mask float32|None)."""
    scale_ref = max(1.0, min(width, height) / 720.0)
    bg_layer = None

    bt = (bg_type or "transparent").lower()
    if bt == "color":
        bgr = _hex_to_bgr(bg_color or "#000000", (0, 0, 0))
        bg_layer = np.zeros((height, width, 4), dtype=np.uint8)
        bg_layer[:, :, :3] = bgr
        bg_layer[:, :, 3] = 255
    elif bt == "image" and bg_img_path and os.path.exists(bg_img_path):
        raw_bg = cv2.imread(bg_img_path, cv2.IMREAD_COLOR)
        if raw_bg is not None:
            bh0, bw0 = raw_bg.shape[:2]
            bg_ratio = bw0 / float(max(1, bh0))
            canvas_ratio = width / float(max(1, height))
            if bg_ratio > canvas_ratio:
                nh = height
                nw = int(round(bw0 * (height / float(max(1, bh0)))))
            else:
                nw = width
                nh = int(round(bh0 * (width / float(max(1, bw0)))))
            resized_bg = cv2.resize(raw_bg, (max(1, nw), max(1, nh)), interpolation=cv2.INTER_LINEAR)
            x0 = max(0, (nw - width) // 2)
            y0 = max(0, (nh - height) // 2)
            crop_bg = resized_bg[y0:y0 + height, x0:x0 + width]
            if crop_bg.shape[0] == height and crop_bg.shape[1] == width:
                bg_layer = np.zeros((height, width, 4), dtype=np.uint8)
                bg_layer[:, :, :3] = crop_bg
                bg_layer[:, :, 3] = 255

    ef = (ef_type or "none").lower()
    if ef == "none":
        return bg_layer, None, None

    margin_px = max(0.0, float(ef_margin) * scale_ref)
    ef_w = max(2.0, width - margin_px * 2.0)
    ef_h = max(2.0, height - margin_px * 2.0)
    max_rad = min(ef_w, ef_h) / 2.0
    rad = max(0.0, min(float(ef_radius) * scale_ref, max_rad))
    op = max(0.0, min(1.0, float(ef_opacity) / 100.0))
    bw = max(0.0, float(ef_border_width) * scale_ref)
    bgr_ef = np.array(_hex_to_bgr(ef_color or "#ffffff", (255, 255, 255)), dtype=np.float32)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx_r = width / 2.0
    cy_r = height / 2.0
    qx = np.maximum(np.abs(xx - cx_r) - (ef_w / 2.0) + rad, 0.0)
    qy = np.maximum(np.abs(yy - cy_r) - (ef_h / 2.0) + rad, 0.0)
    dist = np.sqrt(qx * qx + qy * qy) - rad

    inside_mask = np.clip(0.5 - dist, 0.0, 1.0)
    border_mask = np.clip(0.5 - np.abs(dist + bw / 2.0) + bw / 2.0, 0.0, 1.0) if bw > 0 else np.zeros((height, width), dtype=np.float32)

    effect_layer = np.zeros((height, width, 4), dtype=np.float32)
    glass_mask = None

    def _paint_border(layer_arr, alpha_mult=1.0):
        if bw > 0:
            a = border_mask * op * alpha_mult
            layer_arr[:, :, :3] = np.where(a[:, :, None] > 0.001, bgr_ef, layer_arr[:, :, :3])
            layer_arr[:, :, 3] = np.maximum(layer_arr[:, :, 3], a)

    if ef == "card_frame":
        fill_a = inside_mask * op * (0x18 / 255.0)
        effect_layer[:, :, :3] = bgr_ef
        effect_layer[:, :, 3] = fill_a
        _paint_border(effect_layer, 1.0)
    elif ef == "glassmorphism":
        glass_mask = inside_mask
        fill_a = inside_mask * op * (0x20 / 255.0)
        effect_layer[:, :, :3] = bgr_ef
        effect_layer[:, :, 3] = fill_a
        _paint_border(effect_layer, 0x45 / 255.0)
    elif ef == "glow_border":
        if bw <= 0:
            bw_glow = 2.0 * scale_ref
            border_mask = np.clip(0.5 - np.abs(dist + bw_glow / 2.0) + bw_glow / 2.0, 0.0, 1.0)
        glow_blur = cv2.GaussianBlur(border_mask, (0, 0), sigmaX=max(4.0, 9.0 * scale_ref))
        glow_a = np.clip(np.maximum(border_mask, glow_blur * 1.6) * op, 0.0, 1.0)
        effect_layer[:, :, :3] = bgr_ef
        effect_layer[:, :, 3] = glow_a
    elif ef == "vignette":
        r_dist = np.sqrt((xx - cx_r) ** 2 + (yy - cy_r) ** 2)
        r0 = ef_w * 0.15
        r1 = ef_w * 0.6
        t_v = np.clip((r_dist - r0) / max(1.0, r1 - r0), 0.0, 1.0)
        effect_layer[:, :, :3] = bgr_ef
        effect_layer[:, :, 3] = inside_mask * t_v * op
    elif ef == "inner_shadow":
        bw_in = max(bw, 4.0 * scale_ref)
        ring = np.clip(0.5 - np.abs(dist + bw_in / 2.0) + bw_in / 2.0, 0.0, 1.0)
        glow_in = cv2.GaussianBlur(ring, (0, 0), sigmaX=max(6.0, 12.0 * scale_ref)) * inside_mask
        effect_layer[:, :, :3] = bgr_ef
        effect_layer[:, :, 3] = np.clip(np.maximum(ring, glow_in * 1.4) * op, 0.0, 1.0)
    elif ef == "gradient_overlay":
        u = np.clip(((xx - (cx_r - ef_w / 2.0)) / ef_w + (yy - (cy_r - ef_h / 2.0)) / ef_h) / 2.0, 0.0, 1.0)
        # Gradiente: 0 -> efColor@40%, 0.5 -> transparente, 1 -> preto@60%
        top_w = np.clip((0.5 - u) / 0.5, 0.0, 1.0) * (0x40 / 255.0)
        bot_w = np.clip((u - 0.5) / 0.5, 0.0, 1.0) * (0x99 / 255.0)
        grad_a = (top_w + bot_w) * inside_mask * op
        denom = np.maximum(top_w + bot_w, 1e-5)
        effect_layer[:, :, :3] = bgr_ef * (top_w / denom)[:, :, None]
        effect_layer[:, :, 3] = grad_a
        _paint_border(effect_layer, 0x40 / 255.0)
    elif ef == "scanlines":
        step = max(4, int(round(6 * scale_ref)))
        thick = max(2, int(round(3 * scale_ref)))
        scan = ((yy.astype(np.int32) - int(cy_r - ef_h / 2.0)) % step < thick).astype(np.float32)
        effect_layer[:, :, :3] = 0.0
        effect_layer[:, :, 3] = scan * inside_mask * op * 0.4
        _paint_border(effect_layer, 0x60 / 255.0)
    elif ef == "cinema_bars":
        bar_h = ef_h * 0.10
        y_top = cy_r - ef_h / 2.0 + bar_h
        y_bot = cy_r + ef_h / 2.0 - bar_h
        bars = ((yy <= y_top) | (yy >= y_bot)).astype(np.float32)
        effect_layer[:, :, :3] = bgr_ef
        effect_layer[:, :, 3] = bars * inside_mask * op
        _paint_border(effect_layer, 0x40 / 255.0)

    return bg_layer, effect_layer, glass_mask


def composite_characters_opencv(
    visual_video_path: str,
    char_out_path: str,
    valid_chars: list,
    bg_img_cache: dict,
    width: int,
    height: int,
    fps: int,
    v_enc_args: list[str],
):
    """Composita fundo + efeito + personagem animado em sub-pixel sobre o vídeo."""
    # Pré-carrega PNGs dos personagens com canal alpha e pré-renderiza camadas de fundo/efeito
    char_img_cache = {}
    bg_effect_cache = {}
    windows = []

    for c_idx, c_path, ch_meta in valid_chars:
        if c_path not in char_img_cache:
            c_img = cv2.imread(c_path, cv2.IMREAD_UNCHANGED)
            if c_img is None:
                continue
            if c_img.ndim == 2:
                c_img = cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGRA)
            elif c_img.shape[2] == 3:
                alpha_ch = np.full((c_img.shape[0], c_img.shape[1], 1), 255, dtype=np.uint8)
                c_img = np.concatenate([c_img, alpha_ch], axis=2)
            char_img_cache[c_path] = c_img

        bg_key = (
            ch_meta.get("backgroundType", "transparent"),
            ch_meta.get("backgroundColor", "#000000"),
            ch_meta.get("backgroundImageUrl", ""),
            ch_meta.get("bgEffectType", "none"),
            ch_meta.get("bgEffectBorderRadius", 16),
            ch_meta.get("bgEffectColor", "#ffffff"),
            ch_meta.get("bgEffectOpacity", 80),
            ch_meta.get("bgEffectMargin", 16),
            ch_meta.get("bgEffectBorderWidth", 2),
        )
        if bg_key not in bg_effect_cache:
            bg_img_path = bg_img_cache.get(ch_meta.get("backgroundImageUrl", ""))
            bg_effect_cache[bg_key] = build_character_bg_and_effect_layers(
                width, height,
                bg_key[0], bg_key[1], bg_img_path,
                bg_key[3], bg_key[4], bg_key[5], bg_key[6], bg_key[7], bg_key[8],
            )

        c_start = float(ch_meta.get("startTime", 0))
        c_end = c_start + float(ch_meta.get("duration", 10))
        windows.append((c_start, c_end, char_img_cache[c_path], ch_meta, bg_effect_cache[bg_key]))

    cap = cv2.VideoCapture(visual_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não conseguiu abrir vídeo concatenado: {visual_video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    ff_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{vid_fps:.3f}",
        "-i", "pipe:0",
        *v_enc_args,
        "-an",
        char_out_path,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    scale_ratio = width / 1280.0
    char_max_h = height * 0.8

    try:
        f_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cur_t = f_idx / float(vid_fps)
            f_idx += 1

            active_win = None
            for w in windows:
                if w[0] <= cur_t <= w[1]:
                    active_win = w
                    break

            if active_win is not None:
                c_start, c_end, c_img, ch_meta, (bg_layer, ef_layer, glass_mask) = active_win

                # 1. Fundo do personagem (Cor sólida ou Imagem personalizada)
                if bg_layer is not None:
                    frame = bg_layer[:, :, :3].copy()

                # 2. Efeito sobre o fundo (com bordas arredondadas)
                if glass_mask is not None:
                    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=10.0)
                    gm = glass_mask[:, :, None]
                    frame = np.clip(frame.astype(np.float32) * (1.0 - gm) + blurred.astype(np.float32) * gm, 0, 255).astype(np.uint8)
                if ef_layer is not None:
                    a = ef_layer[:, :, 3:4]
                    frame = np.clip(frame.astype(np.float32) * (1.0 - a) + ef_layer[:, :, :3] * a, 0, 255).astype(np.uint8)

                # 3. Personagem animado em sub-pixel (espela FinalizarVideo.tsx)
                src_h, src_w = c_img.shape[:2]
                c_scale = max(0.1, min(3.0, float(ch_meta.get("scale", 100)) / 100.0))
                render_h = char_max_h * c_scale
                render_w = float(src_w) * (render_h / float(max(1, src_h)))

                pos = str(ch_meta.get("position", "center")).lower()
                padding = width * 0.02
                if pos in ("left", "bottom-left"):
                    cx = -width / 2.0 + padding + render_w / 2.0
                elif pos in ("right", "bottom-right"):
                    cx = width / 2.0 - padding - render_w / 2.0
                else:
                    cx = 0.0

                if pos.startswith("bottom"):
                    cy = height / 2.0 - height * 0.05 - render_h / 2.0
                else:
                    cy = -height * 0.025

                t_local = cur_t - c_start
                dx, dy, d_scale, d_rot, d_opacity = compute_character_anim(
                    ch_meta.get("animation", "none"), t_local, width, height
                )

                ct = ch_meta.get("customTransform") or {}
                ct_x = float(ct.get("x", 0.0) or 0.0) * scale_ratio
                ct_y = float(ct.get("y", 0.0) or 0.0) * scale_ratio
                ct_sx = float(ct.get("scaleX", 1.0) or 1.0)
                ct_sy = float(ct.get("scaleY", 1.0) or 1.0)
                ct_rot = float(ct.get("rotation", 0.0) or 0.0)

                # Matriz exata: Translate(W/2 + cx + dx + ct_x, H/2 + cy + dy + ct_y)
                #   @ Rotate(d_rot + ct_rot)
                #   @ Translate(0, render_h/2) @ Scale(d_scale*ct_sx, d_scale*ct_sy) @ Translate(0, -render_h/2)
                #   @ Fit(src -> -render_w/2..-render_h/2)
                t_fit = np.array([
                    [render_w / float(max(1, src_w)), 0.0, -render_w / 2.0],
                    [0.0, render_h / float(max(1, src_h)), -render_h / 2.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

                sx_tot = d_scale * ct_sx
                sy_tot = d_scale * ct_sy
                t_piv = np.array([
                    [sx_tot, 0.0, 0.0],
                    [0.0, sy_tot, (render_h / 2.0) * (1.0 - sy_tot)],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

                rad_rot = math.radians(d_rot + ct_rot)
                cos_r, sin_r = math.cos(rad_rot), math.sin(rad_rot)
                tx_abs = (width / 2.0) + cx + dx + ct_x
                ty_abs = (height / 2.0) + cy + dy + ct_y
                t_world = np.array([
                    [cos_r, -sin_r, tx_abs],
                    [sin_r, cos_r, ty_abs],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

                mat = (t_world @ t_piv @ t_fit)[:2, :]

                # Recorta ROI delimitadora para compositar apenas onde há personagem
                corners = np.array([
                    [0.0, 0.0, 1.0],
                    [float(src_w), 0.0, 1.0],
                    [0.0, float(src_h), 1.0],
                    [float(src_w), float(src_h), 1.0],
                ], dtype=np.float64).T
                proj = mat @ corners
                bx0 = max(0, int(math.floor(np.min(proj[0]))) - 4)
                by0 = max(0, int(math.floor(np.min(proj[1]))) - 4)
                bx1 = min(width, int(math.ceil(np.max(proj[0]))) + 4)
                by1 = min(height, int(math.ceil(np.max(proj[1]))) + 4)

                if bx1 > bx0 and by1 > by0:
                    roi_mat = mat.copy()
                    roi_mat[0, 2] -= bx0
                    roi_mat[1, 2] -= by0
                    warped_char = cv2.warpAffine(
                        c_img, roi_mat, (bx1 - bx0, by1 - by0),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0),
                    )
                    alpha_f = (warped_char[:, :, 3:4].astype(np.float32) / 255.0) * d_opacity
                    roi_frame = frame[by0:by1, bx0:bx1].astype(np.float32)
                    blended = roi_frame * (1.0 - alpha_f) + warped_char[:, :, :3].astype(np.float32) * alpha_f
                    frame[by0:by1, bx0:bx1] = np.clip(blended, 0, 255).astype(np.uint8)

            proc.stdin.write(frame.tobytes())
    finally:
        cap.release()
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

    _, stderr_bytes = proc.communicate()
    if proc.returncode != 0:
        err_tail = (stderr_bytes or b"").decode("utf-8", errors="ignore")[-2000:]
        raise RuntimeError(f"Erro OpenCV+FFmpeg ao aplicar personagens (code {proc.returncode}):\n{err_tail}")


def build_ken_burns_filter(movement: str, duration: float, fps: int, width: int, height: int) -> str:
    frames = max(2, int(math.ceil(duration * fps)))
    mov = (movement or "none").lower()
    # Fallback FFmpeg puro caso OpenCV não esteja instalado: 3x supersampling + ciclo 8s ping-pong
    pre_w = min(int(width * 3), 3840)
    pre_h = min(int(height * 3), 2160) if width >= height else min(int(height * 3), 3840)
    pre = f"scale={pre_w}:{pre_h}:force_original_aspect_ratio=increase,crop={pre_w}:{pre_h},setsar=1"
    if duration > 8.0:
        p_expr = f"if(lte(mod(on/{fps}/8,2),1),mod(on/{fps}/8,2),2-mod(on/{fps}/8,2))"
    else:
        p_expr = f"(on/{frames})"

    if mov == "zoom_in":
        zp = f"zoompan=z='1+0.15*{p_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "zoom_out":
        zp = f"zoompan=z='1.15-0.15*{p_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_left":
        zp = f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)-(iw*(0.05-0.1*{p_expr})/1.2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_right":
        zp = f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)-(-iw*0.05+iw*0.1*{p_expr})/1.2':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_up":
        zp = f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)-(ih*(0.05-0.1*{p_expr})/1.2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_down":
        zp = f"zoompan=z='1.2':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)-(-ih*0.05+ih*0.1*{p_expr})/1.2':d={frames}:s={width}x{height}:fps={fps}"
    else:
        return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,fps={fps}"

    return f"{pre},{zp}"


# ==============================================================================
# UPLOAD DO VÍDEO FINALIZADO
# ==============================================================================

def upload_to_s3_if_configured(file_path: str, job_id: str) -> str | None:
    bucket = os.environ.get("S3_BUCKET", "").strip()
    access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    if not (bucket and access_key and secret_key):
        return None

    import boto3
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "").strip() or None
    region = os.environ.get("S3_REGION", "auto").strip()
    public_base = os.environ.get("S3_PUBLIC_URL", "").strip()

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    key = f"renders/{job_id}_{int(time.time())}.mp4"
    s3.upload_file(file_path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    if public_base:
        return f"{public_base.rstrip('/')}/{key}"
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400)


def upload_to_callback(file_path: str, callback_url: str) -> str | None:
    if not callback_url:
        return None
    try:
        print(f"[RunPod] Enviando MP4 renderizado de volta para o app: {callback_url}")
        with open(file_path, "rb") as f:
            resp = requests.post(
                callback_url,
                data=f,
                headers={"Content-Type": "video/mp4"},
                timeout=300,
            )
        if resp.ok:
            data = resp.json()
            return data.get("url")
        print(f"[RunPod] Callback retornou {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[RunPod] Falha no callback upload: {e}")
    return None


def upload_to_temp_cloud(file_path: str, job_id: str = "job") -> tuple[str | None, list[str]]:
    errors = []
    size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)

    # 1. filebin.net (suporta arquivos pesados / multi-GB via stream direto)
    try:
        bin_id = f"rp_{str(job_id)[:10]}_{int(time.time())}"
        fb_url = f"https://filebin.net/{bin_id}/video_{str(job_id)[:8]}.mp4"
        with open(file_path, "rb") as f:
            r = requests.post(
                fb_url,
                data=f,
                headers={
                    "Content-Type": "video/mp4",
                    "Accept": "application/json",
                    "cid": f"rp-{job_id}",
                    "User-Agent": "RunPod-VideoRenderer/1.0",
                },
                timeout=600,
            )
        if r.status_code in (200, 201):
            print(f"[RunPod] Upload filebin.net concluído ({size_mb} MB): {fb_url}")
            return fb_url, errors
        errors.append(f"filebin:{r.status_code}:{r.text[:120]}")
    except Exception as e:
        errors.append(f"filebin:{e}")
        print(f"[RunPod] filebin fallback falhou: {e}")

    # 2. qu.ax (até 256 MB)
    if size_mb <= 250:
        try:
            with open(file_path, "rb") as f:
                r = requests.post(
                    "https://qu.ax/upload.php",
                    files={"files[]": ("video_renderizado.mp4", f, "video/mp4")},
                    headers={"User-Agent": "RunPod-VideoRenderer/1.0"},
                    timeout=300,
                )
            if r.ok:
                files = r.json().get("files", [])
                if files and files[0].get("url"):
                    return files[0]["url"], errors
            errors.append(f"quax:{r.status_code}:{r.text[:120]}")
        except Exception as e:
            errors.append(f"quax:{e}")
            print(f"[RunPod] qu.ax fallback falhou: {e}")

    # 3. uguu.se (até 128 MB)
    if size_mb <= 125:
        try:
            with open(file_path, "rb") as f:
                r = requests.post(
                    "https://uguu.se/upload.php",
                    files={"files[]": ("video_renderizado.mp4", f, "video/mp4")},
                    headers={"User-Agent": "RunPod-VideoRenderer/1.0"},
                    timeout=300,
                )
            if r.ok:
                files = r.json().get("files", [])
                if files and files[0].get("url"):
                    return files[0]["url"], errors
            errors.append(f"uguu:{r.status_code}:{r.text[:120]}")
        except Exception as e:
            errors.append(f"uguu:{e}")
            print(f"[RunPod] uguu fallback falhou: {e}")

    # 4. tmpfiles.org (até 100 MB)
    if size_mb <= 98:
        try:
            with open(file_path, "rb") as f:
                r = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": ("video_renderizado.mp4", f, "video/mp4")},
                    headers={"User-Agent": "RunPod-VideoRenderer/1.0"},
                    timeout=300,
                )
            if r.ok:
                url = r.json().get("data", {}).get("url", "")
                if url:
                    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/"), errors
            errors.append(f"tmpfiles:{r.status_code}:{r.text[:120]}")
        except Exception as e:
            errors.append(f"tmpfiles:{e}")
            print(f"[RunPod] tmpfiles fallback falhou: {e}")

    return None, errors


# ==============================================================================
# HANDLER PRINCIPAL RUNPOD
# ==============================================================================

def handler(job):
    start_ts = time.time()
    job_id = job.get("id", str(uuid.uuid4())[:8])
    inp = job.get("input", {})

    work_dir = tempfile.mkdtemp(prefix=f"rp_render_{job_id}_")
    print(f"[RunPod] Iniciando Job {job_id} em {work_dir}")

    try:
        canvas = inp.get("canvas", {})
        width = int(canvas.get("width", 1920))
        height = int(canvas.get("height", 1080))
        fps = int(inp.get("fps", 30))
        preset_mode = inp.get("quality", "balanced")  # high | balanced | fast
        total_duration = float(inp.get("totalDuration", 0))
        clips = inp.get("clips", [])
        audio_tracks = inp.get("audioTracks", [])
        background_music = inp.get("backgroundMusic")
        subtitles_ass = inp.get("subtitlesAss", "")
        subtitles_srt = inp.get("subtitlesSrt", "")
        callback_upload_url = inp.get("callbackUploadUrl", "")
        authentic_intro_secs = float(inp.get("authenticIntroSecs", 0))

        if not clips and total_duration <= 0:
            return {"error": "Nenhum clipe ou duração válida enviada no payload."}

        use_nvenc = has_nvenc()
        vcodec = "h264_nvenc" if use_nvenc else "libx264"
        if use_nvenc:
            preset_flag = "p6" if preset_mode == "high" else ("p1" if preset_mode == "fast" else "p4")
            cq_val = "19" if preset_mode == "high" else ("25" if preset_mode == "fast" else "21")
            v_enc_args = ["-c:v", "h264_nvenc", "-preset", preset_flag, "-cq", cq_val, "-pix_fmt", "yuv420p"]
        else:
            preset_flag = "slow" if preset_mode == "high" else ("ultrafast" if preset_mode == "fast" else "medium")
            crf_val = "18" if preset_mode == "high" else ("26" if preset_mode == "fast" else "21")
            v_enc_args = ["-c:v", "libx264", "-preset", preset_flag, "-crf", crf_val, "-pix_fmt", "yuv420p"]

        print(f"[RunPod] Encoder selecionado: {vcodec} | {width}x{height} @ {fps}fps | {len(clips)} clipes | {len(audio_tracks)} áudios")

        # 1. Baixar todas as mídias e áudios em paralelo
        download_tasks = []
        media_dir = os.path.join(work_dir, "media")
        os.makedirs(media_dir, exist_ok=True)

        clip_paths = {}
        audio_paths = {}
        bgm_path = None

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for idx, clip in enumerate(clips):
                url = clip.get("url", "")
                if not url or clip.get("isBlackScreen"):
                    continue
                ext = ".mp4" if clip.get("type") == "video" else ".png"
                dest = os.path.join(media_dir, f"clip_{idx}{ext}")
                futures[pool.submit(download_file, url, dest)] = ("clip", idx, dest)

            for idx, aud in enumerate(audio_tracks):
                url = aud.get("url", "")
                if not url:
                    continue
                dest = os.path.join(media_dir, f"audio_{idx}.bin")
                futures[pool.submit(download_file, url, dest)] = ("audio", idx, dest)

            if background_music and background_music.get("url"):
                dest = os.path.join(media_dir, "bgm.bin")
                futures[pool.submit(download_file, background_music["url"], dest)] = ("bgm", 0, dest)

            for fut in as_completed(futures):
                kind, idx, dest = futures[fut]
                try:
                    fut.result()
                    if kind == "clip":
                        clip_paths[idx] = dest
                    elif kind == "audio":
                        audio_paths[idx] = dest
                    elif kind == "bgm":
                        bgm_path = dest
                except Exception as dl_err:
                    print(f"[RunPod Warning] Falha ao baixar {kind} #{idx}: {dl_err}")

        # 2. Renderizar cada clipe individualmente (com Ken Burns, Filtro, Overlay, Duração exata)
        segment_files = []

        # Se houver introdução autêntica (tela preta inicial para vídeos externos)
        if authentic_intro_secs > 0.05:
            black_seg = os.path.join(work_dir, "seg_0000_black.mp4")
            run_cmd([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={authentic_intro_secs:.3f}",
                *v_enc_args, "-an", black_seg
            ], "Gerar tela preta inicial")
            segment_files.append(black_seg)

        def render_single_clip(idx: int, clip: dict) -> str:
            dur = max(0.2, float(clip.get("duration", 3.0)))
            seg_out = os.path.join(work_dir, f"seg_{idx+1:04d}.mp4")
            src_path = clip_paths.get(idx)

            if not src_path or not os.path.exists(src_path):
                run_cmd([
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={dur:.3f}",
                    *v_enc_args, "-an", seg_out
                ], f"Clipe {idx+1} (tela preta)")
                return seg_out

            vf_post_parts = []
            vis_filter = get_visual_filter_expr(clip.get("visualFilter", "none"))
            if vis_filter:
                vf_post_parts.append(vis_filter)

            ov_filter = get_overlay_filter_expr(clip.get("overlayEffect", "none"), width, height)
            if ov_filter:
                vf_post_parts.append(ov_filter)

            movement = clip.get("movement", "none")
            camera_effect = clip.get("cameraEffect", "none")
            custom_transform = clip.get("customTransform")
            needs_motion = (
                (movement and movement != "none")
                or (camera_effect and camera_effect != "none")
                or bool(custom_transform)
            )

            if HAS_OPENCV and (clip.get("type") != "video" or needs_motion):
                render_clip_with_opencv(
                    idx, clip, src_path, seg_out, width, height, fps, dur, vf_post_parts, v_enc_args
                )
                return seg_out

            vf_parts = []
            if clip.get("type") == "video":
                speed = float(clip.get("playbackSpeed", 1.0) or 1.0)
                if abs(speed - 1.0) > 0.05:
                    vf_parts.append(f"setpts={1.0/speed:.4f}*PTS")
                vf_parts.append(f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,fps={fps}")
                input_args = ["-stream_loop", "-1", "-i", src_path, "-t", f"{dur:.3f}"]
            else:
                kb = build_ken_burns_filter(movement, dur, fps, width, height)
                vf_parts.append(kb)
                input_args = ["-loop", "1", "-i", src_path, "-t", f"{dur:.3f}"]

            vf_parts.extend(vf_post_parts)
            vf_str = ",".join([p for p in vf_parts if p])

            run_cmd([
                "ffmpeg", "-y",
                *input_args,
                "-vf", vf_str,
                *v_enc_args,
                "-an",
                seg_out
            ], f"Renderizar clipe {idx+1}/{len(clips)}")
            return seg_out

        rendered_by_idx = {}
        with ThreadPoolExecutor(max_workers=4 if use_nvenc else 2) as rpool:
            rfuts = {rpool.submit(render_single_clip, idx, clip): idx for idx, clip in enumerate(clips)}
            for fut in as_completed(rfuts):
                c_idx = rfuts[fut]
                rendered_by_idx[c_idx] = fut.result()

        for idx in range(len(clips)):
            segment_files.append(rendered_by_idx[idx])

        if not segment_files:
            return {"error": "Nenhum segmento de vídeo pôde ser gerado."}

        # 3. Concatenar segmentos visuais
        concat_list_path = os.path.join(work_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        visual_video_path = os.path.join(work_dir, "visual_concat.mp4")
        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy",
            visual_video_path
        ], "Concatenar segmentos visuais")

        # 3b. Aplicar overlays de personagem (PIP + Animação + Fundo + Efeito) se configurados
        character_overlays = inp.get("characterOverlays", [])
        if character_overlays:
            char_files = {}
            bg_img_cache = {}
            url_to_local = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                cfuts = {}
                for c_idx, ch in enumerate(character_overlays):
                    c_url = ch.get("url", "")
                    if c_url:
                        if c_url in url_to_local:
                            char_files[c_idx] = url_to_local[c_url]
                        else:
                            dest = os.path.join(media_dir, f"char_u_{len(url_to_local)}.png")
                            url_to_local[c_url] = dest
                            cfuts[pool.submit(download_file, c_url, dest)] = ("char", c_idx, c_url, dest)
                    bg_url = ch.get("backgroundImageUrl", "")
                    if bg_url and bg_url not in bg_img_cache:
                        bg_dest = os.path.join(media_dir, f"char_bg_{len(bg_img_cache)}.png")
                        bg_img_cache[bg_url] = bg_dest
                        cfuts[pool.submit(download_file, bg_url, bg_dest)] = ("charbg", c_idx, bg_url, bg_dest)

                for fut in as_completed(cfuts):
                    kind, c_idx, src_url, dest = cfuts[fut]
                    try:
                        fut.result()
                        if kind == "char":
                            char_files[c_idx] = dest
                        elif kind == "charbg":
                            bg_img_cache[src_url] = dest
                    except Exception as cerr:
                        print(f"[RunPod Warning] Falha ao baixar {kind} #{c_idx}: {cerr}")
                        if kind == "charbg":
                            bg_img_cache.pop(src_url, None)

            # Preenche entradas repetidas que compartilharam o mesmo URL
            for c_idx, ch in enumerate(character_overlays):
                c_url = ch.get("url", "")
                if c_idx not in char_files and c_url in url_to_local and os.path.exists(url_to_local[c_url]):
                    char_files[c_idx] = url_to_local[c_url]

            valid_chars = [(i, char_files[i], character_overlays[i]) for i in sorted(char_files.keys())]
            if valid_chars:
                char_out_path = os.path.join(work_dir, "visual_with_chars.mp4")
                if HAS_OPENCV:
                    composite_characters_opencv(
                        visual_video_path, char_out_path, valid_chars, bg_img_cache, width, height, fps, v_enc_args
                    )
                else:
                    c_cmd = ["ffmpeg", "-y", "-i", visual_video_path]
                    fc_parts = []
                    last_v = "[0:v]"
                    for n_idx, (_, c_path, ch_meta) in enumerate(valid_chars, start=1):
                        c_cmd.extend(["-i", c_path])
                        c_start = float(ch_meta.get("startTime", 0))
                        c_end = c_start + float(ch_meta.get("duration", 10))
                        c_scale = max(0.1, min(3.0, float(ch_meta.get("scale", 100)) / 100.0))
                        target_h = int(height * 0.80 * c_scale)
                        pos = ch_meta.get("position", "center")
                        anim = str(ch_meta.get("animation", "none")).lower()
                        if pos in ("left", "bottom-left"):
                            x_expr = f"{int(width * 0.02)}"
                        elif pos in ("right", "bottom-right"):
                            x_expr = f"W-w-{int(width * 0.02)}"
                        else:
                            x_expr = "(W-w)/2"
                        base_y = "H-h" if str(pos).startswith("bottom") else f"(H-h)/2-{int(height * 0.025)}"
                        if anim in ("floating", "jumping", "bobbing"):
                            freq = "2*PI" if anim == "jumping" else ("2*PI/1.5" if anim == "bobbing" else "PI")
                            amp = max(10, int(10 * min(width, height) / 720.0))
                            y_expr = f"'{base_y}-{amp}*(1-cos((t-{c_start:.2f})*{freq}))'"
                        else:
                            y_expr = base_y

                        scaled_lbl = f"[ch{n_idx}]"
                        out_lbl = "[vchars]" if n_idx == len(valid_chars) else f"[vc{n_idx}]"
                        fc_parts.append(f"[{n_idx}:v]scale=-1:{target_h}{scaled_lbl}")
                        fc_parts.append(f"{last_v}{scaled_lbl}overlay=x={x_expr}:y={y_expr}:enable='between(t,{c_start:.2f},{c_end:.2f})'{out_lbl}")
                        last_v = out_lbl

                    c_cmd.extend([
                        "-filter_complex", ";".join(fc_parts),
                        "-map", last_v,
                        *v_enc_args,
                        "-an",
                        char_out_path
                    ])
                    run_cmd(c_cmd, "Aplicar personagens PIP")
                visual_video_path = char_out_path

        # 4. Queimar legendas (.ASS prioritário, fallback .SRT) se fornecidas
        subtitled_video_path = visual_video_path
        sub_file_path = None
        if subtitles_ass and subtitles_ass.strip():
            sub_file_path = os.path.join(work_dir, "subtitles.ass")
            with open(sub_file_path, "w", encoding="utf-8") as f:
                f.write(subtitles_ass)
        elif subtitles_srt and subtitles_srt.strip():
            sub_file_path = os.path.join(work_dir, "subtitles.srt")
            with open(sub_file_path, "w", encoding="utf-8") as f:
                f.write(subtitles_srt)

        if sub_file_path:
            subtitled_video_path = os.path.join(work_dir, "visual_subtitled.mp4")
            escaped_sub = sub_file_path.replace("\\", "/").replace(":", "\\:")
            sub_filter = f"ass={escaped_sub}" if sub_file_path.endswith(".ass") else f"subtitles={escaped_sub}"
            run_cmd([
                "ffmpeg", "-y",
                "-i", visual_video_path,
                "-vf", sub_filter,
                *v_enc_args,
                "-an",
                subtitled_video_path
            ], "Queimar legendas no vídeo")

        # 5. Mixar todas as trilhas de áudio (narrações + música de fundo) e unir ao vídeo final
        final_output_path = os.path.join(work_dir, f"final_runpod_{job_id}.mp4")
        valid_audios = [(idx, audio_paths[idx], audio_tracks[idx]) for idx in sorted(audio_paths.keys())]

        if valid_audios or bgm_path:
            cmd = ["ffmpeg", "-y", "-i", subtitled_video_path]
            filter_complex_parts = []
            mix_labels = []
            input_cursor = 1

            for _, a_path, a_meta in valid_audios:
                cmd.extend(["-i", a_path])
                delay_ms = max(0, int(round(float(a_meta.get("startTime", 0)) * 1000)))
                vol = float(a_meta.get("volume", 1.0))
                lbl = f"aud{input_cursor}"
                filter_complex_parts.append(
                    f"[{input_cursor}:a]adelay={delay_ms}|{delay_ms},volume={vol:.2f},aformat=sample_rates=44100:channel_layouts=stereo[{lbl}]"
                )
                mix_labels.append(f"[{lbl}]")
                input_cursor += 1

            if bgm_path:
                cmd.extend(["-stream_loop", "-1", "-i", bgm_path])
                bgm_vol = float(background_music.get("volume", 0.15))
                lbl = f"bgm{input_cursor}"
                filter_complex_parts.append(
                    f"[{input_cursor}:a]volume={bgm_vol:.2f},aformat=sample_rates=44100:channel_layouts=stereo[{lbl}]"
                )
                mix_labels.append(f"[{lbl}]")
                input_cursor += 1

            if len(mix_labels) == 1:
                filter_complex_parts.append(f"{mix_labels[0]}acopy[aout]")
            else:
                filter_complex_parts.append(
                    f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=2:normalize=0[aout]"
                )

            cmd.extend([
                "-filter_complex", ";".join(filter_complex_parts),
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                final_output_path
            ])
            run_cmd(cmd, "Mixar áudio e finalizar MP4")
        else:
            run_cmd([
                "ffmpeg", "-y", "-i", subtitled_video_path,
                "-c:v", "copy", "-movflags", "+faststart", final_output_path
            ], "Finalizar MP4 (sem áudio)")

        size_mb = round(os.path.getsize(final_output_path) / (1024 * 1024), 2)
        elapsed = round(time.time() - start_ts, 2)
        print(f"[RunPod] Concluído em {elapsed}s | Tamanho: {size_mb} MB | GPU NVENC: {use_nvenc}")

        # 6. Retornar o vídeo renderizado (Prioridade: S3 -> Callback App -> Temp Cloud -> Base64 se < 8MB)
        video_url = upload_to_s3_if_configured(final_output_path, job_id)
        delivery_method = "s3"

        if not video_url and callback_upload_url:
            video_url = upload_to_callback(final_output_path, callback_upload_url)
            delivery_method = "callback"

        upload_errors = []
        if not video_url:
            video_url, upload_errors = upload_to_temp_cloud(final_output_path, job_id)
            delivery_method = "temp_cloud"

        result = {
            "status": "completed",
            "jobId": job_id,
            "encoder": vcodec,
            "gpuAccelerated": use_nvenc,
            "elapsedSeconds": elapsed,
            "sizeMb": size_mb,
            "deliveryMethod": delivery_method,
        }

        if video_url:
            result["videoUrl"] = video_url
        elif size_mb <= 12.0:
            with open(final_output_path, "rb") as f:
                result["videoBase64"] = base64.b64encode(f.read()).decode("utf-8")
            result["deliveryMethod"] = "base64"
        else:
            err_detail = " | ".join(upload_errors) if upload_errors else "Sem host disponível"
            return {"error": f"Vídeo gerado ({size_mb} MB em {elapsed}s), mas o upload falhou: {err_detail}. Configure S3/R2 nas variáveis do Endpoint."}

        return result

    except Exception as e:
        print(f"[RunPod ERROR] {e}")
        return {"error": str(e)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
