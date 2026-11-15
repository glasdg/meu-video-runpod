#!/usr/bin/env python3
"""
RunPod Serverless GPU Worker - Renderizador de Vídeo Completo (Timeline + Legendas + Efeitos + Áudio)
Recebe o projeto JSON da aba "6. Finalizar Vídeo -> Capcut (Timeline)",
baixa todas as mídias/áudios, renderiza os clipes com movimentos Ken Burns, filtros,
personagens, mixagem de áudio/música de fundo e queima as legendas estilizadas (.ASS/.SRT) via FFmpeg GPU (NVENC).
"""

import os
import re
import json
import uuid
import math
import time
import random
import shutil
import base64
import tempfile
import threading
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import requests
import runpod

try:
    import numpy as np
    import cv2
    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    HAS_PIL = True
except Exception:
    HAS_PIL = False

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
            print("[RunPod NVENC] h264_nvenc não listado em ffmpeg -encoders.")
            return False
        # Usa 256x256 pois GPUs Ampere/Turing (RTX A5000, RTX 4090, L4) exigem largura mínima >= 145px no NVENC H.264
        test = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p", "-f", "null", "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if test.returncode != 0:
            print(f"[RunPod NVENC] Teste 256x256 falhou (code {test.returncode}): {(test.stderr or '')[-500:]}")
            return False
        return True
    except Exception as e:
        print(f"[RunPod NVENC] Exceção ao testar NVENC: {e}")
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
    sw = float(max(1, src_w))
    sh = float(max(1, src_h))
    img_ratio = sw / sh
    canvas_ratio = width / float(max(1, height))
    if img_ratio > canvas_ratio:
        draw_w = float(width)
        draw_h = sh * (width / sw)
    else:
        draw_h = float(height)
        draw_w = sw * (height / sh)

    f_sx = draw_w / sw
    f_sy = draw_h / sh
    f_tx = -draw_w * 0.5
    f_ty = -draw_h * 0.5

    # Combina t_kb @ t_fit escalarmente (zero alocação extra de matrizes NumPy)
    kb_m00 = scale * f_sx
    kb_m02 = scale * f_tx + tx
    kb_m11 = scale * f_sy
    kb_m12 = scale * f_ty + ty

    if not custom_transform:
        cx0 = width * 0.5
        cy0 = height * 0.5
        return np.array([
            [kb_m00, 0.0, kb_m02 + cx0],
            [0.0, kb_m11, kb_m12 + cy0],
        ], dtype=np.float64)

    ct = custom_transform
    scale_ratio = width / 1280.0
    cx = (width * 0.5) + float(ct.get("x", 0.0) or 0.0) * scale_ratio
    cy = (height * 0.5) + float(ct.get("y", 0.0) or 0.0) * scale_ratio
    csx = float(ct.get("scaleX", 1.0) or 1.0)
    csy = float(ct.get("scaleY", 1.0) or 1.0)
    rot_rad = math.radians(float(ct.get("rotation", 0.0) or 0.0))
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)

    a = csx * cos_r
    b = -csy * sin_r
    c = csx * sin_r
    d = csy * cos_r

    return np.array([
        [a * kb_m00, b * kb_m11, a * kb_m02 + b * kb_m12 + cx],
        [c * kb_m00, d * kb_m11, c * kb_m02 + d * kb_m12 + cy],
    ], dtype=np.float64)


def _start_ffmpeg_pipe(ff_cmd: list[str]) -> tuple[subprocess.Popen, list[bytes], threading.Thread]:
    """Inicia FFmpeg com stdin=PIPE e drena stderr em thread dedicada para nunca travar por buffer de 64KB."""
    cmd = list(ff_cmd)
    if len(cmd) >= 1 and cmd[0] == "ffmpeg" and "-loglevel" not in cmd:
        cmd = [cmd[0], "-loglevel", "error", "-nostats", *cmd[1:]]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024 * 4,
    )
    stderr_chunks: list[bytes] = []

    def _drain_stderr():
        try:
            if proc.stderr is not None:
                while True:
                    chunk = proc.stderr.read(8192)
                    if not chunk:
                        break
                    if len(stderr_chunks) < 64:
                        stderr_chunks.append(chunk)
        except Exception:
            pass

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()
    return proc, stderr_chunks, t


def _close_and_wait_ffmpeg(
    proc: subprocess.Popen,
    stderr_chunks: list[bytes],
    stderr_thread: threading.Thread,
    err_prefix: str,
) -> None:
    """Fecha stdin com segurança, aguarda FFmpeg e coleta stderr sem risco de flush of closed file ou deadlock."""
    if proc.stdin is not None:
        try:
            if not proc.stdin.closed:
                proc.stdin.flush()
        except Exception:
            pass
        try:
            if not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        proc.stdin = None

    proc.wait()
    stderr_thread.join(timeout=5.0)
    try:
        if proc.stderr is not None and not proc.stderr.closed:
            proc.stderr.close()
    except Exception:
        pass

    if proc.returncode != 0:
        err_tail = b"".join(stderr_chunks).decode("utf-8", errors="ignore")[-2000:]
        raise RuntimeError(f"{err_prefix} (code {proc.returncode}):\n{err_tail}")


def _resize_oversized_src(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Reduz imagens/vídeos muito acima do tamanho de destino (~1.28x) para acelerar drasticamente warpAffine."""
    h, w = img.shape[:2]
    max_w = int(width * 1.28)
    max_h = int(height * 1.28)
    if w > max_w and h > max_h:
        ratio = max(max_w / float(w), max_h / float(h))
        nw = max(width, int(round(w * ratio)))
        nh = max(height, int(round(h * ratio)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def _write_raw_frame(proc_stdin, frame: np.ndarray, use_yuv420p: bool) -> None:
    if use_yuv420p:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        proc_stdin.write(memoryview(yuv))
    else:
        proc_stdin.write(memoryview(np.ascontiguousarray(frame)))


def get_prev_clip_end_frame(
    idx: int,
    clips: list[dict],
    clip_paths: dict[int, str],
    width: int,
    height: int,
    authentic_intro_secs: float,
) -> np.ndarray | None:
    """Extrai rapidamente apenas o último frame do clipe anterior (sem decodificar o MP4 inteiro) para transições em paralelo."""
    if idx == 0:
        if authentic_intro_secs > 0.05:
            return np.zeros((height, width, 3), dtype=np.uint8)
        return None

    prev_idx = idx - 1
    prev_clip = clips[prev_idx]
    prev_src = clip_paths.get(prev_idx)
    if not prev_src or not os.path.exists(prev_src):
        return np.zeros((height, width, 3), dtype=np.uint8)

    prev_dur = max(0.2, float(prev_clip.get("duration", 3.0)))
    prev_mov = prev_clip.get("movement", "none")
    prev_cam = prev_clip.get("cameraEffect", "none")
    prev_ct = prev_clip.get("customTransform")
    prev_rng = random.Random(10007 * (prev_idx + 1))
    scale, tx, ty = compute_capcut_transform(prev_mov, prev_cam, 1.0, width, height, prev_dur, prev_rng)

    if prev_clip.get("type") != "video":
        img = cv2.imread(prev_src, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        img = _resize_oversized_src(img, width, height)
        src_h, src_w = img.shape[:2]
        mat = build_affine_matrix(src_w, src_h, width, height, scale, tx, ty, prev_ct)
        return cv2.warpAffine(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    cap = cv2.VideoCapture(prev_src)
    if not cap.isOpened():
        return np.zeros((height, width, 3), dtype=np.uint8)
    try:
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        speed = float(prev_clip.get("playbackSpeed", 1.0) or 1.0)
        target_idx = int((prev_dur * speed) * vid_fps)
        if n_vid > 0:
            target_idx = target_idx % n_vid
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_idx - 1))
        ok, frm = cap.read()
        if not ok or frm is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frm = cap.read()
        if not ok or frm is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        src_h, src_w = frm.shape[:2]
        mat = build_affine_matrix(src_w, src_h, width, height, scale, tx, ty, prev_ct)
        return cv2.warpAffine(frm, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    finally:
        cap.release()


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
    prev_end_frame: np.ndarray | None = None,
    tr_name: str = "none",
    tr_dur: float = 0.0,
    pil_subs: list[dict] | None = None,
    sub_config: dict | None = None,
    clip_timeline_start: float = 0.0,
    work_dir: str | None = None,
) -> None:
    """Renderiza clipe usando OpenCV warpAffine sub-pixel + transição opcional + legendas Pillow idênticas ao Canvas + FFmpeg NVENC/x264."""
    is_video = clip.get("type") == "video"
    movement = clip.get("movement", "none")
    camera_effect = clip.get("cameraEffect", "none")
    custom_transform = clip.get("customTransform")
    speed = float(clip.get("playbackSpeed", 1.0) or 1.0)
    total_frames = max(2, int(round(dur * fps)))

    trans_frames = 0
    if prev_end_frame is not None and tr_name and tr_name != "none" and tr_dur > 0.05:
        max_safe = max(2, int(total_frames * 0.45))
        trans_frames = min(max(2, int(round(tr_dur * fps))), max_safe)

    has_motion = (
        (movement and movement != "none")
        or (camera_effect and camera_effect != "none")
    )

    use_pil_subs = bool(HAS_PIL and pil_subs)
    pil_overlay_cache: list[tuple[float, float, dict]] = []
    temp_sub_pngs: list[str] = []

    if use_pil_subs:
        pil_overlay_cache = _prepare_clip_pil_overlays(pil_subs, sub_config or {}, width, height)

    # Caso haja filtros pós-OpenCV (cinematic, noir, vignette etc.) + legendas Pillow,
    # salva as legendas em PNGs recortados e aplica via FFmpeg overlay DEPOIS do filtro para manter as cores originais da legenda.
    use_ffmpeg_png_subs = bool(use_pil_subs and vf_post_parts and pil_overlay_cache and work_dir)
    stamp_pil_in_opencv = bool(use_pil_subs and pil_overlay_cache and not vf_post_parts)

    def _build_filtered_png_ffmpeg_args(base_input_args: list[str], extra_vf: list[str]) -> list[str]:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *base_input_args]
        active_chunks: list[tuple[int, float, float, dict]] = []
        for k_i, (s_t, e_t, ov) in enumerate(pil_overlay_cache):
            rel_s = max(0.0, s_t - clip_timeline_start)
            rel_e = min(dur, e_t - clip_timeline_start)
            if rel_e > rel_s:
                active_chunks.append((k_i, rel_s, rel_e, ov))

        fc = []
        last_v = "[0:v]"
        if extra_vf:
            fc.append(f"[0:v]{','.join(extra_vf)}[vbase]")
            last_v = "[vbase]"
        in_idx = 1
        for c_pos, (k_i, rel_s, rel_e, ov) in enumerate(active_chunks):
            png_path = os.path.join(work_dir, f"seg_{idx+1:04d}_sub_{k_i}.png")
            _save_overlay_png(ov, png_path)
            temp_sub_pngs.append(png_path)
            cmd.extend(["-i", png_path])
            out_lbl = "[vout]" if c_pos == len(active_chunks) - 1 else f"[vs{c_pos}]"
            fc.append(
                f"{last_v}[{in_idx}:v]overlay=x={ov['bx0']}:y={ov['by0']}:enable='between(t,{rel_s:.3f},{rel_e:.3f})'{out_lbl}"
            )
            last_v = out_lbl
            in_idx += 1
        if in_idx > 1:
            cmd.extend(["-filter_complex", ";".join(fc), "-map", last_v])
        elif extra_vf:
            cmd.extend(["-vf", ",".join(extra_vf)])
        cmd.extend([*v_enc_args, "-an", seg_out])
        return cmd

    try:
        # Atalho ultra-rápido para imagem estática sem movimento nem transição:
        if not is_video and not has_motion and trans_frames == 0:
            img = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"OpenCV não conseguiu abrir imagem: {src_path}")
            img = _resize_oversized_src(img, width, height)
            src_h, src_w = img.shape[:2]
            mat = build_affine_matrix(src_w, src_h, width, height, 1.0, 0.0, 0.0, custom_transform)
            static_frame = cv2.warpAffine(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            if not stamp_pil_in_opencv:
                static_png = f"{seg_out}.static.png"
                cv2.imwrite(static_png, static_frame)
                try:
                    if use_ffmpeg_png_subs:
                        cmd = _build_filtered_png_ffmpeg_args(
                            ["-loop", "1", "-framerate", str(fps), "-i", static_png, "-t", f"{dur:.3f}"],
                            vf_post_parts,
                        )
                    else:
                        cmd = [
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-loop", "1", "-framerate", str(fps), "-i", static_png,
                            "-t", f"{dur:.3f}",
                        ]
                        if vf_post_parts:
                            cmd.extend(["-vf", ",".join(vf_post_parts)])
                        cmd.extend([*v_enc_args, "-an", seg_out])
                    run_cmd(cmd, f"Renderizar clipe estático {idx+1}")
                finally:
                    try:
                        if os.path.exists(static_png):
                            os.remove(static_png)
                    except Exception:
                        pass
                return

            # Imagem estática + legendas Pillow diretas na RAM (muito rápido e fiel)
            use_yuv = (width % 2 == 0 and height % 2 == 0)
            ff_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p" if use_yuv else "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
                *v_enc_args, "-an", seg_out,
            ]
            proc, stderr_chunks, stderr_thread = _start_ffmpeg_pipe(ff_cmd)
            try:
                base_yuv = cv2.cvtColor(static_frame, cv2.COLOR_BGR2YUV_I420) if use_yuv else None
                for f_idx in range(total_frames):
                    cur_t = clip_timeline_start + (f_idx / float(fps))
                    active_sub = _find_active_pil_overlay(pil_overlay_cache, cur_t)
                    try:
                        if active_sub is None and base_yuv is not None:
                            proc.stdin.write(memoryview(base_yuv))
                        elif active_sub is None:
                            proc.stdin.write(memoryview(np.ascontiguousarray(static_frame)))
                        else:
                            frm = static_frame.copy()
                            _stamp_pil_subtitle(frm, active_sub[2], cur_t, active_sub[0], active_sub[1])
                            _write_raw_frame(proc.stdin, frm, use_yuv)
                    except BrokenPipeError:
                        break
            finally:
                _close_and_wait_ffmpeg(proc, stderr_chunks, stderr_thread, f"Erro OpenCV+FFmpeg estático clipe {idx+1}")
            return

        use_yuv = (width % 2 == 0 and height % 2 == 0)
        base_pipe_args = [
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p" if use_yuv else "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
        ]
        if use_ffmpeg_png_subs:
            ff_cmd = _build_filtered_png_ffmpeg_args(base_pipe_args, vf_post_parts)
        else:
            ff_cmd = ["ffmpeg", "-y", *base_pipe_args]
            if vf_post_parts:
                ff_cmd.extend(["-vf", ",".join(vf_post_parts)])
            ff_cmd.extend([*v_enc_args, "-an", seg_out])

        proc, stderr_chunks, stderr_thread = _start_ffmpeg_pipe(ff_cmd)
        rng = random.Random(10007 * (idx + 1))

        try:
            if not is_video:
                img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"OpenCV não conseguiu abrir imagem: {src_path}")
                img = _resize_oversized_src(img, width, height)
                src_h, src_w = img.shape[:2]
                for f_idx in range(total_frames):
                    progress = f_idx / float(max(1, total_frames - 1))
                    scale, tx, ty = compute_capcut_transform(movement, camera_effect, progress, width, height, dur, rng)
                    mat = build_affine_matrix(src_w, src_h, width, height, scale, tx, ty, custom_transform)
                    frame = cv2.warpAffine(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    if trans_frames > 0 and f_idx < trans_frames:
                        p = (f_idx + 1) / float(trans_frames + 1)
                        frame = blend_transition_frames(prev_end_frame, frame, tr_name, p, width, height)
                    if stamp_pil_in_opencv:
                        cur_t = clip_timeline_start + (f_idx / float(fps))
                        active_sub = _find_active_pil_overlay(pil_overlay_cache, cur_t)
                        if active_sub is not None:
                            _stamp_pil_subtitle(frame, active_sub[2], cur_t, active_sub[0], active_sub[1])
                    try:
                        _write_raw_frame(proc.stdin, frame, use_yuv)
                    except BrokenPipeError:
                        break
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
                    vid_frames.append(_resize_oversized_src(frm, width, height))
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
                    if trans_frames > 0 and f_idx < trans_frames:
                        p = (f_idx + 1) / float(trans_frames + 1)
                        frame = blend_transition_frames(prev_end_frame, frame, tr_name, p, width, height)
                    if stamp_pil_in_opencv:
                        cur_t = clip_timeline_start + t_sec
                        active_sub = _find_active_pil_overlay(pil_overlay_cache, cur_t)
                        if active_sub is not None:
                            _stamp_pil_subtitle(frame, active_sub[2], cur_t, active_sub[0], active_sub[1])
                    try:
                        _write_raw_frame(proc.stdin, frame, use_yuv)
                    except BrokenPipeError:
                        break
        finally:
            _close_and_wait_ffmpeg(proc, stderr_chunks, stderr_thread, f"Erro OpenCV+FFmpeg clipe {idx+1}")
    finally:
        for tp in temp_sub_pngs:
            try:
                if os.path.exists(tp):
                    os.remove(tp)
            except Exception:
                pass
        return




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
    sub_filter: str | None = None,
    subtitle_items: list[dict] | None = None,
    subtitle_config: dict | None = None,
):
    """Composita fundo + efeito + personagem animado em sub-pixel (+ legendas Pillow Canvas-idênticas em 1 único passe)."""
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
            c_img = _resize_oversized_src(c_img, width, height)
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
            bg_layer, ef_layer, glass_mask = build_character_bg_and_effect_layers(
                width, height,
                bg_key[0], bg_key[1], bg_img_path,
                bg_key[3], bg_key[4], bg_key[5], bg_key[6], bg_key[7], bg_key[8],
            )
            static_bg_bgr = None
            ef_inv_u16 = None
            ef_premult_u16 = None
            if bg_layer is not None and glass_mask is None:
                base_f = bg_layer[:, :, :3].astype(np.float32)
                if ef_layer is not None:
                    a = ef_layer[:, :, 3:4]
                    base_f = base_f * (1.0 - a) + ef_layer[:, :, :3] * a
                static_bg_bgr = np.clip(base_f, 0, 255).astype(np.uint8)
            elif ef_layer is not None:
                a_f = ef_layer[:, :, 3:4]
                ef_inv_u16 = np.clip(np.round((1.0 - a_f) * 256.0), 0, 256).astype(np.uint16)
                ef_premult_u16 = np.clip(np.round(ef_layer[:, :, :3] * a_f * 256.0), 0, 65535).astype(np.uint16)
            glass_u16 = np.clip(np.round(glass_mask[:, :, None] * 256.0), 0, 256).astype(np.uint16) if glass_mask is not None else None
            glass_inv_u16 = (256 - glass_u16) if glass_u16 is not None else None
            bg_effect_cache[bg_key] = (bg_layer, static_bg_bgr, ef_inv_u16, ef_premult_u16, glass_u16, glass_inv_u16)

        c_start = float(ch_meta.get("startTime", 0))
        c_end = c_start + float(ch_meta.get("duration", 10))
        windows.append((c_start, c_end, char_img_cache[c_path], ch_meta, bg_effect_cache[bg_key]))

    use_pil_subs = bool(HAS_PIL and subtitle_items)
    pil_sub_windows: list[tuple[float, float, str]] = []
    pil_sub_rendered: dict[int, dict | None] = {}
    if use_pil_subs:
        for item in subtitle_items:
            txt = str(item.get("text") or "").strip()
            s_t = float(item.get("start") or 0.0)
            e_t = max(s_t + 0.05, float(item.get("end") or (s_t + 1.0)))
            if txt:
                pil_sub_windows.append((s_t, e_t, txt))

    cap = cv2.VideoCapture(visual_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não conseguiu abrir vídeo concatenado: {visual_video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    use_yuv = (width % 2 == 0 and height % 2 == 0)
    ff_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p" if use_yuv else "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{vid_fps:.3f}",
        "-i", "pipe:0",
    ]
    if sub_filter and not use_pil_subs:
        ff_cmd.extend(["-vf", sub_filter])
    ff_cmd.extend([
        *v_enc_args,
        "-an",
        char_out_path,
    ])
    proc, stderr_chunks, stderr_thread = _start_ffmpeg_pipe(ff_cmd)

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
                c_start, c_end, c_img, ch_meta, (bg_layer, static_bg_bgr, ef_inv_u16, ef_premult_u16, glass_u16, glass_inv_u16) = active_win

                # 1. Fundo + Efeito otimizados (zero alocação float32 por frame)
                if static_bg_bgr is not None:
                    frame = static_bg_bgr.copy()
                else:
                    if bg_layer is not None:
                        frame = bg_layer[:, :, :3].copy()
                    if glass_u16 is not None:
                        half = cv2.resize(frame, (max(2, width // 2), max(2, height // 2)), interpolation=cv2.INTER_LINEAR)
                        blurred_half = cv2.GaussianBlur(half, (0, 0), sigmaX=5.0)
                        blurred = cv2.resize(blurred_half, (width, height), interpolation=cv2.INTER_LINEAR)
                        frame = ((frame.astype(np.uint16) * glass_inv_u16 + blurred.astype(np.uint16) * glass_u16) >> 8).astype(np.uint8)
                    if ef_inv_u16 is not None:
                        frame = ((frame.astype(np.uint16) * ef_inv_u16 + ef_premult_u16) >> 8).astype(np.uint8)

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
                    alpha_u16 = ((warped_char[:, :, 3:4].astype(np.uint16) * int(round(d_opacity * 256.0))) >> 8)
                    inv_alpha_u16 = 255 - alpha_u16
                    roi_u16 = frame[by0:by1, bx0:bx1].astype(np.uint16)
                    char_u16 = warped_char[:, :, :3].astype(np.uint16)
                    blended = ((roi_u16 * inv_alpha_u16 + char_u16 * alpha_u16) * 257) >> 16
                    frame[by0:by1, bx0:bx1] = blended.astype(np.uint8)

            if use_pil_subs and pil_sub_windows:
                for s_idx, (ss_t, se_t, s_txt) in enumerate(pil_sub_windows):
                    if ss_t <= cur_t < se_t:
                        if s_idx not in pil_sub_rendered:
                            pil_sub_rendered[s_idx] = _render_pil_subtitle_overlay(
                                s_txt, subtitle_config or {}, width, height
                            )
                        ov = pil_sub_rendered[s_idx]
                        if ov is not None:
                            _stamp_pil_subtitle(frame, ov, cur_t, ss_t, se_t)
                        break

            try:
                _write_raw_frame(proc.stdin, frame, use_yuv)
            except BrokenPipeError:
                break
    finally:
        cap.release()
        _close_and_wait_ffmpeg(proc, stderr_chunks, stderr_thread, "Erro OpenCV+FFmpeg ao aplicar personagens")


def _ease_in_out(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return p * p * (3.0 - 2.0 * p)


def _make_film_burn_mask(width: int, height: int, p: float) -> np.ndarray:
    """Gera máscara suave de vazamento de luz quente (Film Burn) em BGR float32."""
    gw, gh = 96, 54
    xs = np.linspace(0.0, 1.0, gw, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, gh, dtype=np.float32)[:, None]
    cx = 0.20 + 0.60 * p
    cy = 0.45
    dist = np.sqrt((xs - cx) ** 2 + ((ys - cy) * (gh / float(gw))) ** 2)
    core = np.exp(-(dist ** 2) / (2.0 * (0.24 ** 2)))
    halo = np.exp(-(dist ** 2) / (2.0 * (0.52 ** 2)))
    # BGR warm golden/amber core + rich orange halo
    small = np.zeros((gh, gw, 3), dtype=np.float32)
    small[:, :, 0] = core * 130.0 + halo * 25.0   # Blue
    small[:, :, 1] = core * 225.0 + halo * 115.0  # Green
    small[:, :, 2] = core * 255.0 + halo * 235.0  # Red
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def blend_transition_frames(prev_f: np.ndarray, cur_f: np.ndarray, trans: str, p_raw: float, width: int, height: int) -> np.ndarray:
    """Mistura o último frame do clipe anterior com o frame atual do novo clipe sem alterar a duração total."""
    p = _ease_in_out(p_raw)
    tr = (trans or "none").lower()

    if tr == "fade":
        return cv2.addWeighted(prev_f, 1.0 - p, cur_f, p, 0)

    if tr == "luma_fade":
        luma = cv2.cvtColor(cur_f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        luma = cv2.GaussianBlur(luma, (15, 15), 0)
        softness = 0.28
        cutoff = 1.0 - p * (1.0 + softness)
        alpha = np.clip((luma - cutoff) / softness, 0.0, 1.0)[..., None]
        out = prev_f.astype(np.float32) * (1.0 - alpha) + cur_f.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    if tr == "film_burn":
        t = max(0.0, min(1.0, (p - 0.30) / 0.40))
        base = cv2.addWeighted(prev_f, 1.0 - t, cur_f, t, 0).astype(np.float32)
        burn_intensity = math.sin(p * math.pi)
        burn_mask = _make_film_burn_mask(width, height, p) * (0.95 * burn_intensity)
        # Screen-style additive glow
        out = 255.0 - ((255.0 - base) * (255.0 - np.clip(burn_mask, 0.0, 255.0)) / 255.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    if tr == "overexposure":
        glow = math.sin(p * math.pi)
        t = max(0.0, min(1.0, (p - 0.42) / 0.16))
        base = cv2.addWeighted(prev_f, 1.0 - t, cur_f, t, 0).astype(np.float32)
        gray = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        desat = 0.42 * glow
        mixed = base * (1.0 - desat) + gray * desat
        out = mixed * (1.0 + 1.85 * glow) + 55.0 * (glow ** 2)
        return np.clip(out, 0, 255).astype(np.uint8)

    if tr == "split_horizontal":
        half_w = int(round(p * (width / 2.0)))
        cx = width // 2
        left = max(0, cx - half_w)
        right = min(width, cx + half_w)
        out = prev_f.copy()
        if right > left:
            out[:, left:right] = cur_f[:, left:right]
        return out

    if tr == "split_vertical":
        half_h = int(round(p * (height / 2.0)))
        cy = height // 2
        top = max(0, cy - half_h)
        bottom = min(height, cy + half_h)
        out = prev_f.copy()
        if bottom > top:
            out[top:bottom, :] = cur_f[top:bottom, :]
        return out

    if tr == "flash_white":
        if p < 0.5:
            wp = p * 2.0
            out = prev_f.astype(np.float32) * (1.0 - wp) + 255.0 * wp
        else:
            wp = (1.0 - p) * 2.0
            out = cur_f.astype(np.float32) * (1.0 - wp) + 255.0 * wp
        return np.clip(out, 0, 255).astype(np.uint8)

    if tr == "flash_black":
        if p < 0.5:
            bp = 1.0 - p * 2.0
            out = prev_f.astype(np.float32) * bp
        else:
            bp = (p - 0.5) * 2.0
            out = cur_f.astype(np.float32) * bp
        return np.clip(out, 0, 255).astype(np.uint8)

    if tr == "slide_left":
        dx = int(round(p * width))
        out = np.zeros_like(cur_f)
        if dx < width:
            out[:, :width - dx] = prev_f[:, dx:]
        if dx > 0:
            out[:, width - dx:] = cur_f[:, :dx]
        return out

    if tr == "slide_right":
        dx = int(round(p * width))
        out = np.zeros_like(cur_f)
        if dx < width:
            out[:, dx:] = prev_f[:, :width - dx]
        if dx > 0:
            out[:, :dx] = cur_f[:, width - dx:]
        return out

    if tr == "slide_up":
        dy = int(round(p * height))
        out = np.zeros_like(cur_f)
        if dy < height:
            out[:height - dy, :] = prev_f[dy:, :]
        if dy > 0:
            out[height - dy:, :] = cur_f[:dy, :]
        return out

    if tr == "slide_down":
        dy = int(round(p * height))
        out = np.zeros_like(cur_f)
        if dy < height:
            out[dy:, :] = prev_f[:height - dy, :]
        if dy > 0:
            out[:dy, :] = cur_f[height - dy:, :]
        return out

    if tr == "wipe_left":
        cut = max(0, min(width, int(round(p * width))))
        out = prev_f.copy()
        if cut > 0:
            out[:, :cut] = cur_f[:, :cut]
        return out

    if tr in ("zoom_in", "zoom_out"):
        s_prev = (1.0 + 0.16 * p) if tr == "zoom_in" else max(0.75, 1.0 - 0.15 * p)
        s_cur = max(0.75, 0.85 + 0.15 * p) if tr == "zoom_in" else (1.18 - 0.18 * p)
        cx, cy = width / 2.0, height / 2.0
        m_prev = np.array([[s_prev, 0.0, cx * (1.0 - s_prev)], [0.0, s_prev, cy * (1.0 - s_prev)]], dtype=np.float64)
        m_cur = np.array([[s_cur, 0.0, cx * (1.0 - s_cur)], [0.0, s_cur, cy * (1.0 - s_cur)]], dtype=np.float64)
        z_prev = cv2.warpAffine(prev_f, m_prev, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        z_cur = cv2.warpAffine(cur_f, m_cur, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(z_prev, 1.0 - p, z_cur, p, 0)

    if tr == "blur":
        sig_prev = max(0.1, p * 14.0)
        sig_cur = max(0.1, (1.0 - p) * 14.0)
        b_prev = cv2.GaussianBlur(prev_f, (0, 0), sigmaX=sig_prev) if sig_prev > 0.5 else prev_f
        b_cur = cv2.GaussianBlur(cur_f, (0, 0), sigmaX=sig_cur) if sig_cur > 0.5 else cur_f
        return cv2.addWeighted(b_prev, 1.0 - p, b_cur, p, 0)

    return cur_f


def concat_with_transitions_opencv(
    segment_files: list[str],
    clip_transitions: list[tuple[str, float]],
    out_path: str,
    width: int,
    height: int,
    fps: int,
    v_enc_args: list[str],
) -> None:
    """Concatena segmentos aplicando transições opcionais entre clipes sem alterar a duração total."""
    ff_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        *v_enc_args,
        "-an",
        out_path,
    ]
    proc, stderr_chunks, stderr_thread = _start_ffmpeg_pipe(ff_cmd)
    prev_last_frame = None

    try:
        for seg_idx, seg_path in enumerate(segment_files):
            cap = cv2.VideoCapture(seg_path)
            if not cap.isOpened():
                continue

            tr_name, tr_dur = clip_transitions[seg_idx] if seg_idx < len(clip_transitions) else ("none", 0.0)
            total_seg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            trans_frames = 0
            if prev_last_frame is not None and tr_name and tr_name != "none" and tr_dur > 0.05:
                max_safe = max(2, int(total_seg_frames * 0.45)) if total_seg_frames > 0 else int(round(tr_dur * fps))
                trans_frames = min(max(2, int(round(tr_dur * fps))), max_safe)

            f_local = 0
            last_valid = None
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                if frm.shape[1] != width or frm.shape[0] != height:
                    frm = cv2.resize(frm, (width, height), interpolation=cv2.INTER_LINEAR)
                last_valid = frm

                try:
                    if trans_frames > 0 and f_local < trans_frames and prev_last_frame is not None:
                        p = (f_local + 1) / float(trans_frames + 1)
                        out_frm = blend_transition_frames(prev_last_frame, frm, tr_name, p, width, height)
                        proc.stdin.write(memoryview(np.ascontiguousarray(out_frm)))
                    else:
                        proc.stdin.write(memoryview(np.ascontiguousarray(frm)))
                except BrokenPipeError:
                    break
                f_local += 1

            cap.release()
            if last_valid is not None:
                prev_last_frame = last_valid
    finally:
        _close_and_wait_ffmpeg(proc, stderr_chunks, stderr_thread, "Erro ao aplicar transições entre imagens")


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
# RENDERIZADOR DE LEGENDAS CANVAS-IDÊNTICO (PILLOW RGBA + FONTES + CANTOS ARREDONDADOS)
# ==============================================================================

_FONT_PATH_CACHE: dict[str, str] = {}


def _hex_to_rgb(hex_str: str, default=(255, 255, 255)) -> tuple[int, int, int]:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) < 6:
        return default
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except Exception:
        return default


def _resolve_pil_font_path(role: str) -> str | None:
    if role in _FONT_PATH_CACHE:
        return _FONT_PATH_CACHE[role] or None

    fc_queries_and_paths = {
        "sans_black": (
            ["Liberation Sans:style=Bold", "Roboto:style=Black", "Roboto:style=Bold", "DejaVu Sans:style=Bold"],
            [
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Black.ttf",
                "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ],
        ),
        "sans_italic_bold": (
            ["Liberation Sans:style=Bold Italic", "Roboto:style=Bold Italic", "DejaVu Sans:style=Bold Oblique"],
            [
                "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf",
                "/usr/share/fonts/truetype/roboto/Roboto-BoldItalic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
            ],
        ),
        "roboto_bold": (
            ["Roboto:style=Black", "Roboto:style=Bold", "Liberation Sans:style=Bold", "DejaVu Sans:style=Bold"],
            [
                "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Black.ttf",
                "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
                "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ],
        ),
        "roboto_med": (
            ["Roboto:style=Medium", "Roboto:style=Regular", "Liberation Sans:style=Regular", "DejaVu Sans:style=Book"],
            [
                "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Medium.ttf",
                "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
                "/usr/share/fonts/truetype/roboto/Roboto-Medium.ttf",
                "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
        ),
        "comic": (
            ["Comic Neue:style=Bold", "Comic Sans MS:style=Bold", "DejaVu Sans:style=Bold"],
            [
                "/usr/share/fonts/truetype/comic-neue/ComicNeue-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ],
        ),
        "serif_italic": (
            ["Liberation Serif:style=Italic", "DejaVu Serif:style=Italic"],
            [
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
            ],
        ),
        "serif_bold": (
            ["Liberation Serif:style=Bold", "DejaVu Serif:style=Bold"],
            [
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            ],
        ),
    }

    queries, fallbacks = fc_queries_and_paths.get(role, fc_queries_and_paths["sans_black"])
    resolved = None
    fc_bin = shutil.which("fc-match")
    if fc_bin:
        for q in queries:
            try:
                p = subprocess.check_output([fc_bin, "-f", "%{file}", q], text=True, stderr=subprocess.DEVNULL).strip()
                if p and os.path.exists(p):
                    resolved = p
                    break
            except Exception:
                pass

    if not resolved:
        for fp in fallbacks:
            if os.path.exists(fp):
                resolved = fp
                break

    _FONT_PATH_CACHE[role] = resolved or ""
    return resolved


def _load_pil_font(role: str, size_px: int):
    sz = max(10, int(round(size_px)))
    fp = _resolve_pil_font_path(role)
    if fp:
        try:
            return ImageFont.truetype(fp, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _render_pil_subtitle_overlay(
    text: str,
    sub_config: dict,
    width: int,
    height: int,
) -> dict | None:
    """Desenha 1 legenda idêntica ao Canvas de FinalizarVideo.tsx em caixa RGBA recortada e pronta para compor em OpenCV."""
    if not HAS_PIL:
        return None

    lines = [ln for ln in str(text or "").split("\n") if ln]
    if not lines:
        return None

    style = str((sub_config or {}).get("style") or "classic")
    sub_color = str((sub_config or {}).get("color") or "")
    sub_scale = float((sub_config or {}).get("scale") or 1.0)

    ref_side = min(width, height)
    x = width / 2.0
    margin = height * 0.07
    y = height - margin

    canvas_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas_img)

    def _draw_shadow_layer(draw_fn, ox: float, oy: float, blur_r: float):
        if blur_r <= 0.4:
            sh = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sh)
            draw_fn(sd, ox, oy)
            canvas_img.alpha_composite(sh)
            return
        sh = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        draw_fn(sd, ox, oy)
        sh = sh.filter(ImageFilter.GaussianBlur(radius=float(blur_r)))
        canvas_img.alpha_composite(sh)

    def _measure_max_w(font_obj) -> float:
        mw = 40.0
        for ln in lines:
            try:
                w_ln = float(draw.textlength(ln, font=font_obj))
            except Exception:
                bb = draw.textbbox((0, 0), ln, font=font_obj)
                w_ln = float(bb[2] - bb[0])
            if w_ln > mw:
                mw = w_ln
        return mw

    is_karaoke = style == "karaoke"
    active_overlay_arr = None
    text_left_x = width
    text_right_x = 0

    if style in ("impact", "mrbeast", "hormozi"):
        font_size = max(12, int(math.floor(ref_side * 0.056 * sub_scale)))
        line_h = font_size * 1.32
        font = _load_pil_font("sans_italic_bold" if style == "impact" else "sans_black", font_size)
        _, descent = font.getmetrics()
        sw_canvas = max(4, int(math.floor(ref_side * 0.014)))
        sw = max(2, int(round(sw_canvas * 0.5)))
        sh_oy = max(2, int(math.floor(ref_side * 0.006)))
        sh_blur = max(1.0, math.floor(ref_side * 0.01) * 0.65)

        def _paint_sh(sd, ox, oy):
            for i, ln in enumerate(lines):
                ly = y - (len(lines) - 1 - i) * line_h
                by = ly - descent
                sd.text((x + ox, by + oy), ln, font=font, fill=(0, 0, 0, 242), stroke_width=sw, stroke_fill=(0, 0, 0, 242), anchor="ms")

        _draw_shadow_layer(_paint_sh, 0.0, float(sh_oy), sh_blur)
        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            by = ly - descent
            if style == "hormozi":
                rgb = (255, 255, 255) if (i % 2 == 0) else _hex_to_rgb(sub_color, (74, 222, 128))
            elif style == "mrbeast":
                rgb = _hex_to_rgb(sub_color, (250, 204, 21))
            else:
                rgb = _hex_to_rgb(sub_color, (255, 255, 255))
            draw.text((x, by), ln, font=font, fill=(*rgb, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 255), anchor="ms")

    elif style in ("tiktok-yellow", "tiktok-black", "capcut-box", "badge-purple", "youtube", "youtube-amarelo", "youtube-shorts"):
        is_yt = style in ("youtube", "youtube-amarelo")
        font_size = max(12, int(math.floor(ref_side * (0.041 if is_yt else 0.046) * sub_scale)))
        line_h = font_size * 1.38
        font = _load_pil_font("roboto_med" if is_yt else "roboto_bold", font_size)
        _, descent = font.getmetrics()
        pad_x = max(16.0, width * 0.024)
        pad_y = max(8.0, height * 0.011)
        max_w = _measure_max_w(font)
        rect_w = max_w + pad_x * 2.0
        total_h = len(lines) * line_h
        rect_h = total_h + pad_y * 1.6
        rx = x - rect_w / 2.0
        ry = y - total_h - pad_y * 0.5
        rad = 6 if is_yt else 12

        box_rgba_map = {
            "tiktok-yellow": (255, 221, 0, 255),
            "tiktok-black": (0, 0, 0, 230),
            "capcut-box": (17, 24, 39, 209),
            "badge-purple": (124, 58, 237, 255),
            "youtube-shorts": (220, 38, 38, 255),
            "youtube": (8, 8, 8, 199),
            "youtube-amarelo": (8, 8, 8, 199),
        }
        box_rgba = box_rgba_map.get(style, (8, 8, 8, 199))
        draw.rounded_rectangle([rx, ry, rx + rect_w, ry + rect_h], radius=rad, fill=box_rgba)

        if style == "tiktok-yellow":
            txt_rgba = (0, 0, 0, 255)
        elif style == "youtube-amarelo":
            txt_rgba = (250, 204, 21, 255)
        else:
            txt_rgba = (255, 255, 255, 255)

        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            by = (ly - pad_y * 0.25) - descent
            draw.text((x, by), ln, font=font, fill=txt_rgba, anchor="ms")

    elif style == "neon":
        font_size = max(12, int(math.floor(ref_side * 0.052 * sub_scale)))
        line_h = font_size * 1.35
        font = _load_pil_font("sans_black", font_size)
        _, descent = font.getmetrics()
        glow_blur = max(4.0, math.floor(ref_side * 0.022) * 0.70)

        def _paint_neon_glow(sd, ox, oy):
            for i, ln in enumerate(lines):
                ly = y - (len(lines) - 1 - i) * line_h
                sd.text((x + ox, (ly - descent) + oy), ln, font=font, fill=(6, 182, 212, 255), anchor="ms")

        def _paint_neon_drop(sd, ox, oy):
            for i, ln in enumerate(lines):
                ly = y - (len(lines) - 1 - i) * line_h
                sd.text((x + ox, (ly - descent) + oy), ln, font=font, fill=(0, 0, 0, 217), anchor="ms")

        _draw_shadow_layer(_paint_neon_glow, 0.0, 0.0, glow_blur)
        _draw_shadow_layer(_paint_neon_glow, 0.0, 0.0, max(2.0, glow_blur * 0.45))
        _draw_shadow_layer(_paint_neon_drop, 2.0, 2.0, 1.5)
        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            draw.text((x, ly - descent), ln, font=font, fill=(34, 211, 238, 255), anchor="ms")

    elif style == "netflix":
        font_size = max(12, int(math.floor(ref_side * 0.042 * sub_scale)))
        line_h = font_size * 1.38
        font = _load_pil_font("sans_black", font_size)
        _, descent = font.getmetrics()
        sw = max(1, int(round(max(2, math.floor(ref_side * 0.004)) * 0.5)))

        def _paint_nf(sd, ox, oy):
            for i, ln in enumerate(lines):
                ly = y - (len(lines) - 1 - i) * line_h
                sd.text((x + ox, (ly - descent) + oy), ln, font=font, fill=(0, 0, 0, 230), stroke_width=sw, stroke_fill=(0, 0, 0, 230), anchor="ms")

        _draw_shadow_layer(_paint_nf, 0.0, 2.0, 2.5)
        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            draw.text((x, ly - descent), ln, font=font, fill=(250, 204, 21, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 255), anchor="ms")

    elif style == "comic":
        font_size = max(12, int(math.floor(ref_side * 0.052 * sub_scale)))
        line_h = font_size * 1.35
        font = _load_pil_font("comic", font_size)
        _, descent = font.getmetrics()
        sw = max(2, int(round(max(3, math.floor(ref_side * 0.01)) * 0.5)))
        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            by = ly - descent
            draw.text((x + 3.0, by + 3.0), ln, font=font, fill=(0, 0, 0, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 255), anchor="ms")
            draw.text((x, by), ln, font=font, fill=(255, 235, 59, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 255), anchor="ms")

    elif style == "karaoke":
        font_size = max(12, int(math.floor(ref_side * 0.056 * sub_scale)))
        line_h = font_size * 1.34
        font = _load_pil_font("sans_black", font_size)
        _, descent = font.getmetrics()
        sw = max(2, int(round(max(4, math.floor(ref_side * 0.014)) * 0.5)))
        sh_oy = max(2, int(math.floor(ref_side * 0.006)))
        sh_blur = max(1.0, math.floor(ref_side * 0.01) * 0.65)
        max_w = _measure_max_w(font)
        text_left_x = int(math.floor(x - max_w / 2.0))
        text_right_x = int(math.ceil(x + max_w / 2.0))

        def _paint_kar_sh(sd, ox, oy):
            for i, ln in enumerate(lines):
                ly = y - (len(lines) - 1 - i) * line_h
                sd.text((x + ox, (ly - descent) + oy), ln, font=font, fill=(0, 0, 0, 242), stroke_width=sw, stroke_fill=(0, 0, 0, 242), anchor="ms")

        _draw_shadow_layer(_paint_kar_sh, 0.0, float(sh_oy), sh_blur)
        active_img = canvas_img.copy()
        active_draw = ImageDraw.Draw(active_img)
        active_rgb = _hex_to_rgb(sub_color, (255, 255, 0))
        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            by = ly - descent
            draw.text((x, by), ln, font=font, fill=(255, 255, 255, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 255), anchor="ms")
            active_draw.text((x, by), ln, font=font, fill=(*active_rgb, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 255), anchor="ms")

        active_overlay_arr = np.array(active_img, dtype=np.uint8)

    elif style in ("elegant", "documentary"):
        font_size = max(12, int(math.floor(ref_side * 0.044 * sub_scale)))
        line_h = font_size * 1.40
        font = _load_pil_font("serif_italic" if style == "elegant" else "serif_bold", font_size)
        _, descent = font.getmetrics()
        sw = max(1, int(round(max(2, math.floor(ref_side * 0.004)) * 0.5)))

        def _paint_ed(sd, ox, oy):
            for i, ln in enumerate(lines):
                ly = y - (len(lines) - 1 - i) * line_h
                sd.text((x + ox, (ly - descent) + oy), ln, font=font, fill=(0, 0, 0, 204), stroke_width=sw, stroke_fill=(0, 0, 0, 204), anchor="ms")

        _draw_shadow_layer(_paint_ed, 0.0, 3.0, 4.5)
        rgb = (248, 250, 252) if style == "documentary" else (255, 255, 255)
        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            draw.text((x, ly - descent), ln, font=font, fill=(*rgb, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 217), anchor="ms")

    else:
        # clean-white, classic & youtube-vlogger
        font_size = max(12, int(math.floor(ref_side * 0.046 * sub_scale)))
        line_h = font_size * 1.38
        font = _load_pil_font("roboto_bold" if style == "youtube-vlogger" else "sans_black", font_size)
        _, descent = font.getmetrics()
        sw = max(1, int(round(max(2, math.floor(ref_side * 0.006)) * 0.5)))

        def _paint_std(sd, ox, oy):
            for i, ln in enumerate(lines):
                ly = y - (len(lines) - 1 - i) * line_h
                sd.text((x + ox, (ly - descent) + oy), ln, font=font, fill=(0, 0, 0, 217), stroke_width=sw, stroke_fill=(0, 0, 0, 217), anchor="ms")

        _draw_shadow_layer(_paint_std, 0.0, 2.0, 2.5)
        rgb = _hex_to_rgb(sub_color, (255, 255, 255)) if style == "clean-white" else (255, 255, 255)
        for i, ln in enumerate(lines):
            ly = y - (len(lines) - 1 - i) * line_h
            draw.text((x, ly - descent), ln, font=font, fill=(*rgb, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 235), anchor="ms")

    bbox = canvas_img.getbbox()
    if not bbox:
        return None

    bx0 = max(0, bbox[0] - 2)
    by0 = max(0, bbox[1] - 2)
    bx1 = min(width, bbox[2] + 2)
    by1 = min(height, bbox[3] + 2)
    if bx1 <= bx0 or by1 <= by0:
        return None

    crop_rgba = np.array(canvas_img.crop((bx0, by0, bx1, by1)), dtype=np.uint8)
    bgr_u16 = crop_rgba[:, :, :3][:, :, ::-1].astype(np.uint16)
    alpha_u16 = crop_rgba[:, :, 3:4].astype(np.uint16)
    inv_u16 = 255 - alpha_u16

    out = {
        "bx0": bx0,
        "by0": by0,
        "bx1": bx1,
        "by1": by1,
        "bgr_u16": bgr_u16,
        "alpha_u16": alpha_u16,
        "inv_u16": inv_u16,
        "rgba": crop_rgba,
        "karaoke": False,
    }

    if is_karaoke and active_overlay_arr is not None:
        act_crop = active_overlay_arr[by0:by1, bx0:bx1]
        out["karaoke"] = True
        out["act_bgr_u16"] = act_crop[:, :, :3][:, :, ::-1].astype(np.uint16)
        out["act_alpha_u16"] = act_crop[:, :, 3:4].astype(np.uint16)
        out["act_inv_u16"] = 255 - out["act_alpha_u16"]
        out["t_left"] = max(0, text_left_x - bx0)
        out["t_right"] = min(bx1 - bx0, text_right_x - bx0)

    return out


def _save_overlay_png(ov: dict, dest_path: str) -> None:
    rgba = ov["rgba"]
    bgra = rgba[:, :, [2, 1, 0, 3]]
    cv2.imwrite(dest_path, bgra)


def _prepare_clip_pil_overlays(pil_subs: list[dict], sub_config: dict, width: int, height: int) -> list[tuple[float, float, dict]]:
    rendered: list[tuple[float, float, dict]] = []
    for item in pil_subs:
        s_t = float(item.get("start") or 0.0)
        e_t = max(s_t + 0.05, float(item.get("end") or (s_t + 1.0)))
        txt = str(item.get("text") or "").strip()
        if not txt:
            continue
        ov = _render_pil_subtitle_overlay(txt, sub_config, width, height)
        if ov is not None:
            rendered.append((s_t, e_t, ov))
    return rendered


def _find_active_pil_overlay(overlays: list[tuple[float, float, dict]], cur_t: float):
    for s_t, e_t, ov in overlays:
        if s_t <= cur_t < e_t:
            return (s_t, e_t, ov)
    return None


def _stamp_pil_subtitle(frame: np.ndarray, ov: dict, cur_t: float, s_t: float, e_t: float) -> None:
    bx0, by0, bx1, by1 = ov["bx0"], ov["by0"], ov["bx1"], ov["by1"]
    roi_u16 = frame[by0:by1, bx0:bx1].astype(np.uint16)

    if ov.get("karaoke"):
        prog = max(0.0, min(1.0, (cur_t - s_t) / max(0.1, e_t - s_t)))
        tl = ov["t_left"]
        tr = ov["t_right"]
        split_x = max(0, min(bx1 - bx0, int(round(tl + (tr - tl) * prog))))
        out_roi = ((roi_u16 * ov["inv_u16"] + ov["bgr_u16"] * ov["alpha_u16"]) * 257) >> 16
        if split_x > 0:
            act_part = ((roi_u16[:, :split_x] * ov["act_inv_u16"][:, :split_x] + ov["act_bgr_u16"][:, :split_x] * ov["act_alpha_u16"][:, :split_x]) * 257) >> 16
            out_roi[:, :split_x] = act_part
        frame[by0:by1, bx0:bx1] = out_roi.astype(np.uint8)
        return

    blended = ((roi_u16 * ov["inv_u16"] + ov["bgr_u16"] * ov["alpha_u16"]) * 257) >> 16
    frame[by0:by1, bx0:bx1] = blended.astype(np.uint8)


# ==============================================================================
# PARSING DE LEGENDAS & WORKER MULTIPROCESSADO DE CLIPES
# ==============================================================================

def _parse_ass_or_srt_time(ts: str) -> float:
    parts = ts.strip().replace(",", ".").split(":")
    if len(parts) != 3:
        return 0.0
    try:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except Exception:
        return 0.0


def _parse_subtitle_intervals(sub_file_path: str | None) -> list[tuple[float, float]]:
    if not sub_file_path or not os.path.exists(sub_file_path):
        return []
    intervals: list[tuple[float, float]] = []
    try:
        with open(sub_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if sub_file_path.endswith(".ass"):
            for line in content.splitlines():
                if line.startswith("Dialogue:"):
                    cols = line.split(",", 9)
                    if len(cols) >= 3:
                        s_t = _parse_ass_or_srt_time(cols[1])
                        e_t = _parse_ass_or_srt_time(cols[2])
                        if e_t > s_t:
                            intervals.append((s_t, e_t))
        else:
            for m in re.finditer(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})", content):
                s_t = _parse_ass_or_srt_time(m.group(1))
                e_t = _parse_ass_or_srt_time(m.group(2))
                if e_t > s_t:
                    intervals.append((s_t, e_t))
    except Exception as e:
        print(f"[RunPod Warning] Falha ao analisar intervalos de legenda: {e}")
    return intervals


def _clip_has_subtitles(clip_start: float, clip_end: float, intervals: list[tuple[float, float]]) -> bool:
    for s_t, e_t in intervals:
        if e_t > clip_start and s_t < clip_end:
            return True
    return False


def _render_clip_worker(task_args: tuple) -> tuple[int, str]:
    (
        idx,
        clip,
        clips,
        clip_paths,
        work_dir,
        width,
        height,
        fps,
        v_enc_args,
        authentic_intro_secs,
        clip_timeline_start,
        sub_filter,
        clip_needs_sub,
        pil_clip_subs,
        sub_config,
    ) = task_args

    if HAS_OPENCV:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    dur = max(0.2, float(clip.get("duration", 3.0)))
    seg_out = os.path.join(work_dir, f"seg_{idx+1:04d}.mp4")
    src_path = clip_paths.get(idx)

    use_pil_here = bool(HAS_PIL and HAS_OPENCV and pil_clip_subs)

    sub_vf = None
    if sub_filter and clip_needs_sub and not use_pil_here:
        sub_vf = f"setpts=PTS+{clip_timeline_start:.3f}/TB,{sub_filter},setpts=PTS-STARTPTS"

    if not src_path or not os.path.exists(src_path):
        if use_pil_here:
            pil_overlays = _prepare_clip_pil_overlays(pil_clip_subs, sub_config or {}, width, height)
            use_yuv = (width % 2 == 0 and height % 2 == 0)
            total_frames = max(2, int(round(dur * fps)))
            ff_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p" if use_yuv else "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
                *v_enc_args, "-an", seg_out,
            ]
            proc, stderr_chunks, stderr_thread = _start_ffmpeg_pipe(ff_cmd)
            try:
                black_frm = np.zeros((height, width, 3), dtype=np.uint8)
                black_yuv = cv2.cvtColor(black_frm, cv2.COLOR_BGR2YUV_I420) if use_yuv else None
                for f_idx in range(total_frames):
                    cur_t = clip_timeline_start + (f_idx / float(fps))
                    active_sub = _find_active_pil_overlay(pil_overlays, cur_t)
                    try:
                        if active_sub is None and black_yuv is not None:
                            proc.stdin.write(memoryview(black_yuv))
                        elif active_sub is None:
                            proc.stdin.write(memoryview(black_frm))
                        else:
                            frm = black_frm.copy()
                            _stamp_pil_subtitle(frm, active_sub[2], cur_t, active_sub[0], active_sub[1])
                            _write_raw_frame(proc.stdin, frm, use_yuv)
                    except BrokenPipeError:
                        break
            finally:
                _close_and_wait_ffmpeg(proc, stderr_chunks, stderr_thread, f"Erro clipe preto+legenda {idx+1}")
            return idx, seg_out

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={dur:.3f}",
        ]
        if sub_vf:
            cmd.extend(["-vf", sub_vf])
        cmd.extend([*v_enc_args, "-an", seg_out])
        run_cmd(cmd, f"Clipe {idx+1} (tela preta)")
        return idx, seg_out

    vf_post_parts = []
    vis_filter = get_visual_filter_expr(clip.get("visualFilter", "none"))
    if vis_filter:
        vf_post_parts.append(vis_filter)

    ov_filter = get_overlay_filter_expr(clip.get("overlayEffect", "none"), width, height)
    if ov_filter:
        vf_post_parts.append(ov_filter)

    if sub_vf:
        vf_post_parts.append(sub_vf)

    movement = clip.get("movement", "none")
    camera_effect = clip.get("cameraEffect", "none")
    custom_transform = clip.get("customTransform")
    needs_motion = (
        (movement and movement != "none")
        or (camera_effect and camera_effect != "none")
        or bool(custom_transform)
    )

    tr_name = str(clip.get("transition") or "none").lower()
    tr_dur = float(clip.get("transitionDuration") or 0.5)
    has_tr_in = (idx > 0 or authentic_intro_secs > 0.05) and tr_name not in ("", "none") and tr_dur > 0.05

    if HAS_OPENCV and (clip.get("type") != "video" or needs_motion or has_tr_in or use_pil_here):
        prev_end_frame = (
            get_prev_clip_end_frame(idx, clips, clip_paths, width, height, authentic_intro_secs)
            if has_tr_in
            else None
        )
        render_clip_with_opencv(
            idx,
            clip,
            src_path,
            seg_out,
            width,
            height,
            fps,
            dur,
            vf_post_parts,
            v_enc_args,
            prev_end_frame=prev_end_frame,
            tr_name=tr_name if has_tr_in else "none",
            tr_dur=tr_dur if has_tr_in else 0.0,
            pil_subs=pil_clip_subs if use_pil_here else None,
            sub_config=sub_config,
            clip_timeline_start=clip_timeline_start,
            work_dir=work_dir,
        )
        return idx, seg_out

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
        "ffmpeg", "-y", "-loglevel", "error",
        *input_args,
        "-vf", vf_str,
        *v_enc_args,
        "-an",
        seg_out
    ], f"Renderizar clipe {idx+1}/{len(clips)}")
    return idx, seg_out


def _report_progress(job: dict, progress_pct: int, stage: str) -> None:
    try:
        runpod.serverless.progress_update(job, {"progress": int(progress_pct), "stage": stage})
    except Exception:
        pass


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
        character_overlays = inp.get("characterOverlays", [])
        subtitles_ass = inp.get("subtitlesAss", "")
        subtitles_srt = inp.get("subtitlesSrt", "")
        subtitle_items = inp.get("subtitleItems") or []
        subtitle_config = inp.get("subtitleConfig") or {}
        callback_upload_url = inp.get("callbackUploadUrl", "")
        authentic_intro_secs = float(inp.get("authenticIntroSecs", 0))

        if not clips and total_duration <= 0:
            return {"error": "Nenhum clipe ou duração válida enviada no payload."}

        use_nvenc = has_nvenc()
        vcodec = "h264_nvenc" if use_nvenc else "libx264"
        if use_nvenc:
            if preset_mode == "high":
                v_enc_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-pix_fmt", "yuv420p"]
            elif preset_mode == "fast":
                v_enc_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", "-rc", "constqp", "-qp", "25", "-bf", "0", "-pix_fmt", "yuv420p"]
            else:
                v_enc_args = ["-c:v", "h264_nvenc", "-preset", "p3", "-cq", "21", "-bf", "0", "-pix_fmt", "yuv420p"]
        else:
            preset_flag = "slow" if preset_mode == "high" else ("ultrafast" if preset_mode == "fast" else "medium")
            crf_val = "18" if preset_mode == "high" else ("26" if preset_mode == "fast" else "21")
            v_enc_args = ["-c:v", "libx264", "-preset", preset_flag, "-crf", crf_val, "-pix_fmt", "yuv420p"]

        use_pil_subs = bool(HAS_PIL and HAS_OPENCV and subtitle_items)
        print(f"[RunPod] Encoder selecionado: {vcodec} ({preset_mode}) | {width}x{height} @ {fps}fps | {len(clips)} clipes | {len(audio_tracks)} áudios | Legendas Canvas(PIL): {use_pil_subs} | CPUs: {os.cpu_count()}")
        _report_progress(job, 55, f"Baixando {len(clips)} mídias e {len(audio_tracks)} áudios...")

        # 1. Baixar todas as mídias e áudios em paralelo
        media_dir = os.path.join(work_dir, "media")
        os.makedirs(media_dir, exist_ok=True)

        clip_paths = {}
        audio_paths = {}
        bgm_path = None

        with ThreadPoolExecutor(max_workers=10) as pool:
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

        # Preparar arquivo de legendas antecipadamente (.ASS/.SRT fallback + lista estruturada Pillow Canvas-idêntica)
        sub_file_path = None
        sub_filter = None
        sub_intervals: list[tuple[float, float]] = []
        if subtitles_ass and subtitles_ass.strip():
            sub_file_path = os.path.join(work_dir, "subtitles.ass")
            with open(sub_file_path, "w", encoding="utf-8") as f:
                f.write(subtitles_ass)
        elif subtitles_srt and subtitles_srt.strip():
            sub_file_path = os.path.join(work_dir, "subtitles.srt")
            with open(sub_file_path, "w", encoding="utf-8") as f:
                f.write(subtitles_srt)

        if sub_file_path:
            escaped_sub = sub_file_path.replace("\\", "/").replace(":", "\\:")
            sub_filter = f"ass={escaped_sub}" if sub_file_path.endswith(".ass") else f"subtitles={escaped_sub}"
            sub_intervals = _parse_subtitle_intervals(sub_file_path)

        has_char_overlays = bool(character_overlays)
        has_any_subs = bool(use_pil_subs or sub_filter)
        burn_subs_in_clips = bool(has_any_subs and not has_char_overlays)

        def _get_overlapping_pil_subs(s_win: float, e_win: float) -> list[dict]:
            if not use_pil_subs:
                return []
            res = []
            for it in subtitle_items:
                st_i = float(it.get("start") or 0.0)
                en_i = max(st_i + 0.05, float(it.get("end") or (st_i + 1.0)))
                if en_i > s_win and st_i < e_win:
                    res.append(it)
            return res

        # 2. Renderizar cada clipe individualmente em multiprocesso (sem GIL!)
        segment_files = []
        intro_dur = authentic_intro_secs if authentic_intro_secs > 0.05 else 0.0

        if intro_dur > 0:
            black_seg = os.path.join(work_dir, "seg_0000_black.mp4")
            intro_pil_subs = _get_overlapping_pil_subs(0.0, intro_dur) if burn_subs_in_clips else []
            if intro_pil_subs:
                pil_ov = _prepare_clip_pil_overlays(intro_pil_subs, subtitle_config, width, height)
                use_yuv_i = (width % 2 == 0 and height % 2 == 0)
                n_ifrm = max(2, int(round(intro_dur * fps)))
                proc_i, err_c_i, err_t_i = _start_ffmpeg_pipe([
                    "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p" if use_yuv_i else "bgr24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", *v_enc_args, "-an", black_seg
                ])
                try:
                    blk_i = np.zeros((height, width, 3), dtype=np.uint8)
                    blk_yuv_i = cv2.cvtColor(blk_i, cv2.COLOR_BGR2YUV_I420) if use_yuv_i else None
                    for fi in range(n_ifrm):
                        ct_i = fi / float(fps)
                        act_i = _find_active_pil_overlay(pil_ov, ct_i)
                        if act_i is None and blk_yuv_i is not None:
                            proc_i.stdin.write(memoryview(blk_yuv_i))
                        elif act_i is None:
                            proc_i.stdin.write(memoryview(blk_i))
                        else:
                            frm_i = blk_i.copy()
                            _stamp_pil_subtitle(frm_i, act_i[2], ct_i, act_i[0], act_i[1])
                            _write_raw_frame(proc_i.stdin, frm_i, use_yuv_i)
                finally:
                    _close_and_wait_ffmpeg(proc_i, err_c_i, err_t_i, "Erro tela preta inicial")
            else:
                intro_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={intro_dur:.3f}",
                ]
                if burn_subs_in_clips and sub_filter and _clip_has_subtitles(0.0, intro_dur, sub_intervals):
                    intro_cmd.extend(["-vf", sub_filter])
                intro_cmd.extend([*v_enc_args, "-an", black_seg])
                run_cmd(intro_cmd, "Gerar tela preta inicial")
            segment_files.append(black_seg)

        clip_tasks = []
        t_cursor = intro_dur
        for idx, clip in enumerate(clips):
            dur = max(0.2, float(clip.get("duration", 3.0)))
            clip_end = t_cursor + dur
            pil_for_clip = _get_overlapping_pil_subs(t_cursor, clip_end) if burn_subs_in_clips else []
            needs_sub = burn_subs_in_clips and (
                bool(pil_for_clip) or _clip_has_subtitles(t_cursor, clip_end, sub_intervals)
            )
            clip_tasks.append((
                idx,
                clip,
                clips,
                clip_paths,
                work_dir,
                width,
                height,
                fps,
                v_enc_args,
                authentic_intro_secs,
                t_cursor,
                sub_filter if burn_subs_in_clips else None,
                needs_sub,
                pil_for_clip,
                subtitle_config,
            ))
            t_cursor = clip_end

        cpu_cnt = os.cpu_count() or 4
        clip_workers = min(8, max(3, cpu_cnt)) if use_nvenc else min(4, max(2, cpu_cnt // 2))
        print(f"[RunPod] Renderizando {len(clips)} clipes com {clip_workers} processos paralelos (legendas no passe dos clipes: {burn_subs_in_clips} | modo PIL: {use_pil_subs})")

        rendered_by_idx = {}
        done_count = 0
        last_prog_ts = 0.0
        total_clips = max(1, len(clip_tasks))

        def _track_clip_done(c_idx: int, res_path: str):
            nonlocal done_count, last_prog_ts
            rendered_by_idx[c_idx] = res_path
            done_count += 1
            now_t = time.time()
            if now_t - last_prog_ts >= 4.0 or done_count == total_clips:
                last_prog_ts = now_t
                pct = 58 + int((done_count / float(total_clips)) * 30)
                _report_progress(job, pct, f"Renderizando clipes na GPU ({done_count}/{total_clips})...")

        if clip_tasks:
            try:
                mp_ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=clip_workers, mp_context=mp_ctx) as rpool:
                    rfuts = {rpool.submit(_render_clip_worker, t): t[0] for t in clip_tasks}
                    for fut in as_completed(rfuts):
                        c_idx, res_path = fut.result()
                        _track_clip_done(c_idx, res_path)
            except Exception as mp_err:
                print(f"[RunPod Warning] ProcessPool fallback para ThreadPool: {mp_err}")
                rendered_by_idx.clear()
                done_count = 0
                with ThreadPoolExecutor(max_workers=min(4, clip_workers)) as tpool:
                    rfuts = {tpool.submit(_render_clip_worker, t): t[0] for t in clip_tasks}
                    for fut in as_completed(rfuts):
                        c_idx, res_path = fut.result()
                        _track_clip_done(c_idx, res_path)

        for idx in range(len(clips)):
            segment_files.append(rendered_by_idx[idx])

        if not segment_files:
            return {"error": "Nenhum segmento de vídeo pôde ser gerado."}

        # 3. Concatenar segmentos visuais (concat -c copy instantâneo)
        _report_progress(job, 90, "Unindo segmentos do vídeo...")
        visual_video_path = os.path.join(work_dir, "visual_concat.mp4")
        concat_list_path = os.path.join(work_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        run_cmd([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy",
            visual_video_path
        ], "Concatenar segmentos visuais")

        # 3b. Aplicar overlays de personagem (e legendas Pillow Canvas-idênticas no mesmo passe se houver personagens)
        subs_already_burned = burn_subs_in_clips
        if character_overlays:
            _report_progress(job, 92, "Aplicando personagens e legendas na GPU...")
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

            for c_idx, ch in enumerate(character_overlays):
                c_url = ch.get("url", "")
                if c_idx not in char_files and c_url in url_to_local and os.path.exists(url_to_local[c_url]):
                    char_files[c_idx] = url_to_local[c_url]

            valid_chars = [(i, char_files[i], character_overlays[i]) for i in sorted(char_files.keys())]
            if valid_chars:
                char_out_path = os.path.join(work_dir, "visual_with_chars.mp4")
                if HAS_OPENCV:
                    composite_characters_opencv(
                        visual_video_path,
                        char_out_path,
                        valid_chars,
                        bg_img_cache,
                        width,
                        height,
                        fps,
                        v_enc_args,
                        sub_filter=sub_filter,
                        subtitle_items=subtitle_items,
                        subtitle_config=subtitle_config,
                    )
                    if has_any_subs:
                        subs_already_burned = True
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
                        out_lbl = "[vchars_raw]" if (n_idx == len(valid_chars) and sub_filter) else ("[vchars]" if n_idx == len(valid_chars) else f"[vc{n_idx}]")
                        fc_parts.append(f"[{n_idx}:v]scale=-1:{target_h}{scaled_lbl}")
                        fc_parts.append(f"{last_v}{scaled_lbl}overlay=x={x_expr}:y={y_expr}:enable='between(t,{c_start:.2f},{c_end:.2f})'{out_lbl}")
                        last_v = out_lbl

                    if sub_filter:
                        fc_parts.append(f"{last_v}{sub_filter}[vchars]")
                        last_v = "[vchars]"
                        subs_already_burned = True

                    c_cmd.extend([
                        "-filter_complex", ";".join(fc_parts),
                        "-map", last_v,
                        *v_enc_args,
                        "-an",
                        char_out_path
                    ])
                    run_cmd(c_cmd, "Aplicar personagens PIP e legendas")
                visual_video_path = char_out_path

        # 4. Fallback de legendas apenas caso existam legendas e não tenham sido queimadas nas etapas anteriores
        subtitled_video_path = visual_video_path
        if has_any_subs and not subs_already_burned:
            _report_progress(job, 94, "Queimando legendas no vídeo...")
            subtitled_video_path = os.path.join(work_dir, "visual_subtitled.mp4")
            if use_pil_subs:
                composite_characters_opencv(
                    visual_video_path,
                    subtitled_video_path,
                    [],
                    {},
                    width,
                    height,
                    fps,
                    v_enc_args,
                    sub_filter=None,
                    subtitle_items=subtitle_items,
                    subtitle_config=subtitle_config,
                )
            elif sub_filter:
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", visual_video_path,
                    "-vf", sub_filter,
                    *v_enc_args,
                    "-an",
                    subtitled_video_path
                ], "Queimar legendas no vídeo")

        _report_progress(job, 96, "Mixando áudios e finalizando MP4...")

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
