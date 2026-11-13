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
import shutil
import base64
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import runpod

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
        "cinematic": "eq=contrast=1.12:saturation=0.88:brightness=-0.02,colorbalance=rs=-0.03:bs=0.05",
        "vintage": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,eq=contrast=0.95:brightness=0.03",
        "bw": "hue=s=0,eq=contrast=1.1",
        "noir": "hue=s=0,eq=contrast=1.45:brightness=-0.06",
        "neon": "eq=saturation=1.6:contrast=1.25",
        "pastel": "eq=saturation=0.65:brightness=0.08:contrast=0.92",
        "warm": "colorbalance=rs=0.08:gs=0.02:bs=-0.06,eq=saturation=1.15",
        "high_contrast": "eq=contrast=1.35:saturation=1.1",
        "dream": "gblur=sigma=0.6,eq=brightness=0.06:saturation=1.12",
        "sepia_old": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,eq=contrast=1.08",
        "technicolor": "eq=saturation=1.45:contrast=1.2,colorbalance=rs=0.05:gs=-0.02:bs=0.04",
    }
    return mapping.get(vf, "")


def get_overlay_filter_expr(overlay: str, width: int, height: int) -> str:
    ov = (overlay or "none").lower()
    if ov == "vignette":
        return "vignette=PI/4"
    elif ov == "letterbox":
        bar_h = max(20, int(height * 0.11))
        return f"drawbox=x=0:y=0:w={width}:h={bar_h}:color=black:t=fill,drawbox=x=0:y={height - bar_h}:w={width}:h={bar_h}:color=black:t=fill"
    elif ov in ("film_grain", "vhs"):
        return "noise=alls=14:allf=t+u"
    elif ov == "scanlines":
        return "drawgrid=w=0:h=4:t=1:c=black@0.35"
    elif ov == "blood_screen":
        return "vignette=PI/3,colorbalance=rs=0.14:gs=-0.04:bs=-0.04"
    return ""


def build_ken_burns_filter(movement: str, duration: float, fps: int, width: int, height: int) -> str:
    frames = max(2, int(math.ceil(duration * fps)))
    mov = (movement or "none").lower()
    pre_w = min(int(width * 1.4), 2688)
    pre_h = min(int(height * 1.4), 1512) if width >= height else min(int(height * 1.4), 2688)
    pre = f"scale={pre_w}:{pre_h}:force_original_aspect_ratio=increase,crop={pre_w}:{pre_h},setsar=1"
    if mov == "zoom_in":
        zp = f"zoompan=z='min(zoom+0.0012,1.25)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "zoom_out":
        zp = f"zoompan=z='if(lte(on,1),1.25,max(1.0,zoom-0.0012))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_left":
        zp = f"zoompan=z='1.18':x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_right":
        zp = f"zoompan=z='1.18':x='(iw-iw/zoom)*(on/{frames})':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_up":
        zp = f"zoompan=z='1.18':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{frames})':d={frames}:s={width}x{height}:fps={fps}"
    elif mov == "pan_down":
        zp = f"zoompan=z='1.18':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(on/{frames})':d={frames}:s={width}x{height}:fps={fps}"
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


def upload_to_temp_cloud(file_path: str) -> str | None:
    # Fallback 1: tmpfiles.org
    try:
        with open(file_path, "rb") as f:
            r = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": ("video_renderizado.mp4", f, "video/mp4")},
                timeout=300,
            )
        if r.ok:
            url = r.json().get("data", {}).get("url", "")
            if url:
                return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    except Exception as e:
        print(f"[RunPod] tmpfiles fallback falhou: {e}")

    # Fallback 2: uguu.se
    try:
        with open(file_path, "rb") as f:
            r = requests.post(
                "https://uguu.se/upload.php",
                files={"files[]": ("video_renderizado.mp4", f, "video/mp4")},
                timeout=300,
            )
        if r.ok:
            files = r.json().get("files", [])
            if files and files[0].get("url"):
                return files[0]["url"]
    except Exception as e:
        print(f"[RunPod] uguu fallback falhou: {e}")

    # Fallback 3: litterbox.catbox.moe
    try:
        with open(file_path, "rb") as f:
            r = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                data={"reqtype": "fileupload", "time": "72h"},
                files={"fileToUpload": ("video_renderizado.mp4", f, "video/mp4")},
                timeout=300,
            )
        if r.ok and r.text.strip().startswith("http"):
            return r.text.strip()
    except Exception as e:
        print(f"[RunPod] litterbox fallback falhou: {e}")

    return None


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

            vf_parts = []
            if clip.get("type") == "video":
                speed = float(clip.get("playbackSpeed", 1.0) or 1.0)
                if abs(speed - 1.0) > 0.05:
                    vf_parts.append(f"setpts={1.0/speed:.4f}*PTS")
                vf_parts.append(f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,fps={fps}")
                input_args = ["-stream_loop", "-1", "-i", src_path, "-t", f"{dur:.3f}"]
            else:
                kb = build_ken_burns_filter(clip.get("movement", "none"), dur, fps, width, height)
                vf_parts.append(kb)
                input_args = ["-loop", "1", "-i", src_path, "-t", f"{dur:.3f}"]

            vis_filter = get_visual_filter_expr(clip.get("visualFilter", "none"))
            if vis_filter:
                vf_parts.append(vis_filter)

            ov_filter = get_overlay_filter_expr(clip.get("overlayEffect", "none"), width, height)
            if ov_filter:
                vf_parts.append(ov_filter)

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

        # 3b. Aplicar overlays de personagem (PIP) se configurados
        character_overlays = inp.get("characterOverlays", [])
        if character_overlays:
            char_files = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                cfuts = {}
                for c_idx, ch in enumerate(character_overlays):
                    c_url = ch.get("url", "")
                    if c_url:
                        dest = os.path.join(media_dir, f"char_{c_idx}.png")
                        cfuts[pool.submit(download_file, c_url, dest)] = (c_idx, dest)
                for fut in as_completed(cfuts):
                    c_idx, dest = cfuts[fut]
                    try:
                        fut.result()
                        char_files[c_idx] = dest
                    except Exception as cerr:
                        print(f"[RunPod Warning] Falha ao baixar personagem #{c_idx}: {cerr}")

            valid_chars = [(i, char_files[i], character_overlays[i]) for i in sorted(char_files.keys())]
            if valid_chars:
                char_out_path = os.path.join(work_dir, "visual_with_chars.mp4")
                c_cmd = ["ffmpeg", "-y", "-i", visual_video_path]
                fc_parts = []
                last_v = "[0:v]"
                for n_idx, (_, c_path, ch_meta) in enumerate(valid_chars, start=1):
                    c_cmd.extend(["-i", c_path])
                    c_start = float(ch_meta.get("startTime", 0))
                    c_end = c_start + float(ch_meta.get("duration", 5))
                    c_scale = max(0.15, min(1.5, float(ch_meta.get("scale", 60)) / 100.0))
                    target_h = int(height * 0.78 * c_scale)
                    pos = ch_meta.get("position", "center")
                    if pos in ("left", "bottom-left"):
                        x_expr = f"{int(width * 0.03)}"
                    elif pos in ("right", "bottom-right"):
                        x_expr = f"W-w-{int(width * 0.03)}"
                    else:
                        x_expr = "(W-w)/2"
                    y_expr = "H-h" if str(pos).startswith("bottom") else f"H-h-{int(height * 0.04)}"

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

        if not video_url:
            video_url = upload_to_temp_cloud(final_output_path)
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
        elif size_mb <= 8.0:
            with open(final_output_path, "rb") as f:
                result["videoBase64"] = base64.b64encode(f.read()).decode("utf-8")
            result["deliveryMethod"] = "base64"
        else:
            return {"error": "Vídeo gerado com sucesso, mas não foi possível fazer upload (>8MB). Configure S3/R2 ou use um app com URL pública."}

        return result

    except Exception as e:
        print(f"[RunPod ERROR] {e}")
        return {"error": str(e)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
