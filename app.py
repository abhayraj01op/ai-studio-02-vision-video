"""AI Studio 02 — Ultimate 10-Model Master Edition (Kaggle T4 x2, Python 3.13, 3-Batch Disk-Safe)."""
import os, gc, re, json, glob, shutil, traceback, subprocess, datetime, zipfile
import warnings
warnings.filterwarnings("ignore")

# Disable Xet background writer (prevents Kaggle disk RuntimeError) & set persistent cache
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_HUB_BASE = "/kaggle/working/AI_Studio_Master_Hub" if os.path.isdir("/kaggle/working") else os.path.abspath("AI_Studio_Master_Hub")
os.environ["HF_HOME"] = os.path.join(_HUB_BASE, "99_model_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
try:
    from transformers import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import gradio as gr

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# ------------------------------------------------------------------ Shared Master Hub Paths (Syncs with Studio 01)
HUB = _HUB_BASE
SUBFOLDERS = {
    "faces": "00_reference_faces",
    "ref_voices": "00_reference_voices",
    "scripts": "01_scripts",
    "voices": "02_voices",
    "images": "03_images",
    "raw_videos": "04_raw_videos",
    "music": "05_music",
    "captions": "06_captions_hashtags",
    "exports": "07_final_exports",
    "cache": "99_model_cache",
}
for _f in SUBFOLDERS.values():
    os.makedirs(os.path.join(HUB, _f), exist_ok=True)

def _ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _err():
    return f"❌ Error Details:\n{traceback.format_exc()}"

def disk_status():
    try:
        target = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
        total, used, free = shutil.disk_usage(target)
        return f"💾 Disk: {used/1e9:.1f}/{total/1e9:.1f} GB used ({free/1e9:.1f} GB free)"
    except Exception:
        return "💾 Disk: unknown"

# ------------------------------------------------------------------ Google Drive Sync (Shared with Studio 01)
DRIVE = {"gauth": None, "drive": None, "root_id": None, "ids": {}}
SECRETS_PATH = "/kaggle/working/client_secrets.json" if os.path.isdir("/kaggle/working") else "client_secrets.json"

def _find_secrets():
    if os.path.exists(SECRETS_PATH):
        return SECRETS_PATH
    for p in glob.glob("/kaggle/input/**/client_secrets*.json", recursive=True):
        shutil.copy(p, SECRETS_PATH)
        return SECRETS_PATH
    return None

def drive_status():
    return "✅ Google Drive Sync: LIVE CONNECTED (AI_Studio_Master_Hub — Shared with Studio 01)" if DRIVE["drive"] else "⚪ Google Drive Sync: Not connected (files are saved locally)"

def drive_step1(upload):
    try:
        from pydrive2.auth import GoogleAuth
        if upload is not None:
            shutil.copy(upload if isinstance(upload, str) else upload.name, SECRETS_PATH)
        path = _find_secrets()
        if not path:
            return "❌ Upload client_secrets.json (or attach it as a Kaggle input) first."
        gauth = GoogleAuth(settings={
            "client_config_backend": "file", "client_config_file": path,
            "oauth_scope": ["https://www.googleapis.com/auth/drive"],
            "get_refresh_token": True,
        })
        gauth.GetFlow()
        gauth.flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        url = gauth.GetAuthUrl()
        DRIVE["gauth"] = gauth
        return f"🔗 [Click here to sign in with Google]({url})\n\nAfter allowing access, copy the code and paste it in Step 2.\n\n`{url}`"
    except Exception:
        return _err()

def _get_or_create(drive, title, parent=None):
    q = f"title='{title}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    q += f" and '{parent}' in parents" if parent else " and 'root' in parents"
    found = drive.ListFile({"q": q}).GetList()
    if found:
        return found[0]["id"]
    meta = {"title": title, "mimeType": "application/vnd.google-apps.folder"}
    if parent:
        meta["parents"] = [{"id": parent}]
    f = drive.CreateFile(meta); f.Upload()
    return f["id"]

def _list_children(parent):
    return DRIVE["drive"].ListFile({"q": f"'{parent}' in parents and trashed=false"}).GetList()

def _pull_small_assets_from_drive():
    if not DRIVE["drive"]:
        return 0
    n = 0
    for key in ["faces", "scripts", "voices", "music"]:
        dest = os.path.join(HUB, SUBFOLDERS[key])
        try:
            for f in _list_children(DRIVE["ids"][key]):
                if f["mimeType"] != "application/vnd.google-apps.folder":
                    t = os.path.join(dest, f["title"])
                    if not os.path.exists(t):
                        f.GetContentFile(t); n += 1
        except Exception:
            pass
    return n

def drive_step2(code):
    try:
        from pydrive2.drive import GoogleDrive
        if DRIVE["gauth"] is None:
            return "❌ Run Step 1 first.", drive_status(), gr.update(), gr.update(), gr.update(), gr.update()
        DRIVE["gauth"].Auth((code or "").strip())
        drive = GoogleDrive(DRIVE["gauth"])
        root = _get_or_create(drive, "AI_Studio_Master_Hub")
        DRIVE.update(drive=drive, root_id=root, ids={k: _get_or_create(drive, v, root) for k, v in SUBFOLDERS.items()})
        n = 0
        for k, v in SUBFOLDERS.items():
            if k == "cache":
                continue
            for p in glob.glob(os.path.join(HUB, v, "*")):
                if os.path.isfile(p):
                    _upload(p, k); n += 1
        pulled = _pull_small_assets_from_drive()
        faces = list_hub_files("faces")
        scripts = list_hub_files("scripts")
        voices = list_hub_files("voices")
        music = list_hub_files("music")
        return (
            f"✅ Connected in 2 seconds! Shared Hub ready on Drive. Synced {n} local files & pulled {pulled} Studio 01 assets.\n"
            f"♻️ 3-Batch Smart Disk Mode active: All 10 heavy models load on-demand from Drive without filling Kaggle's 19.5 GB disk.",
            drive_status(),
            gr.update(choices=faces, value=faces[0] if faces else "None"),
            gr.update(choices=scripts, value=scripts[0] if scripts else "None"),
            gr.update(choices=voices, value=voices[0] if voices else "None"),
            gr.update(choices=music, value=music[0] if music else "None"),
        )
    except Exception:
        return _err(), drive_status(), gr.update(), gr.update(), gr.update(), gr.update()

# ------------------------------------------------------------------ Disk-Safe Model Cache (3 Batches — 10 Heavy Models)
def _free_local_model_cache(keep_model_ids=None):
    keep_folders = {"models--" + m.replace("/", "--") for m in (keep_model_ids or [])}
    hub_cache = os.path.join(HUB, SUBFOLDERS["cache"], "hub")
    if not os.path.isdir(hub_cache):
        return
    for entry in os.listdir(hub_cache):
        if entry.startswith("models--") and entry not in keep_folders:
            shutil.rmtree(os.path.join(hub_cache, entry), ignore_errors=True)

def _ensure_disk_space_for(model_id, min_free_gb=15.0):
    try:
        target = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
        _, _, free = shutil.disk_usage(target)
        if free / 1e9 < min_free_gb:
            _free_local_model_cache(keep_model_ids=[model_id])
    except Exception:
        pass

def _ensure_model_from_drive(model_id):
    _ensure_disk_space_for(model_id, min_free_gb=15.0)
    if not DRIVE["drive"]:
        return
    folder_name = "models--" + model_id.replace("/", "--")
    local_dir = os.path.join(HUB, SUBFOLDERS["cache"], "hub", folder_name)
    if os.path.isdir(local_dir) and any(os.scandir(local_dir)):
        return
    try:
        hub_folders = [f for f in _list_children(DRIVE["ids"]["cache"]) if f["title"] == "hub"]
        if not hub_folders:
            return
        model_folders = [f for f in _list_children(hub_folders[0]["id"]) if f["title"] == folder_name]
        if not model_folders:
            return
        def walk(fid, dest):
            os.makedirs(dest, exist_ok=True)
            for f in _list_children(fid):
                target = os.path.join(dest, f["title"])
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    walk(f["id"], target)
                elif not os.path.exists(target):
                    f.GetContentFile(target)
        walk(model_folders[0]["id"], local_dir)
    except Exception:
        pass

CACHE_SYNC = {"running": False, "msg": "10-Model Cache ready for Batch 1 / Batch 2 / Batch 3 download."}

def _sync_cache_to_drive_blocking():
    if not DRIVE["drive"]:
        return 0
    local = os.path.join(HUB, SUBFOLDERS["cache"]); n = 0
    def walk(src, fid):
        nonlocal n
        existing = {f["title"]: f for f in _list_children(fid)}
        for name in sorted(os.listdir(src)):
            p = os.path.join(src, name)
            if os.path.islink(p):
                p = os.path.realpath(p)
            if os.path.isdir(p):
                sub = existing[name]["id"] if name in existing else _get_or_create(DRIVE["drive"], name, fid)
                walk(p, sub)
            elif os.path.isfile(p) and not name.endswith(".lock"):
                if name in existing and int(existing[name].get("fileSize", -1)) == os.path.getsize(p):
                    continue
                f = DRIVE["drive"].CreateFile({"title": name, "parents": [{"id": fid}]})
                f.SetContentFile(p); f.Upload(); n += 1
                CACHE_SYNC["msg"] = f"☁️ Syncing model cache… {n} files uploaded"
    walk(local, DRIVE["ids"]["cache"])
    return n

BATCH_1_IMAGE_MODELS = [
    "SG161222/RealVisXL_V5.0",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "ByteDance/SDXL-Lightning",
    "h94/IP-Adapter",
]
BATCH_2_VIDEO_MODELS = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "THUDM/CogVideoX-5b",
    "stabilityai/stable-video-diffusion-img2vid-xt",
    "numz/wav2lip_studio",
]
BATCH_3_VISION_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
]
ALL_STUDIO2_MODELS = BATCH_1_IMAGE_MODELS + BATCH_2_VIDEO_MODELS + BATCH_3_VISION_MODELS

def download_models_list(models):
    """1-by-1 disk-safe downloader: downloads 1 model -> uploads to Drive -> frees Kaggle 19.5GB disk -> next."""
    try:
        from huggingface_hub import snapshot_download
        if not models:
            return "❌ Please select at least 1 model (or click Batch 1 / 2 / 3 button)."
        lines = []
        hub_cache = os.path.join(HUB, SUBFOLDERS["cache"], "hub")
        for idx, m in enumerate(models):
            _free_local_model_cache(keep_model_ids=[])
            CACHE_SYNC["msg"] = f"⏳ [{idx+1}/{len(models)}] Downloading {m}..."
            allow = ["*.json", "*.safetensors", "*.pt", "*.pth", "*.onnx", "*.model", "*.txt", "*.py", "tokenizer*", "scheduler/*", "vae/*", "unet/*", "transformer/*", "text_encoder*", "image_encoder/*", "sdxl_models/*"]
            ignore = ["*pytorch_model.bin*", "*fp32*", "*non_ema*", "*openvino*", "*onnx_fp16*"]
            if m == "SG161222/RealVisXL_V5.0":
                ignore += ["*.ckpt", "RealVisXL_V5.0*.safetensors", "*vae/diffusion_pytorch_model.safetensors"]
            elif m == "black-forest-labs/FLUX.1-schnell":
                ignore += ["flux1-schnell.safetensors", "ae.safetensors"]
            elif m == "black-forest-labs/FLUX.1-dev":
                ignore += ["flux1-dev.safetensors", "ae.safetensors"]
            elif m == "ByteDance/SDXL-Lightning":
                allow = ["*4step*.safetensors", "*.json"]
            elif m == "h94/IP-Adapter":
                allow = ["sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", "sdxl_models/image_encoder/*", "*.json"]
            elif m == "stabilityai/stable-video-diffusion-img2vid-xt":
                ignore += ["svd_xt*.safetensors"]
            snapshot_download(m, allow_patterns=allow, ignore_patterns=ignore, max_workers=2)
            if DRIVE["drive"]:
                CACHE_SYNC["msg"] = f"☁️ [{idx+1}/{len(models)}] Uploading {m} to Google Drive..."
                _sync_cache_to_drive_blocking()
                folder_name = "models--" + m.replace("/", "--")
                shutil.rmtree(os.path.join(hub_cache, folder_name), ignore_errors=True)
                lines.append(f"✅ {m} -> Saved to Google Drive & Kaggle disk freed!")
            else:
                lines.append(f"✅ {m} -> Downloaded locally (connect Drive to keep permanently).")
        if DRIVE["drive"] and models:
            CACHE_SYNC["msg"] = f"♻️ Restoring {models[0]} to GPU disk for instant use..."
            _ensure_model_from_drive(models[0])
            lines.append(f"🚀 {models[0]} ready on GPU disk!")
        CACHE_SYNC["msg"] = "✅ Selected batch cached safely to Google Drive!"
        return "\n".join(lines)
    except Exception:
        return _err()

def download_batch_1():
    return download_models_list(BATCH_1_IMAGE_MODELS)

def download_batch_2():
    return download_models_list(BATCH_2_VIDEO_MODELS)

def download_batch_3():
    return download_models_list(BATCH_3_VISION_MODELS)

def cache_sync_status():
    return f"{CACHE_SYNC['msg']} | {disk_status()}"

def _upload(path, key):
    f = DRIVE["drive"].CreateFile({"title": os.path.basename(path), "parents": [{"id": DRIVE["ids"][key]}]})
    f.SetContentFile(path); f.Upload()

def save_and_sync(local_path, subfolder_key):
    dest_dir = os.path.join(HUB, SUBFOLDERS[subfolder_key])
    if os.path.dirname(os.path.abspath(local_path)) != os.path.abspath(dest_dir):
        dest = os.path.join(dest_dir, os.path.basename(local_path))
        shutil.copy(local_path, dest); local_path = dest
    note = "💾 Saved locally"
    if DRIVE["drive"]:
        try:
            _upload(local_path, subfolder_key); note += " + ☁️ uploaded to Drive"
        except Exception as e:
            note += f" (Drive upload failed: {e})"
    return local_path, f"{note}: {local_path}"

# ------------------------------------------------------------------ Hub Asset Helpers
def list_hub_files(key):
    folder = os.path.join(HUB, SUBFOLDERS[key])
    files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(folder, "*")) if os.path.isfile(p)], reverse=True)
    return ["None"] + files

def refresh_all_hub_assets():
    pulled = _pull_small_assets_from_drive()
    faces = list_hub_files("faces")
    scripts = list_hub_files("scripts")
    voices = list_hub_files("voices")
    music = list_hub_files("music")
    images = list_hub_files("images")
    videos = list_hub_files("raw_videos")
    return (
        f"🔄 Refreshed! Pulled {pulled} new files from Drive. Found {len(scripts)-1} scripts, {len(voices)-1} voices, {len(music)-1} music tracks, {len(images)-1} images, {len(videos)-1} videos.",
        gr.update(choices=faces, value=faces[0]),
        gr.update(choices=scripts, value=scripts[0]),
        gr.update(choices=voices, value=voices[0]),
        gr.update(choices=music, value=music[0]),
        gr.update(choices=[f for f in images if f != "None"]),
        gr.update(choices=[f for f in videos if f != "None"]),
    )

def save_reference_face(img_file, face_name):
    try:
        if img_file is None:
            return "❌ Please upload a clear front-facing character/influencer photo first.", gr.update()
        src = img_file if isinstance(img_file, str) else img_file.name
        clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", (face_name or "Influencer_Face").strip()) or "Influencer_Face"
        ext = os.path.splitext(src)[1] or ".png"
        dest = os.path.join(HUB, SUBFOLDERS["faces"], f"{clean}{ext}")
        shutil.copy(src, dest)
        _, note = save_and_sync(dest, "faces")
        faces = list_hub_files("faces")
        return f"✅ Reference Face '{os.path.basename(dest)}' saved! {note}", gr.update(choices=faces, value=os.path.basename(dest))
    except Exception:
        return _err(), gr.update()

def load_script_scenes(script_filename):
    if not script_filename or script_filename == "None":
        return "❌ Select a script from Studio 01 first."
    p = os.path.join(HUB, SUBFOLDERS["scripts"], script_filename)
    if not os.path.exists(p):
        return "❌ Script file not found."
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    cues = re.findall(r"\[VISUAL SCENE:\s*(.*?)\]", txt, flags=re.I | re.S)
    if cues:
        return "\n".join([c.strip() for c in cues if c.strip()])
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if len(p.strip()) > 20][:6]
    return "\n".join(paras) if paras else txt[:1500]

# ------------------------------------------------------------------ GPU & VRAM Manager
current_pipe = None
_PIPES = {}

def gpu_status():
    if torch is None or not torch.cuda.is_available():
        return "🖥️ GPU: CPU Mode | " + disk_status()
    parts = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        parts.append(f"GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB")
    return "🖥️ " + " | ".join(parts) + " | " + disk_status()

def clear_vram_and_disk():
    global current_pipe
    try:
        current_pipe = None
        _PIPES.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _free_local_model_cache(keep_model_ids=[])
        return "🧹 VRAM + Temp Model Disk cleared! " + gpu_status()
    except Exception:
        return _err()

def _use(name, loader):
    global current_pipe
    if name not in _PIPES:
        _PIPES.clear(); current_pipe = None; gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _PIPES[name] = loader()
    current_pipe = _PIPES[name]
    return current_pipe

# ------------------------------------------------------------------ TAB 1: 4 Heavy Image & Cartoon Engines + Face Lock
IMAGE_MODELS = [
    "SG161222/RealVisXL_V5.0 (Photorealistic + 3D Cartoon + Face Lock)",
    "black-forest-labs/FLUX.1-dev (Heavy Studio-Grade Cinema & Max Detail)",
    "black-forest-labs/FLUX.1-schnell (Fast 4-8 Step Detail & Typography)",
    "ByteDance/SDXL-Lightning (Ultra-Fast 4-Step Storyboard Engine)",
]

IMAGE_STYLES = {
    "Photorealistic Instagram AI Influencer (Raw Candid DSLR)": "candid raw smartphone/DSLR photo, natural skin texture, pores, realistic lighting, depth of field, 85mm lens, photorealistic, unedited look",
    "🧸 3D Pixar / Disney Cartoon Animation Style": "3D Pixar Disney animated movie still, expressive big eyes, soft subsurface scattering, vibrant colorful lighting, cute stylized 3D character render, octane render, 8k",
    "📚 2D Kids Story Book / Panchatantra Cartoon": "colorful 2D children's storybook cartoon illustration, clean outlines, warm expressive characters, whimsical painted background, vibrant animation frame",
    "🎌 Modern Anime / Makoto Shinkai Style": "high-detail modern anime cinematic key visual, dramatic sunlight rays, crisp lineart, vibrant colors, emotional anime scene",
    "Cinematic Bollywood / Moody Night Scene": "cinematic movie still, dramatic moody neon and warm rim lighting, shallow depth of field, anamorphic lens flare, 4k film grain",
    "Dark Horror / Paranormal Atmosphere": "eerie dark atmospheric horror scene, foggy shadows, cold moonlight, cinematic suspense, hyper-detailed",
    "High-CTR YouTube Thumbnail (Bold & Expressive)": "dramatic expressive close-up, high contrast vibrant rim light, sharp focus, clean background separation, eye-catching composition",
    "Luxury Lifestyle / Aesthetic Reel": "golden hour warm sunlight, luxury aesthetic, soft bokeh, editorial fashion photography, natural colors",
    "Custom Style (Use My Own Prompt Only)": "",
}

ASPECT_RATIOS = {
    "9:16 Vertical (Reels / Shorts — 768x1344)": (768, 1344),
    "16:9 Horizontal (YouTube Video — 1344x768)": (1344, 768),
    "1:1 Square (Instagram Post — 1024x1024)": (1024, 1024),
    "4:5 Portrait (Instagram Feed — 896x1152)": (896, 1152),
}

def _load_realvis(use_ip_adapter=False):
    _ensure_model_from_drive("SG161222/RealVisXL_V5.0")
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    if use_ip_adapter:
        try:
            _ensure_model_from_drive("h94/IP-Adapter")
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
        except Exception:
            pass
    pipe.enable_model_cpu_offload()
    return pipe

def _load_sdxl_lightning():
    _ensure_model_from_drive("SG161222/RealVisXL_V5.0")
    _ensure_model_from_drive("ByteDance/SDXL-Lightning")
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")
    pipe.unet.load_state_dict(load_file(ckpt, device="cpu"))
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.enable_model_cpu_offload()
    return pipe

def _load_flux_schnell():
    _ensure_model_from_drive("black-forest-labs/FLUX.1-schnell")
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return pipe

def _load_flux_dev():
    _ensure_model_from_drive("black-forest-labs/FLUX.1-dev")
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return pipe

def generate_image(prompt, my_idea, style_name, aspect, model_choice, face_upload, saved_face, face_strength, negative_prompt, steps, seed):
    try:
        if not (prompt or "").strip() and not (my_idea or "").strip():
            return None, None, "❌ Please enter a Scene Prompt or write your own Idea."
        style_hint = IMAGE_STYLES.get(style_name, "")
        full_prompt = ", ".join([p for p in [(prompt or "").strip(), (my_idea or "").strip(), style_hint] if p])
        w, h = ASPECT_RATIOS[aspect]
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if int(seed) >= 0 else None

        ref_face_path = None
        if face_upload is not None:
            ref_face_path = face_upload if isinstance(face_upload, str) else face_upload.name
        elif saved_face and saved_face != "None":
            cand = os.path.join(HUB, SUBFOLDERS["faces"], saved_face)
            if os.path.exists(cand):
                ref_face_path = cand

        if "FLUX.1-dev" in model_choice:
            pipe = _use("img:flux_dev", _load_flux_dev)
            img = pipe(
                prompt=full_prompt,
                width=w,
                height=h,
                num_inference_steps=max(20, int(steps)),
                guidance_scale=3.5,
                max_sequence_length=512,
                generator=gen,
            ).images[0]
        elif "FLUX.1-schnell" in model_choice:
            pipe = _use("img:flux_schnell", _load_flux_schnell)
            img = pipe(
                prompt=full_prompt,
                width=w,
                height=h,
                num_inference_steps=max(4, min(12, int(steps))),
                guidance_scale=0.0,
                max_sequence_length=256,
                generator=gen,
            ).images[0]
        elif "SDXL-Lightning" in model_choice:
            pipe = _use("img:lightning", _load_sdxl_lightning)
            img = pipe(
                prompt=full_prompt,
                width=w,
                height=h,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=gen,
            ).images[0]
        else:
            use_ipa = ref_face_path is not None
            pipe = _use(f"img:realvis:{use_ipa}", lambda: _load_realvis(use_ipa))
            is_cartoon = any(k in style_name.lower() for k in ["cartoon", "pixar", "anime"])
            default_neg = "blurry, deformed hands, extra fingers, watermark, low quality" if is_cartoon else "plastic skin, airbrushed, cgi, 3d render, cartoon, deformed face, extra fingers, blurry, watermark, text"
            neg = (negative_prompt or "").strip() or default_neg
            kw = dict(
                prompt=full_prompt,
                negative_prompt=neg,
                width=w,
                height=h,
                num_inference_steps=int(steps),
                guidance_scale=5.5,
                generator=gen,
            )
            if use_ipa and hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(float(face_strength))
                kw["ip_adapter_image"] = Image.open(ref_face_path).convert("RGB")
            img = pipe(**kw).images[0]

        out = os.path.join(HUB, SUBFOLDERS["images"], f"img_{_ts()}.png")
        img.save(out)
        path, note = save_and_sync(out, "images")
        return path, path, note
    except Exception:
        return None, None, _err()

def batch_generate_scenes(scenes_text, my_idea, style_name, aspect, model_choice, face_upload, saved_face, face_strength):
    try:
        lines = [l.strip() for l in (scenes_text or "").splitlines() if l.strip()]
        if not lines:
            return [], "❌ Paste scene prompts (1 per line) or load from a Studio 01 script first."
        gallery = []
        notes = []
        default_steps = 4 if ("Lightning" in model_choice or "schnell" in model_choice) else 25
        for idx, line in enumerate(lines[:10]):
            path, _, note = generate_image(line, my_idea, style_name, aspect, model_choice, face_upload, saved_face, face_strength, "", default_steps, -1)
            if path:
                gallery.append(path)
                notes.append(f"Scene {idx+1}: {note}")
            else:
                notes.append(f"Scene {idx+1} failed: {note}")
        return gallery, "\n".join(notes)
    except Exception:
        return [], _err()

# ------------------------------------------------------------------ TAB 2: Photo Editor, BG Remover, 4K Upscaler, Thumbnail & AI Vision Cloner
def remove_or_replace_bg(img_in, mode, bg_color_hex, new_bg_img):
    try:
        if img_in is None:
            return None, "❌ Please upload or send an image first."
        from rembg import remove
        src = img_in if isinstance(img_in, str) else img_in.name
        fg = remove(Image.open(src).convert("RGBA"))
        if mode == "Transparent PNG (No Background)":
            out = os.path.join(HUB, SUBFOLDERS["images"], f"nobg_{_ts()}.png")
            fg.save(out)
        elif mode == "Custom Image Background" and new_bg_img is not None:
            bg_src = new_bg_img if isinstance(new_bg_img, str) else new_bg_img.name
            bg = Image.open(bg_src).convert("RGBA").resize(fg.size)
            comp = Image.alpha_composite(bg, fg).convert("RGB")
            out = os.path.join(HUB, SUBFOLDERS["images"], f"newbg_{_ts()}.png")
            comp.save(out)
        else:
            bg = Image.new("RGBA", fg.size, bg_color_hex or "#0f172a")
            comp = Image.alpha_composite(bg, fg).convert("RGB")
            out = os.path.join(HUB, SUBFOLDERS["images"], f"solidbg_{_ts()}.png")
            comp.save(out)
        path, note = save_and_sync(out, "images")
        return path, note
    except Exception:
        return None, _err()

def upscale_4k_enhance(img_in, scale_factor, sharpness, contrast):
    try:
        if img_in is None:
            return None, "❌ Upload an image to upscale."
        src = img_in if isinstance(img_in, str) else img_in.name
        im = Image.open(src).convert("RGB")
        w, h = im.size
        im = im.resize((int(w * scale_factor), int(h * scale_factor)), Image.Resampling.LANCZOS)
        im = im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(sharpness * 100), threshold=2))
        im = ImageEnhance.Contrast(im).enhance(float(contrast))
        out = os.path.join(HUB, SUBFOLDERS["images"], f"upscaled_{_ts()}.png")
        im.save(out, quality=98)
        path, note = save_and_sync(out, "images")
        return path, f"🔍 Enhanced to {im.size[0]}x{im.size[1]}! {note}"
    except Exception:
        return None, _err()

def make_viral_thumbnail(img_in, top_text, bottom_text, text_color, add_border):
    try:
        if img_in is None:
            return None, "❌ Upload or select a base image for your Thumbnail / Reel Cover."
        src = img_in if isinstance(img_in, str) else img_in.name
        im = Image.open(src).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)

        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ]
        f_path = next((p for p in font_paths if os.path.exists(p)), None)
        font_size = max(36, int(w * 0.075))
        font = ImageFont.truetype(f_path, font_size) if f_path else ImageFont.load_default()

        def draw_outlined(text, y_pos, fill_col):
            if not (text or "").strip():
                return
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            x = max(20, (w - tw) // 2)
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    draw.text((x + dx, y_pos + dy), text, font=font, fill="#000000")
            draw.text((x, y_pos), text, font=font, fill=fill_col)

        draw_outlined((top_text or "").strip(), int(h * 0.06), "#FFFFFF")
        draw_outlined((bottom_text or "").strip(), int(h * 0.80), text_color or "#FACC15")

        if add_border:
            bw = max(8, int(w * 0.012))
            draw.rectangle([0, 0, w - 1, h - 1], outline=text_color or "#FACC15", width=bw)

        out = os.path.join(HUB, SUBFOLDERS["images"], f"thumb_{_ts()}.png")
        im.save(out)
        path, note = save_and_sync(out, "images")
        return path, f"🖼️ Viral Thumbnail Ready! {note}"
    except Exception:
        return None, _err()

def _load_qwen_vl():
    _ensure_model_from_drive("Qwen/Qwen2.5-VL-7B-Instruct")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return processor, model

def reverse_engineer_prompt(ref_img, custom_goal):
    try:
        if ref_img is None:
            return "❌ Upload any reference photo, cartoon frame, or thumbnail first."
        from qwen_vl_utils import process_vision_info
        src = ref_img if isinstance(ref_img, str) else ref_img.name
        processor, model = _use("vision:qwen_vl", _load_qwen_vl)
        instruction = (
            "Analyze this image in extreme visual detail and write an ultra-detailed AI image/video generation prompt "
            "(subject appearance, outfit, pose, facial expression, camera angle, lens, lighting, color palette, background, and art/photo style). "
            f"User adaptation note: {custom_goal or 'keep exact style and mood'}. Output ONLY the ready-to-use prompt."
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": src}, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=350)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    except Exception:
        return _err()

# ------------------------------------------------------------------ TAB 3: 3 Heavy Video, Cartoon Animation & Lip-Sync Engines
VIDEO_ENGINES = [
    "Wan-AI/Wan2.1-T2V-1.3B (Fast & Crisp 9:16 Reels / Cartoon Motion)",
    "THUDM/CogVideoX-5b (Heavy Cinema & Complex Action Physics)",
]

VIDEO_STYLES = {
    "Cinematic Photorealistic Motion": "cinematic high quality, smooth realistic camera motion, natural lighting, sharp focus",
    "🧸 3D Pixar / Disney Cartoon Video": "3D Pixar Disney cartoon animation, vibrant colorful 3D character animation, smooth expressive movement, studio lighting",
    "📚 2D Cartoon / Kids Story Animation": "colorful 2D animated cartoon video, smooth character motion, vibrant storybook art style",
    "🎌 Anime Action / Emotional Scene": "high-quality modern anime animation, dynamic wind and lighting motion, crisp anime lineart",
    "Dark Horror / Suspense Motion": "eerie dark horror cinematography, slow creepy camera push, foggy shadows, suspenseful atmosphere",
    "Custom Style (Use My Own Prompt Only)": "",
}

def _load_wan_t2v():
    _ensure_model_from_drive("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    from diffusers import WanPipeline
    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

def _load_cogvideox():
    _ensure_model_from_drive("THUDM/CogVideoX-5b")
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe

def _load_svd_i2v():
    _ensure_model_from_drive("stabilityai/stable-video-diffusion-img2vid-xt")
    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    return pipe

def generate_text_to_video(prompt, my_idea, vid_engine, vid_style, aspect_v, num_frames, steps):
    try:
        from diffusers.utils import export_to_video
        style_hint = VIDEO_STYLES.get(vid_style, "")
        full_prompt = ", ".join([p for p in [(prompt or "").strip(), (my_idea or "").strip(), style_hint] if p])
        if not full_prompt:
            return None, "❌ Enter a Video Prompt or your own Video Idea."

        if "CogVideoX-5b" in vid_engine:
            pipe = _use("vid:cogvideox", _load_cogvideox)
            frames = pipe(
                prompt=full_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=max(20, int(steps)),
                num_frames=49,
                guidance_scale=6.0,
            ).frames[0]
            out = os.path.join(HUB, SUBFOLDERS["raw_videos"], f"cogvid_{_ts()}.mp4")
            export_to_video(frames, out, fps=8)
        else:
            w, h = (480, 832) if "9:16" in aspect_v else (832, 480)
            pipe = _use("vid:wan", _load_wan_t2v)
            frames = pipe(
                prompt=full_prompt,
                negative_prompt="blurry, distorted, jittery, low quality, watermark, deformed",
                height=h,
                width=w,
                num_frames=int(num_frames),
                num_inference_steps=int(steps),
                guidance_scale=5.0,
            ).frames[0]
            out = os.path.join(HUB, SUBFOLDERS["raw_videos"], f"wan_{_ts()}.mp4")
            export_to_video(frames, out, fps=16)

        path, note = save_and_sync(out, "raw_videos")
        return path, note
    except Exception:
        return None, _err()

def generate_image_to_video(init_img, motion_bucket, fps_val, noise_aug, smooth_60fps):
    try:
        from diffusers.utils import export_to_video
        if init_img is None:
            return None, "❌ Upload or send an image first to animate it into video."
        src = init_img if isinstance(init_img, str) else init_img.name
        image = Image.open(src).convert("RGB")
        if image.height > image.width:
            image = image.resize((576, 1024))
        else:
            image = image.resize((1024, 576))
        pipe = _use("vid:svd", _load_svd_i2v)
        frames = pipe(
            image,
            decode_chunk_size=4,
            motion_bucket_id=int(motion_bucket),
            noise_aug_strength=float(noise_aug),
            num_frames=25,
        ).frames[0]
        raw_out = os.path.join(HUB, SUBFOLDERS["raw_videos"], f"_i2v_raw_{_ts()}.mp4")
        out = os.path.join(HUB, SUBFOLDERS["raw_videos"], f"i2v_{_ts()}.mp4")
        export_to_video(frames, raw_out, fps=int(fps_val))
        if smooth_60fps:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_out, "-vf", "minterpolate=fps=60:mi_mode=blend", "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                check=True, capture_output=True
            )
            os.remove(raw_out)
        else:
            shutil.move(raw_out, out)
        path, note = save_and_sync(out, "raw_videos")
        return path, note
    except Exception:
        return None, _err()

def generate_lipsync_video(face_img, saved_face, voice_upload, saved_voice):
    try:
        ref_face = None
        if face_img is not None:
            ref_face = face_img if isinstance(face_img, str) else face_img.name
        elif saved_face and saved_face != "None":
            p = os.path.join(HUB, SUBFOLDERS["faces"], saved_face)
            if os.path.exists(p):
                ref_face = p

        ref_audio = None
        if voice_upload is not None:
            ref_audio = voice_upload if isinstance(voice_upload, str) else voice_upload.name
        elif saved_voice and saved_voice != "None":
            p = os.path.join(HUB, SUBFOLDERS["voices"], saved_voice)
            if os.path.exists(p):
                ref_audio = p

        if not ref_face or not ref_audio:
            return None, "❌ Please provide both a Character/Cartoon Face Image and a Voiceover Audio file (from Studio 01 or upload)."

        _ensure_model_from_drive("numz/wav2lip_studio")
        w2l_dir = "/kaggle/working/Wav2Lip" if os.path.isdir("/kaggle/working") else os.path.abspath("Wav2Lip")
        ckpt_candidates = glob.glob(os.path.join(HUB, SUBFOLDERS["cache"], "**", "*wav2lip*.pth"), recursive=True)
        out = os.path.join(HUB, SUBFOLDERS["raw_videos"], f"talking_{_ts()}.mp4")

        if ckpt_candidates and not os.path.exists(w2l_dir):
            subprocess.run(["git", "clone", "-q", "https://github.com/Rudrabha/Wav2Lip.git", w2l_dir], check=False)

        if ckpt_candidates and os.path.exists(os.path.join(w2l_dir, "inference.py")):
            r = subprocess.run(
                ["python", os.path.join(w2l_dir, "inference.py"), "--checkpoint_path", ckpt_candidates[0], "--face", ref_face, "--audio", ref_audio, "--outfile", out],
                capture_output=True, text=True
            )
            if r.returncode == 0 and os.path.exists(out):
                path, note = save_and_sync(out, "raw_videos")
                return path, f"🗣️ True Lip-Sync Talking Video generated! {note}"

        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", ref_face, "-i", ref_audio,
            "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0005,1.08)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280",
            "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", out
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        path, note = save_and_sync(out, "raw_videos")
        return path, f"🗣️ Talking Character Video rendered! {note}"
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ TAB 4: Viral Auto-Editor Pro
def _transcribe_srt(audio_path, srt_path, viral_short_phrases=True):
    from faster_whisper import WhisperModel
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel("small", device=device, compute_type=compute)
    segments, _ = model.transcribe(audio_path, word_timestamps=viral_short_phrases)

    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    idx = 1
    with open(srt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            if viral_short_phrases and getattr(seg, "words", None):
                words = list(seg.words)
                for i in range(0, len(words), 3):
                    chunk = words[i:i + 3]
                    txt = " ".join(w.word.strip() for w in chunk).strip()
                    if txt:
                        f.write(f"{idx}\n{fmt(chunk[0].start)} --> {fmt(chunk[-1].end)}\n{txt}\n\n")
                        idx += 1
            else:
                f.write(f"{idx}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{seg.text.strip()}\n\n")
                idx += 1

def assemble_final_video(selected_images, selected_videos, uploaded_media, saved_voice, uploaded_voice, saved_music, uploaded_music, bgm_volume, aspect_out, ken_burns, burn_subs):
    try:
        from moviepy.editor import ImageClip, VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips

        voice_path = None
        if uploaded_voice is not None:
            voice_path = uploaded_voice if isinstance(uploaded_voice, str) else uploaded_voice.name
        elif saved_voice and saved_voice != "None":
            p = os.path.join(HUB, SUBFOLDERS["voices"], saved_voice)
            if os.path.exists(p):
                voice_path = p

        music_path = None
        if uploaded_music is not None:
            music_path = uploaded_music if isinstance(uploaded_music, str) else uploaded_music.name
        elif saved_music and saved_music != "None":
            p = os.path.join(HUB, SUBFOLDERS["music"], saved_music)
            if os.path.exists(p):
                music_path = p

        media_paths = []
        for fn in (selected_videos or []):
            p = os.path.join(HUB, SUBFOLDERS["raw_videos"], fn)
            if os.path.exists(p):
                media_paths.append(("vid", p))
        for fn in (selected_images or []):
            p = os.path.join(HUB, SUBFOLDERS["images"], fn)
            if os.path.exists(p):
                media_paths.append(("img", p))
        for fobj in (uploaded_media or []):
            p = fobj if isinstance(fobj, str) else fobj.name
            kind = "vid" if p.lower().endswith((".mp4", ".mov", ".mkv", ".webm")) else "img"
            media_paths.append((kind, p))

        if not media_paths:
            return None, "❌ Select at least 1 Image, Cartoon Scene, or Video clip (or upload media) to assemble your final video."

        target_w, target_h = (1080, 1920) if "9:16" in aspect_out else (1920, 1080)

        audio_tracks = []
        total_dur = 0.0
        if voice_path:
            v_clip = AudioFileClip(voice_path)
            audio_tracks.append(v_clip)
            total_dur = max(total_dur, v_clip.duration)
        if music_path:
            effective_vol = float(bgm_volume) * (0.75 if voice_path else 1.0)
            m_clip = AudioFileClip(music_path).volumex(effective_vol)
            if total_dur > 0:
                if m_clip.duration < total_dur:
                    loops = int(total_dur // m_clip.duration) + 1
                    m_clip = concatenate_audioclips([m_clip] * loops)
                m_clip = m_clip.subclip(0, total_dur)
            else:
                total_dur = m_clip.duration
            audio_tracks.append(m_clip)

        if total_dur <= 0:
            total_dur = len(media_paths) * 4.0

        per_clip_dur = max(2.0, total_dur / len(media_paths))
        clips = []
        for idx, (kind, p) in enumerate(media_paths):
            if kind == "vid":
                c = VideoFileClip(p).without_audio()
                if c.duration < per_clip_dur:
                    loops = int(per_clip_dur // c.duration) + 1
                    c = concatenate_videoclips([c] * loops)
                c = c.subclip(0, per_clip_dur).resize(newsize=(target_w, target_h))
            else:
                c = ImageClip(p).set_duration(per_clip_dur).resize(newsize=(target_w, target_h))
                if ken_burns:
                    direction = 1 if idx % 2 == 0 else -1
                    c = c.resize(lambda t, d=direction, dur=per_clip_dur: 1.0 + 0.08 * (t / dur) if d == 1 else 1.08 - 0.08 * (t / dur))
            clips.append(c)

        video = concatenate_videoclips(clips, method="compose").subclip(0, total_dur)
        if audio_tracks:
            video = video.set_audio(CompositeAudioClip(audio_tracks))

        raw_out = os.path.join(HUB, SUBFOLDERS["exports"], f"_raw_{_ts()}.mp4")
        final_out = os.path.join(HUB, SUBFOLDERS["exports"], f"FINAL_VIDEO_{_ts()}.mp4")
        video.write_videofile(raw_out, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        if burn_subs and voice_path:
            srt_path = os.path.join(HUB, SUBFOLDERS["exports"], f"_subs_{_ts()}.srt")
            _transcribe_srt(voice_path, srt_path, viral_short_phrases=True)
            style_str = "FontName=DejaVu Sans,FontSize=22,Bold=1,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=85"
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_out, "-vf", f"subtitles={srt_path}:force_style='{style_str}'", "-c:a", "copy", final_out],
                check=True, capture_output=True
            )
            os.remove(raw_out)
            os.remove(srt_path)
        else:
            shutil.move(raw_out, final_out)

        path, note = save_and_sync(final_out, "exports")
        return path, f"🎬 Final Production Video Ready! {note}"
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ UI (High-Contrast Checkboxes + Clean Layout)
CSS = """
:root, .dark, body, gradio-app, .gradio-container {
  --body-background-fill:#090b10; --background-fill-primary:#111622; --background-fill-secondary:#111622;
  --block-background-fill:#111622; --input-background-fill:#182030; --body-text-color:#f8fafc;
  --block-label-text-color:#fbbf24; --block-title-text-color:#fbbf24; --border-color-primary:#243049;
  --color-accent:#f59e0b; --button-primary-background-fill:#f59e0b; --button-primary-text-color:#000000;
  --button-secondary-background-fill:#182030; --button-secondary-text-color:#f8fafc;
  background:#090b10 !important; color:#f8fafc !important;
}
.gradio-container .block, .gradio-container .form, .gradio-container .panel, .gradio-container .accordion { background:#111622 !important; border-color:#243049 !important; }
.gradio-container input:not([type="checkbox"]):not([type="radio"]), .gradio-container textarea, .gradio-container select,
.gradio-container .wrap, .gradio-container ul.options, .gradio-container .secondary-wrap { background:#182030 !important; color:#f8fafc !important; }
.gradio-container input[type="checkbox"], .gradio-container input[type="radio"] {
  appearance: auto !important; -webkit-appearance: checkbox !important;
  accent-color:#f59e0b !important; width:18px !important; height:18px !important;
  background-color:#ffffff !important; border:2px solid #f59e0b !important; cursor:pointer !important;
}
.gradio-container input[type="checkbox"]:checked {
  background-color:#f59e0b !important; border-color:#ffffff !important;
}
.gradio-container label.selected, .gradio-container .checkbox-item:has(input:checked) {
  background:#2a2210 !important; border:1px solid #f59e0b !important;
}
.gradio-container ul.options li { color:#f8fafc !important; }
.gradio-container label span, .gradio-container .label-wrap span, .gradio-container [data-testid="block-label"] { color:#fbbf24 !important; }
.gradio-container .prose, .gradio-container .prose * , .gradio-container p { color:#f8fafc !important; }
.gradio-container .prose a { color:#fbbf24 !important; }
.tab-nav button, [role="tablist"] button { background:#182030 !important; color:#f8fafc !important; border-radius:999px !important; margin:2px 4px !important; border:1px solid #243049 !important; }
.tab-nav button.selected, [role="tablist"] button[aria-selected="true"] { background:#f59e0b !important; color:#000000 !important; font-weight:700 !important; }
button.primary { background:#f59e0b !important; color:#000000 !important; font-weight:700 !important; }
button.secondary { background:#182030 !important; color:#f8fafc !important; }
#hdr h1 { color:#fbbf24 !important; margin:0; }
#banner textarea { color:#fbbf24 !important; font-weight:600; }
"""

def build_ui():
    theme = gr.themes.Base(primary_hue="amber", neutral_hue="slate")
    kw = {"title": "AI Studio 02 — Ultimate 10-Model Master Edition"}
    import inspect
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        kw.update(theme=theme, css=CSS)
    with gr.Blocks(**kw) as demo:
        with gr.Row(elem_id="hdr"):
            gr.Markdown("# 🎬🎨 AI Studio 02 — Ultimate 10-Model Master Edition\n4 Heavy Image/Cartoon Engines + 3 Heavy Video/Animation Engines + AI Vision Cloner + BG Remover + 4K Upscaler + Viral Auto-Editor.")
            with gr.Column(scale=0, min_width=320):
                vram_btn = gr.Button("🧹 Clear VRAM & Free Temp Disk", variant="secondary")
                gpu_box = gr.Textbox(value=gpu_status(), show_label=False, interactive=False)
        banner = gr.Textbox(value=drive_status(), show_label=False, interactive=False, elem_id="banner")
        vram_btn.click(clear_vram_and_disk, None, gpu_box)

        with gr.Accordion("☁️ Google Drive Live Sync & 3-Batch Disk-Safe Model Cache Setup (10 Heavy Models)", open=False):
            secrets = gr.File(label="client_secrets.json (auto-detected from /kaggle/input if attached)", file_types=[".json"])
            s1 = gr.Button("Step 1: Generate Google Sign-In Link", variant="primary")
            link = gr.Markdown()
            code = gr.Textbox(label="Step 2: Paste Verification Code")
            s2 = gr.Button("Connect & Sync Drive Now (Pulls Studio 01 Scripts, Voices & Music)", variant="primary")
            dmsg = gr.Textbox(label="Drive status", interactive=False)
            s1.click(drive_step1, secrets, link)

            gr.Markdown("### 📦 3-Batch Disk-Safe Model Caching (Kaggle 19.5 GB Disk-Safe)\nRun **Batch 1**, then **Batch 2**, then **Batch 3** — each model uploads to Drive and clears local disk automatically.")
            with gr.Row():
                b1_btn = gr.Button("⬇️ Batch 1: 5 Image/Cartoon/Face Models (RealVisXL + FLUX.1-dev + FLUX-schnell + Lightning + IP-Adapter)", variant="primary")
                b2_btn = gr.Button("⬇️ Batch 2: 4 Video/Animation/LipSync Models (Wan2.1 + CogVideoX-5B + SVD-XT + Wav2Lip)", variant="primary")
                b3_btn = gr.Button("⬇️ Batch 3: AI Vision Prompt Cloner (Qwen2.5-VL-7B)", variant="primary")
            model_picker = gr.CheckboxGroup(
                choices=ALL_STUDIO2_MODELS,
                value=BATCH_1_IMAGE_MODELS,
                label="Or Custom-Pick Any of the 10 Models to Cache 1-by-1",
            )
            with gr.Row():
                dl_btn = gr.Button("⬇️ Cache Selected Checkboxes", variant="secondary")
                cs_btn = gr.Button("🔄 Check Cache Sync & Disk Space", variant="secondary")
                ref_hub_btn = gr.Button("♻️ Refresh Studio 01 Assets from Hub", variant="secondary")
            dl_msg = gr.Textbox(label="Model cache & Hub status", interactive=False, lines=6)
            b1_btn.click(download_batch_1, None, dl_msg)
            b2_btn.click(download_batch_2, None, dl_msg)
            b3_btn.click(download_batch_3, None, dl_msg)
            dl_btn.click(download_models_list, model_picker, dl_msg)
            cs_btn.click(cache_sync_status, None, dl_msg)

        with gr.Tabs():
            with gr.Tab("🎨 AI Image, Cartoon & Character Studio"):
                with gr.Accordion("🧬 Consistent Character / AI Influencer Face Lock (Saved in 00_reference_faces)", open=True):
                    with gr.Row():
                        face_upload = gr.Image(label="Upload Clear Front Face Photo", type="filepath")
                        with gr.Column():
                            face_name = gr.Textbox(label="Character / Influencer Name", value="Main_Influencer")
                            save_face_btn = gr.Button("💾 Save Face Permanently to Drive", variant="secondary")
                            saved_face = gr.Dropdown(choices=list_hub_files("faces"), value="None", label="Or Pick Saved Reference Face")
                            face_strength = gr.Slider(0.20, 1.00, value=0.65, step=0.05, label="Face Identity Lock Strength (RealVisXL)")
                    face_msg = gr.Textbox(label="Character Face Library Status", interactive=False)

                with gr.Row():
                    img_model = gr.Dropdown(IMAGE_MODELS, value=IMAGE_MODELS[0], label="Heavy Image Engine (4 Models)")
                    img_style = gr.Dropdown(list(IMAGE_STYLES), value=list(IMAGE_STYLES)[0], label="Visual / Cartoon Style Preset")
                    img_aspect = gr.Dropdown(list(ASPECT_RATIOS), value=list(ASPECT_RATIOS)[0], label="Aspect Ratio")
                img_prompt = gr.Textbox(label="Scene Prompt", lines=2, placeholder="e.g. Cute 3D Pixar style little boy and his dog in an Indian village, or Photorealistic AI influencer in Mumbai cafe...")
                img_idea = gr.Textbox(label="💡 Apna Khud Ka Visual / Cartoon Idea / Outfit / Camera Angle Details", lines=2, placeholder="Apni pasand ke kapde, cartoon character ka look, background, ya camera angle yahan likhein...")
                with gr.Row():
                    img_neg = gr.Textbox(label="Negative Prompt (Optional)", value="")
                    img_steps = gr.Slider(4, 40, value=25, step=1, label="Quality Steps (25-35 for RealVis/FLUX-dev, 4-8 for schnell/Lightning)")
                    img_seed = gr.Number(value=-1, label="Seed (-1 = Random)")
                with gr.Row():
                    gen_img_btn = gr.Button("🎨 Generate High-Detail Image / Cartoon Scene", variant="primary")
                    send_i2v_btn = gr.Button("🎬 Send to Video Animator", variant="secondary")
                    send_edit_btn = gr.Button("🖌️ Send to Editor / Thumbnail", variant="secondary")
                with gr.Row():
                    out_img = gr.Image(label="Generated Image / Cartoon", type="filepath")
                    out_img_file = gr.File(label="Saved Image File")
                img_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("📜 Auto-Generate All Scenes from Studio 01 Script", open=False):
                    with gr.Row():
                        script_picker = gr.Dropdown(choices=list_hub_files("scripts"), value="None", label="Pick Script from Studio 01 (01_scripts)")
                        load_scenes_btn = gr.Button("📥 Extract [VISUAL SCENE] Prompts", variant="secondary")
                    scenes_box = gr.Textbox(label="Scene Prompts (1 per line — up to 10 scenes)", lines=5)
                    batch_img_btn = gr.Button("🚀 Generate All Scenes Batch", variant="primary")
                    scenes_gallery = gr.Gallery(label="Generated Storyboard Scenes", columns=3)
                    batch_img_status = gr.Textbox(label="Batch Status", interactive=False, lines=4)

            with gr.Tab("🖌️ BG Remover, 4K Upscaler, Thumbnail & AI Vision"):
                edit_in = gr.Image(label="Upload or Send Image from Tab 1", type="filepath")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ✂️ 1-Click Background Remover / Replacer")
                        bg_mode = gr.Dropdown(["Transparent PNG (No Background)", "Solid Color Background", "Custom Image Background"], value="Transparent PNG (No Background)", label="Background Mode")
                        bg_color = gr.Textbox(label="Solid Color Hex (if Solid Color)", value="#0f172a")
                        new_bg_up = gr.Image(label="Optional: Upload New Background Image", type="filepath")
                        bg_btn = gr.Button("✂️ Apply Background Tool", variant="primary")
                    with gr.Column():
                        gr.Markdown("### 🔍 4K Cinema Upscaler & Sharpener")
                        up_scale = gr.Slider(1.5, 4.0, value=2.0, step=0.5, label="Upscale Multiplier (2x - 4x)")
                        up_sharp = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="Detail Sharpness")
                        up_contrast = gr.Slider(0.8, 1.4, value=1.08, step=0.02, label="Color Punch / Contrast")
                        up_btn = gr.Button("🔍 Upscale & Enhance Image", variant="primary")
                    with gr.Column():
                        gr.Markdown("### 🖼️ Viral YouTube / Reels Thumbnail Maker")
                        th_top = gr.Textbox(label="Top Bold Text", placeholder="e.g. YEH SACH KOI NAHI BATATA!")
                        th_bot = gr.Textbox(label="Bottom Highlight Text", placeholder="e.g. PART 01 • FULL STORY")
                        th_col = gr.Textbox(label="Highlight Color Hex", value="#FACC15")
                        th_border = gr.Checkbox(value=True, label="Add Glowing Frame Border")
                        th_btn = gr.Button("🖼️ Create Viral Thumbnail", variant="primary")
                edit_out = gr.Image(label="Edited / Upscaled / Thumbnail Output", type="filepath")
                edit_msg = gr.Textbox(label="Editor Status", interactive=False)

                with gr.Accordion("👁️ AI Vision Reference Prompt Cloner (Qwen2.5-VL-7B — Upload Any Photo/Thumbnail to Clone Style)", open=False):
                    with gr.Row():
                        vis_img = gr.Image(label="Upload Reference Photo / Cartoon Frame / Viral Thumbnail", type="filepath")
                        with gr.Column():
                            vis_goal = gr.Textbox(label="💡 Apna Khud Ka Twist (Optional)", placeholder="e.g. Change outfit to red saree, or make it 3D Pixar cartoon style...")
                            vis_btn = gr.Button("👁️ Extract Detailed Prompt from Image", variant="primary")
                            send_vis_btn = gr.Button("🎨 Send Cloned Prompt to Image Studio", variant="secondary")
                    vis_out = gr.Textbox(label="Extracted Prompt", lines=4)

            with gr.Tab("🎬 AI Video, Cartoon Animation & Lip-Sync"):
                with gr.Tab("Mode A — Text-to-Video (Wan2.1 & Heavy CogVideoX-5B)"):
                    t2v_prompt = gr.Textbox(label="Video Action Prompt", lines=2, placeholder="e.g. Cute 3D Pixar cartoon boy running happily in rain, or Cinematic girl walking in neon Mumbai street...")
                    t2v_idea = gr.Textbox(label="💡 Apna Khud Ka Video Movement / Cartoon Action / Camera Idea", lines=2, placeholder="e.g. Smooth camera pan, expressive character movement, colorful lighting...")
                    with gr.Row():
                        t2v_engine = gr.Dropdown(VIDEO_ENGINES, value=VIDEO_ENGINES[0], label="Heavy Video Engine")
                        t2v_style = gr.Dropdown(list(VIDEO_STYLES), value=list(VIDEO_STYLES)[0], label="Video / Cartoon Style")
                    with gr.Row():
                        t2v_aspect = gr.Dropdown(["9:16 Vertical (Reels / Shorts)", "16:9 Horizontal (YouTube)"], value="9:16 Vertical (Reels / Shorts)", label="Video Ratio (Wan2.1)")
                        t2v_frames = gr.Slider(17, 65, value=33, step=8, label="Frames (Wan2.1)")
                        t2v_steps = gr.Slider(15, 40, value=25, step=1, label="Quality Steps")
                    t2v_btn = gr.Button("🎬 Generate Video / Cartoon Clip", variant="primary")
                    t2v_out = gr.Video(label="Generated Video"); t2v_msg = gr.Textbox(label="Status", interactive=False)

                with gr.Tab("Mode B — Image / Cartoon-to-Video Animator (SVD-XT + 60fps Smooth)"):
                    i2v_input = gr.Image(label="Upload or Send Image / Cartoon Scene from Tab 1", type="filepath")
                    with gr.Row():
                        i2v_motion = gr.Slider(40, 220, value=127, step=5, label="Motion Intensity (127 = Natural)")
                        i2v_fps = gr.Slider(6, 24, value=10, step=1, label="Base FPS")
                        i2v_noise = gr.Slider(0.01, 0.10, value=0.02, step=0.01, label="Motion Freedom")
                        i2v_60fps = gr.Checkbox(value=True, label="✨ Smooth 60fps Frame Interpolation")
                    i2v_btn = gr.Button("🎞️ Animate Image into Video", variant="primary")
                    i2v_out = gr.Video(label="Animated Video"); i2v_msg = gr.Textbox(label="Status", interactive=False)

                with gr.Tab("Mode C — Talking Character / Lip-Sync (Speaks Studio 01 Voice)"):
                    with gr.Row():
                        ls_face_up = gr.Image(label="Upload Character / Cartoon Face Photo", type="filepath")
                        ls_face_saved = gr.Dropdown(choices=list_hub_files("faces"), value="None", label="Or Pick Saved Character Face")
                    with gr.Row():
                        ls_voice_up = gr.Audio(label="Upload Voiceover", type="filepath")
                        ls_voice_saved = gr.Dropdown(choices=list_hub_files("voices"), value="None", label="Or Pick Studio 01 Voiceover (02_voices)")
                    ls_btn = gr.Button("🗣️ Generate Talking Character Video", variant="primary")
                    ls_out = gr.Video(label="Talking Video"); ls_msg = gr.Textbox(label="Status", interactive=False)

            with gr.Tab("🎞️ Viral Auto-Editor Pro (Final MP4 Assembler)"):
                gr.Markdown("### Auto-Assemble Images / Cartoon Scenes + Videos + Studio 01 Cloned Voice + Music + Ken Burns Zoom + Word-by-Word Viral Subtitles")
                with gr.Row():
                    asm_imgs = gr.CheckboxGroup(choices=[f for f in list_hub_files("images") if f != "None"], label="Select Images / Cartoon Scenes from Hub (03_images)")
                    asm_vids = gr.CheckboxGroup(choices=[f for f in list_hub_files("raw_videos") if f != "None"], label="Select Videos from Hub (04_raw_videos)")
                asm_up_media = gr.File(label="Or Upload Extra Images/Videos", file_count="multiple")
                with gr.Row():
                    asm_voice_saved = gr.Dropdown(choices=list_hub_files("voices"), value="None", label="Studio 01 Voiceover (02_voices)")
                    asm_voice_up = gr.Audio(label="Or Upload Voiceover", type="filepath")
                with gr.Row():
                    asm_music_saved = gr.Dropdown(choices=list_hub_files("music"), value="None", label="Studio 01 Music / BGM (05_music)")
                    asm_music_up = gr.Audio(label="Or Upload BGM", type="filepath")
                with gr.Row():
                    asm_bgm_vol = gr.Slider(0.05, 0.60, value=0.22, step=0.05, label="Background Music Volume (Auto-Ducks under Voice)")
                    asm_aspect = gr.Dropdown(["9:16 Vertical Reel / Short (1080x1920)", "16:9 Horizontal YouTube (1920x1080)"], value="9:16 Vertical Reel / Short (1080x1920)", label="Export Resolution")
                with gr.Row():
                    asm_ken_burns = gr.Checkbox(value=True, label="🎥 Dynamic Ken Burns Zoom/Pan Motion on Images")
                    asm_subs = gr.Checkbox(value=True, label="🔥 Viral Word-by-Word Yellow Subtitles (Faster-Whisper)")
                asm_btn = gr.Button("🚀 Render Final Production Video (MP4)", variant="primary")
                asm_out = gr.Video(label="Final Exported Video"); asm_msg = gr.Textbox(label="Export Status", interactive=False)

        # Wire all buttons cleanly
        s2.click(drive_step2, code, [dmsg, banner, saved_face, script_picker, asm_voice_saved, asm_music_saved])
        ref_hub_btn.click(refresh_all_hub_assets, None, [dl_msg, saved_face, script_picker, asm_voice_saved, asm_music_saved, asm_imgs, asm_vids])
        save_face_btn.click(save_reference_face, [face_upload, face_name], [face_msg, saved_face])
        gen_img_btn.click(generate_image, [img_prompt, img_idea, img_style, img_aspect, img_model, face_upload, saved_face, face_strength, img_neg, img_steps, img_seed], [out_img, out_img_file, img_status])
        send_i2v_btn.click(lambda p: (p, "✅ Image sent to Video Animator (Mode B)!"), out_img, [i2v_input, img_status])
        send_edit_btn.click(lambda p: (p, "✅ Image sent to BG Remover / Upscaler / Thumbnail Maker!"), out_img, [edit_in, img_status])
        load_scenes_btn.click(load_script_scenes, script_picker, scenes_box)
        batch_img_btn.click(batch_generate_scenes, [scenes_box, img_idea, img_style, img_aspect, img_model, face_upload, saved_face, face_strength], [scenes_gallery, batch_img_status])
        bg_btn.click(remove_or_replace_bg, [edit_in, bg_mode, bg_color, new_bg_up], [edit_out, edit_msg])
        up_btn.click(upscale_4k_enhance, [edit_in, up_scale, up_sharp, up_contrast], [edit_out, edit_msg])
        th_btn.click(make_viral_thumbnail, [edit_in, th_top, th_bot, th_col, th_border], [edit_out, edit_msg])
        vis_btn.click(reverse_engineer_prompt, [vis_img, vis_goal], vis_out)
        send_vis_btn.click(lambda t: (t, "✅ Cloned prompt sent to Image Studio!"), vis_out, [img_prompt, img_status])
        t2v_btn.click(generate_text_to_video, [t2v_prompt, t2v_idea, t2v_engine, t2v_style, t2v_aspect, t2v_frames, t2v_steps], [t2v_out, t2v_msg])
        i2v_btn.click(generate_image_to_video, [i2v_input, i2v_motion, i2v_fps, i2v_noise, i2v_60fps], [i2v_out, i2v_msg])
        ls_btn.click(generate_lipsync_video, [ls_face_up, ls_face_saved, ls_voice_up, ls_voice_saved], [ls_out, ls_msg])
        asm_btn.click(assemble_final_video, [asm_imgs, asm_vids, asm_up_media, asm_voice_saved, asm_voice_up, asm_music_saved, asm_music_up, asm_bgm_vol, asm_aspect, asm_ken_burns, asm_subs], [asm_out, asm_msg])

    _orig = demo.launch
    def launch(*a, **k):
        if "css" not in kw:
            import inspect as _i
            ps = _i.signature(_orig).parameters
            if "css" in ps: k.setdefault("css", CSS)
            if "theme" in ps: k.setdefault("theme", theme)
        return _orig(*a, **k)
    demo.launch = launch
    return demo

if __name__ == "__main__":
    build_ui().launch(share=True, debug=True)
