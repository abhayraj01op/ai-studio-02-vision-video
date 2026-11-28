# 🎬🎨 AI Studio 02 — Ultimate All-Tools Master Edition

Repository Name: **`ai-studio-02-vision-video`**

## ▶️ [1-Click Launch on Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-studio-02-vision-video/blob/main/Studio_02.ipynb)

### Fixes Included
- **Automatic Cloudflare Backup Tunnel (`trycloudflare.com`):** If Gradio's public share server fails due to SSL/TLS certificate expiry (`x509: certificate has expired`), `app.py` automatically launches locally and generates a working clickable `https://...trycloudflare.com` public link.
- **Zero Hardcoded Token Expiry:** Paste any fresh HuggingFace `Read` token inside the UI box (only needed for gated `FLUX.1-schnell` / `FLUX.1-dev`). All public models use `token=None`.
- **19.5 GB Kaggle Disk-Safe:** Exact single-copy weight filters + automatic local cache cleanup after Drive upload.
