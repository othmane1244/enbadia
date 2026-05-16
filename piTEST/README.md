# piTEST — Session Handoff

Test scripts and notes for getting YOLO + camera + Hailo running on a Raspberry Pi 5.
Read this top to bottom before continuing — it explains where things stand and why.

---

## Hardware

| Component | Detail |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1 (8 GB RAM) |
| AI accelerator | Hailo-10H (PCIe `0001:01:00.0`) — detected, not yet used |
| Camera | OV5647 (5 MP, Pi Camera v1) via CSI |
| Storage | 57 GB SD card |
| Pi IP | `192.168.0.102` |
| User | `ai` |

Camera modes (from `libcamera-hello --list-cameras`):
- 640×480 @ 58.9 fps
- 1296×972 @ 46.3 fps
- 1920×1080 @ 32.8 fps
- 2592×1944 @ 15.6 fps

---

## Software state

| | Detail |
|---|---|
| OS | **Debian 13 (Trixie)** — kernel `6.12.75+rpt-rpi-2712` |
| Python | **3.13.5** |
| Venv | `/opt/ai_venv` (created with `--system-site-packages`) |
| Hailo PCIe driver | **Not installed** (we never finished `install_dependencies.sh`) |
| Hailo Python bindings | Not installed |
| `picamera2` | Installed (system) |
| `onnxruntime` | Installed in venv |
| `ultralytics` / `torch` | **NOT installed cleanly** — see "The core problem" |

---

## The core problem — READ THIS

On **aarch64 + Python 3.13**, the only PyTorch versions on PyPI are **2.6+**, and they **mandate CUDA** at import time (built for Jetson, not Pi). On a Pi5 with no CUDA, torch refuses to load.

This blocks `pip install ultralytics` from working, which blocks exporting `.pt → .onnx` on the Pi.

### Why this didn't bite other people
Everyone running YOLO on Pi uses **Raspberry Pi OS Bookworm (Debian 12) with Python 3.11**, where torch 2.0–2.4 is available CPU-only without CUDA. Our Pi has Trixie, which is the source of every dependency headache in this session.

### Recommended fix
**Reflash the SD card with Raspberry Pi OS Bookworm (64-bit Lite).** ~20 minutes. Restores supported config:
- Python 3.11 → `pip install ultralytics` just works
- Hailo packages (`hailo-h10-all`) are tested on Bookworm
- All Ultralytics + Raspberry Pi docs assume Bookworm

### Workarounds if you can't reflash
1. **Miniconda + Python 3.11** in `~/miniconda3` — gets ultralytics working, leaves rest of system on Trixie
2. **Docker** (`ultralytics/ultralytics:latest-arm64`) — clean container, but camera/Hailo passthrough is fiddly

---

## What works right now

### 1. Camera streaming (no AI)
```bash
source /opt/ai_venv/bin/activate
python3 /home/ai/ai/piTEST/camera_stream.py
```
Then open `http://192.168.0.102:8080` on any device on the same WiFi.
Settings: 1280×720 @ 30 fps (sweet spot for this project — see `subject.txt`).

### 2. YOLO inference — **blocked** until `yolov8n.onnx` is on disk
```bash
python3 /home/ai/ai/piTEST/yolo_camera.py
```
Script is ready and uses `onnxruntime` (no torch). It needs `yolov8n.onnx` in this folder.

To produce the ONNX file (any of):
- **On a PC with ultralytics:**
  ```bash
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
  scp yolov8n.onnx ai@192.168.0.102:/home/ai/ai/piTEST/
  ```
- **Via Docker on the Pi (no torch install needed):**
  ```bash
  sudo docker run --rm -v /home/ai/ai/piTEST:/output \
    ultralytics/ultralytics:latest-arm64 \
    python3 -c "from ultralytics import YOLO; import shutil; \
      YOLO('yolov8n.pt').export(format='onnx'); \
      shutil.copy('yolov8n.onnx', '/output/yolov8n.onnx')"
  ```

---

## Files in this folder

| File | Purpose |
|---|---|
| `camera_stream.py` | MJPEG live stream, no AI. **Works.** |
| `yolo_camera.py` | Camera + YOLO inference via onnxruntime. Needs `yolov8n.onnx`. |
| `export_model.py` | Helper to export a custom `.pt` → `.onnx` from a PC and SCP to Pi |
| `export_to_pi.bat` | Same as above, but as a double-click Windows batch script |
| `README.md` | This file |

---

## Project context (don't lose sight of this)

The bigger project (`/home/ai/ai`) is **ENSA Béni Mellal — IA & Cybersécurité 2025-2026**: a smart surveillance system that must detect intrusions, falls, abandoned objects, and suspicious behavior in real time on this Pi + Hailo. Target: **≥15 FPS** with YOLO via Hailo-10H. See `subject.txt`.

`piTEST/` is throwaway exploration — once we have a working pipeline (camera → Hailo → YOLO → alerts), it gets integrated into `main.py`.

---

## Next steps (in order)

1. **Reflash to Debian 12 Bookworm** (or commit to a workaround).
2. Run `install_dependencies.sh` from project root — installs Hailo stack + camera + Python deps. Already patched for safety (no blind `apt upgrade`, uses a venv, etc.).
3. Get `yolov8n.onnx` on the Pi and verify `yolo_camera.py` shows detections in the browser.
4. Move YOLO inference from `onnxruntime` (CPU, ~5 fps) to **Hailo-10H** (~30+ fps):
   - Export model to `.hef` using Hailo Dataflow Compiler (DFC) on PC
   - Replace `onnxruntime.InferenceSession` with `hailo_platform.VDevice`
5. Integrate the working pipeline into `/home/ai/ai/main.py` and wire up FastAPI alerts.

---

## Useful one-liners

```bash
# Quick camera sanity check
rpicam-hello --list-cameras
rpicam-jpeg -o /tmp/test.jpg --timeout 1000

# Get the Pi's IP
hostname -I

# Reactivate the venv
source /opt/ai_venv/bin/activate

# Check Hailo on PCIe
lspci | grep -i hailo

# Tail logs of a backgrounded script
tail -f /home/ai/ai/logs/*.log
```
