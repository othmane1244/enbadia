#!/bin/bash
# ==============================================================
# Raspberry Pi 5 — Hailo-10H + OV5647 Camera — Full installer
# Target OS: Raspberry Pi OS Bookworm 13 (Debian Trixie)
# ==============================================================
#
# Installs:
#   - System tools, build essentials
#   - Camera stack (libcamera, picamera2, rpicam-apps)
#   - Hailo-10H runtime + PCIe driver + firmware + Python bindings
#   - System Python deps via apt: numpy, opencv, scipy
#       → avoids ABI conflicts with picamera2/hailo_platform
#   - Main Python venv at /opt/ai_venv (Python 3.13, system-site-packages)
#       Pure-Python web/db/util deps only — no torch, no opencv pip
#   - Miniconda at /opt/miniconda3 with Python 3.11 env "yolo"
#       PyTorch CPU 2.4.1, ultralytics, ncnn, ONNX — model work + CPU fallback
#
# Run:  sudo bash install_dependencies.sh
# Re-runnable: yes (idempotent)
# ==============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
step()  { echo -e "\n${BLUE}════════ $1 ════════${NC}"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --------------------------------------------------------------
# 0. Sanity checks
# --------------------------------------------------------------
step "0. Sanity checks"

[[ $EUID -eq 0 ]] || error "Run as root: sudo bash $0"

. /etc/os-release
if [[ "$VERSION_CODENAME" != "trixie" ]]; then
    error "This script targets Debian 13 (Trixie). You are on: $PRETTY_NAME"
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "System: $PRETTY_NAME, Python $PYVER, kernel $(uname -r)"

# --------------------------------------------------------------
# 1. System tools + system Python deps via apt
# --------------------------------------------------------------
step "1. System tools and Python deps"

apt update -qq
warn "Skipping 'apt upgrade' — run it manually later to avoid kernel changes mid-script"

apt install -y \
    build-essential cmake git curl wget pkg-config dkms \
    python3 python3-pip python3-venv python3-dev \
    v4l-utils ffmpeg libcap-dev \
    libgl1 libglib2.0-0

# System Python packages — used by the main venv via --system-site-packages.
# We install these via apt (not pip) so they match the system numpy ABI,
# which is what picamera2 and hailo_platform were compiled against.
apt install -y \
    python3-numpy \
    python3-opencv \
    python3-scipy \
    python3-pil \
    python3-psutil

# --------------------------------------------------------------
# 2. Camera (OV5647 / libcamera / picamera2)
# --------------------------------------------------------------
step "2. Camera stack"

apt install -y \
    libcamera-dev python3-libcamera python3-picamera2 \
    rpicam-apps

CONFIG=/boot/firmware/config.txt
if [[ -f $CONFIG ]]; then
    if grep -q "camera_auto_detect=0" "$CONFIG"; then
        warn "camera_auto_detect=0 found — enabling"
        sed -i 's/camera_auto_detect=0/camera_auto_detect=1/' "$CONFIG"
    elif ! grep -q "camera_auto_detect=1" "$CONFIG"; then
        warn "appending camera_auto_detect=1 to $CONFIG"
        cp "$CONFIG" "${CONFIG}.bak.$(date +%s)"
        echo -e "\n# Camera auto-detect\ncamera_auto_detect=1" >> "$CONFIG"
    fi
fi

# --------------------------------------------------------------
# 3. Hailo-10H stack
# --------------------------------------------------------------
step "3. Hailo-10H stack"

# hailo-h10-all pulls in: h10-hailort, h10-hailort-pcie-driver (with firmware),
# hailo-tappas-core, rpicam-apps-hailo-postprocess, python3-h10-hailort
apt install -y hailo-h10-all

# Verify firmware files are present
if [[ -d /lib/firmware/hailo/hailo10h ]]; then
    info "Hailo-10H firmware files installed: $(ls /lib/firmware/hailo/hailo10h | wc -l) files"
else
    warn "Hailo-10H firmware directory missing — driver will fail to load"
fi

# Try to load the driver. May fail until reboot if DKMS module needs to build.
modprobe hailo_pci 2>/dev/null || warn "modprobe hailo_pci failed — reboot needed"

# --------------------------------------------------------------
# 4. Main Python venv at /opt/ai_venv (Python 3.13)
# --------------------------------------------------------------
step "4. Main venv at /opt/ai_venv"

VENV_DIR=/opt/ai_venv
if [[ -d $VENV_DIR ]]; then
    warn "$VENV_DIR exists — recreating"
    rm -rf "$VENV_DIR"
fi

# --system-site-packages so venv inherits picamera2, hailo_platform,
# numpy, opencv, scipy from system apt packages
python3 -m venv --system-site-packages "$VENV_DIR"
PIP="$VENV_DIR/bin/pip install --no-cache-dir"

# Only pure-Python deps (or ones with safe wheels) — no opencv, scipy, numpy
$PIP \
    "fastapi" "uvicorn[standard]" "pydantic" "pydantic-settings" \
    "python-dotenv" "websockets" "httpx" \
    "supabase" "postgrest" \
    "requests" "python-multipart" "aiofiles"

info "Main venv ready: source $VENV_DIR/bin/activate"

# --------------------------------------------------------------
# 5. Miniconda + Python 3.11 env "yolo" (for ultralytics + torch)
# --------------------------------------------------------------
step "5. Miniconda + Python 3.11 env for YOLO"

CONDA_DIR=/opt/miniconda3
if [[ ! -d $CONDA_DIR ]]; then
    info "Downloading Miniconda for aarch64..."
    cd /tmp
    wget -q --show-progress \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
        -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_DIR"
    rm miniconda.sh
else
    info "Miniconda already at $CONDA_DIR"
fi

CONDA="$CONDA_DIR/bin/conda"

if ! "$CONDA" env list | grep -q "^yolo "; then
    info "Creating conda env 'yolo' with Python 3.11"
    "$CONDA" create -y -n yolo python=3.11
fi

CONDA_PIP="$CONDA_DIR/envs/yolo/bin/pip install --no-cache-dir"

# torch 2.4.1 — last version before NVIDIA hard-deps in 2.6+
$CONDA_PIP "torch==2.4.1" "torchvision==0.19.1"

# YOLO stack
$CONDA_PIP \
    "ultralytics" "ncnn" \
    "onnx" "onnxruntime" "onnxsim" \
    "opencv-python" "lap"

ln -sf "$CONDA_DIR/bin/conda" /usr/local/bin/conda

info "Conda env ready: source /opt/miniconda3/bin/activate yolo"

# --------------------------------------------------------------
# 6. Verification
# --------------------------------------------------------------
step "6. Verification"

echo ""
echo "--- Hardware ---"
if lspci | grep -qi hailo; then
    echo -e "  ${GREEN}OK${NC}    Hailo on PCIe: $(lspci | grep -i hailo | head -1)"
else
    echo -e "  ${RED}FAIL${NC}  Hailo device not found on PCIe"
fi

if ls /dev/hailo* &>/dev/null; then
    echo -e "  ${GREEN}OK${NC}    Hailo device node: $(ls /dev/hailo*)"
else
    echo -e "  ${YELLOW}WARN${NC}  /dev/hailo* not found — reboot required"
fi

if libcamera-hello --list-cameras 2>&1 | grep -qE "ov5647|imx"; then
    echo -e "  ${GREEN}OK${NC}    Camera detected"
else
    echo -e "  ${YELLOW}WARN${NC}  No camera detected (check ribbon cable)"
fi

echo ""
echo "--- Main venv (Python 3.13) ---"
"$VENV_DIR/bin/python3" -c "import picamera2"      2>/dev/null && echo "  picamera2     OK" || echo -e "  ${RED}FAIL${NC} picamera2"
"$VENV_DIR/bin/python3" -c "import cv2"            2>/dev/null && echo "  opencv        OK" || echo -e "  ${RED}FAIL${NC} cv2"
"$VENV_DIR/bin/python3" -c "import numpy"          2>/dev/null && echo "  numpy         OK" || echo -e "  ${RED}FAIL${NC} numpy"
"$VENV_DIR/bin/python3" -c "import scipy"          2>/dev/null && echo "  scipy         OK" || echo -e "  ${RED}FAIL${NC} scipy"
"$VENV_DIR/bin/python3" -c "import fastapi"        2>/dev/null && echo "  fastapi       OK" || echo -e "  ${RED}FAIL${NC} fastapi"
"$VENV_DIR/bin/python3" -c "import supabase"       2>/dev/null && echo "  supabase      OK" || echo -e "  ${RED}FAIL${NC} supabase"
"$VENV_DIR/bin/python3" -c "import hailo_platform" 2>/dev/null && echo "  hailo_platform OK" || echo -e "  ${YELLOW}WARN${NC} hailo_platform (reboot may be needed)"

echo ""
echo "--- Conda env 'yolo' (Python 3.11) ---"
"$CONDA_DIR/envs/yolo/bin/python3" -c "import torch"       2>/dev/null && echo "  torch         OK" || echo -e "  ${RED}FAIL${NC} torch"
"$CONDA_DIR/envs/yolo/bin/python3" -c "import ultralytics" 2>/dev/null && echo "  ultralytics   OK" || echo -e "  ${RED}FAIL${NC} ultralytics"
"$CONDA_DIR/envs/yolo/bin/python3" -c "import ncnn"        2>/dev/null && echo "  ncnn          OK" || echo -e "  ${RED}FAIL${NC} ncnn"

echo ""
step "Done"
echo ""
echo "Two Python environments are now installed:"
echo ""
echo -e "  1) ${GREEN}/opt/ai_venv${NC}  — Python 3.13, project + Hailo"
echo "        source /opt/ai_venv/bin/activate"
echo "        → main surveillance pipeline (camera, Hailo inference, API)"
echo ""
echo -e "  2) ${GREEN}/opt/miniconda3/envs/yolo${NC}  — Python 3.11, YOLO toolchain"
echo "        source /opt/miniconda3/bin/activate yolo"
echo "        → model export (.pt → .onnx/.ncnn), CPU inference fallback"
echo ""
if ! ls /dev/hailo* &>/dev/null; then
    warn "Reboot required to finish loading the Hailo PCIe driver:"
    echo -e "  ${YELLOW}sudo reboot${NC}"
fi
