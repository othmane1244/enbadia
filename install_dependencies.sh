#!/bin/bash
# ==============================================================
# Raspberry Pi 5 — Hailo-10H + OV5647 Camera
# Full dependency installer
# ==============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --------------------------------------------------------------
# 0. Root check
# --------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    error "Run as root: sudo bash install_dependencies.sh"
fi

info "Starting installation on Raspberry Pi 5 — Debian $(. /etc/os-release && echo $VERSION_ID)"

# --------------------------------------------------------------
# 1. System update
# --------------------------------------------------------------
info "Updating package lists..."
apt update -qq

warn "Skipping 'apt upgrade' — run it manually after installation to avoid"
warn "upgrading the kernel mid-script which can break the Hailo PCIe driver."
warn "  sudo apt upgrade"

# --------------------------------------------------------------
# 2. System tools & build essentials
# --------------------------------------------------------------
info "Installing system tools..."
apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    dkms \
    v4l-utils \
    ffmpeg \
    libcap-dev

# --------------------------------------------------------------
# 3. Camera — OV5647 / libcamera stack
# --------------------------------------------------------------
info "Installing camera dependencies..."
apt install -y \
    libcamera-apps \
    libcamera-dev \
    python3-libcamera \
    python3-picamera2 \
    rpicam-apps

# Enable CSI camera in config if not already enabled
CONFIG=/boot/firmware/config.txt
if grep -q "camera_auto_detect=0" "$CONFIG" 2>/dev/null; then
    warn "camera_auto_detect=0 found — enabling it..."
    sed -i 's/camera_auto_detect=0/camera_auto_detect=1/' "$CONFIG"
elif ! grep -q "camera_auto_detect=1" "$CONFIG" 2>/dev/null; then
    warn "camera_auto_detect not found in $CONFIG — appending..."
    cp "$CONFIG" "${CONFIG}.bak"
    echo -e "\n# Camera auto-detect\ncamera_auto_detect=1" >> "$CONFIG"
fi

# --------------------------------------------------------------
# 4. Hailo-10H — runtime, PCIe driver, Python bindings, TAPPAS
# --------------------------------------------------------------
info "Installing Hailo-10H stack..."
apt install -y \
    hailo-h10-all \
    h10-hailort \
    h10-hailort-pcie-driver \
    python3-h10-hailort \
    hailo-tappas-core \
    python3-hailo-tappas \
    hailo-models \
    rpicam-apps-hailo-postprocess

# --------------------------------------------------------------
# 5. Python project dependencies (RPi5 adapted)
# --------------------------------------------------------------
info "Installing Python dependencies into virtualenv..."

VENV_DIR="/opt/ai_venv"
python3 -m venv --system-site-packages "$VENV_DIR"
PIP="$VENV_DIR/bin/pip install"

# Activate for subsequent python3 calls in this script
export PATH="$VENV_DIR/bin:$PATH"

# Core AI / Vision
$PIP \
    "ultralytics==8.3.161" \
    "onnx==1.16.2" \
    "onnxruntime==1.26.0" \
    "onnxsim==0.4.36" \
    "numpy==1.26.4" \
    "opencv-python==4.10.0.84"

# API Backend
$PIP \
    "fastapi==0.115.5" \
    "uvicorn[standard]==0.32.1" \
    "pydantic==2.9.2" \
    "pydantic-settings==2.6.1" \
    "python-dotenv==1.0.1" \
    "websockets==13.1" \
    "httpx==0.27.2"

# Database
$PIP \
    "supabase==2.9.1" \
    "postgrest==0.18.0"

# Tracking
$PIP \
    "lap==0.4.0" \
    "scipy==1.14.1"

# Utilities
$PIP \
    "Pillow==10.4.0" \
    "requests==2.32.3" \
    "python-multipart==0.0.12" \
    "aiofiles==24.1.0" \
    "psutil==5.9.6"

info "Virtualenv ready at $VENV_DIR"
info "Activate it with: source $VENV_DIR/bin/activate"

# --------------------------------------------------------------
# 6. Verify installations
# --------------------------------------------------------------
info "Running verification checks..."

echo ""
echo "--- Camera ---"
if libcamera-hello --list-cameras 2>&1 | grep -q "ov5647\|imx"; then
    echo -e "  ${GREEN}OK${NC} Camera detected"
else
    echo -e "  ${YELLOW}WARN${NC} No camera detected (check ribbon cable / enable CSI)"
fi

echo ""
echo "--- Hailo-10H ---"
if lspci | grep -qi hailo; then
    echo -e "  ${GREEN}OK${NC} Hailo device on PCIe bus"
else
    echo -e "  ${RED}FAIL${NC} Hailo device not found on PCIe"
fi

if ls /dev/hailo* &>/dev/null; then
    echo -e "  ${GREEN}OK${NC} Hailo device node: $(ls /dev/hailo*)"
else
    echo -e "  ${YELLOW}WARN${NC} /dev/hailo* not found — driver may need a reboot"
fi

echo ""
echo "--- Python bindings ---"
python3 -c "import hailo_platform; print('  hailo_platform OK')" 2>/dev/null \
    || echo -e "  ${YELLOW}WARN${NC} hailo_platform not importable yet (reboot may be required)"

python3 -c "import picamera2; print('  picamera2 OK')" 2>/dev/null \
    || echo -e "  ${YELLOW}WARN${NC} picamera2 not importable"

python3 -c "import cv2; print('  opencv', cv2.__version__, 'OK')" 2>/dev/null \
    || echo -e "  ${RED}FAIL${NC} opencv not importable"

python3 -c "import ultralytics; print('  ultralytics OK')" 2>/dev/null \
    || echo -e "  ${RED}FAIL${NC} ultralytics not importable"

python3 -c "import fastapi; print('  fastapi OK')" 2>/dev/null \
    || echo -e "  ${RED}FAIL${NC} fastapi not importable"

echo ""
info "Installation complete. A reboot is recommended to load the Hailo PCIe driver."
echo -e "  Run: ${YELLOW}sudo reboot${NC}"
