#!/bin/bash
# Install the ReSpeaker sound daemon as a system service on this Pi.
#
#   ./install.sh
#
# Run it as the user the service should run as, not with sudo: the unit is
# written for whoever runs this. Safe to re-run; it is what tools/deploy.sh
# runs after every sync.
set -e

SERVICE_NAME="respeaker-console"
SCRIPT_NAME="server.py"

if [ "$EUID" -eq 0 ]; then
    echo "Run as a regular user, not sudo — the unit runs as whoever installs it."
    exit 1
fi

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER=$(whoami)
CARD=$(grep -E '^RESPEAKER_CARD=' /etc/default/${SERVICE_NAME} 2>/dev/null | cut -d= -f2 | tr -d '"')
CARD=${CARD:-seeed2micvoicec}

echo "Install directory: ${INSTALL_DIR}"
echo "User:              ${RUN_USER}"
echo "Card:              ${CARD}"

# ── packages ─────────────────────────────────────────────────────────────
# numpy from apt, not pip: on a Pi Zero a pip build of numpy takes an hour.
need=()
/usr/bin/python3 -c "import numpy" 2>/dev/null || need+=(python3-numpy)
command -v arecord >/dev/null || need+=(alsa-utils)
command -v curl >/dev/null || need+=(curl)
/usr/bin/python3 -m pip --version >/dev/null 2>&1 || need+=(python3-pip)
if [ ${#need[@]} -gt 0 ]; then
    echo "Installing: ${need[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${need[@]}"
fi

# ── sound ID runtime (optional) ──────────────────────────────────────────
# ai-edge-litert publishes aarch64 wheels for CPython 3.11 and 3.12 only:
# 64-bit Raspberry Pi OS Bookworm (3.11) or Ubuntu 24.04 (3.12). Anything
# else runs without sound ID; direction finding does not need it.
PY=$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
ARCH=$(uname -m)
if /usr/bin/python3 -c "import ai_edge_litert" 2>/dev/null; then
    echo "ai-edge-litert present"
elif [ "$ARCH" = "aarch64" ] && { [ "$PY" = "3.11" ] || [ "$PY" = "3.12" ]; }; then
    echo "Installing ai-edge-litert for ${RUN_USER} (Python ${PY})"
    /usr/bin/python3 -m pip install --user --break-system-packages ai-edge-litert || \
        echo "WARNING: ai-edge-litert install failed — sound ID will be off"
else
    echo "WARNING: no LiteRT wheel for ${ARCH} / Python ${PY} — sound ID will be off."
    echo "         It needs a 64-bit OS with Python 3.11 or 3.12."
fi

if [ ! -s "${INSTALL_DIR}/models/yamnet.tflite" ]; then
    "${INSTALL_DIR}/download_model.sh" || echo "WARNING: model download failed — sound ID will be off"
fi

# ── hardware check ───────────────────────────────────────────────────────
if ! arecord -l 2>/dev/null | grep -q "${CARD}"; then
    echo
    echo "WARNING: no ALSA card named '${CARD}'. arecord -l shows:"
    arecord -l 2>/dev/null | grep '^card' || echo "  (no capture cards)"
    echo "  The v2.0 HAT needs its overlay in /boot/firmware/config.txt:"
    echo "      dtoverlay=respeaker-2mic-v2_0"
    echo "  then a reboot. If the card has another name, set RESPEAKER_CARD and"
    echo "  RESPEAKER_DEVICE in /etc/default/${SERVICE_NAME}."
    echo
fi

# ── config ───────────────────────────────────────────────────────────────
if [ ! -f /etc/default/${SERVICE_NAME} ]; then
    echo "Installing /etc/default/${SERVICE_NAME} from config/${SERVICE_NAME}.env.example"
    sudo install -m 644 "${INSTALL_DIR}/config/${SERVICE_NAME}.env.example" /etc/default/${SERVICE_NAME}
else
    echo "/etc/default/${SERVICE_NAME} exists — left alone"
fi

# ── unit ─────────────────────────────────────────────────────────────────
echo "Installing ${SERVICE_NAME}.service"
sed -e "s#^User=.*#User=${RUN_USER}#" \
    -e "s#^WorkingDirectory=.*#WorkingDirectory=${INSTALL_DIR}#" \
    -e "s#^ExecStart=.*#ExecStart=/usr/bin/python3 ${INSTALL_DIR}/${SCRIPT_NAME}#" \
    "${INSTALL_DIR}/systemd/${SERVICE_NAME}.service" | sudo tee /etc/systemd/system/${SERVICE_NAME}.service >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}
sleep 2
systemctl --no-pager -l status ${SERVICE_NAME} | head -15

PORT=$(grep -E '^RESPEAKER_HTTP_PORT=' /etc/default/${SERVICE_NAME} 2>/dev/null | cut -d= -f2 | tr -d '"')
echo
echo "  http://$(hostname).local:${PORT:-8082}/"
