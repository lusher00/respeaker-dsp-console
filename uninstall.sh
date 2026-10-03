#!/bin/bash
# Remove the respeaker-console service. Leaves the tree, /etc/default and the
# saved settings in /var/lib/respeaker-console unless --purge.
set -e
SERVICE_NAME="respeaker-console"
sudo systemctl disable --now ${SERVICE_NAME} 2>/dev/null || true
sudo rm -f /etc/systemd/system/${SERVICE_NAME}.service
sudo systemctl daemon-reload
if [ "${1:-}" = "--purge" ]; then
    sudo rm -f /etc/default/${SERVICE_NAME}
    sudo rm -rf /var/lib/${SERVICE_NAME}
    echo "Removed service, /etc/default/${SERVICE_NAME} and saved settings."
else
    echo "Removed service. Config and settings kept (--purge to remove)."
fi
