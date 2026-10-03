#!/bin/bash
# Deploy respeaker-dsp-console from this Mac to the Pi with the HAT.
#
#   tools/deploy.sh [install|sync|test|status|stop|start|logs]
#
#   install  (default) rsync, run ./install.sh on the Pi, show its status
#   sync     rsync the tree only; the running service is not touched
#   test     rsync, then run the test suite on the Pi
#   status   service state and /health; no sync
#   stop     stop the service
#   start    start the service
#   logs     follow the service journal
#
# Git lives on the Mac. The tree lands in ~/respeaker-dsp-console on the Pi
# as plain files; --delete keeps it an exact copy, which is why the Pi's own
# settings live in /etc/default/respeaker-console and /var/lib/respeaker-console,
# not in the tree. The downloaded model is excluded, so --delete leaves it.
#
# Host is an ssh alias (see ~/.ssh/config); override with
#   RESPEAKER_PI_SSH=... tools/deploy.sh ...
set -euo pipefail

PI=${RESPEAKER_PI_SSH:-pi0}
DIR=respeaker-dsp-console
PORT=${RESPEAKER_HTTP_PORT:-8082}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

action=${1:-install}

EXCLUDES=(--exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store
          --exclude .vscode --exclude '*.swp' --exclude 'models/*.tflite'
          --exclude settings.json)

do_sync() {
    echo "== sync -> $PI:~/$DIR"
    rsync -az --delete "${EXCLUDES[@]}" ./ "$PI:$DIR/"
}

case "$action" in
  sync)   do_sync ;;
  install)
    do_sync
    echo "== install on $PI"
    ssh -t "$PI" "cd ~/$DIR && ./install.sh"
    ;;
  test)
    do_sync
    echo "== test on $PI"
    # Fake capture and a fake classifier: no HAT needed, and it does not
    # fight the running service for the card.
    ssh "$PI" "cd ~/$DIR && ./tests/run_tests.sh"
    ;;
  status)
    ssh "$PI" "systemctl status -n 0 --no-pager respeaker-console; echo; curl -s localhost:$PORT/health | python3 -m json.tool"
    ;;
  stop)   ssh -t "$PI" "sudo systemctl stop respeaker-console" ;;
  start)  ssh -t "$PI" "sudo systemctl start respeaker-console" ;;
  logs)   ssh -t "$PI" "journalctl -u respeaker-console -f -n 50" ;;
  *)      sed -n '2,22p' "$0"; exit 2 ;;
esac
