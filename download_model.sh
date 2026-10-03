#!/bin/bash
# Fetch the YAMNet TFLite model (Google, Apache 2.0) into models/.
#
#   ./download_model.sh            skip if already present
#   ./download_model.sh --force    re-download
#
# 521 AudioSet classes, 0.975 s of 16 kHz mono in, ~4 MB float32. The class
# names are in models/yamnet_class_map.csv (from tensorflow/models), in the
# same order as the model's output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

URL=${YAMNET_URL:-https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite}
OUT=models/yamnet.tflite

if [ -s "$OUT" ] && [ "${1:-}" != "--force" ]; then
    echo "$OUT present ($(stat -c %s "$OUT") bytes) — --force to re-download"
    exit 0
fi

mkdir -p models
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
echo "Downloading $URL"
if command -v curl >/dev/null; then curl -fL --retry 3 -o "$tmp" "$URL"
else wget -O "$tmp" "$URL"; fi

# A TFLite flatbuffer carries "TFL3" at byte 4. An HTML error page does not.
if [ "$(dd if="$tmp" bs=1 skip=4 count=4 2>/dev/null)" != "TFL3" ]; then
    echo "Downloaded file is not a TFLite model — check the URL" >&2
    exit 1
fi
mv "$tmp" "$OUT"
trap - EXIT
echo "Saved $OUT ($(stat -c %s "$OUT") bytes)"
