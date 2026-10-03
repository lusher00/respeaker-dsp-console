"""Sound identification with YAMNet (AudioSet, 521 classes) on LiteRT.

Input: 15600 samples (0.975 s) of mono float32 at 16 kHz, -1..1.
Output: 521 scores, one per class in models/yamnet_class_map.csv order.

The model file is not in the repo; download_model.sh fetches it on the Pi.
Without it, or without a TFLite runtime, the classifier reports itself
unavailable and everything else keeps running.

Runtime lookup order: ai_edge_litert (current name, pip ai-edge-litert),
tflite_runtime (old name), tensorflow.lite. ai-edge-litert ships aarch64
wheels for Python 3.11 and 3.12 only — 64-bit OS required, no armv6/armv7.
"""

import csv
import os
import time

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 15600


def load_labels(path):
    with open(path, newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


def _interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai_edge_litert"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite_runtime"
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter      # type: ignore
        return Interpreter, "tensorflow"
    except ImportError:
        return None, None


class YamNet:
    def __init__(self, model_path, labels_path, threads=2, interpreter=None):
        self.model_path = model_path
        self.labels = load_labels(labels_path)
        self.error = None
        self.runtime = None
        self.interp = None
        self.last_ms = None
        if interpreter is not None:                 # tests inject a fake
            self.interp = interpreter
            self.runtime = "injected"
        else:
            cls, name = _interpreter_class()
            if cls is None:
                self.error = "no TFLite runtime (pip install ai-edge-litert)"
                return
            if not os.path.isfile(model_path):
                self.error = f"model not found: {model_path} (run ./download_model.sh)"
                return
            try:
                self.interp = cls(model_path=model_path, num_threads=int(threads))
            except Exception as e:                  # noqa: BLE001
                self.error = f"model load failed: {e}"
                return
            self.runtime = name
        self._bind()

    def _bind(self):
        it = self.interp
        inp = it.get_input_details()[0]
        shape = list(inp["shape"])
        sig = list(inp.get("shape_signature", shape))
        # YAMNet exports differ: [15600] (TF Hub) or [1, 15600]. A -1 in the
        # signature means the length is dynamic and has to be set once.
        want = [WINDOW] if len(shape) == 1 else [1, WINDOW]
        if shape != want or -1 in sig:
            it.resize_tensor_input(inp["index"], want)
        it.allocate_tensors()
        self.in_index = inp["index"]
        self.in_shape = want
        outs = it.get_output_details()
        # The score tensor is the one whose last dimension is the class count;
        # the full-graph export also returns embeddings (1024) and a spectrogram.
        cand = [o for o in outs if int(o["shape"][-1]) == len(self.labels)]
        if not cand:
            self.error = (f"no output with {len(self.labels)} classes: "
                          f"{[list(o['shape']) for o in outs]}")
            self.interp = None
            return
        self.out_index = cand[0]["index"]

    @property
    def ready(self):
        return self.interp is not None

    def scores(self, wave16k):
        """wave16k: >= WINDOW float32 samples. Returns 521 scores (mean over
        the model's internal patches if it returns more than one row)."""
        x = np.asarray(wave16k[-WINDOW:], np.float32).reshape(self.in_shape)
        t = time.perf_counter()
        self.interp.set_tensor(self.in_index, x)
        self.interp.invoke()
        s = np.array(self.interp.get_tensor(self.out_index), np.float32)
        self.last_ms = round((time.perf_counter() - t) * 1000, 1)
        s = s.reshape(-1, len(self.labels))
        return s.mean(axis=0)

    def top(self, scores, k=5, ignore=()):
        order = np.argsort(scores)[::-1]
        out = []
        for i in order:
            name = self.labels[int(i)]
            if name in ignore:
                continue
            out.append({"label": name, "score": round(float(scores[i]), 3)})
            if len(out) >= k:
                break
        return out
