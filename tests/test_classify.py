import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from classify import WINDOW, YamNet, load_labels  # noqa: E402

LABELS = os.path.join(os.path.dirname(HERE), "models", "yamnet_class_map.csv")


class FakeInterp:
    """Mimics the LiteRT Interpreter surface YamNet uses."""

    def __init__(self, in_shape, outs, hot=58):
        self.in_shape = list(in_shape)
        self.outs = outs                  # list of output shapes
        self.hot = hot
        self.resized = None
        self.x = None

    def get_input_details(self):
        return [{"index": 0, "shape": np.array(self.in_shape), "shape_signature": np.array(self.in_shape)}]

    def get_output_details(self):
        return [{"index": 10 + i, "shape": np.array(s)} for i, s in enumerate(self.outs)]

    def resize_tensor_input(self, i, shape):
        self.resized = list(shape)

    def allocate_tensors(self):
        pass

    def set_tensor(self, i, x):
        self.x = x

    def invoke(self):
        pass

    def get_tensor(self, i):
        shape = self.outs[i - 10]
        out = np.zeros(shape, np.float32)
        if shape[-1] == 521:
            out[..., self.hot] = 0.8
            out[..., 0] = 0.1
        return out


class ClassifyTest(unittest.TestCase):
    def test_labels(self):
        lab = load_labels(LABELS)
        self.assertEqual(len(lab), 521)
        self.assertEqual(lab[0], "Speech")

    def test_tfhub_shape_and_output_pick(self):
        it = FakeInterp([WINDOW], [[1, 1024], [1, 521], [96, 64]])
        y = YamNet("x", LABELS, interpreter=it)
        self.assertTrue(y.ready)
        s = y.scores(np.zeros(20000, np.float32))
        self.assertEqual(it.x.shape, (WINDOW,))
        top = y.top(s, 2)
        self.assertEqual(top[0]["score"], 0.8)
        self.assertEqual(top[1]["label"], "Speech")

    def test_batched_input_and_ignore(self):
        it = FakeInterp([1, WINDOW], [[1, 521]], hot=0)
        y = YamNet("x", LABELS, interpreter=it)
        y.scores(np.zeros(WINDOW, np.float32))
        self.assertEqual(it.x.shape, (1, WINDOW))
        self.assertNotIn("Speech", [c["label"] for c in y.top(y.scores(np.zeros(WINDOW)), 3, {"Speech"})])

    def test_resizes_wrong_input(self):
        it = FakeInterp([1, 1], [[1, 521]])
        YamNet("x", LABELS, interpreter=it)
        self.assertEqual(it.resized, [1, WINDOW])

    def test_wrong_model_is_reported(self):
        y = YamNet("x", LABELS, interpreter=FakeInterp([WINDOW], [[1, 10]]))
        self.assertFalse(y.ready)
        self.assertIn("521", y.error)

    def test_missing_model_is_not_fatal(self):
        y = YamNet("/nonexistent.tflite", LABELS)
        self.assertFalse(y.ready)
        self.assertTrue(y.error)


if __name__ == "__main__":
    unittest.main()
