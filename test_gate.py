import json
import unittest
import urllib.error

import numpy as np

import pavlok_test
from core import AudioRing, SpeakerGate, cosine


class Emb:
    def __init__(self, v): self.v = np.asarray(v, dtype=np.float32)
    def embed(self, s): return self.v


class GateTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)
        self.assertEqual(cosine([1, 0], [1, 0, 0]), 0.0)

    def test_threshold(self):
        ref = [1.0, 0.0]
        audio = np.zeros(16000, dtype=np.float32)
        ok, s = SpeakerGate(ref, Emb([1, 0.1]), 0.5).check(audio)
        self.assertTrue(ok and s > 0.9)
        ok, s = SpeakerGate(ref, Emb([0.2, 1.0]), 0.5).check(audio)
        self.assertFalse(ok)
        self.assertTrue(SpeakerGate(ref, Emb([0.2, 1.0]), 0.1).check(audio)[0])

    def test_short_audio_fails_closed(self):
        ok, s = SpeakerGate([1, 0], Emb([1, 0]), 0.5).check(np.zeros(100, dtype=np.float32))
        self.assertFalse(ok); self.assertIsNone(s)

    def test_ring(self):
        r = AudioRing(seconds=1.0)
        r.add(np.ones(16000 * 2, dtype=np.float32))
        self.assertEqual(len(r.buf), 16000)
        self.assertEqual(len(r.last(0.5)), 8000)


class PavlokFallback(unittest.TestCase):
    def test_bearer_then_bare(self):
        seen = []

        class R:
            status = 200
            def __enter__(s): return s
            def __exit__(s, *a): pass
            def read(s): return b"{}"

        def opener(req, timeout=0):
            seen.append(req.get_header("Authorization"))
            if len(seen) == 1:
                raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)
            return R()

        status, _ = pavlok_test.run("KEY", 10, opener)
        self.assertEqual(status, 200)
        self.assertEqual(seen, ["Bearer KEY", "KEY"])


if __name__ == "__main__":
    unittest.main()
