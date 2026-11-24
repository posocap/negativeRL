import unittest
from core import Limiter, Matcher, UtteranceTracker, normalize, load_words


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


class T(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Sh!t,"), "sht")
        self.assertEqual(normalize("DAMN."), "damn")

    def test_matcher(self):
        m = Matcher({"damn", "shit"})
        self.assertTrue(m.hit("Damn!"))
        self.assertTrue(m.hit("shitty"))
        self.assertFalse(m.hit("damnation"))
        self.assertFalse(m.hit("class"))
        self.assertFalse(Matcher({"damn"}, min_conf=0.8).hit("damn", 0.5))
        self.assertTrue(Matcher({"damn"}, min_conf=0.8).hit("damn", 0.9))

    def test_default_list_loads(self):
        self.assertIn("fuck", load_words("words.txt"))

    def test_limiter(self):
        c = Clock()
        l = Limiter(cooldown_s=10, per_min=2, per_hour=3, clock=c)
        self.assertTrue(l.allow()); l.record()
        c.t = 5;  self.assertFalse(l.allow())        # cooldown
        c.t = 11; self.assertTrue(l.allow()); l.record()
        c.t = 22; self.assertFalse(l.allow())        # per-minute cap (2 within 60s)
        c.t = 80; self.assertTrue(l.allow()); l.record()
        c.t = 100; self.assertFalse(l.allow())       # per-hour cap (3)
        c.t = 4000; self.assertTrue(l.allow())       # window expired

    def test_tracker_holds_last_word_and_dedupes(self):
        u = UtteranceTracker()
        self.assertEqual(u.update("oh damn"), ["oh"])
        self.assertEqual(u.update("oh damn it"), ["damn"])
        self.assertEqual(u.update("oh damn it", final=True), ["it"])
        self.assertEqual(u.update("hello", final=True), ["hello"])  # reset


if __name__ == "__main__":
    unittest.main()
