"""Model-free core: normalization, word matching, rate limiting, kill switch."""
import os
import re
import signal
import time
from collections import deque

_NON_ALPHA = re.compile(r"[^a-z']")
_SUFFIXES = ("", "s", "es", "ing", "in", "ed", "er", "ers", "y", "ty")


def normalize(word: str) -> str:
    return _NON_ALPHA.sub("", word.lower()).strip("'")


def load_words(path: str) -> set:
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                words.add(normalize(line))
    words.discard("")
    return words


class Matcher:
    """Exact-word match against the list plus simple suffix variants.

    Whole-word only (no substring matching), so 'class' never matches 'ass'.
    """

    def __init__(self, words, min_conf: float = 0.0, use_variants: bool = True):
        self.min_conf = min_conf
        self.targets = set()
        for w in words:
            if use_variants:
                self.targets.update(w + s for s in _SUFFIXES)
            else:
                self.targets.add(w)

    def hit(self, word: str, conf=None) -> bool:
        if conf is not None and conf < self.min_conf:
            return False
        return normalize(word) in self.targets


class Limiter:
    """Cooldown + per-minute + per-hour caps. Clock is injectable for tests."""

    def __init__(self, cooldown_s=10.0, per_min=3, per_hour=20, clock=time.monotonic):
        self.cooldown_s, self.per_min, self.per_hour = cooldown_s, per_min, per_hour
        self.clock = clock
        self._events = deque()

    def allow(self) -> bool:
        now = self.clock()
        while self._events and now - self._events[0] > 3600:
            self._events.popleft()
        if self._events and now - self._events[-1] < self.cooldown_s:
            return False
        if sum(1 for t in self._events if now - t <= 60) >= self.per_min:
            return False
        if len(self._events) >= self.per_hour:
            return False
        return True

    def record(self):
        self._events.append(self.clock())


class KillSwitch:
    """Stops on SIGINT/SIGTERM or when the kill file exists."""

    def __init__(self, path="KILL"):
        self.path = path
        self._flag = False

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self, "_flag", True))

    def tripped(self) -> bool:
        return self._flag or os.path.exists(self.path)


class UtteranceTracker:
    """Turns repeated partial transcripts into each word exactly once.

    Partials hold back the last word (it may still change, e.g. 'ass' ->
    'assignment'); the endpoint flush releases it.
    """

    def __init__(self):
        self.seen = 0

    def update(self, text: str, final: bool = False):
        words = text.split()
        ready = words if final else words[:-1]
        new = ready[self.seen:]
        self.seen = max(self.seen, len(ready))
        if final:
            self.seen = 0
        return new


# ---- speaker gating (model-free parts) ----
import numpy as _np


def cosine(a, b) -> float:
    a = _np.asarray(a, dtype=_np.float64).ravel()
    b = _np.asarray(b, dtype=_np.float64).ravel()
    na, nb = _np.linalg.norm(a), _np.linalg.norm(b)
    if na == 0 or nb == 0 or a.shape != b.shape:
        return 0.0
    return float(a @ b / (na * nb))


class AudioRing:
    """Keeps the most recent `seconds` of 16 kHz mono float32 audio."""

    def __init__(self, seconds=6.0, rate=16000):
        self.rate, self.cap = rate, int(seconds * rate)
        self.buf = _np.zeros(0, dtype=_np.float32)

    def add(self, samples):
        self.buf = _np.concatenate([self.buf, _np.asarray(samples, dtype=_np.float32)])[-self.cap:]

    def last(self, seconds):
        return self.buf[-int(seconds * self.rate):]


class SpeakerGate:
    """Passes only if the recent audio's embedding is close enough to the enrolled voice.

    embedder: object with embed(float32 samples) -> vector. ref: enrolled vector.
    Audio shorter than min_s fails closed (score None).
    """

    def __init__(self, ref, embedder, threshold=0.5, min_s=0.5, rate=16000):
        self.ref, self.embedder, self.threshold = _np.asarray(ref), embedder, threshold
        self.min_samples = int(min_s * rate)

    def check(self, samples):
        if len(samples) < self.min_samples:
            return False, None
        score = cosine(self.ref, self.embedder.embed(samples))
        return score >= self.threshold, score
