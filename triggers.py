"""Trigger backends. All strength/pulse values are hard-capped here."""
import json
import os
import time
import urllib.request

MAX_PULSE_MS = 300      # hard cap regardless of CLI
MAX_PAVLOK_STRENGTH = 50


class LogTrigger:
    def fire(self, word: str):
        print(f"[trigger:log] would fire on {word!r}", flush=True)


class DryRun:
    """Wraps a real backend; logs instead of firing (default mode)."""

    def __init__(self, inner):
        self.inner = inner

    def fire(self, word: str):
        print(f"[dry-run] would fire {type(self.inner).__name__} on {word!r}", flush=True)


class GpioTrigger:
    def __init__(self, pin: int, pulse_ms: int = 100, active_high: bool = True):
        from gpiozero import OutputDevice  # imported lazily: Pi only
        self.dev = OutputDevice(pin, active_high=active_high, initial_value=False)
        self.pulse_s = min(pulse_ms, MAX_PULSE_MS) / 1000.0

    def fire(self, word: str):
        self.dev.on()
        time.sleep(self.pulse_s)
        self.dev.off()
        print(f"[trigger:gpio] pulse {self.pulse_s*1000:.0f} ms on {word!r}", flush=True)

    def close(self):
        self.dev.off()
        self.dev.close()


class PavlokTrigger:
    """POST https://api.pavlok.com/api/v5/stimulus/send (per Pavlok docs).

    Auth value format is NOT specified in the docs page we could read.
    TODO: confirm; set PAVLOK_AUTH_SCHEME to "" for a bare key, default "Bearer".
    """

    URL = "https://api.pavlok.com/api/v5/stimulus/send"

    def __init__(self, strength: int = 20, kind: str = "zap"):
        self.token = os.environ.get("PAVLOK_TOKEN")
        if not self.token:
            raise SystemExit("Set PAVLOK_TOKEN env var for --trigger pavlok")
        scheme = os.environ.get("PAVLOK_AUTH_SCHEME", "Bearer")
        self.auth = f"{scheme} {self.token}".strip()
        self.kind = kind
        self.strength = max(1, min(strength, MAX_PAVLOK_STRENGTH))

    def fire(self, word: str):
        body = json.dumps({"stimulus": {
            "stimulusType": self.kind,
            "stimulusValue": self.strength,
            "reason": "profanity detector",
        }}).encode()
        req = urllib.request.Request(self.URL, data=body, method="POST", headers={
            "Authorization": self.auth, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                print(f"[trigger:pavlok] {r.status} on {word!r}", flush=True)
        except Exception as e:  # never crash the listener
            print(f"[trigger:pavlok] ERROR {e}", flush=True)
