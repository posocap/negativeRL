#!/usr/bin/env python3
"""Enroll your voice: record ~30 s (or --wav) -> speaker embedding -> voice.npy.

Read normally (include some of the words you'll be flagged on, at your usual volume).
Embeds 3 s chunks and averages them; prints the worst chunk-vs-mean similarity to help
pick --verify-threshold (set it a bit below that number).
"""
import argparse
import sys
import wave

import numpy as np

from core import cosine
from speaker import SherpaEmbedder

RATE = 16000


def read_wav(path):
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            sys.exit("WAV must be 16 kHz, mono, 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def record(seconds):
    import sounddevice as sd
    print(f"Recording {seconds}s... speak naturally now.", flush=True)
    a = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    return a[:, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="speaker embedding .onnx file")
    p.add_argument("--wav")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--chunk", type=float, default=3.0)
    p.add_argument("--out", default="voice.npy")
    a = p.parse_args()

    audio = read_wav(a.wav) if a.wav else record(a.seconds)
    n = int(a.chunk * RATE)
    chunks = [audio[i:i + n] for i in range(0, len(audio) - n + 1, n)]
    if len(chunks) < 3:
        sys.exit("Need at least ~9 s of audio.")
    emb = SherpaEmbedder(a.model)
    vecs = np.stack([emb.embed(c) for c in chunks])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    mean = vecs.mean(axis=0)
    sims = [cosine(mean, v) for v in vecs]
    np.save(a.out, mean)
    print(f"saved {a.out}  chunks={len(chunks)}  similarity min={min(sims):.2f} mean={np.mean(sims):.2f}")
    print("Suggest --verify-threshold a little below the min above (default 0.5).")


if __name__ == "__main__":
    main()
