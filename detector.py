#!/usr/bin/env python3
"""Profanity detector prototype: mic/wav -> streaming ASR -> word match -> trigger.

Dry-run by default. Pass --live to make triggers actually fire.
"""
import argparse
import queue
import sys
import wave

import numpy as np

from core import AudioRing, KillSwitch, Limiter, Matcher, SpeakerGate, UtteranceTracker, load_words
import triggers


def build_asr(args, words):
    if args.asr == "sherpa":
        from asr import SherpaASR
        return SherpaASR(args.model, threads=args.threads)
    from asr import VoskASR
    return VoskASR(args.model, grammar_words=words if args.grammar else None)


def build_trigger(args):
    if args.trigger == "gpio":
        t = triggers.GpioTrigger(args.pin, args.pulse_ms)
    elif args.trigger == "pavlok":
        t = triggers.PavlokTrigger(args.strength)
    else:
        t = triggers.LogTrigger()
    return t if args.live or args.trigger == "log" else triggers.DryRun(t)


def wav_chunks(path, chunk=1600):
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            sys.exit("WAV must be 16 kHz, mono, 16-bit (ffmpeg -i in.x -ar 16000 -ac 1 out.wav)")
        while True:
            data = w.readframes(chunk)
            if not data:
                return
            yield np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def mic_chunks(kill, chunk=1600):
    import sounddevice as sd
    q = queue.Queue()
    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=chunk, callback=cb):
        while not kill.tripped():
            try:
                yield q.get(timeout=0.2)
            except queue.Empty:
                continue


def run(args):
    words = load_words(args.words)
    matcher = Matcher(words, min_conf=args.min_conf)
    limiter = Limiter(args.cooldown, args.per_min, args.per_hour)
    kill = KillSwitch(args.kill_file)
    kill.install()
    asr = build_asr(args, words)
    trig = build_trigger(args)
    tracker = UtteranceTracker()
    ring = AudioRing(max(6.0, args.verify_window + 1))
    gate = None
    if args.verify:
        from speaker import SherpaEmbedder
        if not args.verify_model:
            sys.exit("--verify needs --verify-model (speaker embedding .onnx)")
        gate = SpeakerGate(np.load(args.verify), SherpaEmbedder(args.verify_model),
                           args.verify_threshold)
    mode = "LIVE" if args.live else "dry-run"
    print(f"[start] asr={asr.name} trigger={args.trigger} mode={mode} words={len(words)}", flush=True)

    def handle(events):
        for text, final in events:
            conf = getattr(asr, "last_conf", {}) if final else {}
            for w in tracker.update(text, final):
                print(f"  heard: {w}", flush=True) if args.verbose else None
                if matcher.hit(w, conf.get(w)):
                    if gate:
                        ok, score = gate.check(ring.last(args.verify_window))
                        shown = "n/a" if score is None else f"{score:.2f}"
                        print(f"[verify] {w!r} similarity={shown} threshold={args.verify_threshold} "
                              f"-> {'match' if ok else 'REJECTED'}", flush=True)
                        if not ok:
                            continue
                    if limiter.allow():
                        limiter.record()
                        trig.fire(w)
                    else:
                        print(f"[limited] suppressed {w!r}", flush=True)

    source = wav_chunks(args.wav) if args.wav else mic_chunks(kill)
    for chunk in source:
        if kill.tripped():
            break
        ring.add(chunk)
        handle(asr.feed(chunk))
    handle(asr.flush())
    if hasattr(trig, "close"):
        trig.close()
    print("[stop]", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asr", choices=["sherpa", "vosk"], default="sherpa")
    p.add_argument("--model", required=True, help="model directory")
    p.add_argument("--words", default="words.txt")
    p.add_argument("--wav", help="run on a 16 kHz mono 16-bit WAV instead of the mic")
    p.add_argument("--trigger", choices=["log", "gpio", "pavlok"], default="log")
    p.add_argument("--live", action="store_true", help="actually fire triggers (default: dry-run)")
    p.add_argument("--min-conf", type=float, default=0.6, help="Vosk final-word confidence floor")
    p.add_argument("--grammar", action="store_true", help="Vosk: restrict vocab to word list + [unk]")
    p.add_argument("--cooldown", type=float, default=10.0)
    p.add_argument("--per-min", type=int, default=3)
    p.add_argument("--per-hour", type=int, default=20)
    p.add_argument("--kill-file", default="KILL", help="touch this file to stop")
    p.add_argument("--pin", type=int, default=17, help="GPIO BCM pin")
    p.add_argument("--pulse-ms", type=int, default=100)
    p.add_argument("--strength", type=int, default=20, help="Pavlok 1-100 (capped at 50)")
    p.add_argument("--verify", metavar="VOICE.NPY", help="only fire if speaker matches enrolled voice")
    p.add_argument("--verify-model", help="speaker embedding .onnx (see README)")
    p.add_argument("--verify-threshold", type=float, default=0.5, help="cosine similarity, 0-1")
    p.add_argument("--verify-window", type=float, default=3.0, help="seconds of audio to embed per hit")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("-v", "--verbose", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
