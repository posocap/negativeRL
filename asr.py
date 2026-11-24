"""Streaming ASR wrappers. Each yields (text, final) from feed(float32 samples)."""
import json
import os

import numpy as np


class SherpaASR:
    name = "sherpa"

    def __init__(self, model_dir: str, threads: int = 2):
        import sherpa_onnx
        def pick(prefix):
            for f in sorted(os.listdir(model_dir)):
                if f.startswith(prefix) and f.endswith(".int8.onnx"):
                    return os.path.join(model_dir, f)
            for f in sorted(os.listdir(model_dir)):
                if f.startswith(prefix) and f.endswith(".onnx"):
                    return os.path.join(model_dir, f)
            raise SystemExit(f"No {prefix}*.onnx in {model_dir}")
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(model_dir, "tokens.txt"),
            encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
            num_threads=threads, sample_rate=16000, decoding_method="greedy_search",
            enable_endpoint_detection=True)
        self.stream = self.rec.create_stream()

    def _text(self):
        r = self.rec.get_result(self.stream)
        return (r if isinstance(r, str) else r.text).strip().lower()

    def feed(self, samples: np.ndarray):
        self.stream.accept_waveform(16000, samples)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = self._text()
        if self.rec.is_endpoint(self.stream):
            self.rec.reset(self.stream)
            return [(text, True)] if text else []
        return [(text, False)] if text else []

    def flush(self):
        self.stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
        self.stream.input_finished()
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = self._text()
        return [(text, True)] if text else []


class VoskASR:
    """Vosk small-en. Optional grammar restricts vocabulary; [unk] absorbs the rest."""
    name = "vosk"

    def __init__(self, model_dir: str, grammar_words=None):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        model = Model(model_dir)
        if grammar_words:
            g = json.dumps(sorted(grammar_words) + ["[unk]"])
            self.rec = KaldiRecognizer(model, 16000, g)
        else:
            self.rec = KaldiRecognizer(model, 16000)
        self.rec.SetWords(True)
        self.last_conf = {}

    def feed(self, samples: np.ndarray):
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        if self.rec.AcceptWaveform(pcm):
            res = json.loads(self.rec.Result())
            self.last_conf = {w["word"]: w.get("conf", 1.0) for w in res.get("result", [])}
            return [(res.get("text", "").strip(), True)] if res.get("text") else []
        part = json.loads(self.rec.PartialResult()).get("partial", "").strip()
        return [(part, False)] if part else []

    def flush(self):
        res = json.loads(self.rec.FinalResult())
        self.last_conf = {w["word"]: w.get("conf", 1.0) for w in res.get("result", [])}
        return [(res.get("text", "").strip(), True)] if res.get("text") else []
