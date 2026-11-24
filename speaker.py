"""Speaker-embedding wrapper (sherpa-onnx). Needs a pretrained model file, e.g. WeSpeaker/3D-Speaker."""
import numpy as np


class SherpaEmbedder:
    def __init__(self, model_path: str, threads: int = 1):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=threads)
        if hasattr(cfg, "validate") and not cfg.validate():
            raise SystemExit(f"Bad speaker model config: {model_path}")
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        s = self.ext.create_stream()
        s.accept_waveform(16000, np.asarray(samples, dtype=np.float32))
        s.input_finished()
        return np.array(self.ext.compute(s), dtype=np.float32)
