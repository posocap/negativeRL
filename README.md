# negativeRL
Negative reinforcement using hot-word detection and electricity. 


Mic or WAV -> streaming ASR -> word-list match -> trigger (log / GPIO / Pavlok).
**Dry-run by default**: nothing fires unless you pass `--live`.

## Setup
    pip install -r requirements.txt          # drop gpiozero off-Pi
    python -m unittest test_core -v          # no models needed

## Models (download manually)
- sherpa-onnx streaming Zipformer (English): pick a small `streaming-zipformer` model from
  https://github.com/k2-fsa/sherpa-onnx/releases (asr-models), untar, pass dir as `--model`.
  Folder needs tokens.txt + encoder/decoder/joiner .onnx (int8 preferred).
- Vosk fallback: https://alphacephei.com/vosk/models -> `vosk-model-small-en-us-0.15`.

- Speaker model (for `--verify`): download a pretrained speaker-embedding ONNX from
  https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
  (e.g. `wespeaker_en_voxceleb_resnet34.onnx` or a 3D-Speaker model); pass the file path.

## Run
    python detector.py --asr sherpa --model ./zipformer-en                  # mic, dry-run, log
    python detector.py --asr vosk --model ./vosk-model-small-en-us-0.15 --grammar -v
    python detector.py --model ./zipformer-en --wav sample.wav -v           # test w/o mic
    python detector.py --model ./zipformer-en --trigger gpio --pin 17 --live
    PAVLOK_TOKEN=... python detector.py --model ./zipformer-en --trigger pavlok --live

## Voice enrollment / speaker gating (only fire on YOUR voice)
    python enroll.py --model ./wespeaker_en_voxceleb_resnet34.onnx            # records 30 s -> voice.npy
    python detector.py --model ./zipformer-en --verify voice.npy --verify-model ./wespeaker_en_voxceleb_resnet34.onnx -v
- Each hit embeds the last `--verify-window` s (default 3) and fires only if cosine similarity
  >= `--verify-threshold` (default 0.5). Scores print on every hit, including dry-run: say the
  words yourself and note your scores, then say them from a speaker/other person and compare.
  `enroll.py` prints the worst chunk similarity; set the threshold a bit below it.
- Unverifiable (<0.5 s audio) fails closed. The window is a recent-audio approximation (no word
  timestamps), so a final result delayed past the window can score low; raise the window if needed.

## Pavlok key test
    PAVLOK_TOKEN=... python pavlok_test.py --yes        # one zap, strength 10 (cap 50)
Tries `Authorization: Bearer <key>`, falls back to the bare key on 401, prints status/body.

## Safety
- Cooldown 10 s, 3/min, 20/hour (`--cooldown --per-min --per-hour`).
- Stop: Ctrl-C/SIGTERM or `touch KILL`.
- GPIO pulse capped at 300 ms, Pavlok strength capped at 50 in code. These are software
  limits only: the real shock stage needs hardware current/energy limiting + physical off switch.
- Whole-word match only; partials hold back the last word so "ass" never fires on "assignment".
- Pavlok: endpoint POST https://api.pavlok.com/api/v5/stimulus/send, body
  {"stimulus":{"stimulusType":"zap","stimulusValue":1-100}}. The Authorization value format
  wasn't in the docs page we could read: default is `Bearer <key>`; set PAVLOK_AUTH_SCHEME=""
  for a bare key. Verify against https://pavlok.readme.io before using.
- Sherpa gives no per-word confidence; `--min-conf` only applies to Vosk finals.

## Next: ESP32-S3 port
Replace ASR + matcher with ESP-SR MultiNet (custom English commands from words.txt, up to
~200) on a XIAO ESP32-S3 Sense (PDM mic). Reuse the same limiter/cooldown logic in C++, and
send the trigger to a forearm ESP over ESP-NOW. Keep this Python version as the accuracy
baseline for tuning the word list.
