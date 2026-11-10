# Deepfake Video and Audio Detection

Undergraduate thesis project: a deepfake detector with a **video branch** (face crops + EfficientNet-B0) and an **audio branch** (frozen wav2vec2-base features + small MLP), evaluated with a protocol that was frozen before any test results were seen, and served through a FastAPI/Docker demo.

The evaluation protocol, every amendment to it, and the reasons for each change are in [`docs/protocol.md`](docs/protocol.md). Each test set is evaluated once per checkpoint; the log is [`reports/eval_ledger.csv`](reports/eval_ledger.csv).

## Results

### Video (video-level AUC, mean ± sample SD over seeds 42/43/44)

| Test set | Baseline | + compression augmentation | V-A (+ social-media augmentation) |
|---|---|---|---|
| FF++ c23 test (in-domain) | 0.9889 ± 0.0042 | 0.9807 ± 0.0015 | 0.9788 ± 0.0009 |
| FF++ c40 test (heavy compression) | 0.7987 ± 0.0163 | 0.8714 ± 0.0113 | 0.8832 ± 0.0075 |
| Celeb-DF v2, fixed 200-video subset (cross-dataset) | 0.7807 ± 0.0214 | 0.8373 ± 0.0111 | 0.8422 ± 0.0154 |
| Celeb-DF subset re-encoded H.264 CRF 30 | 0.7176 ± 0.0183 | 0.7649 ± 0.0161 | 0.7929 ± 0.0263 |

Trained on FaceForensics++ c23 (official splits), 16 frames per video, MediaPipe face crops, decision threshold chosen on FF++ validation. Augmentation was added after the baseline results were seen; this is recorded in the protocol before retraining. V-A was trained after these test sets had been seen, so its numbers on them are reported only, not used for any choice.

### Audio (EER, mean ± SD over 3 seeds)

| Model | ASVspoof 2019 LA eval | In-the-Wild (31,779 files) |
|---|---|---|
| Original (layer 2, first 4 s) | 1.57% ± 0.03% | 30.03% ± 0.21% (AUC 0.7695) |
| v2 A: silence trimmed | 8.89% ± 1.02% | 21.83% ± 1.58% (AUC 0.8579) |
| v2 B: silence trimmed + noise/reverb/codec augmentation | 10.93% ± 0.27% | 22.56% ± 0.50% (AUC 0.8620) |

Frozen wav2vec2-base, mean-pooled hidden layer, MLP 768-256-1, trained on ASVspoof 2019 LA. Trimming leading/trailing silence explains most of the In-the-Wild gain and raises ASVspoof EER, which indicates the original 1.57% partly relied on a silence shortcut.

### Realistic test sets (test-only, defined in the protocol before evaluation)

| Set | Contents | Result (API checkpoints) |
|---|---|---|
| R1 | 149 real clips from Wikimedia Commons (free licenses), 15 s each | Video false alarm 10.8% [6.3–18.0] (111 clips with enough faces); audio false alarm 42.4% [34.6–50.5] |
| R2 | R1 compressed like a video call (≤640×360, 15 fps, H.264 300 kbps, Opus 16 kbps) | Video false alarm 12.6% [7.7–20.1]; audio false alarm 55.2% [47.1–63.2] |
| R3 | Celeb-DF 200 subset, call-compressed | Video AUC: baseline 0.7816, augmentation 0.8319, V-A 0.8347 (mean of 3 seeds) |
| R4 | 5,000 In-the-Wild files, Opus 16 kbps | Audio EER: original 33.51%, v2 A 25.95%, v2 B 25.26% (mean of 3 seeds) |

Brackets are Wilson 95% intervals at each checkpoint's EER threshold. At the API's 5%-false-alarm threshold (chosen on validation), R3 video gives 15.2% false alarm and detects 57.0% of fakes; R4 audio gives 42.0% false alarm and detects 86.4% of spoofs. Thresholds chosen on lab data do not hold in these conditions. Per-file scores are in `reports/eval_realworld/`.

### Known limitations

- Cross-dataset AUC is well below in-domain AUC, as expected for this setup.
- The audio branch is not reliable on real-world speech: even the v2 model used by the API flags 42% of real Commons clips as spoofed (55% after call compression). All bona fide training speech comes from one clean studio source; the original model also relied partly on leading silence.
- The video model was trained only on face swap / face reenactment (FF++). It is not designed for fully AI-generated (diffusion) video.
- R1/R2 contain real clips only (no fakes), and their realness cannot be verified absolutely. The Commons candidate list is skewed towards the first search keywords (see protocol).
- Results on a few individual real-world clips are illustrations, not error rates.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Preprocessing, training, evaluation, Grad-CAM, audio pipeline, API (`api.py`, `api_core.py`) |
| `scripts/` | Manifest building, CRF 30 re-encoding, API parity checks, latency benchmark |
| `configs/` | Manifests, the fixed Celeb-DF test subset and the R1–R4 lists (file names; Commons titles, URLs and licenses) |
| `docs/protocol.md` | Frozen evaluation protocol and dated amendments |
| `reports/` | Per-video predictions, evaluation ledger, sanity checks, latency, training histories |

Run any script with `--help` for its arguments.

## Not included

- **Datasets.** FaceForensics++, Celeb-DF v2, ASVspoof 2019 LA and In-the-Wild must be obtained from their owners under their own terms. Commons clips can be re-downloaded with `scripts/collect_commons.py`. No videos, audio, face crops or derived images from any of these sources are in this repository.
- **Model weights.** Trained checkpoints (`*.pt`) are not published.
- **Third-party model files** expected under `models/`: MediaPipe `blaze_face_short_range.tflite` and `face_landmarker.task`, and `facebook/wav2vec2-base` (config + weights) in `models/wav2vec2-base/`.

## Running the API

Requires the checkpoints and model files above.

```bash
docker build -t deepfake-api:cpu .
docker run --rm --cpus=4 --memory=4g -p 8000:8000 deepfake-api:cpu
```

Then open `http://localhost:8000/docs`.

- `POST /predict`: video file; returns a fake probability, label, per-frame scores and an optional Grad-CAM heatmap. Returns `no_reliable_face` when fewer than 4 frames contain a usable face.
- `POST /predict_audio`: `.wav`/`.flac` or a video (audio extracted with ffmpeg); uses the v2 B seed-43 model (chosen on validation), trims silence, returns a spoof probability and label, or `no_audio`.
- Every result also has a three-level `verdict`: `high_suspicion` (score at or above the 5%-false-alarm threshold on validation data), `no_sign_detected`, or `inconclusive` (too few faces, no audio, or too short).

Measured CPU latency in Docker (n = 20): video mean 1.68 s; audio file mean 0.15 s, audio from video mean 0.28 s. Results are for reference only and are not a forensic judgement.

## Environment

Python 3.14, PyTorch 2.11 (CUDA 12.8 for training, CPU in Docker). Pinned versions in `requirements.txt` (includes the PyTorch cu128 index), `requirements-api.txt` and `requirements-dev.txt`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Tests

```bash
python -m pytest tests -q
```

`tests/test_splits.py` needs no data: it checks that FF++ train/val/test identities are disjoint, that the manifests follow the official splits, and that the Celeb-DF test subset is fixed at 100 real + 100 fake. The API and regression tests are skipped automatically when the checkpoints, model files or datasets are not present.

## License

Code: MIT (see `LICENSE`). Datasets and third-party models keep their own licenses.
