# Deepfake Video and Audio Detection

Undergraduate thesis project: a deepfake detector with a **video branch** (face crops + EfficientNet-B0) and an **audio branch** (frozen wav2vec2-base features + small MLP), evaluated with a protocol that was frozen before any test results were seen, and served through a FastAPI/Docker demo.

The evaluation protocol, every amendment to it, and the reasons for each change are in [`docs/protocol.md`](docs/protocol.md). Each test set is evaluated once per checkpoint; the log is [`reports/eval_ledger.csv`](reports/eval_ledger.csv).

## Results

### Video (video-level AUC, mean ± sample SD over seeds 42/43/44)

| Test set | Baseline | + compression augmentation |
|---|---|---|
| FF++ c23 test (in-domain) | 0.9889 ± 0.0042 | 0.9807 ± 0.0015 |
| FF++ c40 test (heavy compression) | 0.7987 ± 0.0163 | 0.8714 ± 0.0113 |
| Celeb-DF v2, fixed 200-video subset (cross-dataset) | 0.7807 ± 0.0214 | 0.8373 ± 0.0111 |
| Celeb-DF subset re-encoded H.264 CRF 30 | 0.7176 ± 0.0183 | 0.7649 ± 0.0161 |

Trained on FaceForensics++ c23 (official splits), 16 frames per video, MediaPipe face crops, decision threshold chosen on FF++ validation. Augmentation was added after the baseline results were seen; this is recorded in the protocol before retraining.

### Audio (ASVspoof 2019 LA eval, 3 seeds)

EER 1.57% ± 0.03%, AUC 0.9987 ± 0.0001. Layer 2 of frozen wav2vec2-base, mean-pooled over the first 4 s, MLP 768-256-1.

### Known limitations

- Cross-dataset AUC is well below in-domain AUC, as expected for this setup.
- The audio branch labels real-world speech (phone recordings, interviews, social-media clips) as spoofed. A diagnostic on ASVspoof dev files showed that AAC compression and resampling do not cause this; leading-silence length contributes partly. The main cause appears to be that all bona fide training speech comes from one clean studio source. The leading-silence shortcut check is recorded in `docs/protocol.md`.
- The video model was trained only on face swap / face reenactment (FF++). It is not designed for fully AI-generated (diffusion) video.
- Results on a few real-world clips are illustrations, not error rates.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Preprocessing, training, evaluation, Grad-CAM, audio pipeline, API (`api.py`, `api_core.py`) |
| `scripts/` | Manifest building, CRF 30 re-encoding, API parity checks, latency benchmark |
| `configs/` | Video manifests and the fixed Celeb-DF test subset (file names only) |
| `docs/protocol.md` | Frozen evaluation protocol and dated amendments |
| `reports/` | Per-video predictions, evaluation ledger, sanity checks, latency, training histories |

Run any script with `--help` for its arguments.

## Not included

- **Datasets.** FaceForensics++, Celeb-DF v2 and ASVspoof 2019 LA must be obtained from their owners under their own terms. No videos, audio, face crops or derived images from these datasets are in this repository.
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
- `POST /predict_audio`: `.wav`/`.flac` or a video (audio extracted with ffmpeg); returns a spoof probability and label, or `no_audio`.

Measured CPU latency in Docker (n = 20): video mean 1.68 s, audio file mean 0.19 s. Results are for reference only and are not a forensic judgement.

## Environment

Python 3.14, PyTorch 2.11 (CUDA 12.8 for training, CPU in Docker). Pinned versions in `requirements.txt` and `requirements-api.txt`.

## License

Code: MIT (see `LICENSE`). Datasets and third-party models keep their own licenses.
