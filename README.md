# Deepfake Video and Audio Detection

Undergraduate thesis project: a deepfake detector with a **video branch** (face crops + EfficientNet-B0) and an **audio branch** (frozen wav2vec2-base features + small MLP), evaluated with a protocol that was frozen before any test results were seen (and amended, with dates and reasons, whenever later experiments were added), and served through a FastAPI/Docker demo.

The evaluation protocol, every amendment to it, and the reasons for each change are in [`docs/protocol.md`](docs/protocol.md). Each test set is evaluated once per checkpoint; the log is [`reports/eval_ledger.csv`](reports/eval_ledger.csv).

## Results

All numbers are measured; mean ± sample SD over seeds 42/43/44 unless a single checkpoint is named. The longer write-up (Vietnamese) is [`docs/BAO_CAO_v2_draft.md`](docs/BAO_CAO_v2_draft.md).

### Video models

| Name | Training data |
|---|---|
| Baseline | FaceForensics++ c23 (official splits), 16 frames per video, MediaPipe face crops |
| Aug | Baseline + compression augmentation |
| V-A | Aug + social-media degradation |
| V-B | V-A data + face swaps made between FF++ identities with inswapper and simswap (`configs/manifest_vb.csv`) — **used by the API (seed 44)** |
| V-SBI | V-B + Self-Blended Images (negative result) |
| V-C | V-B + MAVOS-DD train and LAV-DF train subsets |

Thresholds are chosen on FF++ validation and never re-chosen on a test set. V-A, V-SBI and V-C were designed after some test sets had been seen; this is recorded in the protocol, and their numbers on those sets are reported only.

**Original test sets (video-level AUC)**

| Test set | Baseline | Aug | V-A | V-B | V-SBI | V-C |
|---|---|---|---|---|---|---|
| FF++ c23 test | 0.9889 ± 0.0042 | 0.9807 ± 0.0015 | 0.9788 ± 0.0009 | 0.9683 ± 0.0017 | 0.9709 ± 0.0028 | 0.9523 ± 0.0020 |
| FF++ c40 test | 0.7987 ± 0.0163 | 0.8714 ± 0.0113 | 0.8832 ± 0.0075 | 0.8866 ± 0.0070 | 0.8730 ± 0.0041 | 0.8346 ± 0.0128 |
| Celeb-DF v2, fixed 200-video subset | 0.7807 ± 0.0214 | 0.8373 ± 0.0111 | 0.8422 ± 0.0154 | 0.8257 ± 0.0203 | 0.8491 ± 0.0140 | 0.6821 ± 0.0159 |
| Celeb-DF subset, H.264 CRF 30 | 0.7176 ± 0.0183 | 0.7649 ± 0.0161 | 0.7929 ± 0.0263 | 0.7714 ± 0.0055 | 0.7764 ± 0.0175 | 0.6250 ± 0.0228 |

**Newer face-swap tools** (`reports/eval_vb/`; detection rate at a 5% false-alarm threshold on FF++ validation; 140 real FF++ test videos)

| Test set | Fake videos | Baseline | Aug | V-A | V-B |
|---|---|---|---|---|---|
| Test-A: inswapper + simswap (tools seen in training, unseen identities) | 268 | 61.2% | 29.2% | 30.1% | 94.0% |
| Test-B: ghost + hyperswap (unseen tools) | 268 | 44.5% | 22.2% | 28.3% | 78.7% |
| Test-C: roop (unseen tool) | 133 | 75.5% | 34.9% | 32.6% | 85.6% |
| Test-A-enh: inswapper + GFPGAN | 134 | 23.0% | 10.4% | 10.6% | 48.0% |
| False alarm on real FF++ test | 140 | 7.7% | 4.6% | 5.3% | 6.2% |

After video-call compression, V-B's false alarm on real videos rises to 17.2%. V-SBI gave no clear gain (Test-B 79.8%) and V-C was lower on every fake set except Test-A-enh (`reports/eval_vb_V-SBI/`, `reports/eval_vb_V-C/`).

### Audio models (EER)

| Model | ASVspoof 2019 LA eval | In-the-Wild (31,779 files) |
|---|---|---|
| Original (layer 2, first 4 s) | 1.57% ± 0.03% | 30.03% ± 0.21% (AUC 0.7695) |
| v2a: silence trimmed | 8.89% ± 1.02% | 21.83% ± 1.58% (AUC 0.8579) |
| v2b: v2a + noise/reverb/codec augmentation | 10.93% ± 0.27% | 22.56% ± 0.50% (AUC 0.8620) |
| v2c: v2b + MAVOS-DD train and LAV-DF train audio — **used by the API (seed 44)** | 10.50% ± 0.02% | 20.00% ± 0.24% (AUC 0.8777) |

Frozen wav2vec2-base, mean-pooled hidden layer, MLP 768-256-1. Trimming leading/trailing silence explains most of the In-the-Wild gain and raises ASVspoof EER, which indicates the original 1.57% partly relied on a silence shortcut.

### Out-of-domain test sets (AUC)

MAVOS-DD test (6,800 videos, 8 languages), AV-Deepfake1M validation subset (6,000), LAV-DF test subset (6,000), SocialDF (991 Instagram videos that could still be downloaded). Video uses the visual label, audio the audio label; SocialDF only has one overall label. Each checkpoint was run once (`reports/eval_ood/`, `reports/eval_ood_vc/`).

| Model | MAVOS-DD | AV-Deepfake1M | LAV-DF | SocialDF |
|---|---|---|---|---|
| Video V-B (API) | 0.512 ± 0.008 | 0.568 ± 0.006 | 0.449 ± 0.008 | 0.651 ± 0.017 |
| Video V-C | 0.885 ± 0.004 (partly in-domain) | 0.527 ± 0.002 | 0.748 ± 0.007 (in-domain) | 0.507 ± 0.050 |
| Audio v2b | 0.604 ± 0.007 | 0.603 ± 0.006 | 0.601 ± 0.007 | 0.464 ± 0.008 |
| Audio v2c (API) | 0.969 ± 0.004 (in-domain) | 0.773 ± 0.007 | 0.782 ± 0.001 (in-domain) | 0.419 ± 0.004 |

- Models trained on FF++ and ASVspoof hardly generalise to these newer deepfakes. Whole-face generators in MAVOS-DD (echomimic, memo, sonic) are almost never detected by V-B (AUC 0.21–0.31).
- V-C improves strongly inside MAVOS-DD and LAV-DF but is worse than V-B on every source it was not trained on (AV-Deepfake1M, SocialDF, Celeb-DF), so the API keeps V-B.
- v2c also improves on AV-Deepfake1M, which it was not trained on, so the API audio branch moved to v2c. Its threshold is conservative out of domain: at the EER threshold it detects only 3.4% of AV-Deepfake1M audio fakes.
- AV-Deepfake1M and LAV-DF edit very short segments (median 0.36 s and 0.69 s); averaging 16 frames / the first 4 s of audio is a design limit for them.

**Untrained fusion** (mean of the two API branches; `reports/eval_ood_vc/fusion_v2c.csv`): AUC 0.687 / 0.648 / 0.639 / 0.641 on the four sets. It is not better than the best single branch on any set.

### Realistic test sets (test-only, defined in the protocol before evaluation)

| Set | Contents | Result |
|---|---|---|
| R1 | 149 real clips from Wikimedia Commons (free licenses), 15 s each | False alarm, API at 5% threshold: video V-B 10.8% [6.3–18.0] (111 clips with enough faces); audio v2c 3.5% [1.5–7.9] (144 clips) |
| R2 | R1 compressed like a video call (≤640×360, 15 fps, H.264 300 kbps, Opus 16 kbps) | Video V-B 10.8% [6.3–18.0]; audio v2c 7.0% [3.8–12.4] (143 clips) |
| R3 | Celeb-DF 200 subset, call-compressed | Video AUC: baseline 0.7816, aug 0.8319, V-A 0.8347 |
| R4 | 5,000 In-the-Wild files, Opus 16 kbps | Audio EER: original 33.5%, v2a 26.0%, v2b 25.3%, v2c 22.6% (AUC 0.851). API v2c: false alarm 12.0% [10.9–13.2], detects 59.2% [56.9–61.4] |

Brackets are Wilson 95% intervals. The previous audio model (v2b) flagged 45.8% (R1) and 55.9% (R2) of real clips at the same kind of threshold; v2c flags far fewer real clips but is also more conservative on fakes, so this shows fewer false alarms, not better real-world detection. Per-file scores are in `reports/eval_realworld/` and `reports/eval_realworld_v2c/`.

### Known limitations

- Cross-dataset results are far below in-domain results; on the newer out-of-domain sets the API video model is close to chance.
- The video models target face swap / reenactment. They are not designed for fully AI-generated (diffusion) video.
- V-B fakes are face swaps between FF++ identities only; detection drops on unseen tools (78.7%) and after GFPGAN enhancement (48.0%), and call compression nearly triples false alarms on real video.
- The audio branch's conservative threshold misses many out-of-domain fakes. Treat both branches as indicators, not proof; "no sign detected" does not mean a file is real.
- R1/R2 contain real clips only, and their realness cannot be verified absolutely. SocialDF labels come from the dataset authors; whether the audio was manipulated is unknown.
- Deepfake-Eval-2024 access was refused, so it was dropped.
- The inswapper and simswap models are marked non-commercial; V-B was trained on their outputs.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Preprocessing, training (`train*.py`, `audio_v2*.py`), evaluation (`evaluate.py`, `eval_vb.py`, `eval_ood.py`, `eval_realworld.py`), Grad-CAM, API (`api.py`, `api_core.py`), demo page (`static/index.html`) |
| `scripts/` | Manifest and subset building, face-swap generation for V-B, dataset download helpers, call-compression, API parity, latency, UI tests (Playwright) |
| `configs/` | Manifests and fixed test/train subsets (file names only), the R1–R4 lists (Commons titles, URLs and licenses) |
| `docs/protocol.md` | Frozen evaluation protocol and dated amendments |
| `reports/` | Per-video predictions, evaluation ledger, summaries, latency, training histories and logs |

Run any script with `--help` for its arguments.

## Not included

- **Datasets.** FaceForensics++, Celeb-DF v2, ASVspoof 2019 LA, In-the-Wild, MAVOS-DD, AV-Deepfake1M, LAV-DF, SocialDF and RIRS_NOISES must be obtained from their owners under their own terms. Commons clips can be re-downloaded with `scripts/collect_commons.py`. No videos, audio, face crops, generated face swaps, heatmaps or other derived images are in this repository.
- **Per-file lists for AV-Deepfake1M and SocialDF** (subset list, per-video predictions, download log) are kept private; only aggregate results are published.
- **Model weights.** Trained checkpoints (`*.pt`) are not published.
- **Third-party model files** expected under `models/`: MediaPipe `blaze_face_short_range.tflite` and `face_landmarker.task`, and `facebook/wav2vec2-base` (config + weights) in `models/wav2vec2-base/`.

## Running the API

Requires the checkpoints and model files above.

```bash
docker build -t deepfake-api:cpu .
docker run --rm --cpus=4 --memory=4g -p 8000:8000 deepfake-api:cpu
```

Open `http://localhost:8000/` for the demo page (Vietnamese) or `http://localhost:8000/docs` for the API.

- `POST /predict`: video file; uses V-B seed 44. Returns a fake probability, label (FF++ validation EER threshold 0.6697), per-frame scores and an optional Grad-CAM heatmap. Returns `no_reliable_face` when fewer than 4 frames contain a usable face, and `unreadable` when no frame can be decoded.
- `POST /predict_audio`: `.wav`/`.flac` or a video (audio extracted with ffmpeg); uses audio v2c seed 44 (layer 4), trims silence, returns a logit, spoof probability and label (EER threshold 6.8268), or `no_audio`.
- Every result also has a three-level `verdict`: `high_suspicion` (score at or above the 5%-false-alarm threshold on validation data: video 0.7163, audio logit 7.4202), `no_sign_detected`, or `inconclusive`.
- Files over 100 MB are rejected (HTTP 413).

Measured CPU latency in Docker (n = 20, 2 warm-up runs excluded): video mean 1.51 s (p95 2.40 s); audio file mean 0.140 s, audio from video mean 0.256 s. API scores match the offline evaluation code (video max |d| 0.0048, audio 0.0000). Results are for reference only and are not a forensic judgement.

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
