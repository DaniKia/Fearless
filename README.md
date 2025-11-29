# Fearless

**Modular ASR on the Fearless Steps Corpus with a clean upgrade path to SID and role‑aware speech‑act classification.**

---

## Why this repo exists

This project gives you a **working, reproducible ASR pipeline** over the Fearless Steps corpus with minimal custom code and only **free models/software**. It starts simple (segment‑based ASR) and scales to streams, SID, and role‑aware speech‑act labeling without refactoring.

---

## Features (current → planned)

* **✓ Segment ASR (ASR_track2)** using an off‑the‑shelf Whisper model; prints WER/CER against provided references.
* **✓ Colab‑friendly**: runs on a T4 GPU with straightforward cells.
* **⬜ JSONL/TSV export** of hypotheses for downstream tasks (SID, acts).
* **⬜ Faster‑Whisper support** for faster, low‑VRAM inference.
* **⬜ SID integration** (merge provided segment→speaker mapping; optional model‑based SID later).
* **⬜ Role‑aware speech‑act labeling** (rules first, then optional prosody/lexical models).
* **⬜ Streams (ASR_track1)** using provided UEM/RTTM/refSAD to avoid building SAD/SD.

> Items marked ⬜ are intended next steps and have stubs/structure to drop in cleanly.

---

## Repository structure

```
.
├─ main.py                      # Orchestrates batch/single ASR and evaluation
├─ config.py                    # All dataset paths & model settings in one place
├─ modules/
│  ├─ drive_connector.py        # (Optional) Google Drive mount helpers
│  ├─ data_loader.py            # Lists audio, loads refs, (planned) SID labels
│  ├─ whisper_transcriber.py    # Whisper inference wrapper (planned: faster-whisper)
│  ├─ evaluator.py              # WER/CER & display utilities
│  ├─ sid_identifier.py         # (planned) SID lookup/model
│  └─ speech_act_classifier.py  # (planned) Rule-based + pluggable ML
└─ README.md
```

---

## Data you need (not bundled)

You must legally obtain the Fearless Steps data and place it under a **consistent layout**. Start with **Segments / ASR_track2** for the lowest friction:

```
<DATA_ROOT>/
  Phase3/
    Audio/
      Segments/
        ASR_track2/
          Dev/*.wav
          Train/*.wav
          Eval/*.wav        # often no references (hyp-only)
    Transcripts/
      ASR_track2/
        Dev/*.txt           # reference text per segment
        Train/*.txt

# (Later for Streams / ASR_track1)
    Audio/Streams/{Dev,Train,Eval}/*.wav
    Transcripts/ASR_track1/{Dev,Train}/*.json
    SD_track2/{UEM,RTTM,refSAD}/...        # for clean region reuse
```

Edit `config.py` to point `DATA_ROOT` and `PHASE` (e.g., "Phase3").

---

## Quick start (local)

1. **Python & GPU:** Python 3.10+ recommended. NVIDIA GPU optional but preferred.
2. **Install deps:**

   ```bash
   pip install -U openai-whisper faster-whisper jiwer pandas tqdm soundfile
   ```
3. **Configure paths:** set `DATA_ROOT` and `PHASE` in `config.py`.
4. **Run ASR on Segments:**

   ```bash
   python main.py --dataset Dev --limit 100        # increase/remove --limit to process all
   # optional model size
   python main.py --dataset Dev --model large-v3
   ```
5. **Review output:** WER/CER prints to console for any segment with a provided reference.

**Notes**

* Fearless Steps audio is **8 kHz**; Whisper/Faster‑Whisper will resample internally.
* Start with `large-v3` for accuracy; drop to `distil-large-v3` if on tight VRAM.

---

## Quick start (Google Colab)

Use a single notebook with these sections:

1. **GPU + installs**

   ```python
   !nvidia-smi -L
   !pip -q install faster-whisper openai-whisper jiwer pandas tqdm soundfile
   ```
2. **Mount Drive & set paths**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   DATA_ROOT = "/content/drive/MyDrive/FS"
   PHASE = "Phase3"
   ```
3. **Import repo** (if using this repo from Drive or `git clone`) and **run `main.py`** with flags as above.
4. **OOM fix**: use `compute_type="int8_float16"` in Faster‑Whisper, or switch to `distil-large-v3`.

---

## CLI

```
python main.py \
  --dataset {Dev|Train|Eval} \
  [--model {tiny|base|small|medium|large-v3|distil-large-v3}] \
  [--limit N]                # limit number of files for quick runs
```

Planned flags (introduced as features land):

```
  [--export jsonl|tsv]       # write hypotheses for downstream tasks
  [--faster]                 # use faster-whisper
  [--sid]                    # attach speaker IDs from SID mapping
  [--classify]               # attach role-aware speech-act labels
  [--streams]                # run over ASR_track1 streams using UEM/RTTM/refSAD
```

---

## Outputs

Current: console WER/CER summary.

Planned exports (one line per utterance):

* **TSV**: `utt_id \t wav_path \t hyp`  → `outputs/<phase>_<split>_asr.tsv`
* **JSONL**: `{utt_id, wav, hyp, words?}` → `outputs/<phase>_<split>_asr.jsonl`

Downstream manifest (for SID/acts):

```json
{"utt_id": "...", "wav": "...", "transcript": "...", "speaker_id": null, "role": null}
```

---

## Evaluation

* **ASR (Segments):** `jiwer` WER/CER against `Transcripts/ASR_track2/{Dev,Train}`.
* **ASR (Streams):** compare to `Transcripts/ASR_track1/*.json` once stream support lands.
* **SID:** accuracy/confusion against SID mapping (once enabled).
* **Acts:** precision/recall/F1 once labeled data or silver labels exist.

---

## Extending the system (roadmap)

1. **JSON/TSV export** (low effort): persist all hypotheses for reuse.
2. **Faster‑Whisper backend** (low effort): optional flag to swap runtime.
3. **SID integration** (low effort):

   * Load segment→speaker mapping from SID pack.
   * Emit `speaker_id` (and, if available, `role`) alongside transcripts.
4. **Role‑aware speech‑act (medium effort):**

   * Rule‑based classifier module using text + optional role priors.
   * Later: use word timestamps for simple prosody cues; optional lexical model.
5. **Streams support** (medium effort):

   * Parse UEM/RTTM/refSAD; slice streams to scoring regions; transcribe; stitch.

All items are designed to be **drop‑in modules**. No refactors to ASR core should be required.

---

## Design principles

* **Minimal bespoke code**: prefer ready models/tools; reuse Fearless references (RTTM/UEM/refSAD) when available.
* **Single source of truth for paths** in `config.py`.
* **Pure functions, clear I/O**: modules return data; `main.py` orchestrates.
* **Stable, inspectable outputs** (JSONL/TSV) for the next stages.

---

## Troubleshooting

* **OOM on Colab T4**: use `distil-large-v3` or Faster‑Whisper with `compute_type="int8_float16"`.
* **Empty WER set**: ensure reference `.txt` files exist for the chosen split and IDs match audio basenames.
* **Weird jargon spellings**: pass a short `initial_prompt` with Apollo terms (CAPCOM, GUIDO, EECOM, go/no‑go).

---

## Contributing

* Keep modules focused; prefer small, testable functions.
* Put new dataset paths/knobs in `config.py`.
* Add unit tests where practical; avoid print‑heavy functions in library code.
* Open a PR with a concise description and sample output.

---

## License & Acknowledgments

* This repo: see `LICENSE`.
* Fearless Steps corpus: subject to its own license/terms.

---

## Citation

If this work helps your research, consider citing the Fearless Steps challenge/corpus and the chosen ASR backend (Whisper/Faster‑Whisper).
