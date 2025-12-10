# ASR + SID Pipeline for Fearless Steps Challenge

## Overview

This project provides a modular Python pipeline for Automatic Speech Recognition (ASR) and Speaker Identification (SID) using the Fearless Steps Challenge Phase 3 dataset. Its primary purpose is to process audio segments to generate transcriptions and identify speakers, with comprehensive evaluation metrics. The system is designed for both development in Replit and execution in Google Colab. The business vision is to provide a robust solution for analyzing speech data, with market potential in research and voice-based applications, aiming for highly accurate and efficient speech processing.

## User Preferences

- Keep code simple and modular
- Focus on terminal output (Option A - no web UI needed)
- Build incrementally with clear separation between ASR and future SID features

## System Architecture

The project employs a modular architecture with distinct Python scripts for ASR (`main.py`), SID (`sid_main.py`), and combined processing (`combined_main.py`), supported by utility functions (`utils.py`) and a centralized configuration (`config.py`).

### Audio Processing Architecture (NEW)

The audio processing is cleanly separated into two layers:

1. **Audio I/O (`modules/audio_io.py`)**: Responsible only for loading audio files from disk
   - `load_audio(path)` - Returns (waveform, sample_rate)
   - `load_audio_safe(path)` - Returns (waveform, sample_rate, error) with error handling

2. **Audio Preprocessing (`modules/audio_preprocessor.py`)**: Configurable preprocessing pipeline
   - `PreprocessConfig` - Dataclass with flags to enable/disable each step
   - `preprocess_audio(audio, sr, config)` - Apply preprocessing to loaded audio
   - `basic_preprocess_file(path, config)` - Convenience function to load + preprocess

### Preprocessing Steps (configurable via PreprocessConfig)

| Step | Function | Description | Config Flag |
|------|----------|-------------|-------------|
| 1 | `convert_to_mono()` | Convert stereo to mono | `enable_mono` |
| 2 | `resample_audio()` | Resample to target rate (16kHz) | `enable_resample` |
| 3 | `remove_dc_offset()` | Subtract mean from signal | `enable_dc_removal` |
| 4 | `apply_bandpass_filter()` | Optional 80Hz-7500Hz bandpass | `enable_bandpass` |
| 5 | `normalize_rms()` | RMS loudness normalization | `enable_rms_normalization` |
| 6 | `trim_silence()` | Trim leading/trailing silence | `enable_trim` |
| 7 | `check_duration()` | Validate minimum duration | (always runs) |

### UI/UX Decisions
The project prioritizes a command-line interface (CLI) for interaction, focusing on clear terminal output for results and progress, rather than a graphical user interface.

### Technical Implementations & Feature Specifications
- **ASR Pipeline**: Utilizes OpenAI's Whisper for transcription, supporting various models. It includes Word Error Rate (WER) and Character Error Rate (CER) evaluation, side-by-side transcript comparison, and batch processing.
- **SID Pipeline**: Leverages pre-trained SpeechBrain ECAPA embeddings for speaker enrollment and identification. It supports GPU-accelerated batch processing for enrollment, provides accuracy metrics (including Top-K and per-speaker precision), and generates confusion matrices for error analysis. Closed-set identification uses cosine similarity and includes resumable enrollment with per-speaker checkpointing.
- **Audio Preprocessing**: Modular preprocessing pipeline with configurable steps for ablation studies. Enable via `--preprocess` flag.
- **Combined Pipeline**: Integrates both ASR and SID for simultaneous processing and evaluation.
- **Flexible Folder Selection**: Supports processing various folders (e.g., ASR_track2, SID, SD_track2) from the Fearless Steps Challenge dataset using a `--folder` parameter.
- **Data Handling**: Manages loading audio, transcripts, and speaker labels from a structured Google Drive dataset, with automatic detection of dataset types and corresponding label formats.

## External Dependencies

- **ASR Model**: `openai-whisper`
- **Speaker Embeddings**: `speechbrain`
- **Deep Learning Framework**: `torch`, `torchaudio`
- **Evaluation Metrics**: `jiwer` (for WER/CER)
- **Audio Processing**: `soundfile`, `librosa`, `scipy`
- **Progress Bars**: `tqdm`
- **Google Drive Integration**: `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`, `requests`