# ASR + SID Pipeline for Fearless Steps Challenge

## Overview

A modular Python pipeline for Automatic Speech Recognition (ASR) and Speaker Identification (SID) using the Fearless Steps Challenge Phase 3 dataset. The system uses OpenAI's Whisper for transcription and SpeechBrain embeddings for speaker identification. Designed to work seamlessly in both Replit (development) and Google Colab (execution) environments.

## Purpose

Process audio segments from the Fearless Steps Challenge Phase 3 dataset to:
1. **ASR**: Generate transcriptions using Whisper and evaluate against ground truth (WER/CER metrics)
2. **SID**: Identify speakers using pre-trained embeddings and evaluate accuracy
3. **Combined**: Perform both ASR and SID on the same audio for comprehensive analysis

## Project Architecture

### Modular Structure

```
.
├── main.py                     # ASR pipeline entry point
├── sid_main.py                 # SID pipeline entry point
├── combined_main.py            # Combined ASR + SID pipeline
├── utils.py                    # Utility functions (speaker counting, etc.)
├── config.py                   # Configuration (paths, model settings)
├── modules/
│   ├── drive_connector.py     # Google Drive access (Replit & Colab)
│   ├── data_loader.py         # Load audio, transcripts, and SID labels
│   ├── whisper_transcriber.py # Whisper ASR inference
│   ├── evaluator.py           # ASR WER/CER calculation
│   ├── speaker_identifier.py  # SID embedding extraction & identification
│   └── sid_evaluator.py       # SID accuracy metrics & confusion matrix
└── requirements.txt           # Python dependencies
```

### Features

**ASR Pipeline (main.py)**
- Whisper-based transcription (configurable models: tiny.en, base.en, large-v3, etc.)
- WER/CER evaluation against ground truth
- Side-by-side transcript comparison
- Batch and single-file processing
- Optional word-level timestamps

**SID Pipeline (sid_main.py)**
- Speaker enrollment from Train set (~200 speakers)
- GPU batch processing for fast enrollment (~1.5-2.5 hours vs 96+ hours)
- Speaker identification using cosine similarity
- Pre-trained SpeechBrain ECAPA embeddings
- Accuracy metrics and confusion matrix
- Closed-set identification (identifies among known speakers)
- Configurable batch size for enrollment (default: 16 files per GPU batch)

**Combined Pipeline (combined_main.py)**
- Simultaneous ASR + SID processing
- Useful for ASR dataset which includes speaker labels
- Comprehensive evaluation (WER/CER + speaker accuracy)

**Utilities (utils.py)**
- Count unique speakers in dataset
- View speaker statistics

## Dataset

**Source**: Fearless Steps Challenge Phase 3  
**Location**: Google Drive folder `Fearless_Steps_Challenge_Phase3`  
**Current Focus**: FSC_P3_Train_Dev/Audio/Segments/ASR_track2/Dev with corresponding transcripts

**Data Structure**:
```
Fearless_Steps_Challenge_Phase3/
└── FSC_P3_Train_Dev/
    ├── Audio/
    │   └── Segments/
    │       ├── ASR_track2/
    │       │   ├── Dev/     (8kHz WAV files - ASR with speaker labels)
    │       │   └── Train/
    │       └── SID/
    │           ├── Dev/     (8kHz WAV files - Speaker ID only)
    │           └── Train/
    └── Transcripts/
        ├── ASR_track2/
        │   ├── Dev/     (Reference transcriptions)
        │   └── Train/
        └── SID/
            ├── fsc_p3_SID_uttID2spkID_Dev    (Speaker labels)
            └── fsc_p3_SID_uttID2spkID_Train
```

**Label Formats**:
- **ASR Transcripts**: `fsc_p3_ASR_track2_transcriptions_{Dataset}.text`
  - Format: `audio_id TRANSCRIPT TEXT`
- **SID Labels**: `fsc_p3_SID_uttID2spkID_{Dataset}`
  - Format: `audio_id SPEAKER_ID` (e.g., `fsc_p3_SID_dev_0010 CAPCOM1`)

## Usage

**Recommended Environment:** Google Colab (where your dataset is stored)

### In Google Colab (Primary Usage)

```python
# Clone from GitHub and mount Drive
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo

from google.colab import drive
drive.mount('/content/drive')

# ============ UTILITIES ============
# Count unique speakers in dataset
!python utils.py --count-speakers --dataset Dev
!python utils.py --count-speakers --dataset Train

# ============ ASR PIPELINE ============
# Process single file
!python main.py --file fsc_p3_ASR_track2_dev_0018.wav

# Batch processing
!python main.py --batch 5                              # 5 files from Dev
!python main.py --dataset Train --batch 10             # 10 from Train

# Use different Whisper models
!python main.py --file audio.wav --whisper-model base.en
!python main.py --file audio.wav --whisper-model large-v3

# ============ SID PIPELINE ============
# Step 1: Enroll speakers (ONE-TIME, ~1.5-2.5 hours with GPU)
!python sid_main.py --enroll --dataset Train

# Custom GPU batch size for enrollment (tune based on GPU memory)
!python sid_main.py --enroll --dataset Train --batch-size 16  # Default (recommended)
!python sid_main.py --enroll --dataset Train --batch-size 32  # Larger batches (needs more GPU memory)
!python sid_main.py --enroll --dataset Train --batch-size 8   # Smaller batches (if you get OOM errors)

# Step 2: Identify speakers (test accuracy)
!python sid_main.py --file fsc_p3_SID_dev_0010.wav     # Single file
!python sid_main.py --batch 5                           # Test on 5 files
!python sid_main.py --dataset Train --batch 10          # Test on 10 files from Train

# ============ COMBINED ASR + SID ============
# Note: Requires speaker enrollment first
!python combined_main.py --file fsc_p3_ASR_track2_dev_0018.wav
!python combined_main.py --batch 5
!python combined_main.py --dataset Train --batch 10 --whisper-model large-v3
```

### In Replit (Development & Code Review)

Replit is primarily used for:
- Writing and reviewing code
- Testing the modular structure
- Viewing usage with `python main.py --help`
- Pushing code to GitHub

**Google Drive Connection in Replit:**
The project has Replit's Google Drive integration installed. This allows the code to access your Drive files through Replit's secure connection API. To use it:

1. The connection is already added to your project
2. You may need to authorize access by clicking the "Connect" button in the Replit integrations panel
3. The connection uses limited scopes, so it may only access files created by the app
4. For full dataset access, Google Colab is still recommended where you can mount your entire Drive

For actual data processing, use Google Colab where your Phase 3 dataset is accessible in Drive.

## Dependencies

**Core Libraries**
- openai-whisper (Whisper ASR model)
- speechbrain (Pre-trained speaker embeddings)
- torch, torchaudio (Deep learning framework)
- jiwer (WER/CER calculation)
- soundfile (Audio file handling)
- tqdm (Progress bars)

**Google Drive Integration (Replit)**
- google-api-python-client
- google-auth, google-auth-oauthlib, google-auth-httplib2
- requests

## Recent Changes

**2025-11-13**: Critical Bug Fixes for SID and Combined Pipeline
- **Fixed shape mismatch bug**: Embeddings now consistently 1D (192,) instead of (1,192)
  - Fixed both enrollment and identification embedding extraction
  - Resolves `ValueError: shapes (192,) and (1,192) not aligned` in cosine similarity
  - **IMPORTANT**: Delete old `speaker_database.pkl` and re-enroll speakers!
- **Fixed calculate_wer return type**: Now returns `(WER, CER)` tuple
  - Resolves `TypeError: cannot unpack non-iterable float` in combined pipeline
  - Updated all callers (display_comparison, combined_main.py)
- **Added speaker progress logging**: Progress bar description updates to show `[1/218] CAPCOM1 (125 files)`
  - Helps understand slow/fast enrollment pattern (speakers with more files take longer)
  - Updates on same line without spamming new lines

**2025-11-13**: Optimized SID Enrollment with GPU Batch Processing
- Replaced ineffective threading with GPU batch processing
- Processes 16 audio files per batch using `encode_batch()` for true GPU parallelism
- Expected speedup: 50-60x faster (96+ hours → ~1.5-2.5 hours)
- Added `--batch-size` CLI flag (default: 16, tune based on GPU memory)
- Fixed database path to save at root directory level

**2025-11-12**: Speaker Identification (SID) Module Added
- Created complete SID pipeline with speaker enrollment and identification
- Implemented speaker_identifier.py using SpeechBrain ECAPA embeddings
- Added speaker enrollment system (builds database from ~200 speakers)
- Closed-set identification using cosine similarity
- Created sid_evaluator.py for accuracy metrics and confusion matrix
- Built sid_main.py CLI with --enroll and identification modes
- Created combined_main.py for simultaneous ASR + SID processing
- Added utils.py for speaker counting and statistics
- Extended data_loader.py to handle SID labels (uttID2spkID format)
- Updated config.py with SID paths and speaker database location
- Updated requirements.txt with torch, speechbrain, torchaudio, tqdm

**2025-11-11**: ASR Transcript Loading Fixed
- Fixed consolidated transcript file parsing (single .text file format)
- Updated data_loader.py to read from fsc_p3_ASR_track2_transcriptions_{Dataset}.text
- Implemented caching for efficient transcript loading
- Propagated dataset parameter through all ASR functions

**2025-11-11**: Google Drive Connection Update
- Updated drive_connector.py to use Replit's Google Drive integration
- Implemented automatic token fetching from Replit Connectors API
- Connection works seamlessly in Replit environment

**2025-01-11**: Initial Project Setup
- Created modular Python architecture
- Implemented Whisper ASR integration
- Added Google Drive connector for both Replit and Colab
- Built evaluation module with WER/CER metrics
- Created main pipeline script with CLI interface

## Workflow

### Complete SID Setup (First Time)
1. Clone repo and mount Drive in Colab
2. **Delete old database** (if exists): `!rm -f speaker_database.pkl`
3. Count speakers: `!python utils.py --count-speakers --dataset Train`
4. Enroll speakers: `!python sid_main.py --enroll --dataset Train` (creates speaker_database.pkl)
5. Test on Dev set: `!python sid_main.py --batch 5`

### Regular Usage
- **ASR only**: `!python main.py --file audio.wav`
- **SID only**: `!python sid_main.py --file audio.wav`
- **Both**: `!python combined_main.py --file audio.wav`

## Next Phase (Future Extensions)

1. ✅ ~~Add SID module for Speaker Identification~~ **COMPLETED**
2. ✅ ~~Combined ASR+SID pipeline~~ **COMPLETED**
3. **Fine-tune Whisper** on Fearless Steps dataset for improved accuracy
4. **Long-form audio processing** on Streams (ASR_track1)
5. **JSON output export** matching reference format
6. **Advanced SID**: Speaker diarization for multi-speaker segments
7. **Summary reports** with aggregate statistics and visualizations

## User Preferences

- Keep code simple and modular
- Focus on terminal output (Option A - no web UI needed)
- Build incrementally with clear separation between ASR and future SID features
