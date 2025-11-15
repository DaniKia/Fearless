# ASR + SID Pipeline for Fearless Steps Challenge

## Overview

This project provides a modular Python pipeline for Automatic Speech Recognition (ASR) and Speaker Identification (SID) using the Fearless Steps Challenge Phase 3 dataset. Its primary purpose is to process audio segments to generate transcriptions (ASR) and identify speakers (SID), with comprehensive evaluation metrics. The system is designed for both development in Replit and execution in Google Colab. The business vision is to provide a robust solution for analyzing speech data, with market potential in research and voice-based applications, aiming for highly accurate and efficient speech processing.

## User Preferences

- Keep code simple and modular
- Focus on terminal output (Option A - no web UI needed)
- Build incrementally with clear separation between ASR and future SID features

## System Architecture

The project employs a modular architecture with distinct Python scripts for ASR (`main.py`), SID (`sid_main.py`), and combined processing (`combined_main.py`), supported by utility functions (`utils.py`) and a centralized configuration (`config.py`). Core modules handle data loading, Whisper-based ASR transcription, SpeechBrain embedding extraction for SID, and comprehensive evaluation.

**Key Features:**

-   **ASR Pipeline**: Utilizes OpenAI's Whisper for transcription, supporting various models. It includes Word Error Rate (WER) and Character Error Rate (CER) evaluation, side-by-side transcript comparison, and batch processing.
-   **SID Pipeline**: Leverages pre-trained SpeechBrain ECAPA embeddings for speaker enrollment and identification. It supports GPU-accelerated batch processing for enrollment, provides accuracy metrics, and generates confusion matrices for error analysis. Closed-set identification is performed using cosine similarity.
-   **Combined Pipeline**: Integrates both ASR and SID for simultaneous processing and evaluation, particularly useful for datasets with both transcription and speaker labels.
-   **Flexible Folder Selection**: Supports processing various folders (e.g., ASR_track2, SID, SD_track2) from the Fearless Steps Challenge dataset using a `--folder` parameter.
-   **Data Handling**: Manages loading audio, transcripts, and speaker labels from a structured Google Drive dataset, with automatic detection of dataset types (ASR_track2 vs. SID) and corresponding label formats.
-   **UI/UX Decisions**: The project prioritizes a command-line interface (CLI) for interaction, focusing on clear terminal output for results and progress, rather than a graphical user interface.

## External Dependencies

-   **ASR Model**: `openai-whisper`
-   **Speaker Embeddings**: `speechbrain`
-   **Deep Learning Framework**: `torch`, `torchaudio`
-   **Evaluation Metrics**: `jiwer` (for WER/CER)
-   **Audio Processing**: `soundfile`
-   **Progress Bars**: `tqdm`
-   **Google Drive Integration**: `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`, `requests` (for Replit's Google Drive connector and Colab drive mounting)

## Dataset

**Source**: Fearless Steps Challenge Phase 3  
**Location**: Google Drive folder `Fearless_Steps_Challenge_Phase3`  
**Current Focus**: FSC_P3_Train_Dev/Audio/Segments/ASR_track2/Dev with corresponding transcripts

**Data Structure** (supports multiple folders via --folder parameter):
```
Fearless_Steps_Challenge_Phase3/
└── FSC_P3_Train_Dev/
    ├── Audio/
    │   └── Segments/
    │       ├── ASR_track2/        (--folder ASR_track2)
    │       │   ├── Dev/           (8kHz WAV files - ASR with speaker labels)
    │       │   └── Train/
    │       ├── SID/               (--folder SID)
    │       │   ├── Dev/           (8kHz WAV files - Speaker ID only)
    │       │   └── Train/
    │       ├── SD_track2/         (--folder SD_track2)
    │       ├── SD_track1/         (--folder SD_track1)
    │       └── SAD/               (--folder SAD)
    └── Transcripts/
        ├── ASR_track2/
        │   ├── Dev/               (Reference transcriptions)
        │   └── Train/
        ├── SID/
        │   ├── fsc_p3_SID_uttID2spkID_Dev    (Speaker labels)
        │   └── fsc_p3_SID_uttID2spkID_Train
        ├── SD_track2/
        └── SAD/
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
# Process single file from ASR_track2 (default)
!python main.py --file fsc_p3_ASR_track2_dev_0018.wav

# Batch processing from ASR_track2
!python main.py --batch 5                              # 5 files from Dev
!python main.py --dataset Train --batch 10             # 10 from Train

# Use different folders
!python main.py --folder ASR_track2 --batch 5          # ASR folder (default)
!python main.py --folder SID --batch 5                 # SID folder
!python main.py --folder SD_track2 --dataset Dev       # Speaker diarization folder
!python main.py --folder SAD --dataset Train           # Speech activity detection

# Use different Whisper models
!python main.py --file audio.wav --whisper-model base.en
!python main.py --file audio.wav --whisper-model large-v3

# ============ SID PIPELINE ============
# Step 1: Enroll speakers from SID folder (ONE-TIME, ~1.5-2.5 hours with GPU)
!python sid_main.py --enroll --folder SID --dataset Train

# Enroll from ASR_track2 (which includes speaker labels)
!python sid_main.py --enroll --folder ASR_track2 --dataset Train

# Custom GPU batch size for enrollment (tune based on GPU memory)
!python sid_main.py --enroll --folder SID --dataset Train --batch-size 16  # Default (recommended)
!python sid_main.py --enroll --folder SID --dataset Train --batch-size 32  # Larger batches (needs more GPU memory)
!python sid_main.py --enroll --folder SID --dataset Train --batch-size 8   # Smaller batches (if you get OOM errors)

# Step 2: Identify speakers from SID folder (test accuracy)
!python sid_main.py --file fsc_p3_SID_dev_0010.wav     # Single file (default folder: SID)
!python sid_main.py --folder SID --dataset Dev         # Process ALL files in Dev set
!python sid_main.py --folder SID --batch 5             # Test on specific number (5 files)
!python sid_main.py --folder SID --dataset Train --batch 10  # Test on 10 files from Train

# Identify speakers from other folders
!python sid_main.py --folder ASR_track2 --dataset Dev  # Process ASR_track2 audio
!python sid_main.py --folder SD_track2 --batch 10      # Process speaker diarization audio

# Step 3: Analyze errors with confusion matrix
!python sid_main.py --folder SID --dataset Dev --confusion-matrix    # Process ALL Dev files with confusion matrix
!python sid_main.py --folder SID --batch 100 --confusion-matrix      # Process specific number with confusion matrix

# ============ COMBINED ASR + SID ============
# Note: Requires speaker enrollment first
!python combined_main.py --folder ASR_track2 --file fsc_p3_ASR_track2_dev_0018.wav
!python combined_main.py --folder ASR_track2 --batch 5
!python combined_main.py --folder SID --dataset Train --batch 10
!python combined_main.py --folder ASR_track2 --dataset Train --batch 10 --whisper-model large-v3
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

**2025-11-15**: Flexible Folder Selection with --folder Parameter
- **Added --folder parameter** to all three main scripts (sid_main.py, main.py, combined_main.py)
  - Supports any folder: SID, ASR_track2, SD_track2, SD_track1, SAD, or custom folders
  - Default folders: SID for sid_main.py, ASR_track2 for main.py and combined_main.py
  - Usage: `!python main.py --folder SD_track2 --dataset Dev`
- **Generic folder path functions** in config.py
  - `get_folder_audio_path(folder, dataset)` - works with any folder name
  - `get_folder_label_path(folder, dataset)` - handles different label directory structures
  - Simplified implementation for clean architecture
- **Colab path structure**: Maps to `/FSC_P3_Train_Dev/Audio/Segments/{folder}/{dataset}`
- **Local cache paths**: `.asr_cache/{folder}_audio/{dataset}` and `.asr_cache/{folder}_labels/{dataset}`
- **All pipelines now support flexible folder selection** for comprehensive dataset analysis

**2025-11-13**: Confusion Matrix Analysis for Speaker Identification
- **Added --confusion-matrix flag** to sid_main.py for detailed error analysis
  - Shows top 20 most-confused speaker pairs (e.g., "FIDO1 confused as NSCTM: 45 times")
  - Usage: `!python sid_main.py --dataset Dev --confusion-matrix`
- **Removed speaker limit** from per-speaker accuracy display
  - Previously only showed stats when ≤20 speakers, now shows all speakers
  - Helps diagnose accuracy issues on a per-speaker basis
- **Changed default batch behavior**: When `--batch` is not specified, processes ALL files in dataset
  - Previous default: 5 files
  - New behavior: ALL files (for comprehensive analysis)
  - Explicit batch still works: `--batch 100` processes exactly 100 files

**2025-11-12**: Speaker Identification (SID) Module
- **GPU-accelerated enrollment** with batch processing (~1.5-2.5 hours vs 96+ hours)
  - Process multiple audio files simultaneously on GPU
  - Configurable batch size via `--batch-size` (default: 16)
  - Shows detailed progress with time estimates
- **Speaker database persistence** (saves to speaker_database.pkl)
- **Closed-set identification** using cosine similarity
- **Comprehensive evaluation**:
  - Per-file identification with similarity scores
  - Batch accuracy metrics
  - Per-speaker accuracy breakdown
- **Separate SID dataset support** (SID folder) with different label format
- **Combined ASR + SID pipeline** (combined_main.py) for simultaneous evaluation

## Quick Start

### First Time Setup (Google Colab)
1. Mount Drive: `drive.mount('/content/drive')`
2. Clone repository: `!git clone <repo-url>`
3. Count speakers: `!python utils.py --count-speakers --dataset Train`
4. Enroll speakers: `!python sid_main.py --enroll --dataset Train` (creates speaker_database.pkl)
5. Test on Dev set: `!python sid_main.py --batch 5`

### Regular Usage
- **ASR only**: `!python main.py --folder ASR_track2 --file audio.wav`
- **SID only**: `!python sid_main.py --folder SID --file audio.wav`
- **Both**: `!python combined_main.py --folder ASR_track2 --file audio.wav`
- **Custom folders**: `!python main.py --folder SD_track2 --batch 10`

## Next Phase (Future Extensions)

1. ✅ ~~Add SID module for Speaker Identification~~ **COMPLETED**
2. ✅ ~~Add evaluation metrics (accuracy, confusion matrix)~~ **COMPLETED**
3. 🔜 Optimize performance (caching, parallel processing)
4. 🔜 Add speaker diarization (SD) module
5. 🔜 Add speech activity detection (SAD) module
6. 🔜 Evaluate on Eval dataset after Dev tuning
7. 🔜 Consider web UI (Option B) if needed later
