# ASR Pipeline with Whisper

## Overview

A simple, modular Python ASR (Automatic Speech Recognition) pipeline that uses OpenAI's Whisper model to transcribe audio segments and compare them with reference transcripts. Designed to work seamlessly in both Replit and Google Colab environments.

## Purpose

Process audio segments from the Fearless Steps Challenge Phase 3 dataset, generate transcriptions using Whisper tiny.en model, and evaluate performance by comparing against ground truth transcripts.

## Project Architecture

### Modular Structure

```
.
├── main.py                     # Main entry point for the pipeline
├── config.py                   # Configuration (paths, model settings)
├── modules/
│   ├── drive_connector.py     # Google Drive access (Replit & Colab)
│   ├── data_loader.py         # Load audio files and transcripts
│   ├── whisper_transcriber.py # Whisper ASR inference
│   └── evaluator.py           # WER/CER calculation and comparison
└── requirements.txt           # Python dependencies
```

### Current Features

- **Dual Environment Support**: Automatically detects and works in both Replit and Google Colab
- **Google Drive Integration**: Accesses Phase 3 dataset from Google Drive
- **Whisper ASR**: Uses Whisper tiny.en model (fast, lightweight, English-only)
- **Interactive Comparison**: Side-by-side display of reference vs Whisper transcripts
- **Evaluation Metrics**: Calculates WER (Word Error Rate) and CER (Character Error Rate)
- **Batch Processing**: Process multiple files or single files
- **Timestamp Support**: Optional word-level timestamps from Whisper

## Dataset

**Source**: Fearless Steps Challenge Phase 3  
**Location**: Google Drive folder `Fearless_Steps_Challenge_Phase3`  
**Current Focus**: Audio/Segments/ASR_track2/Dev with corresponding transcripts

**Data Structure**:
- Audio: 8kHz, 16-bit WAV files (short utterances)
- Transcripts: Reference transcriptions in Transcripts/ASR_track2/Dev

## Usage

**Recommended Environment:** Google Colab (where your dataset is stored)

### In Google Colab (Primary Usage)

```python
# Clone from GitHub
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Run the pipeline (examples)
!python main.py --batch 5                    # Process 5 files from Dev set
!python main.py --dataset Train --batch 10   # Process 10 from Train set
!python main.py --file filename.wav          # Process specific file
!python main.py --no-timestamps              # Disable timestamp display
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

- openai-whisper (Whisper ASR model)
- jiwer (WER/CER calculation)
- google-api-python-client (Google Drive API for Replit)
- google-auth, google-auth-oauthlib, google-auth-httplib2 (Authentication)
- soundfile (Audio file handling)

## Recent Changes

**2025-11-11**: Google Drive Connection Update
- Updated drive_connector.py to use Replit's Google Drive integration
- Implemented automatic token fetching from Replit Connectors API
- Connection works seamlessly in Replit environment
- Added requests dependency for API calls
- Updated documentation with connection setup instructions

**2025-01-11**: Initial project setup
- Created modular Python architecture
- Implemented Whisper tiny.en integration
- Added Google Drive connector for both Replit and Colab
- Built evaluation module with WER/CER metrics
- Created main pipeline script with CLI interface

## Next Phase (Future Extensions)

1. **Extend to Train set** for larger-scale analysis
2. **Add SID module** for Speaker Identification using Audio/Segments/SID data
3. **Combined ASR+SID pipeline** on Streams (ASR_track1) for long-form audio
4. **JSON output export** matching reference format
5. **Summary reports** with aggregate statistics across datasets

## User Preferences

- Keep code simple and modular
- Focus on terminal output (Option A - no web UI needed)
- Build incrementally with clear separation between ASR and future SID features
