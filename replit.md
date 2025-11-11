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

### In Replit

```bash
# Process 5 files from Dev set (default)
python main.py

# Process specific dataset
python main.py --dataset Train

# Process specific audio file
python main.py --file filename.wav

# Process more files in batch
python main.py --batch 10

# Disable timestamps display
python main.py --no-timestamps
```

### In Google Colab

```python
# Clone from GitHub
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Run the pipeline
!python main.py --batch 5
```

## Dependencies

- openai-whisper (Whisper ASR model)
- jiwer (WER/CER calculation)
- google-api-python-client (Google Drive API for Replit)
- google-auth, google-auth-oauthlib, google-auth-httplib2 (Authentication)
- soundfile (Audio file handling)

## Recent Changes

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
