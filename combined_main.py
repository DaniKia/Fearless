"""
Combined ASR + SID pipeline.
Processes audio files to generate both transcription and speaker identification.
"""

import os
import sys
import argparse
import config
from modules.data_loader import load_reference_transcript, load_speaker_label_auto, get_audio_files_with_transcripts
from modules.whisper_transcriber import transcribe_audio
from modules.evaluator import calculate_wer
from modules.speaker_identifier import SpeakerIdentifier

def process_single_file(audio_path, transcript_dir, label_dir, dataset='Dev', whisper_model=None):
    """
    Process a single audio file with both ASR and SID.
    
    Args:
        audio_path: Path to audio file
        transcript_dir: Directory containing ASR transcripts
        label_dir: Directory containing SID labels (not used, kept for compatibility)
        dataset: Dataset name
        whisper_model: Whisper model name (optional)
    """
    audio_filename = os.path.basename(audio_path)
    audio_dir = os.path.dirname(audio_path)
    
    print("\n" + "="*80)
    print(f"File: {audio_filename}")
    print("="*80)
    
    reference_transcript = load_reference_transcript(transcript_dir, audio_filename, dataset=dataset)
    reference_speaker = load_speaker_label_auto(audio_dir, audio_filename, dataset=dataset)
    
    if not reference_transcript:
        print(f"Warning: No reference transcript found")
    
    if not reference_speaker:
        print(f"Warning: No reference speaker label found")
    
    database_path = config.get_speaker_database_path()
    if not os.path.exists(database_path):
        print(f"\nWarning: Speaker database not found at {database_path}")
        print("Run 'python sid_main.py --enroll' to create speaker database")
        predicted_speaker = None
        similarity = None
    else:
        identifier = SpeakerIdentifier()
        if identifier.load_database(database_path):
            result = identifier.identify_speaker(audio_path, top_k=1)
            if result:
                predicted_speaker, similarity = result
            else:
                predicted_speaker, similarity = None, None
        else:
            predicted_speaker, similarity = None, None
    
    model_name = whisper_model or config.WHISPER_MODEL
    hypothesis_transcript = transcribe_audio(audio_path, model_name=model_name)
    
    print("\n" + "-"*80)
    print("SPEAKER IDENTIFICATION")
    print("-"*80)
    if reference_speaker:
        print(f"Reference Speaker : {reference_speaker}")
    if predicted_speaker:
        speaker_match = "✓" if (reference_speaker == predicted_speaker) else "✗"
        print(f"Predicted Speaker : {predicted_speaker} {speaker_match}")
        if similarity:
            print(f"Similarity Score  : {similarity:.4f}")
    
    print("\n" + "-"*80)
    print("AUTOMATIC SPEECH RECOGNITION")
    print("-"*80)
    if reference_transcript:
        print(f"Reference: {reference_transcript}")
    
    if hypothesis_transcript and 'text' in hypothesis_transcript:
        print(f"Whisper  : {hypothesis_transcript['text']}")
        
        if reference_transcript:
            wer, cer = calculate_wer(reference_transcript, hypothesis_transcript['text'])
            print(f"\nWER: {wer:.2f}% | CER: {cer:.2f}%")
    
    print("="*80)

def process_batch(audio_dir, transcript_dir, label_dir, limit=5, dataset='Dev', whisper_model=None):
    """
    Process multiple audio files with both ASR and SID.
    
    Args:
        audio_dir: Directory with audio files
        transcript_dir: Directory with ASR transcripts
        label_dir: Directory with SID labels
        limit: Maximum number of files to process
        dataset: Dataset name
        whisper_model: Whisper model name (optional)
    """
    pairs = get_audio_files_with_transcripts(audio_dir, transcript_dir, limit=limit, dataset=dataset)
    
    if not pairs:
        print("No audio files with transcripts found")
        return
    
    database_path = config.get_speaker_database_path()
    if not os.path.exists(database_path):
        print(f"\nWarning: Speaker database not found at {database_path}")
        print("Run 'python sid_main.py --enroll' to create speaker database")
        print("Proceeding with ASR only...\n")
        use_sid = False
    else:
        use_sid = True
        identifier = SpeakerIdentifier()
        if not identifier.load_database(database_path):
            use_sid = False
    
    print(f"\nProcessing {len(pairs)} files...\n")
    
    asr_results = []
    sid_results = []
    
    for audio_path, reference_transcript in pairs:
        audio_filename = os.path.basename(audio_path)
        
        predicted_speaker = None
        similarity = None
        reference_speaker = None
        
        if use_sid:
            reference_speaker = load_speaker_label_auto(audio_dir, audio_filename, dataset=dataset)
            if reference_speaker:
                result = identifier.identify_speaker(audio_path, top_k=1)
                if result:
                    predicted_speaker, similarity = result
        
        model_name = whisper_model or config.WHISPER_MODEL
        hypothesis_transcript = transcribe_audio(audio_path, model_name=model_name)
        
        print("\n" + "="*80)
        print(f"File: {audio_filename}")
        print("="*80)
        
        if use_sid and reference_speaker:
            speaker_match = "✓" if (reference_speaker == predicted_speaker) else "✗"
            print(f"Speaker: {reference_speaker} (predicted: {predicted_speaker} {speaker_match})")
            sid_results.append({
                'reference': reference_speaker,
                'predicted': predicted_speaker,
                'correct': (reference_speaker == predicted_speaker)
            })
        
        if hypothesis_transcript and 'text' in hypothesis_transcript:
            print(f"Reference: {reference_transcript}")
            print(f"Whisper  : {hypothesis_transcript['text']}")
            
            wer, cer = calculate_wer(reference_transcript, hypothesis_transcript['text'])
            print(f"WER: {wer:.2f}% | CER: {cer:.2f}%")
            
            asr_results.append({
                'wer': wer,
                'cer': cer
            })
    
    if asr_results or sid_results:
        print("\n" + "="*80)
        print("BATCH SUMMARY")
        print("="*80)
        
        if asr_results:
            avg_wer = sum(r['wer'] for r in asr_results) / len(asr_results)
            avg_cer = sum(r['cer'] for r in asr_results) / len(asr_results)
            print(f"ASR - Average WER: {avg_wer:.2f}% | Average CER: {avg_cer:.2f}%")
        
        if sid_results:
            correct = sum(1 for r in sid_results if r['correct'])
            total = len(sid_results)
            accuracy = (correct / total) * 100
            print(f"SID - Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Combined ASR + SID Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file with both ASR and SID
  python combined_main.py --file fsc_p3_ASR_track2_dev_0018.wav
  
  # Batch process 5 files
  python combined_main.py --batch 5
  
  # Use different Whisper model
  python combined_main.py --file audio.wav --whisper-model base.en
  
  # Process from Train dataset
  python combined_main.py --dataset Train --batch 10
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Process a single audio file'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=5,
        help='Number of files to process in batch mode (default: 5)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='Dev',
        choices=['Dev', 'Train', 'Eval'],
        help='Dataset to use (default: Dev)'
    )
    
    parser.add_argument(
        '--whisper-model',
        type=str,
        default=None,
        help=f'Whisper model to use (default: {config.WHISPER_MODEL})'
    )
    
    args = parser.parse_args()
    
    audio_dir = config.get_audio_path(args.dataset)
    transcript_dir = config.get_transcript_path(args.dataset)
    label_dir = config.get_sid_label_path(args.dataset)
    
    print(f"\n{'='*60}")
    print(f"Combined ASR + SID Pipeline - {args.dataset} Dataset")
    print(f"{'='*60}")
    print(f"Audio directory: {audio_dir}")
    print(f"Transcript directory: {transcript_dir}")
    print(f"Label directory: {label_dir}")
    
    if args.file:
        audio_path = os.path.join(audio_dir, args.file)
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        
        process_single_file(
            audio_path, 
            transcript_dir, 
            label_dir, 
            dataset=args.dataset,
            whisper_model=args.whisper_model
        )
    else:
        process_batch(
            audio_dir,
            transcript_dir,
            label_dir,
            limit=args.batch,
            dataset=args.dataset,
            whisper_model=args.whisper_model
        )

if __name__ == "__main__":
    main()
