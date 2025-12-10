"""
Speaker identification module using pre-trained embeddings.
Uses SpeechBrain ECAPA embeddings with batch processing for efficiency.
"""

import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import config

class SpeakerIdentifier:
    """Speaker identification system with enrollment and prediction."""
    
    def __init__(self, model_name=None, batch_size=16):
        """
        Initialize the speaker identifier.
        
        Args:
            model_name: Name of the embedding model (default from config)
            batch_size: Number of files to process per GPU batch (default: 16)
        """
        self.model_name = model_name or config.SPEAKER_EMBEDDING_MODEL
        self.model = None
        self.speaker_database = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
    def load_model(self):
        """Load the pre-trained embedding model."""
        if self.model is not None:
            return
        
        try:
            import torchaudio
            if not hasattr(torchaudio, 'list_audio_backends'):
                torchaudio.list_audio_backends = lambda: ['soundfile']
            
            from speechbrain.pretrained import EncoderClassifier
            print(f"Loading speaker embedding model: {self.model_name}")
            print(f"Using device: {self.device}")
            
            self.model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_embedding_from_waveform(self, audio, sample_rate=None):
        """
        Extract speaker embedding from audio waveform.
        
        Args:
            audio: Audio waveform as numpy array (must be mono)
            sample_rate: Sample rate (not used by model, for API consistency)
            
        Returns:
            Numpy array containing the embedding vector, or None on error
        """
        if self.model is None:
            self.load_model()
        
        try:
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
                
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def enroll_speakers(self, speaker_files_dict, save_path=None, preprocess_config=None):
        """
        Enroll speakers by creating averaged embeddings from their audio files.
        Uses GPU batch processing for faster enrollment.
        
        Args:
            speaker_files_dict: Dictionary mapping speaker_id to list of audio paths
            save_path: Path to save the speaker database (optional)
            preprocess_config: Optional PreprocessConfig for audio preprocessing
            
        Returns:
            Dictionary mapping speaker_id to averaged embedding
        """
        if self.model is None:
            self.load_model()
        
        speaker_database = {}
        processed_speakers = set()

        if save_path and os.path.exists(save_path):
            print(f"Existing database found at {save_path}. Resuming enrollment...")
            if self.load_database(save_path):
                speaker_database = self.speaker_database or {}
                processed_speakers = set(speaker_database.keys())
            else:
                print("Warning: Failed to load existing database. Starting fresh enrollment.")
                speaker_database = {}

        self.speaker_database = speaker_database

        total_speakers = len(speaker_files_dict)
        total_files = sum(len(files) for files in speaker_files_dict.values())
        processed_files = sum(
            len(files) for speaker_id, files in speaker_files_dict.items()
            if speaker_id in processed_speakers
        )
        remaining_speakers = total_speakers - len(processed_speakers)
        
        print(f"\n{'='*60}")
        print(f"Enrolling {total_speakers} speakers from {total_files} audio files")
        if processed_speakers:
            print(f"Resuming: {len(processed_speakers)} speakers already enrolled, {remaining_speakers} remaining")
        print(f"Using batch size: {self.batch_size}")
        if preprocess_config:
            print(f"Preprocessing enabled: {preprocess_config.enable_mono}, resample={preprocess_config.enable_resample}")
        print(f"{'='*60}\n")
        
        speaker_count = 0
        with tqdm(total=total_files, desc="Processing audio files", unit="file", initial=processed_files) as pbar:
            for speaker_id, audio_files in speaker_files_dict.items():
                speaker_count += 1
                num_files = len(audio_files)
                pbar.set_description(f"[{speaker_count}/{total_speakers}] {speaker_id} ({num_files} files)")

                if speaker_id in processed_speakers:
                    continue

                embeddings = self._process_files_in_batches(audio_files, pbar, preprocess_config)

                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    speaker_database[speaker_id] = avg_embedding
                    self.speaker_database = speaker_database
                    if save_path:
                        self.save_database(save_path)
                else:
                    print(f"\nWarning: No valid embeddings for speaker {speaker_id}")

        print(f"\n{'='*60}")
        print(f"Enrollment complete!")
        print(f"Enrolled {len(speaker_database)} speakers")
        print(f"{'='*60}\n")
        
        return speaker_database
    
    def _process_files_in_batches(self, audio_files, pbar, preprocess_config=None):
        """
        Process audio files in batches using GPU batch processing.
        
        Args:
            audio_files: List of audio file paths
            pbar: Progress bar to update
            preprocess_config: Optional PreprocessConfig for audio preprocessing
            
        Returns:
            List of embeddings
        """
        from modules.audio_io import load_audio_safe
        from modules.audio_preprocessor import preprocess_audio
        
        if self.model is None:
            self.load_model()
        
        all_embeddings = []
        
        for i in range(0, len(audio_files), self.batch_size):
            batch_files = audio_files[i:i + self.batch_size]
            batch_audio = []
            batch_lengths = []
            valid_indices = []
            
            for idx, audio_path in enumerate(batch_files):
                audio, sr, error = load_audio_safe(audio_path)
                
                if error:
                    print(f"\nWarning: Error loading {audio_path}: {error}")
                    continue
                
                if preprocess_config is not None:
                    result = preprocess_audio(audio, sr, preprocess_config)
                    if not result.is_valid:
                        print(f"\nWarning: Invalid audio after preprocessing {audio_path}")
                        continue
                    audio = result.waveform
                else:
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                
                batch_audio.append(torch.FloatTensor(audio))
                batch_lengths.append(len(audio))
                valid_indices.append(idx)
            
            if not batch_audio:
                pbar.update(len(batch_files))
                continue
            
            max_length = max(batch_lengths)
            padded_batch = []
            for audio in batch_audio:
                current_len = len(audio)
                if current_len < max_length:
                    padding = torch.zeros(max_length - current_len)
                    audio = torch.cat([audio, padding])
                elif current_len > max_length:
                    audio = audio[:max_length]
                padded_batch.append(audio)
            
            batch_tensor = torch.stack(padded_batch).to(self.device)
            
            relative_lengths = torch.tensor(batch_lengths, dtype=torch.float) / max_length
            lengths_tensor = torch.clamp(relative_lengths, max=1.0).to(self.device)
            
            try:
                with torch.no_grad():
                    embeddings = self.model.encode_batch(batch_tensor, lengths_tensor)
                    embeddings = embeddings.cpu().numpy()
                    
                    for emb in embeddings:
                        if emb.ndim > 1:
                            emb = emb.flatten()
                        all_embeddings.append(emb)
                        
            except Exception as e:
                print(f"\nWarning: Error processing batch: {e}")
            
            pbar.update(len(batch_files))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_embeddings
    
    def save_database(self, save_path):
        """Save speaker database to file."""
        if self.speaker_database is None:
            print("No speaker database to save")
            return
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(self.speaker_database, f)
            print(f"Speaker database saved to: {save_path}")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self, database_path):
        """Load speaker database from file."""
        try:
            with open(database_path, 'rb') as f:
                self.speaker_database = pickle.load(f)
            print(f"Loaded speaker database with {len(self.speaker_database)} speakers")
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def identify_speaker_from_waveform(self, audio, sample_rate, top_k=1):
        """
        Identify speaker from audio waveform using cosine similarity.
        
        Args:
            audio: Audio waveform as numpy array (mono)
            sample_rate: Sample rate of audio
            top_k: Number of top candidates to return
            
        Returns:
            If top_k=1: (speaker_id, similarity_score)
            If top_k>1: List of (speaker_id, similarity_score) tuples
        """
        if self.speaker_database is None:
            print("Error: No speaker database loaded")
            return None
        
        if self.model is None:
            self.load_model()
        
        test_embedding = self.extract_embedding_from_waveform(audio, sample_rate)
        if test_embedding is None:
            return None
        
        similarities = {}
        for speaker_id, speaker_embedding in self.speaker_database.items():
            similarity = self._cosine_similarity(test_embedding, speaker_embedding)
            similarities[speaker_id] = similarity
        
        sorted_speakers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        if top_k == 1:
            return sorted_speakers[0]
        else:
            return sorted_speakers[:top_k]
    
    def identify_speaker(self, audio_path, top_k=1, preprocess_config=None):
        """
        Identify speaker from audio file using cosine similarity.
        
        Args:
            audio_path: Path to audio file
            top_k: Number of top candidates to return
            preprocess_config: Optional PreprocessConfig for audio preprocessing
            
        Returns:
            If top_k=1: (speaker_id, similarity_score)
            If top_k>1: List of (speaker_id, similarity_score) tuples
        """
        from modules.audio_io import load_audio_safe
        from modules.audio_preprocessor import preprocess_audio
        
        audio, sr, error = load_audio_safe(audio_path)
        if error:
            print(f"Error loading {audio_path}: {error}")
            return None
        
        if preprocess_config is not None:
            result = preprocess_audio(audio, sr, preprocess_config)
            if not result.is_valid:
                print(f"Error: Invalid audio after preprocessing {audio_path}")
                return None
            audio = result.waveform
            sr = result.sample_rate
        else:
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        
        return self.identify_speaker_from_waveform(audio, sr, top_k)
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
