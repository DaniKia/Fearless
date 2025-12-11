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

def normalize_embedding(embedding, method='l2', global_mean=None):
    """
    Normalize an embedding vector.
    
    Args:
        embedding: Numpy array containing the embedding vector
        method: Normalization method - 'l2', 'l2-centered', or None
        global_mean: Optional global mean for centering (required for l2-centered)
        
    Returns:
        Normalized embedding vector
    """
    if method is None or method == 'none':
        return embedding
    
    if method == 'l2-centered':
        if global_mean is not None:
            embedding = embedding - global_mean
    
    if method in ('l2', 'l2-centered'):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    
    return embedding


class SpeakerIdentifier:
    """Speaker identification system with enrollment and prediction."""
    
    def __init__(self, model_name=None, batch_size=16, normalize_method=None):
        """
        Initialize the speaker identifier.
        
        Args:
            model_name: Name of the embedding model (default from config)
            batch_size: Number of files to process per GPU batch (default: 16)
            normalize_method: Embedding normalization method ('l2', 'l2-centered', or None)
        """
        self.model_name = model_name or config.SPEAKER_EMBEDDING_MODEL
        self.model = None
        self.speaker_database = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.normalize_method = normalize_method
        self.global_mean = None
        
    def load_model(self):
        """Load the pre-trained embedding model."""
        if self.model is not None:
            return
        
        try:
            import torchaudio
            if not hasattr(torchaudio, 'list_audio_backends'):
                torchaudio.list_audio_backends = lambda: ['soundfile']
            
            from speechbrain.inference import EncoderClassifier
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
    
    def enroll_speakers(self, speaker_files_dict, save_path=None, preprocess_config=None, metadata=None):
        """
        Enroll speakers by creating averaged embeddings from their audio files.
        Uses GPU batch processing for faster enrollment.
        
        Args:
            speaker_files_dict: Dictionary mapping speaker_id to list of audio paths
            save_path: Path to save the speaker database (optional)
            preprocess_config: Optional PreprocessConfig for audio preprocessing
            metadata: Optional metadata dictionary to store with embeddings
            
        Returns:
            Dictionary mapping speaker_id to averaged embedding
        """
        if self.model is None:
            self.load_model()
        
        speaker_database = {}
        processed_speakers = set()

        if save_path and os.path.exists(save_path):
            if self.normalize_method == 'l2-centered':
                print(f"Warning: l2-centered normalization requires fresh enrollment.")
                print(f"         Existing database at {save_path} will be overwritten.")
            else:
                print(f"Existing database found at {save_path}. Resuming enrollment...")
                if self.load_database(save_path):
                    speaker_database = self.speaker_database or {}
                    processed_speakers = set(speaker_database.keys())
                else:
                    print("Warning: Failed to load existing database. Starting fresh enrollment.")
                    speaker_database = {}

        self.speaker_database = speaker_database
        self.metadata = metadata

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
        if self.normalize_method:
            print(f"Embedding normalization: {self.normalize_method}")
        print(f"{'='*60}\n")
        
        if self.normalize_method == 'l2-centered':
            speaker_database, all_embeddings_flat = self._enroll_two_pass(
                speaker_files_dict, processed_speakers, total_speakers, total_files, 
                processed_files, preprocess_config, save_path
            )
        else:
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
                        if self.normalize_method == 'l2':
                            embeddings = [normalize_embedding(emb, 'l2') for emb in embeddings]
                        
                        avg_embedding = np.mean(embeddings, axis=0)
                        
                        if self.normalize_method == 'l2':
                            avg_embedding = normalize_embedding(avg_embedding, 'l2')
                        
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
    
    def _enroll_two_pass(self, speaker_files_dict, processed_speakers, total_speakers, 
                         total_files, processed_files, preprocess_config, save_path):
        """
        Two-pass enrollment for l2-centered normalization.
        Pass 1: Collect all embeddings and compute global mean
        Pass 2: Center embeddings, compute centroids, and normalize
        """
        speaker_embeddings_raw = {}
        all_embeddings_flat = []
        
        print("Pass 1: Extracting embeddings and computing global mean...")
        speaker_count = 0
        with tqdm(total=total_files, desc="Extracting embeddings", unit="file", initial=processed_files) as pbar:
            for speaker_id, audio_files in speaker_files_dict.items():
                speaker_count += 1
                num_files = len(audio_files)
                pbar.set_description(f"[{speaker_count}/{total_speakers}] {speaker_id} ({num_files} files)")
                
                if speaker_id in processed_speakers:
                    continue
                
                embeddings = self._process_files_in_batches(audio_files, pbar, preprocess_config)
                
                if embeddings:
                    speaker_embeddings_raw[speaker_id] = embeddings
                    all_embeddings_flat.extend(embeddings)
                else:
                    print(f"\nWarning: No valid embeddings for speaker {speaker_id}")
        
        if not all_embeddings_flat:
            return {}, []
        
        self.global_mean = np.mean(all_embeddings_flat, axis=0)
        print(f"Global mean computed from {len(all_embeddings_flat)} embeddings")
        
        print("Pass 2: Centering and normalizing...")
        speaker_database = self.speaker_database or {}
        
        for speaker_id, embeddings in tqdm(speaker_embeddings_raw.items(), desc="Computing centroids"):
            centered_embeddings = [emb - self.global_mean for emb in embeddings]
            normalized_embeddings = [normalize_embedding(emb, 'l2') for emb in centered_embeddings]
            
            avg_embedding = np.mean(normalized_embeddings, axis=0)
            avg_embedding = normalize_embedding(avg_embedding, 'l2')
            
            speaker_database[speaker_id] = avg_embedding
        
        self.speaker_database = speaker_database
        if save_path:
            self.save_database(save_path)
        
        return speaker_database, all_embeddings_flat
    
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
        """Save speaker database to file with metadata."""
        if self.speaker_database is None:
            print("No speaker database to save")
            return
        
        try:
            data = {
                'embeddings': self.speaker_database,
                'metadata': self.metadata,
                'global_mean': self.global_mean
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self, database_path):
        """Load speaker database from file (supports both old and new formats)."""
        try:
            with open(database_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'embeddings' in data:
                self.speaker_database = data['embeddings']
                self.metadata = data.get('metadata')
                self.global_mean = data.get('global_mean')
            else:
                self.speaker_database = data
                self.metadata = None
                self.global_mean = None
            
            print(f"Loaded speaker database with {len(self.speaker_database)} speakers")
            if self.metadata:
                print(f"  Enrollment: {self.metadata.get('folder', 'N/A')}/{self.metadata.get('dataset', 'N/A')}")
                print(f"  Date: {self.metadata.get('date', 'N/A')}")
                preproc = self.metadata.get('preprocessing', {})
                print(f"  Preprocessing: {'Enabled' if preproc.get('enabled') else 'Disabled'}")
                norm_method = self.metadata.get('normalize_method')
                if norm_method:
                    self.normalize_method = norm_method
                    print(f"  Normalization: {norm_method}")
                if self.global_mean is not None:
                    print(f"  Global mean: stored ({len(self.global_mean)} dims)")
            else:
                print("  (Legacy format - no metadata)")
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
        
        if self.normalize_method == 'l2-centered' and self.global_mean is not None:
            test_embedding = test_embedding - self.global_mean
            test_embedding = normalize_embedding(test_embedding, 'l2')
        elif self.normalize_method == 'l2':
            test_embedding = normalize_embedding(test_embedding, 'l2')
        
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
