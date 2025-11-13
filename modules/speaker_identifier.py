"""
Speaker identification module using pre-trained embeddings.
Uses pyannote.audio for speaker embedding extraction.
"""

import os
import torch
import numpy as np
import pickle
import soundfile as sf
from tqdm import tqdm
import config

class SpeakerIdentifier:
    """Speaker identification system with enrollment and prediction."""
    
    def __init__(self, model_name=None):
        """
        Initialize the speaker identifier.
        
        Args:
            model_name: Name of the embedding model (default from config)
        """
        self.model_name = model_name or config.SPEAKER_EMBEDDING_MODEL
        self.model = None
        self.speaker_database = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the pre-trained embedding model."""
        if self.model is not None:
            return
        
        try:
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
    
    def extract_embedding(self, audio_path):
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Numpy array containing the embedding vector
        """
        if self.model is None:
            self.load_model()
        
        try:
            audio, sr = sf.read(audio_path)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding from {audio_path}: {e}")
            return None
    
    def enroll_speakers(self, speaker_files_dict, save_path=None):
        """
        Enroll speakers by creating averaged embeddings from their audio files.
        
        Args:
            speaker_files_dict: Dictionary mapping speaker_id to list of audio paths
            save_path: Path to save the speaker database (optional)
            
        Returns:
            Dictionary mapping speaker_id to averaged embedding
        """
        if self.model is None:
            self.load_model()
        
        speaker_database = {}
        speaker_items = list(speaker_files_dict.items())
        total_speakers = len(speaker_items)
        total_audio_files = sum(len(files) for _, files in speaker_items)

        print(f"\n{'='*60}")
        print(f"Enrolling {total_speakers} speakers...")
        print(f"{'='*60}\n")

        speaker_bar = tqdm(
            total=total_speakers,
            desc="Enrolling speakers",
            position=0,
            dynamic_ncols=True,
        )
        audio_bar = tqdm(
            total=total_audio_files,
            desc="Audio files processed",
            position=1,
            dynamic_ncols=True,
            leave=True,
        )

        try:
            for speaker_id, audio_files in speaker_items:
                embeddings = []

                for audio_path in audio_files:
                    embedding = self.extract_embedding(audio_path)
                    if embedding is not None:
                        embeddings.append(embedding)
                    audio_bar.update(1)

                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    speaker_database[speaker_id] = avg_embedding
                else:
                    print(f"Warning: No valid embeddings for speaker {speaker_id}")

                speaker_bar.update(1)
                speaker_bar.set_postfix({"last_speaker": speaker_id})
        finally:
            speaker_bar.close()
            audio_bar.close()
        
        self.speaker_database = speaker_database
        
        if save_path:
            self.save_database(save_path)
        
        print(f"\n{'='*60}")
        print(f"Enrollment complete!")
        print(f"Enrolled {len(speaker_database)} speakers")
        print(f"{'='*60}\n")
        
        return speaker_database
    
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
    
    def identify_speaker(self, audio_path, top_k=1):
        """
        Identify speaker from audio file using cosine similarity.
        
        Args:
            audio_path: Path to audio file
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
        
        test_embedding = self.extract_embedding(audio_path)
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
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
