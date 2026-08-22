"""
Audio Processing Module for Beko RGS Noise Detection
Handles feature extraction and audio augmentation
"""
import numpy as np
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, sample_rate=22050, duration=5, n_mels=128, n_mfcc=40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = sample_rate * duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
    
    def load_audio(self, file_path):
        """Load and normalize audio file"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            # Pad or trim to fixed length
            if len(y) < self.samples:
                y = np.pad(y, (0, self.samples - len(y)), mode='constant')
            else:
                y = y[:self.samples]
            return y
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def extract_features(self, y):
        """Extract combined audio features (Mel spectrogram + MFCCs)"""
        try:
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_features = np.mean(mel_spec_db, axis=1)
            
            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc
            )
            mfcc_features = np.mean(mfccs, axis=1)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Combine all features
            features = np.concatenate([
                mel_features,
                mfcc_features,
                [spectral_centroid, spectral_rolloff, zero_crossing_rate]
            ])
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def augment_audio(self, y, num_augmentations=5):
        """Generate augmented versions of audio"""
        augmented = [y]  # Include original
        
        try:
            # Pitch shift
            for n_steps in [2, -2]:
                augmented.append(librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=n_steps))
            
            # Time stretch
            for rate in [0.9, 1.1]:
                stretched = librosa.effects.time_stretch(y, rate=rate)
                if len(stretched) < self.samples:
                    stretched = np.pad(stretched, (0, self.samples - len(stretched)))
                else:
                    stretched = stretched[:self.samples]
                augmented.append(stretched)
            
            # Add noise
            noise = np.random.normal(0, 0.005, y.shape)
            augmented.append(y + noise)
            
            # Dynamic range compression
            augmented.append(librosa.effects.percussive(y))
            
            logger.info(f"Generated {len(augmented)} augmented samples")
            return augmented[:num_augmentations + 1]  # Limit to desired number
        except Exception as e:
            logger.error(f"Error during augmentation: {e}")
            return [y]  # Return original if augmentation fails
