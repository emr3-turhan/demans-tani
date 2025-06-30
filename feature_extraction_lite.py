#!/usr/bin/env python3
"""
🎵 Lightweight Feature Extraction for Production
Avoids problematic librosa functions that cause numba issues
"""

import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from audio_converter import convert_m4a_to_wav
import warnings
warnings.filterwarnings('ignore')

class LightweightFeatureExtractor:
    """
    Production-safe feature extractor using only stable librosa functions
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path):
        """Load audio file safely"""
        try:
            # Convert to wav if needed
            if str(file_path).endswith('.m4a'):
                wav_path = convert_m4a_to_wav(str(file_path))
                if wav_path:
                    file_path = wav_path
            
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            print(f"❌ Audio loading error: {e}")
            raise
    
    def extract_basic_features(self, y, sr):
        """Extract basic features without problematic numba functions"""
        features = {}
        
        # 1. Basic statistical features
        features['mean'] = float(np.mean(y))
        features['std'] = float(np.std(y))
        features['max'] = float(np.max(y))
        features['min'] = float(np.min(y))
        features['energy'] = float(np.sum(y**2))
        features['rms'] = float(np.sqrt(np.mean(y**2)))
        
        # 2. Zero crossing rate (safe)
        try:
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
        except:
            features['zcr_mean'] = 0.0
            features['zcr_std'] = 0.0
        
        # 3. Spectral centroid (usually safe)
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        except:
            features['spectral_centroid_mean'] = 0.0
            features['spectral_centroid_std'] = 0.0
        
        # 4. Spectral rolloff (usually safe)
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        except:
            features['spectral_rolloff_mean'] = 0.0
            features['spectral_rolloff_std'] = 0.0
        
        # 5. MFCC (first few coefficients only - safer)
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)  # Reduced from 13
            for i in range(8):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        except Exception as e:
            print(f"⚠️ MFCC extraction failed: {e}")
            for i in range(8):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
        
        # 6. Duration and basic tempo
        features['duration'] = float(len(y) / sr)
        
        # Simple tempo estimation without beat tracking
        try:
            # Use onset detection instead of beat tracking
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                intervals = np.diff(onset_times)
                features['avg_onset_interval'] = float(np.mean(intervals))
                features['onset_rate'] = float(len(onset_frames) / features['duration'])
            else:
                features['avg_onset_interval'] = 0.0
                features['onset_rate'] = 0.0
        except:
            features['avg_onset_interval'] = 0.0
            features['onset_rate'] = 0.0
        
        # 7. Chroma features (first few only)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(6):  # First 6 chroma features
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
        except:
            for i in range(6):
                features[f'chroma_{i}_mean'] = 0.0
        
        # 8. Spectral contrast (reduced)
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(4):  # First 4 contrast bands
                features[f'contrast_{i}_mean'] = float(np.mean(contrast[i]))
        except:
            for i in range(4):
                features[f'contrast_{i}_mean'] = 0.0
        
        return features
    
    def extract_all_features(self, file_path):
        """Extract all features with production-safe methods"""
        print(f"🎵 Lightweight feature extraction: {file_path}")
        
        try:
            # Load audio
            y, sr = self.load_audio(file_path)
            
            # Extract features
            features = self.extract_basic_features(y, sr)
            
            # Add file info
            features['file_path'] = str(file_path)
            features['sample_rate'] = float(sr)
            
            # Ensure we have exactly 60 features for model compatibility
            feature_count = len(features)
            target_count = 60
            
            if feature_count < target_count:
                # Pad with zeros
                for i in range(feature_count, target_count):
                    features[f'pad_feature_{i}'] = 0.0
            
            print(f"✅ Extracted {len(features)} lightweight features")
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            # Return minimal features
            features = {}
            for i in range(60):
                features[f'minimal_feature_{i}'] = 0.0
            features['file_path'] = str(file_path)
            features['duration'] = 0.0
            return features

def main():
    """Test the lightweight extractor"""
    extractor = LightweightFeatureExtractor()
    
    # Test with a file
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        features = extractor.extract_all_features(file_path)
        print(f"Feature count: {len(features)}")
        print("First 10 features:")
        for i, (key, value) in enumerate(list(features.items())[:10]):
            print(f"  {key}: {value}")
    else:
        print("Usage: python feature_extraction_lite.py audio_file.wav")

if __name__ == "__main__":
    main() 