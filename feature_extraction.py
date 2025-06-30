#!/usr/bin/env python3
"""
Ses dosyalarından demans tespiti için özellik çıkarımı yapan Python scripti
"""

import os
# 🔧 Production fix: Set environment variables before importing librosa
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
os.environ['LIBROSA_CACHE_DIR'] = '/tmp'
os.environ['LIBROSA_CACHE_LEVEL'] = '10'

import numpy as np
import librosa
import librosa.display
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from scipy import stats
from audio_converter import convert_m4a_to_wav
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Ses dosyalarından özellik çıkarımı yapan sınıf"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path):
        """
        Ses dosyasını yükler (.m4a dosyaları otomatik olarak .wav'a çevrilir)
        
        Args:
            file_path (str): Ses dosyasının yolu
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        file_path = Path(file_path)
        
        # .m4a dosyalarını .wav'a çevir
        if file_path.suffix.lower() == '.m4a':
            print(f"🔄 .m4a dosyası tespit edildi, .wav'a çevriliyor: {file_path}")
            wav_path = convert_m4a_to_wav(str(file_path))
            if wav_path:
                file_path = Path(wav_path)
            else:
                raise ValueError(f"Dosya çevrilemedi: {file_path}")
        
        # Ses dosyasını yükle
        y, sr = librosa.load(str(file_path), sr=self.sample_rate)
        return y, sr
    
    def extract_mfcc_features(self, y, sr):
        """
        MFCC (Mel-Frequency Cepstral Coefficients) özelliklerini çıkarır
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: MFCC istatistikleri
        """
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        features = {}
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_skew'] = stats.skew(mfccs[i])
            features[f'mfcc_{i}_kurtosis'] = stats.kurtosis(mfccs[i])
        
        return features
    
    def extract_spectral_features(self, y, sr):
        """
        Spektral özellikler çıkarır
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Spektral özellikler
        """
        features = {}
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_pitch_features(self, y, sr):
        """
        Pitch (temel frekans) özelliklerini çıkarır
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Pitch özellikleri
        """
        features = {}
        
        # Fundamental frequency using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Extract pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_range'] = 0
        
        return features
    
    def extract_temporal_features(self, y, sr):
        """
        Zamansal özellikler çıkarır (konuşma hızı, duraklama süreleri)
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Zamansal özellikler
        """
        features = {}
        
        # Energy-based voice activity detection
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)     # 10ms hop
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Voice activity detection (simple threshold-based)
        threshold = np.mean(rms) * 0.3
        voice_frames = rms > threshold
        
        # Speaking segments
        speaking_segments = []
        pause_segments = []
        
        in_speech = False
        segment_start = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_speech:
                # Start of speech segment
                if segment_start < i:
                    pause_segments.append((segment_start, i))
                segment_start = i
                in_speech = True
            elif not is_voice and in_speech:
                # End of speech segment
                speaking_segments.append((segment_start, i))
                segment_start = i
                in_speech = False
        
        # Convert frame indices to time
        frame_to_time = hop_length / sr
        
        speaking_durations = [(end - start) * frame_to_time for start, end in speaking_segments]
        pause_durations = [(end - start) * frame_to_time for start, end in pause_segments]
        
        # Calculate features
        total_duration = len(y) / sr
        total_speaking_time = sum(speaking_durations) if speaking_durations else 0
        total_pause_time = sum(pause_durations) if pause_durations else 0
        
        features['total_duration'] = total_duration
        features['speaking_time'] = total_speaking_time
        features['pause_time'] = total_pause_time
        features['speaking_rate'] = total_speaking_time / total_duration if total_duration > 0 else 0
        features['pause_rate'] = total_pause_time / total_duration if total_duration > 0 else 0
        
        if speaking_durations:
            features['avg_speaking_duration'] = np.mean(speaking_durations)
            features['std_speaking_duration'] = np.std(speaking_durations)
            features['num_speaking_segments'] = len(speaking_durations)
        else:
            features['avg_speaking_duration'] = 0
            features['std_speaking_duration'] = 0
            features['num_speaking_segments'] = 0
        
        if pause_durations:
            features['avg_pause_duration'] = np.mean(pause_durations)
            features['std_pause_duration'] = np.std(pause_durations)
            features['num_pauses'] = len(pause_durations)
        else:
            features['avg_pause_duration'] = 0
            features['std_pause_duration'] = 0
            features['num_pauses'] = 0
        
        return features
    
    def extract_rhythm_features(self, y, sr):
        """
        Ritim ve tempo özelliklerini çıkarır
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Ritim özellikleri
        """
        features = {}
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Beat intervals
        if len(beats) > 1:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times)
            features['beat_interval_mean'] = np.mean(beat_intervals)
            features['beat_interval_std'] = np.std(beat_intervals)
        else:
            features['beat_interval_mean'] = 0
            features['beat_interval_std'] = 0
        
        return features
    
    def extract_all_features(self, file_path):
        """
        Tüm özellikleri çıkarır
        
        Args:
            file_path (str): Ses dosyasının yolu
            
        Returns:
            dict: Tüm özellikler
        """
        print(f"🎵 Özellik çıkarımı başlıyor: {file_path}")
        
        # Ses dosyasını yükle
        y, sr = self.load_audio(file_path)
        
        # Tüm özellikleri çıkar
        features = {}
        
        print("  📊 MFCC özellikleri çıkarılıyor...")
        features.update(self.extract_mfcc_features(y, sr))
        
        print("  🌈 Spektral özellikler çıkarılıyor...")
        features.update(self.extract_spectral_features(y, sr))
        
        print("  🎼 Pitch özellikleri çıkarılıyor...")
        features.update(self.extract_pitch_features(y, sr))
        
        print("  ⏰ Zamansal özellikler çıkarılıyor...")
        features.update(self.extract_temporal_features(y, sr))
        
        print("  🎵 Ritim özellikleri çıkarılıyor...")
        features.update(self.extract_rhythm_features(y, sr))
        
        # Dosya bilgilerini ekle
        features['file_path'] = str(file_path)
        features['duration'] = len(y) / sr
        
        print(f"✅ Toplam {len(features)} özellik çıkarıldı")
        
        return features
    
    def process_dataset(self, audio_directory, output_csv=None):
        """
        Bir dizindeki tüm ses dosyalarını işler
        
        Args:
            audio_directory (str): Ses dosyalarının bulunduğu dizin
            output_csv (str, optional): Çıktı CSV dosyasının adı
            
        Returns:
            pandas.DataFrame: Özellikler DataFrame'i
        """
        audio_path = Path(audio_directory)
        
        if not audio_path.exists():
            raise ValueError(f"Dizin bulunamadı: {audio_directory}")
        
        # Ses dosyalarını bul
        audio_extensions = ['.wav', '.m4a', '.mp3', '.flac']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(audio_path.glob(f"*{ext}")))
            audio_files.extend(list(audio_path.glob(f"*{ext.upper()}")))
        
        if not audio_files:
            raise ValueError(f"Ses dosyası bulunamadı: {audio_directory}")
        
        print(f"📁 {len(audio_files)} ses dosyası bulundu")
        
        # Her dosyayı işle
        all_features = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] İşleniyor: {audio_file.name}")
            
            try:
                features = self.extract_all_features(audio_file)
                all_features.append(features)
            except Exception as e:
                print(f"❌ Hata: {audio_file.name} - {str(e)}")
                continue
        
        # DataFrame oluştur
        if all_features:
            df = pd.DataFrame(all_features)
            
            if output_csv:
                df.to_csv(output_csv, index=False)
                print(f"\n💾 Özellikler kaydedildi: {output_csv}")
            
            print(f"\n📊 Özellik özeti:")
            print(f"   - Dosya sayısı: {len(df)}")
            print(f"   - Özellik sayısı: {len(df.columns)}")
            
            return df
        else:
            print("❌ Hiçbir dosya işlenemedi")
            return None

def main():
    """Ana fonksiyon"""
    import sys
    
    if len(sys.argv) < 2:
        print("Ses Özellik Çıkarıcı")
        print("Kullanım:")
        print("  Tek dosya: python feature_extraction.py dosya.wav")
        print("  Dizin: python feature_extraction.py /ses/dizini")
        print("  CSV çıktısı ile: python feature_extraction.py /ses/dizini cikti.csv")
        return
    
    extractor = AudioFeatureExtractor()
    input_path = sys.argv[1]
    
    if Path(input_path).is_file():
        # Tek dosya işleme
        features = extractor.extract_all_features(input_path)
        
        print(f"\n📊 Çıkarılan Özellikler:")
        for key, value in features.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    elif Path(input_path).is_dir():
        # Dizin işleme
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "audio_features.csv"
        df = extractor.process_dataset(input_path, output_csv)
        
        if df is not None:
            print(f"\n📈 Özellik İstatistikleri:")
            print(df.describe())
    
    else:
        print(f"❌ Dosya veya dizin bulunamadı: {input_path}")

if __name__ == "__main__":
    main() 