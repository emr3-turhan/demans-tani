#!/usr/bin/env python3
"""
Eğitilmiş Sentetik Model Test Scripti
Yeni ses dosyalarını eğitilmiş model ile tahmin eder
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

from feature_extraction import AudioFeatureExtractor

class TrainedModelTester:
    """Eğitilmiş model test sınıfı"""
    
    def __init__(self, model_file):
        """
        Args:
            model_file (str): Eğitilmiş model dosyası (.pkl)
        """
        self.model_file = Path(model_file)
        self.load_model()
        self.feature_extractor = AudioFeatureExtractor()
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_file}")
        
        print(f"📥 Model yükleniyor: {self.model_file}")
        
        with open(self.model_file, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.label_encoder = self.model_data['label_encoder']
        self.model_name = self.model_data['model_name']
        self.training_date = self.model_data['training_date']
        
        print(f"✅ {self.model_name} modeli yüklendi")
        print(f"📅 Eğitim tarihi: {self.training_date}")
        print(f"🏷️ Sınıflar: {self.label_encoder.classes_}")
        
        # Feature columns'u öğren (training features'dan)
        self._determine_feature_columns()
        
    def _determine_feature_columns(self):
        """Feature sütunlarını belirle"""
        # Training data'dan feature columns'u çıkar
        # Bu gerçek uygulamada model metadata'sında saklanmalı
        # Şimdilik mevcut feature extractor'dan çıkaralım
        
        print("🔍 Feature sütunları belirleniyor...")
        
        # Test için dummy features
        dummy_features = {
            'mfcc_0_mean': 0, 'mfcc_0_std': 0, 'mfcc_0_skew': 0, 'mfcc_0_kurtosis': 0,
            'mfcc_1_mean': 0, 'mfcc_1_std': 0, 'mfcc_1_skew': 0, 'mfcc_1_kurtosis': 0,
            'mfcc_2_mean': 0, 'mfcc_2_std': 0, 'mfcc_2_skew': 0, 'mfcc_2_kurtosis': 0,
            'mfcc_3_mean': 0, 'mfcc_3_std': 0, 'mfcc_3_skew': 0, 'mfcc_3_kurtosis': 0,
            'mfcc_4_mean': 0, 'mfcc_4_std': 0, 'mfcc_4_skew': 0, 'mfcc_4_kurtosis': 0,
            'mfcc_5_mean': 0, 'mfcc_5_std': 0, 'mfcc_5_skew': 0, 'mfcc_5_kurtosis': 0,
            'mfcc_6_mean': 0, 'mfcc_6_std': 0, 'mfcc_6_skew': 0, 'mfcc_6_kurtosis': 0,
            'mfcc_7_mean': 0, 'mfcc_7_std': 0, 'mfcc_7_skew': 0, 'mfcc_7_kurtosis': 0,
            'mfcc_8_mean': 0, 'mfcc_8_std': 0, 'mfcc_8_skew': 0, 'mfcc_8_kurtosis': 0,
            'mfcc_9_mean': 0, 'mfcc_9_std': 0, 'mfcc_9_skew': 0, 'mfcc_9_kurtosis': 0,
            'mfcc_10_mean': 0, 'mfcc_10_std': 0, 'mfcc_10_skew': 0, 'mfcc_10_kurtosis': 0,
            'mfcc_11_mean': 0, 'mfcc_11_std': 0, 'mfcc_11_skew': 0, 'mfcc_11_kurtosis': 0,
            'mfcc_12_mean': 0, 'mfcc_12_std': 0, 'mfcc_12_skew': 0, 'mfcc_12_kurtosis': 0,
            'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
            'spectral_rolloff_mean': 0, 'spectral_rolloff_std': 0,
            'spectral_contrast_mean': 0, 'spectral_contrast_std': 0,
            'zcr_mean': 0, 'zcr_std': 0,
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0, 'pitch_max': 0, 'pitch_range': 0,
            'speaking_rate': 0, 'avg_pause_duration': 0, 'voice_activity_ratio': 0,
            'total_segments': 0, 'avg_segment_duration': 0,
            'tempo_mean': 0, 'tempo_std': 0, 'beat_intervals_mean': 0, 'beat_intervals_std': 0
        }
        
        # Numeric feature columns
        self.feature_columns = []
        for col in dummy_features.keys():
            if col not in ['filename', 'class', 'text']:
                self.feature_columns.append(col)
        
        print(f"📊 {len(self.feature_columns)} feature sütunu belirlendi")
        
    def predict_single_file(self, audio_file):
        """
        Tek bir ses dosyası için tahmin yap
        
        Args:
            audio_file (str): Ses dosyası yolu
            
        Returns:
            dict: Tahmin sonuçları
        """
        audio_file = Path(audio_file)
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_file}")
        
        print(f"🎵 Özellik çıkarımı: {audio_file.name}")
        
        # Özellik çıkar
        features = self.feature_extractor.extract_all_features(str(audio_file))
        
        # Feature vektörü hazırla
        feature_vector = []
        missing_features = []
        
        for col in self.feature_columns:
            if col in features:
                value = features[col]
                # NaN kontrolü
                if pd.isna(value) or np.isnan(value):
                    value = 0.0
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
                missing_features.append(col)
        
        if missing_features:
            print(f"⚠️ {len(missing_features)} eksik özellik 0 ile dolduruldu")
        
        # Tahmin yap
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        prediction = self.model.predict(feature_vector)
        probability = self.model.predict_proba(feature_vector)[0]
        
        predicted_class = self.label_encoder.inverse_transform(prediction)[0]
        
        # Sınıf probabiliteleri
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = float(probability[i])
        
        confidence = float(np.max(probability))
        
        result = {
            'file': str(audio_file),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'features_extracted': len([k for k, v in features.items() if isinstance(v, (int, float))]),
            'missing_features': len(missing_features),
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, audio_files, output_file=None):
        """
        Birden fazla ses dosyası için tahmin yap
        
        Args:
            audio_files (list): Ses dosyalarının listesi
            output_file (str, optional): Sonuçları kaydetmek için dosya
            
        Returns:
            list: Tüm tahmin sonuçları
        """
        print(f"🎯 Toplu tahmin başlıyor: {len(audio_files)} dosya")
        
        all_results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] İşleniyor: {Path(audio_file).name}")
            
            try:
                result = self.predict_single_file(audio_file)
                all_results.append(result)
                
                print(f"   📊 Tahmin: {result['predicted_class']} (Güven: {result['confidence']:.3f})")
                
            except Exception as e:
                print(f"   ❌ Hata: {str(e)}")
                all_results.append({
                    'file': str(audio_file),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Sonuçları kaydet
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Sonuçlar kaydedildi: {output_file}")
        
        return all_results
    
    def print_prediction_summary(self, results):
        """Tahmin sonuçlarını özetle"""
        print(f"\n📈 Tahmin Özeti:")
        
        successful_predictions = [r for r in results if 'predicted_class' in r]
        failed_predictions = [r for r in results if 'error' in r]
        
        print(f"   ✅ Başarılı: {len(successful_predictions)}")
        print(f"   ❌ Başarısız: {len(failed_predictions)}")
        
        if successful_predictions:
            # Sınıf dağılımı
            class_counts = {}
            for result in successful_predictions:
                predicted_class = result['predicted_class']
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            
            print(f"\n📊 Tahmin Dağılımı:")
            for class_name, count in class_counts.items():
                print(f"   {class_name}: {count} dosya")
            
            # Ortalama güven
            avg_confidence = sum(r['confidence'] for r in successful_predictions) / len(successful_predictions)
            print(f"\n🎯 Ortalama Güven: {avg_confidence:.3f}")

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Eğitilmiş Sentetik Model Test Scripti")
    parser.add_argument('model', help='Eğitilmiş model dosyası (.pkl)')
    parser.add_argument('audio_files', nargs='+', help='Test edilecek ses dosyaları')
    parser.add_argument('--output', '-o', help='Sonuçları kaydetmek için JSON dosyası')
    parser.add_argument('--verbose', '-v', action='store_true', help='Detaylı çıktı')
    
    args = parser.parse_args()
    
    try:
        # Model test edici
        tester = TrainedModelTester(args.model)
        
        # Ses dosyalarını topla
        audio_files = []
        for pattern in args.audio_files:
            path = Path(pattern)
            if path.is_file():
                audio_files.append(str(path))
            elif path.is_dir():
                # Dizindeki ses dosyalarını bul
                for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
                    audio_files.extend([str(f) for f in path.glob(ext)])
            else:
                print(f"⚠️ Dosya/dizin bulunamadı: {pattern}")
        
        if not audio_files:
            print("❌ Hiçbir ses dosyası bulunamadı!")
            return 1
        
        print(f"📁 {len(audio_files)} ses dosyası bulundu")
        
        # Tahmin yap
        if len(audio_files) == 1:
            result = tester.predict_single_file(audio_files[0])
            
            print(f"\n🎯 Tahmin Sonucu:")
            print(f"   📁 Dosya: {Path(result['file']).name}")
            print(f"   🏷️ Sınıf: {result['predicted_class']}")
            print(f"   🎯 Güven: {result['confidence']:.3f}")
            print(f"   📊 Sınıf Olasılıkları:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"      {class_name}: {prob:.3f}")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Sonuç kaydedildi: {args.output}")
        
        else:
            results = tester.predict_batch(audio_files, args.output)
            tester.print_prediction_summary(results)
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 