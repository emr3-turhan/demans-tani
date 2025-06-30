#!/usr/bin/env python3
"""
Basit Model Tahmin Scripti
Eğitilmiş sentetik model ile yeni ses dosyalarını tahmin eder
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

from feature_extraction import AudioFeatureExtractor

def load_training_features(dataset_dir):
    """Training features'ı yükle"""
    features_file = Path(dataset_dir) / "features.csv"
    if features_file.exists():
        df = pd.read_csv(features_file)
        # Numeric feature columns'u belirle
        feature_columns = []
        for col in df.columns:
            if col not in ['filename', 'class', 'text'] and df[col].dtype in ['float64', 'int64']:
                feature_columns.append(col)
        return feature_columns
    return None

def predict_with_model(model_file, audio_file, dataset_dir=None):
    """
    Model ile tahmin yap
    
    Args:
        model_file (str): Model dosyası
        audio_file (str): Ses dosyası
        dataset_dir (str): Training dataset dizini
    """
    
    # Model yükle
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    model_name = model_data['model_name']
    
    print(f"📥 {model_name} modeli yüklendi")
    print(f"🏷️ Sınıflar: {label_encoder.classes_}")
    
    # Training features'ı belirle
    if dataset_dir:
        feature_columns = load_training_features(dataset_dir)
        if feature_columns:
            print(f"📊 Training features yüklendi: {len(feature_columns)} özellik")
        else:
            print("⚠️ Training features bulunamadı")
    else:
        feature_columns = None
    
    # Özellik çıkar
    extractor = AudioFeatureExtractor()
    features = extractor.extract_all_features(audio_file)
    
    if feature_columns:
        # Training ile aynı sırayla feature vector oluştur
        feature_vector = []
        for col in feature_columns:
            if col in features:
                value = features[col]
                if pd.isna(value):
                    value = 0.0
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
    else:
        # Fallback: mevcut tüm numeric features
        feature_vector = []
        for key, value in features.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                feature_vector.append(float(value))
    
    # Tahmin yap
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    print(f"🔢 Feature vektör boyutu: {feature_vector.shape}")
    
    try:
        prediction = model.predict(feature_vector)
        probability = model.predict_proba(feature_vector)[0]
        
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(probability))
        
        # Sınıf probabiliteleri
        class_probs = {}
        for i, class_name in enumerate(label_encoder.classes_):
            class_probs[class_name] = float(probability[i])
        
        result = {
            'file': str(audio_file),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probs,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Tahmin hatası: {e}")
        return None

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Basit Model Tahmin")
    parser.add_argument('model', help='Model dosyası (.pkl)')
    parser.add_argument('audio', help='Ses dosyası')
    parser.add_argument('--dataset', help='Training dataset dizini')
    parser.add_argument('--output', '-o', help='Sonucu kaydet (JSON)')
    
    args = parser.parse_args()
    
    print(f"🎵 Ses dosyası: {args.audio}")
    
    result = predict_with_model(args.model, args.audio, args.dataset)
    
    if result:
        print(f"\n🎯 Tahmin Sonucu:")
        print(f"   📁 Dosya: {Path(result['file']).name}")
        print(f"   🏷️ Sınıf: {result['predicted_class']}")
        print(f"   🎯 Güven: {result['confidence']:.3f}")
        print(f"   📊 Sınıf Olasılıkları:")
        for class_name, prob in result['class_probabilities'].items():
            emoji = "🟢" if class_name == result['predicted_class'] else "⚪"
            print(f"      {emoji} {class_name}: {prob:.3f}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Sonuç kaydedildi: {args.output}")
    
    return 0 if result else 1

if __name__ == "__main__":
    exit(main()) 