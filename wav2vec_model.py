#!/usr/bin/env python3
"""
Wav2Vec2 modeli kullanarak demans tespiti yapan Python scripti
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import librosa
import pickle
import warnings
warnings.filterwarnings('ignore')

from audio_converter import convert_m4a_to_wav

class Wav2VecDemantiaDetector:
    """Wav2Vec2 tabanlı demans tespit sınıfı"""
    
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """
        Wav2Vec2 modelini başlatır
        
        Args:
            model_name (str): Kullanılacak pre-trained model
        """
        print(f"🤖 Wav2Vec2 model yükleniyor: {model_name}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Cihaz: {self.device}")
        
        # Wav2Vec2 model ve processor'ı yükle
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        
        # Klasifikasyon modeli
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        print("✅ Model hazır!")
    
    def load_and_preprocess_audio(self, file_path, target_sr=16000):
        """
        Ses dosyasını yükler ve Wav2Vec2 için hazırlar
        
        Args:
            file_path (str): Ses dosyasının yolu
            target_sr (int): Hedef sampling rate
            
        Returns:
            numpy.array: Preprocessed audio
        """
        file_path = Path(file_path)
        
        # .m4a dosyalarını .wav'a çevir
        if file_path.suffix.lower() == '.m4a':
            print(f"🔄 .m4a dosyası .wav'a çevriliyor: {file_path}")
            wav_path = convert_m4a_to_wav(str(file_path))
            if wav_path:
                file_path = Path(wav_path)
            else:
                raise ValueError(f"Dosya çevrilemedi: {file_path}")
        
        # Ses dosyasını yükle
        audio, sr = librosa.load(str(file_path), sr=target_sr)
        
        # Wav2Vec2 için normalize et
        audio = audio / np.max(np.abs(audio))  # Normalize
        
        return audio
    
    def extract_wav2vec_embeddings(self, audio_array):
        """
        Wav2Vec2 embeddings çıkarır
        
        Args:
            audio_array (numpy.array): Audio waveform
            
        Returns:
            numpy.array: Wav2Vec2 embeddings
        """
        # Processor ile input hazırla
        inputs = self.processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        # GPU'ya taşı
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Model ile embeddings çıkar
        with torch.no_grad():
            outputs = self.wav2vec_model(**inputs)
            
        # Son hidden state'i al ve pool et
        last_hidden_states = outputs.last_hidden_state  # (batch, time, features)
        
        # Global average pooling
        embeddings = torch.mean(last_hidden_states, dim=1)  # (batch, features)
        
        return embeddings.cpu().numpy()
    
    def process_audio_file(self, file_path):
        """
        Tek bir ses dosyasını işler ve embeddings çıkarır
        
        Args:
            file_path (str): Ses dosyasının yolu
            
        Returns:
            numpy.array: Feature vector
        """
        print(f"🎵 İşleniyor: {file_path}")
        
        # Ses dosyasını yükle
        audio = self.load_and_preprocess_audio(file_path)
        
        # Wav2Vec2 embeddings çıkar
        embeddings = self.extract_wav2vec_embeddings(audio)
        
        return embeddings.flatten()
    
    def prepare_dataset(self, data_directory, labels_csv=None):
        """
        Veri setini hazırlar
        
        Args:
            data_directory (str): Ses dosyalarının bulunduğu dizin
            labels_csv (str, optional): Etiketlerin bulunduğu CSV dosyası
            
        Returns:
            tuple: (features, labels) veya sadece features
        """
        data_path = Path(data_directory)
        
        if not data_path.exists():
            raise ValueError(f"Dizin bulunamadı: {data_directory}")
        
        # Ses dosyalarını bul
        audio_extensions = ['.wav', '.m4a', '.mp3', '.flac']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(data_path.glob(f"*{ext}")))
            audio_files.extend(list(data_path.glob(f"*{ext.upper()}")))
        
        if not audio_files:
            raise ValueError(f"Ses dosyası bulunamadı: {data_directory}")
        
        print(f"📁 {len(audio_files)} ses dosyası bulundu")
        
        # Etiketleri yükle (varsa)
        labels_dict = {}
        if labels_csv and Path(labels_csv).exists():
            labels_df = pd.read_csv(labels_csv)
            print(f"📋 Etiketler yüklendi: {labels_csv}")
            
            # Dosya adı ve etiket mapping'i oluştur
            for _, row in labels_df.iterrows():
                filename = Path(row['filename']).stem  # Extension olmadan
                label = row['label']
                labels_dict[filename] = label
        
        # Feature extraction
        features = []
        labels = []
        processed_files = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] İşleniyor: {audio_file.name}")
            
            try:
                # Embeddings çıkar
                embedding = self.process_audio_file(audio_file)
                features.append(embedding)
                processed_files.append(audio_file.name)
                
                # Etiket varsa ekle
                if labels_dict:
                    filename_stem = audio_file.stem
                    if filename_stem in labels_dict:
                        labels.append(labels_dict[filename_stem])
                    else:
                        # Dosya adından otomatik etiket çıkarmayı dene
                        filename_lower = filename_stem.lower()
                        if 'normal' in filename_lower or 'healthy' in filename_lower:
                            labels.append('Normal')
                        elif 'mci' in filename_lower or 'mild' in filename_lower:
                            labels.append('MCI')
                        elif 'dementia' in filename_lower or 'alzheimer' in filename_lower:
                            labels.append('Dementia')
                        else:
                            labels.append('Unknown')
                
            except Exception as e:
                print(f"❌ Hata: {audio_file.name} - {str(e)}")
                continue
        
        if not features:
            raise ValueError("Hiçbir dosya işlenemedi!")
        
        features = np.array(features)
        
        print(f"\n📊 Veri seti hazırlandı:")
        print(f"   - İşlenen dosya sayısı: {len(features)}")
        print(f"   - Feature boyutu: {features.shape[1]}")
        
        if labels:
            labels = np.array(labels)
            print(f"   - Etiket dağılımı: {np.unique(labels, return_counts=True)}")
            return features, labels, processed_files
        else:
            return features, None, processed_files
    
    def train_classifier(self, features, labels, model_type='random_forest', test_size=0.2):
        """
        Klasifikasyon modelini eğitir
        
        Args:
            features (numpy.array): Feature matrix
            labels (numpy.array): Labels
            model_type (str): Model türü ('random_forest', 'svm', 'logistic')
            test_size (float): Test veri oranı
            
        Returns:
            dict: Eğitim sonuçları
        """
        if labels is None:
            raise ValueError("Etiketler bulunamadı! Eğitim için etiketler gerekli.")
        
        print(f"🎯 Model eğitiliyor: {model_type}")
        
        # Etiketleri encode et
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_encoded, test_size=test_size, random_state=42, stratify=labels_encoded
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model seçimi
        if model_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_type == 'logistic':
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Desteklenmeyen model türü: {model_type}")
        
        # Eğitim
        print("📚 Model eğitiliyor...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Test
        print("🧪 Model test ediliyor...")
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Sonuçları değerlendir
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        results = {
            'model_type': model_type,
            'accuracy': report['accuracy'],
            'classification_report': classification_report(y_test, y_pred, target_names=target_names),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': None
        }
        
        # Feature importance (Random Forest için)
        if model_type == 'random_forest':
            results['feature_importance'] = self.classifier.feature_importances_
        
        print(f"\n📈 Eğitim Tamamlandı!")
        print(f"Doğruluk: {results['accuracy']:.4f}")
        print("\nKlasifikasyon Raporu:")
        print(results['classification_report'])
        
        return results
    
    def predict(self, audio_file):
        """
        Tek bir ses dosyası için tahmin yapar
        
        Args:
            audio_file (str): Ses dosyasının yolu
            
        Returns:
            dict: Tahmin sonucu
        """
        if self.classifier is None:
            raise ValueError("Model henüz eğitilmedi! Önce train_classifier() çağırın.")
        
        print(f"🔮 Tahmin yapılıyor: {audio_file}")
        
        # Feature extraction
        features = self.process_audio_file(audio_file)
        features = features.reshape(1, -1)  # Single sample için reshape
        
        # Scaling
        features_scaled = self.scaler.transform(features)
        
        # Tahmin
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Sonucu decode et
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Olasılıkları etiketlerle eşleştir
        prob_dict = {}
        for i, label in enumerate(self.label_encoder.classes_):
            prob_dict[label] = probabilities[i]
        
        result = {
            'file': audio_file,
            'prediction': predicted_label,
            'confidence': np.max(probabilities),
            'probabilities': prob_dict
        }
        
        print(f"✅ Tahmin: {predicted_label} (Güven: {result['confidence']:.4f})")
        
        return result
    
    def save_model(self, model_path):
        """Eğitilmiş modeli kaydeder"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model kaydedildi: {model_path}")
    
    def load_model(self, model_path):
        """Kaydedilmiş modeli yükler"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        print(f"📂 Model yüklendi: {model_path}")

def create_sample_labels_csv():
    """Örnek etiket dosyası oluşturur"""
    sample_data = {
        'filename': [
            'normal_001.wav',
            'normal_002.wav', 
            'mci_001.wav',
            'mci_002.wav',
            'dementia_001.wav',
            'dementia_002.wav'
        ],
        'label': [
            'Normal',
            'Normal',
            'MCI', 
            'MCI',
            'Dementia',
            'Dementia'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_labels.csv', index=False)
    print("📄 Örnek etiket dosyası oluşturuldu: sample_labels.csv")

def main():
    """Ana fonksiyon"""
    import sys
    
    if len(sys.argv) < 2:
        print("Wav2Vec2 Demans Tespit Sistemi")
        print("Kullanım:")
        print("  Eğitim: python wav2vec_model.py train /ses/dizini labels.csv")
        print("  Tahmin: python wav2vec_model.py predict model.pkl ses_dosyasi.wav")
        print("  Örnek etiket dosyası: python wav2vec_model.py create_labels")
        return
    
    command = sys.argv[1]
    detector = Wav2VecDemantiaDetector()
    
    if command == "train":
        if len(sys.argv) < 3:
            print("❌ Ses dizini belirtiniz!")
            return
        
        data_dir = sys.argv[2]
        labels_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Veri setini hazırla
        features, labels, files = detector.prepare_dataset(data_dir, labels_file)
        
        if labels is not None:
            # Model eğit
            results = detector.train_classifier(features, labels)
            
            # Modeli kaydet
            model_name = f"dementia_model_{results['model_type']}.pkl"
            detector.save_model(model_name)
        else:
            print("⚠️ Etiketler bulunamadı. Sadece feature extraction yapıldı.")
    
    elif command == "predict":
        if len(sys.argv) < 4:
            print("❌ Model dosyası ve ses dosyası belirtiniz!")
            return
        
        model_file = sys.argv[2]
        audio_file = sys.argv[3]
        
        # Modeli yükle
        detector.load_model(model_file)
        
        # Tahmin yap
        result = detector.predict(audio_file)
        
        print(f"\n🎯 Tahmin Sonucu:")
        print(f"   Dosya: {result['file']}")
        print(f"   Tahmin: {result['prediction']}")
        print(f"   Güven: {result['confidence']:.4f}")
        print(f"   Olasılıklar:")
        for label, prob in result['probabilities'].items():
            print(f"     {label}: {prob:.4f}")
    
    elif command == "create_labels":
        create_sample_labels_csv()
    
    else:
        print(f"❌ Bilinmeyen komut: {command}")

if __name__ == "__main__":
    main() 