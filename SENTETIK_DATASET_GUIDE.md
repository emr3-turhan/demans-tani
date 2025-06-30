# 🧠 Sentetik Audio Dataset ve Model Eğitimi Rehberi

Bu rehber, Text-to-Speech ile sentetik audio dataset oluşturma ve demans tespit modeli eğitimi sürecini açıklar.

## 📋 İçindekiler

1. [Sistem Özeti](#sistem-özeti)
2. [Kurulum](#kurulum)
3. [Sentetik Dataset Oluşturma](#sentetik-dataset-oluşturma)
4. [Model Eğitimi](#model-eğitimi)
5. [Model Kullanımı](#model-kullanımı)
6. [Dosya Yapısı](#dosya-yapısı)
7. [Sonuç ve Analiz](#sonuç-ve-analiz)

## 🎯 Sistem Özeti

Bu sistem, demans riskini ses analizi ile tespit etmek için:

### ✅ **Yapabilecekleri:**

- **TTS ile Sentetik Ses Üretimi**: Normal, MCI ve Demans sınıfları için farklı konuşma tarzları
- **Özellik Çıkarımı**: 90+ audio feature (MFCC, spektral, temporal, pitch)
- **Çoklu Model Eğitimi**: RandomForest, SVM, LogisticRegression, GradientBoosting
- **Model Değerlendirmesi**: Cross-validation, confusion matrix, performance metrics
- **Gerçek Zamanlı Tahmin**: Yeni ses dosyaları için sınıf tahmini

### 🎯 **Sınıflar:**

- **Normal**: Akıcı, net konuşma
- **MCI (Mild Cognitive Impairment)**: Orta düzey duraksamalar
- **Dementia**: Bozuk yapı, çok duraklama, tekrarlar

## 🔧 Kurulum

### Gereksinimler:

```bash
# Conda ile
conda install librosa pandas numpy scikit-learn matplotlib seaborn scipy -c conda-forge
conda install pytorch torchvision torchaudio transformers -c pytorch -c conda-forge
pip install gtts pyttsx3 soundfile pydub
```

### Dosyalar:

- `synthetic_dataset_generator.py` - Sentetik dataset üretici
- `train_synthetic_model.py` - Model eğitim scripti
- `model_inference.py` - Model tahmin scripti
- `feature_extraction.py` - Özellik çıkarımı (mevcut)

## 🎵 Sentetik Dataset Oluşturma

### Hızlı Test (15 örnek):

```bash
python synthetic_dataset_generator.py --test --output test_dataset
```

### Tam Dataset (150 örnek - 50 her sınıf):

```bash
python synthetic_dataset_generator.py --samples 50 --output full_dataset
```

### Büyük Dataset (300 örnek - 100 her sınıf):

```bash
python synthetic_dataset_generator.py --samples 100 --output large_dataset
```

### Dataset Yapısı:

```
full_dataset/
├── normal/
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
├── mci/
│   ├── 001.wav
│   └── ...
├── dementia/
│   ├── 001.wav
│   └── ...
├── labels.csv          # Dosya ve sınıf etiketleri
├── features.csv        # Çıkarılmış özellikler
└── dataset_info.json   # Dataset metadatası
```

### Sentetik Özellikler:

| Sınıf        | Konuşma Özellikleri             | TTS Parametreleri        |
| ------------ | ------------------------------- | ------------------------ |
| **Normal**   | Akıcı, düzgün telaffuz          | Hız: 1.0, Normal ton     |
| **MCI**      | Hafif duraksamalar, tereddütler | Hız: 0.8, Bazı tekrarlar |
| **Dementia** | Uzun duraklamalar, bozuk yapı   | Hız: 0.6, Çok tekrar     |

## 🤖 Model Eğitimi

### Basit Eğitim:

```bash
python train_synthetic_model.py --dataset full_dataset
```

### Görselleştirmesiz (Hızlı):

```bash
python train_synthetic_model.py --dataset full_dataset --no-viz
```

### Eğitim Süreci:

1. **Özellik Çıkarımı**: 150 ses dosyasından 90+ özellik
2. **Veri Bölme**: %80 eğitim, %20 test
3. **Model Eğitimi**: 4 farklı algoritma
4. **Değerlendirme**: Cross-validation ve test accuracy
5. **Model Kaydetme**: En iyi model pickle olarak

### Sonuç Dosyaları:

```
full_dataset/trained_models/
├── best_model_randomforest.pkl
├── training_results.json
└── model_results/
    ├── confusion_matrix.png
    ├── model_comparison.png
    └── feature_importance.png
```

## 📊 Model Kullanımı

### Tek Dosya Tahmini:

```bash
python model_inference.py \
    full_dataset/trained_models/best_model_randomforest.pkl \
    test_audio.wav \
    --dataset full_dataset
```

### Toplu Tahmin:

```bash
python model_inference.py \
    full_dataset/trained_models/best_model_randomforest.pkl \
    "test_audio/*.wav" \
    --dataset full_dataset \
    --output predictions.json
```

### Python'da Kullanım:

```python
import pickle
from feature_extraction import AudioFeatureExtractor

# Model yükle
with open('full_dataset/trained_models/best_model_randomforest.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']

# Tahmin yap
extractor = AudioFeatureExtractor()
features = extractor.extract_all_features('test.wav')

# Feature vector hazırla (training features ile aynı sıra)
# ... (feature alignment kodu)

prediction = model.predict([feature_vector])
predicted_class = label_encoder.inverse_transform(prediction)[0]
print(f"Tahmin: {predicted_class}")
```

## 📁 Dosya Yapısı

### Ana Dosyalar:

```
project/
├── synthetic_dataset_generator.py    # Sentetik dataset üretici
├── train_synthetic_model.py          # Model eğitim scripti
├── model_inference.py               # Model tahmin scripti
├── feature_extraction.py            # Özellik çıkarımı (mevcut)
├── dementia_detection_pipeline.py   # Ana pipeline (mevcut)
└── SENTETIK_DATASET_GUIDE.md       # Bu rehber
```

### Oluşturulan Dosyalar:

```
project/
├── test_synthetic_dataset/         # Test dataset (15 örnek)
├── full_synthetic_dataset/         # Tam dataset (150 örnek)
│   ├── normal/ (50 wav dosyası)
│   ├── mci/ (50 wav dosyası)
│   ├── dementia/ (50 wav dosyası)
│   ├── labels.csv
│   ├── features.csv
│   ├── dataset_info.json
│   └── trained_models/
│       ├── best_model_randomforest.pkl
│       └── training_results.json
└── run_analysis.sh                 # Kullanım scripti (mevcut)
```

## 📈 Sonuç ve Analiz

### Test Sonuçları (150 örnek dataset):

| Model              | Test Accuracy | CV Mean | CV Std |
| ------------------ | ------------- | ------- | ------ |
| **RandomForest**   | **100%**      | 100%    | 0.000  |
| GradientBoosting   | 100%          | 92.5%   | 0.035  |
| SVM                | 100%          | 99.2%   | 0.012  |
| LogisticRegression | 100%          | 99.2%   | 0.012  |

### Örnek Tahmin Sonuçları:

#### Normal Sınıfı (001.wav):

```
🎯 Sınıf: normal
🎯 Güven: 83.7%
📊 Olasılıklar:
   🟢 normal: 0.837
   ⚪ mci: 0.133
   ⚪ dementia: 0.030
```

#### MCI Sınıfı (001.wav):

```
🎯 Sınıf: mci
🎯 Güven: 67.9%
📊 Olasılıklar:
   ⚪ normal: 0.069
   🟢 mci: 0.679
   ⚪ dementia: 0.253
```

#### Dementia Sınıfı (001.wav):

```
🎯 Sınıf: dementia
🎯 Güven: 77.9%
📊 Olasılıklar:
   ⚪ normal: 0.050
   ⚪ mci: 0.172
   🟢 dementia: 0.779
```

### Önemli Özellikler:

1. **MFCC coefficients** (0-12): Spektral envelope
2. **Spectral features**: Centroid, rolloff, contrast
3. **Temporal features**: Speaking rate, pause duration
4. **Pitch features**: Mean, std, range

## ⚠️ Önemli Notlar

### ✅ Başarılı Yanlar:

- **Mükemmel Sınıflandırma**: %100 test accuracy
- **Gerçekçi Sentetik Data**: TTS ile farklı konuşma tarzları
- **Kapsamlı Özellikler**: 90+ audio feature
- **Çoklu Model Desteği**: 4 farklı ML algoritması
- **Kullanıma Hazır**: Basit Python interface

### ⚠️ Limitasyonlar:

1. **Sentetik Veri**: Gerçek hasta verisi değil
2. **Küçük Dataset**: 150 örnek (gerçek uygulamalar için az)
3. **Overfitting Riski**: %100 accuracy şüpheli
4. **TTS Limitasyonları**: Sınırlı konuşma varyasyonu
5. **Validation**: Gerçek klinik veri ile test edilmemiş

### 🎯 Geliştime Önerileri:

1. **Daha Büyük Dataset**: 1000+ örnek per class
2. **Daha Çeşitli TTS**: Farklı sesler, tonlar
3. **Gerçek Veri Entegrasyonu**: Klinik dataset ekleme
4. **Advanced Features**: Deep learning embeddings
5. **Cross-validation**: Daha detaylı değerlendirme

## 🚀 Kullanım Senaryoları

### 1. **Araştırma Prototipri**:

- Yeni feature'ları test etme
- Algoritma karşılaştırması
- Proof-of-concept geliştirme

### 2. **Eğitim Amaçlı**:

- ML pipeline öğrenme
- Audio processing eğitimi
- Feature engineering pratiği

### 3. **Baseline Model**:

- Gerçek veri gelene kadar
- Performance benchmark
- Initial system validation

## 📞 Destek

Sorularınız için:

- Feature extraction: `feature_extraction.py`
- Dataset generation: `synthetic_dataset_generator.py`
- Model training: `train_synthetic_model.py`
- Model inference: `model_inference.py`

---

**✨ Bu sistem sentetik veri ile proof-of-concept amaçlıdır. Klinik kullanım için gerçek hasta verisi ve medical validation gereklidir!**
