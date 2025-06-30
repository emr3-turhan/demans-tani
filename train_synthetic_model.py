#!/usr/bin/env python3
"""
Sentetik Dataset ile Model Eğitimi
Ses özelliklerini çıkarır ve çoklu ML modelleri eğitir
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

# ML kütüphaneleri
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# Ses işleme
from feature_extraction import AudioFeatureExtractor

class SyntheticModelTrainer:
    """Sentetik dataset ile model eğitici"""
    
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (str): Sentetik dataset dizini
        """
        self.dataset_dir = Path(dataset_dir)
        self.labels_file = self.dataset_dir / "labels.csv"
        self.feature_extractor = AudioFeatureExtractor()
        
        # Model tanımları
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        }
        
        self.results = {}
        
    def load_dataset(self):
        """Dataset ve etiketleri yükle"""
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels dosyası bulunamadı: {self.labels_file}")
        
        print(f"📊 Dataset yükleniyor: {self.dataset_dir}")
        
        # Labels'ı yükle
        self.labels_df = pd.read_csv(self.labels_file)
        
        print(f"✅ {len(self.labels_df)} örnek yüklendi")
        print(f"📈 Sınıf dağılımı:")
        
        class_counts = self.labels_df['class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} örnek")
        
        return self.labels_df
    
    def extract_features_from_dataset(self, force_recompute=False):
        """
        Dataset'ten özellik çıkar
        
        Args:
            force_recompute (bool): Özellikler varsa yeniden hesapla
        """
        features_file = self.dataset_dir / "features.csv"
        features_raw_file = self.dataset_dir / "features_raw.pkl"
        
        if features_file.exists() and not force_recompute:
            print(f"✅ Özellikler zaten mevcut: {features_file}")
            features_df = pd.read_csv(features_file)
            return features_df
        
        print(f"🔬 Özellik çıkarımı başlıyor...")
        
        all_features = []
        failed_files = []
        
        for idx, row in self.labels_df.iterrows():
            audio_file = self.dataset_dir / row['filename']
            
            if not audio_file.exists():
                print(f"❌ Dosya bulunamadı: {audio_file}")
                failed_files.append(row['filename'])
                continue
            
            try:
                # Özellik çıkar
                features = self.feature_extractor.extract_all_features(str(audio_file))
                
                # Metadata ekle
                features['filename'] = row['filename']
                features['class'] = row['class']
                features['text'] = row['text']
                
                all_features.append(features)
                
                if (idx + 1) % 50 == 0:
                    print(f"   ✅ {idx+1}/{len(self.labels_df)} işlendi")
                    
            except Exception as e:
                print(f"❌ Özellik çıkarma hatası {audio_file}: {e}")
                failed_files.append(row['filename'])
                continue
        
        if not all_features:
            raise ValueError("Hiçbir dosyadan özellik çıkarılamadı!")
        
        # DataFrame oluştur
        features_df = pd.DataFrame(all_features)
        
        # Kaydet
        features_df.to_csv(features_file, index=False)
        
        # Raw features'ı da kaydet (pickle ile)
        with open(features_raw_file, 'wb') as f:
            pickle.dump(all_features, f)
        
        print(f"✅ Özellik çıkarımı tamamlandı!")
        print(f"📊 Başarılı: {len(all_features)}")
        print(f"❌ Başarısız: {len(failed_files)}")
        print(f"💾 Özellikler kaydedildi: {features_file}")
        
        if failed_files:
            print(f"⚠️ Başarısız dosyalar: {failed_files[:5]}...")
        
        return features_df
    
    def prepare_data(self, features_df):
        """Eğitim için veri hazırla"""
        print(f"🔧 Veri hazırlığı...")
        
        # Özellik sütunlarını belirle (numeric olanlar)
        feature_columns = []
        for col in features_df.columns:
            if col not in ['filename', 'class', 'text'] and features_df[col].dtype in ['float64', 'int64']:
                feature_columns.append(col)
        
        print(f"📊 {len(feature_columns)} özellik seçildi")
        
        # X ve y oluştur
        X = features_df[feature_columns].fillna(0)  # NaN değerleri 0 ile doldur
        y = features_df['class']
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"🏷️ Sınıflar: {self.label_encoder.classes_}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"📈 Eğitim seti: {len(X_train)} örnek")
        print(f"📈 Test seti: {len(X_test)} örnek")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Modelleri eğit"""
        print(f"🤖 Model eğitimi başlıyor...")
        
        self.trained_models = {}
        self.scalers = {}
        
        for model_name, model in self.models.items():
            print(f"\n🎯 {model_name} eğitiliyor...")
            
            # Scaling için pipeline oluştur
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Eğit
            pipeline.fit(X_train, y_train)
            
            # Test et
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
            
            # Metrikler
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation (adaptif fold sayısı)
            from collections import Counter
            class_counts = Counter(y_train)
            min_class_count = min(class_counts.values())
            n_folds = min(3, min_class_count)  # En az 2, en fazla 3 fold
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=n_folds)
            except ValueError:
                # Eğer hala hata varsa, sadece accuracy kullan
                cv_scores = np.array([accuracy])
                print(f"   ⚠️ CV skipped due to small dataset")
            
            # Sonuçları kaydet
            self.results[model_name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_prob': y_prob,
                'classification_report': classification_report(
                    y_test, y_pred, 
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
            }
            
            self.trained_models[model_name] = pipeline
            
            print(f"   ✅ Accuracy: {accuracy:.3f}")
            print(f"   🔄 CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return self.results
    
    def evaluate_models(self, X_test, y_test):
        """Modelleri değerlendir"""
        print(f"\n📊 Model değerlendirmesi...")
        
        # Sonuçları özetle
        results_summary = []
        
        for model_name, result in self.results.items():
            results_summary.append({
                'Model': model_name,
                'Test Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        print(f"\n🏆 Model Karşılaştırması:")
        print(results_df.to_string(index=False, float_format='%.3f'))
        
        # En iyi modeli belirle
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n🥇 En iyi model: {best_model_name}")
        
        return results_df, best_model_name
    
    def create_visualizations(self, X_test, y_test, output_dir=None):
        """Görselleştirmeler oluştur"""
        if output_dir is None:
            output_dir = self.dataset_dir / "model_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"📈 Görselleştirmeler oluşturuluyor...")
        
        # 1. Model karşılaştırması
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy karşılaştırması
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0].bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        axes[0].bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confusion matrix (en iyi model için)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_result = self.results[best_model_name]
        
        cm = confusion_matrix(y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[1])
        axes[1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sınıf bazında performans
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_report = best_result['classification_report']
        classes = self.label_encoder.classes_
        
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [class_report[cls][metric] for cls in classes]
            ax.bar(x + i*width, values, width, label=metric.title(), alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Performance - {best_model_name}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Görselleştirmeler kaydedildi: {output_dir}")
        
        return output_dir
    
    def save_best_model(self, best_model_name, output_dir=None):
        """En iyi modeli kaydet"""
        if output_dir is None:
            output_dir = self.dataset_dir / "trained_models"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        best_model = self.trained_models[best_model_name]
        
        # Model ve metadata'yı kaydet
        model_data = {
            'model': best_model,
            'label_encoder': self.label_encoder,
            'model_name': best_model_name,
            'training_date': datetime.now().isoformat(),
            'performance': self.results[best_model_name],
            'dataset_info': str(self.dataset_dir)
        }
        
        model_file = output_dir / f"best_model_{best_model_name.lower()}.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 En iyi model kaydedildi: {model_file}")
        
        # Tüm sonuçları da kaydet
        results_file = output_dir / "training_results.json"
        
        # JSON serializable hale getir
        json_results = {}
        for name, result in self.results.items():
            json_results[name] = {
                'accuracy': float(result['accuracy']),
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std']),
                'classification_report': result['classification_report']
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Eğitim sonuçları kaydedildi: {results_file}")
        
        return model_file, results_file

def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentetik Dataset ile Model Eğitimi")
    parser.add_argument('--dataset', '-d', required=True,
                       help='Sentetik dataset dizini')
    parser.add_argument('--output', '-o', 
                       help='Çıktı dizini (varsayılan: dataset/model_results)')
    parser.add_argument('--force-features', '-f', action='store_true',
                       help='Özellik çıkarımını yeniden yap')
    parser.add_argument('--no-viz', action='store_true',
                       help='Görselleştirme oluşturma')
    
    args = parser.parse_args()
    
    try:
        # Model eğitici
        trainer = SyntheticModelTrainer(args.dataset)
        
        # Dataset yükle
        labels_df = trainer.load_dataset()
        
        # Özellik çıkar
        features_df = trainer.extract_features_from_dataset(
            force_recompute=args.force_features
        )
        
        # Veri hazırla
        X_train, X_test, y_train, y_test, feature_columns = trainer.prepare_data(features_df)
        
        # Modelleri eğit
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Değerlendir
        results_df, best_model_name = trainer.evaluate_models(X_test, y_test)
        
        # Görselleştir
        if not args.no_viz:
            viz_dir = trainer.create_visualizations(X_test, y_test, args.output)
        
        # En iyi modeli kaydet
        model_file, results_file = trainer.save_best_model(best_model_name, args.output)
        
        print(f"\n✅ Model eğitimi tamamlandı!")
        print(f"🏆 En iyi model: {best_model_name}")
        print(f"💾 Model dosyası: {model_file}")
        print(f"📊 Sonuçlar: {results_file}")
        
        # Test için örnek kod
        print(f"\n🧪 Test için örnek kod:")
        print(f"```python")
        print(f"import pickle")
        print(f"from feature_extraction import AudioFeatureExtractor")
        print(f"")
        print(f"# Model yükle")
        print(f"with open('{model_file}', 'rb') as f:")
        print(f"    model_data = pickle.load(f)")
        print(f"")
        print(f"model = model_data['model']")
        print(f"label_encoder = model_data['label_encoder']")
        print(f"")
        print(f"# Yeni ses dosyası tahmin et")
        print(f"extractor = AudioFeatureExtractor()")
        print(f"features = extractor.extract_all_features('test.wav')")
        print(f"")
        print(f"# Özellik vektorü hazırla")
        print(f"feature_vector = [features[col] for col in feature_columns]")
        print(f"prediction = model.predict([feature_vector])")
        print(f"predicted_class = label_encoder.inverse_transform(prediction)[0]")
        print(f"```")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 