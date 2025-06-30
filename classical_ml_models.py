#!/usr/bin/env python3
"""
Klasik makine öğrenmesi modelleriyle demans tespiti yapan Python scripti
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_extraction import AudioFeatureExtractor

class ClassicalMLDemantiaDetector:
    """Klasik makine öğrenmesi yöntemleriyle demans tespit sınıfı"""
    
    def __init__(self):
        """Sınıfı başlatır"""
        self.feature_extractor = AudioFeatureExtractor()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.best_model = None
        self.best_model_name = None
        
    def prepare_features_and_labels(self, data_directory, labels_csv):
        """
        Özellik çıkarımı ve etiket hazırlığı yapar
        
        Args:
            data_directory (str): Ses dosyalarının bulunduğu dizin
            labels_csv (str): Etiketlerin bulunduğu CSV dosyası
            
        Returns:
            tuple: (features_df, labels)
        """
        print("🎵 Özellik çıkarımı başlıyor...")
        
        # Özellik çıkarımı
        features_df = self.feature_extractor.process_dataset(data_directory)
        
        if features_df is None:
            raise ValueError("Özellik çıkarımı başarısız!")
        
        # Etiketleri yükle
        if not Path(labels_csv).exists():
            raise ValueError(f"Etiket dosyası bulunamadı: {labels_csv}")
        
        labels_df = pd.read_csv(labels_csv)
        print(f"📋 Etiketler yüklendi: {labels_csv}")
        
        # Dosya adları ile etiketleri eşleştir
        labels_dict = {}
        for _, row in labels_df.iterrows():
            filename = Path(row['filename']).stem
            label = row['label']
            labels_dict[filename] = label
        
        # Etiketleri features_df'e ekle
        matched_labels = []
        matched_indices = []
        
        for idx, row in features_df.iterrows():
            file_path = Path(row['file_path'])
            filename = file_path.stem
            
            if filename in labels_dict:
                matched_labels.append(labels_dict[filename])
                matched_indices.append(idx)
            else:
                # Dosya adından otomatik etiket çıkarmayı dene
                filename_lower = filename.lower()
                if 'normal' in filename_lower or 'healthy' in filename_lower:
                    matched_labels.append('Normal')
                    matched_indices.append(idx)
                elif 'mci' in filename_lower or 'mild' in filename_lower:
                    matched_labels.append('MCI')
                    matched_indices.append(idx)
                elif 'dementia' in filename_lower or 'alzheimer' in filename_lower:
                    matched_labels.append('Dementia')
                    matched_indices.append(idx)
        
        # Sadece eşleşen örnekleri al
        features_df = features_df.iloc[matched_indices].reset_index(drop=True)
        labels = np.array(matched_labels)
        
        print(f"📊 Veri eşleştirmesi tamamlandı:")
        print(f"   - Toplam örnek sayısı: {len(features_df)}")
        print(f"   - Etiket dağılımı: {np.unique(labels, return_counts=True)}")
        
        return features_df, labels
    
    def preprocess_features(self, features_df):
        """
        Özellikleri ön işleme yapar
        
        Args:
            features_df (pandas.DataFrame): Özellik DataFrame'i
            
        Returns:
            numpy.array: İşlenmiş özellik matrisi
        """
        # Numerik olmayan sütunları çıkar
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_matrix = features_df[numeric_columns].values
        
        # NaN değerleri median ile doldur
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        feature_matrix = imputer.fit_transform(feature_matrix)
        
        # Sonsuz değerleri kaldır
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"🔧 Özellik matrisi hazırlandı: {feature_matrix.shape}")
        
        return feature_matrix, numeric_columns.tolist()
    
    def feature_selection(self, X, y, method='univariate', k=50):
        """
        Özellik seçimi yapar
        
        Args:
            X (numpy.array): Özellik matrisi
            y (numpy.array): Etiketler
            method (str): Seçim yöntemi ('univariate', 'rfe')
            k (int): Seçilecek özellik sayısı
            
        Returns:
            numpy.array: Seçilmiş özellikler
        """
        print(f"🎯 Özellik seçimi yapılıyor: {method}")
        
        if method == 'univariate':
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
        else:
            raise ValueError(f"Desteklenmeyen özellik seçimi yöntemi: {method}")
        
        # Etiketleri encode et
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Özellik seçimi uygula
        X_selected = self.feature_selector.fit_transform(X, y_encoded)
        
        print(f"✅ {X.shape[1]} özellikten {X_selected.shape[1]} tanesi seçildi")
        
        return X_selected
    
    def initialize_models(self):
        """Model sözlüğünü başlatır"""
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'NaiveBayes': GaussianNB()
        }
    
    def train_and_evaluate_models(self, X, y, test_size=0.2, cv_folds=5):
        """
        Tüm modelleri eğitir ve değerlendirir
        
        Args:
            X (numpy.array): Özellik matrisi
            y (numpy.array): Etiketler
            test_size (float): Test veri oranı
            cv_folds (int): Cross-validation fold sayısı
            
        Returns:
            dict: Model performans sonuçları
        """
        print("🚀 Model eğitimi ve değerlendirmesi başlıyor...")
        
        # Etiketleri encode et
        y_encoded = self.label_encoder.transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 {model_name} eğitiliyor...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
            
            # Model eğitimi
            model.fit(X_train_scaled, y_train)
            
            # Test tahmini
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Performans metrikleri
            accuracy = model.score(X_test_scaled, y_test)
            
            # Classification report
            target_names = self.label_encoder.classes_
            class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # ROC AUC (multi-class için macro average)
            roc_auc = None
            if y_pred_proba is not None and len(np.unique(y_encoded)) > 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            elif y_pred_proba is not None and len(np.unique(y_encoded)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            results[model_name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'roc_auc': roc_auc
            }
            
            print(f"   CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"   Test Accuracy: {accuracy:.4f}")
            if roc_auc:
                print(f"   ROC AUC: {roc_auc:.4f}")
        
        # En iyi modeli seç
        best_accuracy = 0
        for model_name, result in results.items():
            if result['test_accuracy'] > best_accuracy:
                best_accuracy = result['test_accuracy']
                self.best_model = result['model']
                self.best_model_name = model_name
        
        print(f"\n🏆 En iyi model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, X, y, model_name='RandomForest'):
        """
        Hiperparametre optimizasyonu yapar
        
        Args:
            X (numpy.array): Özellik matrisi
            y (numpy.array): Etiketler
            model_name (str): Optimize edilecek model
            
        Returns:
            dict: En iyi parametreler ve sonuçlar
        """
        print(f"🔧 {model_name} için hiperparametre optimizasyonu...")
        
        y_encoded = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2']
            }
        }
        
        if model_name not in param_grids:
            print(f"❌ {model_name} için parametre grid'i tanımlanmamış")
            return None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_scaled, y_encoded)
        
        result = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
        
        print(f"✅ En iyi parametreler: {result['best_params']}")
        print(f"✅ En iyi CV score: {result['best_score']:.4f}")
        
        # En iyi modeli güncelle
        self.models[model_name] = result['best_model']
        
        return result
    
    def plot_results(self, results, save_path=None):
        """
        Sonuçları görselleştirir
        
        Args:
            results (dict): Model sonuçları
            save_path (str, optional): Grafiklerin kaydedileceği dizin
        """
        # Model karşılaştırması
        model_names = list(results.keys())
        accuracies = [results[name]['test_accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model accuracy karşılaştırması
        axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Test Accuracy Karşılaştırması')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        
        # 2. Cross-validation sonuçları
        axes[0, 1].errorbar(model_names, cv_means, yerr=cv_stds, 
                           fmt='o-', capsize=5, capthick=2)
        axes[0, 1].set_title('Cross-Validation Sonuçları')
        axes[0, 1].set_ylabel('CV Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. En iyi modelin confusion matrix'i
        best_result = results[self.best_model_name]
        conf_matrix = best_result['confusion_matrix']
        target_names = self.label_encoder.classes_
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names,
                   ax=axes[1, 0])
        axes[1, 0].set_title(f'{self.best_model_name} - Confusion Matrix')
        axes[1, 0].set_ylabel('Gerçek')
        axes[1, 0].set_xlabel('Tahmin')
        
        # 4. ROC AUC karşılaştırması (varsa)
        roc_aucs = [results[name]['roc_auc'] for name in model_names if results[name]['roc_auc'] is not None]
        valid_models = [name for name in model_names if results[name]['roc_auc'] is not None]
        
        if roc_aucs:
            axes[1, 1].bar(valid_models, roc_aucs, color='lightcoral', alpha=0.7)
            axes[1, 1].set_title('ROC AUC Karşılaştırması')
            axes[1, 1].set_ylabel('ROC AUC')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            for i, auc in enumerate(roc_aucs):
                axes[1, 1].text(i, auc + 0.01, f'{auc:.3f}', ha='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'ROC AUC\nHesaplanamadı', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ROC AUC')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
            print(f"📊 Grafik kaydedildi: {save_path}/model_comparison.png")
        
        plt.show()
    
    def predict(self, audio_file):
        """
        Tek bir ses dosyası için tahmin yapar
        
        Args:
            audio_file (str): Ses dosyasının yolu
            
        Returns:
            dict: Tahmin sonucu
        """
        if self.best_model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        print(f"🔮 Tahmin yapılıyor: {audio_file}")
        
        # Özellik çıkarımı
        features = self.feature_extractor.extract_all_features(audio_file)
        
        # DataFrame'e çevir ve ön işle
        features_df = pd.DataFrame([features])
        feature_matrix, _ = self.preprocess_features(features_df)
        
        # Özellik seçimi uygula (eğer yapılmışsa)
        if self.feature_selector:
            feature_matrix = self.feature_selector.transform(feature_matrix)
        
        # Scaling
        feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        # Tahmin
        prediction = self.best_model.predict(feature_matrix_scaled)[0]
        probabilities = self.best_model.predict_proba(feature_matrix_scaled)[0]
        
        # Sonucu decode et
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Olasılıkları etiketlerle eşleştir
        prob_dict = {}
        for i, label in enumerate(self.label_encoder.classes_):
            prob_dict[label] = probabilities[i]
        
        result = {
            'file': audio_file,
            'model': self.best_model_name,
            'prediction': predicted_label,
            'confidence': np.max(probabilities),
            'probabilities': prob_dict
        }
        
        print(f"✅ Tahmin: {predicted_label} (Güven: {result['confidence']:.4f})")
        
        return result
    
    def save_model(self, model_path):
        """Eğitilmiş modeli kaydeder"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'models': self.models
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model kaydedildi: {model_path}")
    
    def load_model(self, model_path):
        """Kaydedilmiş modeli yükler"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_selector = model_data.get('feature_selector', None)
        self.models = model_data.get('models', {})
        
        print(f"📂 Model yüklendi: {model_path}")
        print(f"🏆 En iyi model: {self.best_model_name}")

def main():
    """Ana fonksiyon"""
    import sys
    
    if len(sys.argv) < 2:
        print("Klasik ML Demans Tespit Sistemi")
        print("Kullanım:")
        print("  Eğitim: python classical_ml_models.py train /ses/dizini labels.csv")
        print("  Tahmin: python classical_ml_models.py predict model.pkl ses_dosyasi.wav")
        print("  Hiperparametre: python classical_ml_models.py tune /ses/dizini labels.csv RandomForest")
        return
    
    command = sys.argv[1]
    detector = ClassicalMLDemantiaDetector()
    
    if command == "train":
        if len(sys.argv) < 4:
            print("❌ Ses dizini ve etiket dosyası belirtiniz!")
            return
        
        data_dir = sys.argv[2]
        labels_file = sys.argv[3]
        
        # Modelleri başlat
        detector.initialize_models()
        
        # Veri hazırlığı
        features_df, labels = detector.prepare_features_and_labels(data_dir, labels_file)
        feature_matrix, feature_names = detector.preprocess_features(features_df)
        
        # Özellik seçimi
        feature_matrix_selected = detector.feature_selection(feature_matrix, labels)
        
        # Model eğitimi
        results = detector.train_and_evaluate_models(feature_matrix_selected, labels)
        
        # Sonuçları görselleştir
        detector.plot_results(results, save_path='.')
        
        # Modeli kaydet
        model_name = f"classical_dementia_model.pkl"
        detector.save_model(model_name)
    
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
        print(f"   Model: {result['model']}")
        print(f"   Tahmin: {result['prediction']}")
        print(f"   Güven: {result['confidence']:.4f}")
        print(f"   Olasılıklar:")
        for label, prob in result['probabilities'].items():
            print(f"     {label}: {prob:.4f}")
    
    elif command == "tune":
        if len(sys.argv) < 5:
            print("❌ Ses dizini, etiket dosyası ve model adı belirtiniz!")
            return
        
        data_dir = sys.argv[2]
        labels_file = sys.argv[3]
        model_name = sys.argv[4]
        
        # Modelleri başlat
        detector.initialize_models()
        
        # Veri hazırlığı
        features_df, labels = detector.prepare_features_and_labels(data_dir, labels_file)
        feature_matrix, feature_names = detector.preprocess_features(features_df)
        feature_matrix_selected = detector.feature_selection(feature_matrix, labels)
        
        # Hiperparametre optimizasyonu
        result = detector.hyperparameter_tuning(feature_matrix_selected, labels, model_name)
        
        if result:
            print(f"\n🎯 Optimizasyon Sonucu:")
            print(f"   En iyi parametreler: {result['best_params']}")
            print(f"   En iyi CV score: {result['best_score']:.4f}")
    
    else:
        print(f"❌ Bilinmeyen komut: {command}")

if __name__ == "__main__":
    main() 