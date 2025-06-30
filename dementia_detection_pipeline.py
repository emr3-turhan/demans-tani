#!/usr/bin/env python3
"""
Ses tabanlı demans tespiti için kapsamlı pipeline scripti
"""

import json
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Kendi modüllerimizi import et
from audio_converter import convert_m4a_to_wav, batch_convert_m4a_to_wav
from feature_extraction import AudioFeatureExtractor

class DemantiaDetectionPipeline:
    """Demans tespiti için kapsamlı pipeline sınıfı"""
    
    def __init__(self, config_file=None, model_file=None):
        """
        Pipeline'ı başlatır
        
        Args:
            config_file (str, optional): Konfigürasyon dosyası
            model_file (str, optional): Eğitilmiş model dosyası
        """
        self.config = self.load_config(config_file)
        self.feature_extractor = AudioFeatureExtractor()
        self.trained_model = None
        self.model_info = None
        
        # Eğitilmiş modeli yükle
        if model_file:
            self.load_trained_model(model_file)
        
        print("🚀 Demans Tespit Pipeline Başlatıldı")
    
    def _json_serializer(self, obj):
        """JSON serialization için custom handler"""
        import numpy as np
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def load_trained_model(self, model_file):
        """
        Eğitilmiş modeli yükle
        
        Args:
            model_file (str): Model dosyası yolu (.pkl)
        """
        model_path = Path(model_file)
        
        if not model_path.exists():
            print(f"⚠️ Model dosyası bulunamadı: {model_file}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.trained_model = model_data['model']
            self.model_info = {
                'model_name': model_data.get('model_name', 'Unknown'),
                'training_date': model_data.get('training_date', 'Unknown'),
                'label_encoder': model_data.get('label_encoder'),
                'feature_columns': model_data.get('feature_columns'),
                'dataset_info': model_data.get('dataset_info', {})
            }
            
            print(f"✅ Eğitilmiş model yüklendi: {self.model_info['model_name']}")
            print(f"📅 Eğitim tarihi: {self.model_info['training_date']}")
            
            if self.model_info['label_encoder']:
                classes = self.model_info['label_encoder'].classes_
                print(f"🏷️ Sınıflar: {list(classes)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)}")
            self.trained_model = None
            self.model_info = None
            return False
        
    def load_config(self, config_file):
        """Konfigürasyon dosyasını yükler"""
        default_config = {
            "audio_processing": {
                "sample_rate": 22050,
                "auto_convert_m4a": True
            },
            "feature_extraction": {
                "mfcc_n_coeffs": 13,
                "include_spectral": True,
                "include_pitch": True,
                "include_temporal": True,
                "include_rhythm": True
            },
            "output": {
                "save_features": True,
                "save_predictions": True,
                "generate_report": True
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Default config'i user config ile güncelle
            default_config.update(user_config)
            print(f"📋 Konfigürasyon yüklendi: {config_file}")
        else:
            print("📋 Varsayılan konfigürasyon kullanılıyor")
        
        return default_config
    
    def create_sample_config(self, output_file="config.json"):
        """Örnek konfigürasyon dosyası oluşturur"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Örnek konfigürasyon dosyası oluşturuldu: {output_file}")
    
    def prepare_audio_data(self, input_path, output_dir=None):
        """
        Ses verilerini hazırlar (.m4a dosyalarını .wav'a çevirir)
        
        Args:
            input_path (str): Giriş dizini veya dosyası
            output_dir (str, optional): Çıkış dizini
            
        Returns:
            list: İşlenmiş ses dosyalarının listesi
        """
        print("🎵 Ses verileri hazırlanıyor...")
        
        input_path = Path(input_path)
        processed_files = []
        
        if input_path.is_file():
            # Tek dosya
            if input_path.suffix.lower() == '.m4a':
                if self.config["audio_processing"]["auto_convert_m4a"]:
                    wav_file = convert_m4a_to_wav(str(input_path))
                    if wav_file:
                        processed_files.append(wav_file)
                else:
                    print(f"⚠️ .m4a dosyası atlandı: {input_path}")
            else:
                processed_files.append(str(input_path))
        
        elif input_path.is_dir():
            # Dizin
            audio_extensions = ['.wav', '.m4a', '.mp3', '.flac']
            for ext in audio_extensions:
                files = list(input_path.glob(f"*{ext}")) + list(input_path.glob(f"*{ext.upper()}"))
                
                for file_path in files:
                    if file_path.suffix.lower() == '.m4a':
                        if self.config["audio_processing"]["auto_convert_m4a"]:
                            wav_file = convert_m4a_to_wav(str(file_path))
                            if wav_file:
                                processed_files.append(wav_file)
                    else:
                        processed_files.append(str(file_path))
        
        print(f"✅ {len(processed_files)} ses dosyası hazırlandı")
        return processed_files
    
    def extract_features_batch(self, audio_files, output_csv=None):
        """
        Toplu özellik çıkarımı yapar
        
        Args:
            audio_files (list): Ses dosyalarının listesi
            output_csv (str, optional): Çıktı CSV dosyası
            
        Returns:
            pandas.DataFrame: Özellikler DataFrame'i
        """
        print("📊 Toplu özellik çıkarımı başlıyor...")
        
        all_features = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] İşleniyor: {Path(audio_file).name}")
            
            try:
                features = self.feature_extractor.extract_all_features(audio_file)
                all_features.append(features)
            except Exception as e:
                print(f"❌ Hata: {Path(audio_file).name} - {str(e)}")
                continue
        
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            if output_csv and self.config["output"]["save_features"]:
                features_df.to_csv(output_csv, index=False)
                print(f"💾 Özellikler kaydedildi: {output_csv}")
            
            return features_df
        else:
            raise ValueError("Hiçbir dosyadan özellik çıkarılamadı!")
    
    def predict_with_trained_model(self, features, dataset_dir=None):
        """
        Eğitilmiş model ile tahmin yapar
        
        Args:
            features (dict): Çıkarılan özellikler
            dataset_dir (str, optional): Training dataset dizini (feature alignment için)
            
        Returns:
            dict: Tahmin sonucu
        """
        if not self.trained_model or not self.model_info:
            raise ValueError("Eğitilmiş model yüklenmemiş! Pipeline'ı model_file parametresi ile başlatın.")
        
        # Feature vektörü hazırla
        feature_vector = self._prepare_feature_vector(features, dataset_dir)
        
        if feature_vector is None:
            raise ValueError("Feature vektörü hazırlanamadı!")
        
        # Tahmin yap
        try:
            # Feature vektörü zaten 2D array olarak hazırlanmış
            prediction = self.trained_model.predict(feature_vector)
            probabilities = self.trained_model.predict_proba(feature_vector)[0]
            
            # Sınıf adlarını al
            label_encoder = self.model_info['label_encoder']
            predicted_class = label_encoder.inverse_transform(prediction)[0]
            confidence = float(np.max(probabilities))
            
            # Sınıf olasılıkları
            class_probabilities = {}
            for i, class_name in enumerate(label_encoder.classes_):
                class_probabilities[class_name] = float(probabilities[i])
            
            return {
                'method': 'trained_model',
                'model_name': self.model_info['model_name'],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'training_date': self.model_info['training_date']
            }
            
        except Exception as e:
            raise ValueError(f"Model tahmin hatası: {str(e)}")
    
    def _prepare_feature_vector(self, features, dataset_dir=None):
        """
        Feature vektörünü model için hazırla
        
        Args:
            features (dict): Çıkarılan özellikler
            dataset_dir (str, optional): Dataset dizini
            
        Returns:
            numpy.ndarray: Feature vektörü (2D array)
        """
        # Model'in beklediği feature names'i kontrol et
        if hasattr(self.trained_model, 'named_steps') and 'scaler' in self.trained_model.named_steps:
            scaler = self.trained_model.named_steps['scaler']
            if hasattr(scaler, 'feature_names_in_'):
                # Model'in beklediği feature isimlerini kullan
                expected_features = list(scaler.feature_names_in_)
                print(f"📊 Model'in beklediği features: {len(expected_features)} özellik")
                
                # Feature vektörü oluştur
                feature_row = []
                missing_features = []
                
                for col in expected_features:
                    if col in features:
                        value = features[col]
                        if pd.isna(value):
                            value = 0.0
                        feature_row.append(float(value))
                    else:
                        feature_row.append(0.0)
                        missing_features.append(col)
                
                if missing_features:
                    print(f"⚠️ {len(missing_features)} eksik özellik 0 ile dolduruldu")
                
                feature_df = pd.DataFrame([feature_row], columns=expected_features)
                print(f"🔢 Feature vektör boyutu: {feature_df.shape}")
                return feature_df.values
        
        # Fallback: dataset'ten feature columns'u yükle
        feature_columns = None
        
        if dataset_dir:
            features_file = Path(dataset_dir) / "features.csv"
            if features_file.exists():
                try:
                    df = pd.read_csv(features_file)
                    feature_columns = []
                    for col in df.columns:
                        if col not in ['filename', 'class', 'text'] and df[col].dtype in ['float64', 'int64']:
                            feature_columns.append(col)
                    print(f"📊 Training features yüklendi: {len(feature_columns)} özellik")
                except Exception as e:
                    print(f"⚠️ Features CSV yüklenemedi: {e}")
        
        # Son çare: mevcut numeric features
        if not feature_columns:
            feature_columns = []
            for key, value in features.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    feature_columns.append(key)
            print(f"📊 Mevcut features kullanılıyor: {len(feature_columns)} özellik")
        
        # Feature vektörü oluştur
        feature_row = []
        missing_features = []
        
        for col in feature_columns:
            if col in features:
                value = features[col]
                if pd.isna(value):
                    value = 0.0
                feature_row.append(float(value))
            else:
                feature_row.append(0.0)
                missing_features.append(col)
        
        if missing_features:
            print(f"⚠️ {len(missing_features)} eksik özellik 0 ile dolduruldu")
        
        feature_df = pd.DataFrame([feature_row], columns=feature_columns)
        print(f"🔢 Feature vektör boyutu: {feature_df.shape}")
        return feature_df.values

    def predict_single_file(self, audio_file, dataset_dir=None):
        """
        Tek bir dosya için tahmin yapar (eğitilmiş model varsa onu kullanır)
        
        Args:
            audio_file (str): Ses dosyası
            dataset_dir (str, optional): Training dataset dizini
            
        Returns:
            dict: Tahmin sonuçları
        """
        print(f"🔮 Özellik çıkarımı yapılıyor: {audio_file}")
        
        # Özellik çıkarımı
        features = self.feature_extractor.extract_all_features(audio_file)
        
        # Eğitilmiş model varsa onu kullan
        if self.trained_model and self.model_info:
            try:
                trained_prediction = self.predict_with_trained_model(features, dataset_dir)
                results = {
                    'file': audio_file,
                    'timestamp': datetime.now().isoformat(),
                    'prediction': trained_prediction,
                    'features_extracted': len([k for k, v in features.items() if isinstance(v, (int, float))]),
                    'note': f'Eğitilmiş {trained_prediction["model_name"]} modeli ile tahmin edildi.'
                }
                return results
            except Exception as e:
                print(f"⚠️ Eğitilmiş model hatası: {e}")
                print("📋 Demo tahmine geçiliyor...")
        
        # Demo tahmin (fallback)
        demo_prediction = self.demo_rule_based_prediction(features)
        
        results = {
            'file': audio_file,
            'timestamp': datetime.now().isoformat(),
            'prediction': demo_prediction,
            'features_extracted': len([k for k, v in features.items() if isinstance(v, (int, float))]),
            'note': 'Bu demo amaçlı basit bir tahmindir. Gerçek kullanım için eğitilmiş model gereklidir.'
        }
        
        return results
    
    def demo_rule_based_prediction(self, features):
        """
        Demo amaçlı basit kural tabanlı tahmin
        
        Args:
            features (dict): Çıkarılan özellikler
            
        Returns:
            dict: Tahmin sonucu
        """
        # Basit kurallar (sadece demo amaçlı)
        
        # Konuşma hızı ve duraklama analizi
        speaking_rate = features.get('speaking_rate', 0.5)
        avg_pause_duration = features.get('avg_pause_duration', 1.0)
        
        # Pitch varyasyonu
        pitch_std = features.get('pitch_std', 50)
        
        # MFCC varyasyonu
        mfcc_0_std = features.get('mfcc_0_std', 5)
        
        # Basit scoring
        score = 0
        factors = []
        
        # Düşük konuşma hızı → demans riski
        if speaking_rate < 0.3:
            score += 2
            factors.append("Düşük konuşma hızı")
        elif speaking_rate < 0.5:
            score += 1
            factors.append("Orta konuşma hızı")
        
        # Uzun duraklamalar → demans riski
        if avg_pause_duration > 2.0:
            score += 2
            factors.append("Uzun duraklamalar")
        elif avg_pause_duration > 1.5:
            score += 1
            factors.append("Orta duraklamalar")
        
        # Düşük pitch varyasyonu → demans riski
        if pitch_std < 30:
            score += 1
            factors.append("Düşük pitch varyasyonu")
        
        # Düşük MFCC varyasyonu → demans riski
        if mfcc_0_std < 3:
            score += 1
            factors.append("Düşük ses varyasyonu")
        
        # Tahmin
        if score >= 4:
            prediction = "Yüksek Risk"
            confidence = 0.8
        elif score >= 2:
            prediction = "Orta Risk"
            confidence = 0.6
        else:
            prediction = "Düşük Risk"
            confidence = 0.7
        
        return {
            'risk_level': prediction,
            'confidence': confidence,
            'risk_score': score,
            'risk_factors': factors,
            'key_metrics': {
                'speaking_rate': speaking_rate,
                'avg_pause_duration': avg_pause_duration,
                'pitch_std': pitch_std,
                'mfcc_0_std': mfcc_0_std
            }
        }
    
    def batch_predict(self, audio_files, output_file=None, dataset_dir=None):
        """
        Toplu tahmin yapar
        
        Args:
            audio_files (list): Ses dosyalarının listesi
            output_file (str, optional): Sonuçların kaydedileceği dosya
            dataset_dir (str, optional): Training dataset dizini
            
        Returns:
            list: Tüm tahmin sonuçları
        """
        print("🎯 Toplu tahmin başlıyor...")
        
        if self.trained_model:
            print(f"🤖 Eğitilmiş model kullanılıyor: {self.model_info['model_name']}")
        else:
            print("📋 Demo kural tabanlı tahmin kullanılıyor")
        
        all_results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Tahmin: {Path(audio_file).name}")
            
            try:
                result = self.predict_single_file(audio_file, dataset_dir)
                all_results.append(result)
                
                # Sonucu özetle
                self.print_prediction_summary(result)
                
            except Exception as e:
                print(f"❌ Hata: {str(e)}")
                all_results.append({
                    'file': audio_file,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Sonuçları kaydet
        if output_file and self.config["output"]["save_predictions"]:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            print(f"\n💾 Tahmin sonuçları kaydedildi: {output_file}")
        
        return all_results
    
    def print_prediction_summary(self, result):
        """Tahmin sonucunu özetler"""
        if 'prediction' in result:
            pred = result['prediction']
            
            # Eğitilmiş model sonucu
            if pred.get('method') == 'trained_model':
                print(f"🤖 Model Tahmini: {pred['predicted_class']} (Güven: {pred['confidence']:.3f})")
                print(f"   🏷️ Sınıf Olasılıkları:")
                for class_name, prob in pred['class_probabilities'].items():
                    emoji = "🟢" if class_name == pred['predicted_class'] else "⚪"
                    print(f"      {emoji} {class_name}: {prob:.3f}")
            
            # Demo tahmin sonucu
            elif 'risk_level' in pred:
                print(f"📊 Demo Tahmin: {pred['risk_level']} (Güven: {pred['confidence']:.2f})")
                print(f"   Risk Skoru: {pred['risk_score']}/6")
                if pred['risk_factors']:
                    print(f"   Risk Faktörleri: {', '.join(pred['risk_factors'])}")
    
    def generate_report(self, results, output_file="dementia_analysis_report.html"):
        """
        HTML rapor oluşturur
        
        Args:
            results (list): Tahmin sonuçları
            output_file (str): Rapor dosyası
        """
        if not self.config["output"]["generate_report"]:
            return
        
        print("📝 HTML rapor oluşturuluyor...")
        
        # İstatistikler
        total_files = len(results)
        successful_predictions = len([r for r in results if 'prediction' in r])
        risk_distribution = {}
        
        for result in results:
            if 'prediction' in result:
                pred = result['prediction']
                if pred.get('method') == 'trained_model':
                    risk_level = pred['predicted_class']
                else:
                    risk_level = pred.get('risk_level', 'Unknown')
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Demans Risk Analizi Raporu</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 15px; text-align: center; }}
                .summary {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .result {{ margin: 15px 0; padding: 20px; background: white; border-radius: 10px;
                          box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .risk-high {{ border-left: 5px solid #e74c3c; }}
                .risk-medium {{ border-left: 5px solid #f39c12; }}
                .risk-low {{ border-left: 5px solid #27ae60; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                           gap: 15px; margin: 15px 0; }}
                .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .error {{ color: #e74c3c; background: #fdf2f2; padding: 15px; border-radius: 8px; }}
                h1, h2, h3 {{ margin-top: 0; }}
                .stats {{ display: flex; justify-content: space-around; flex-wrap: wrap; }}
                .stat-item {{ text-align: center; margin: 10px; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 Demans Risk Analizi Raporu</h1>
                <p>Oluşturulma Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>📊 Genel Özet</h2>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-number">{total_files}</div>
                        <div>Toplam Dosya</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{successful_predictions}</div>
                        <div>Başarılı Analiz</div>
                    </div>
                    {''.join([f'<div class="stat-item"><div class="stat-number">{count}</div><div>{risk}</div></div>' 
                             for risk, count in risk_distribution.items()])}
                </div>
            </div>
        """
        
        # Her sonuç için detay
        for i, result in enumerate(results, 1):
            file_name = Path(result['file']).name
            
            if 'error' in result:
                html_content += f"""
                <div class="result">
                    <h3>📁 Dosya {i}: {file_name}</h3>
                    <div class="error">
                        <strong>Hata:</strong> {result['error']}
                    </div>
                </div>
                """
            elif 'prediction' in result:
                pred = result['prediction']
                
                # Eğitilmiş model sonucu
                if pred.get('method') == 'trained_model':
                    class_color = {'normal': 'success', 'mci': 'warning', 'dementia': 'danger'}.get(pred['predicted_class'], 'info')
                    
                    html_content += f"""
                    <div class="result risk-{class_color}">
                        <h3>🤖 Dosya {i}: {file_name}</h3>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                            <div>
                                <h4 style="margin: 0; color: #2c3e50;">Tahmin: {pred['predicted_class'].upper()}</h4>
                                <p style="margin: 5px 0; color: #7f8c8d;">Model: {pred['model_name']} | Güven: {pred['confidence']:.1%}</p>
                            </div>
                        </div>
                        
                        <div style="margin: 15px 0;">
                            <h5>📊 Sınıf Olasılıkları:</h5>
                            {''.join([f'<div style="margin: 5px 0;"><strong>{cls}:</strong> {prob:.1%}</div>' 
                                     for cls, prob in pred['class_probabilities'].items()])}
                        </div>
                        
                        <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 20px;">
                            📝 <strong>Not:</strong> {result.get('note', 'Eğitilmiş model ile tahmin.')}
                        </p>
                    </div>
                    """
                
                # Demo tahmin sonucu
                elif 'risk_level' in pred:
                    risk_class = pred['risk_level'].lower().replace(' ', '-').replace('ü', 'u').replace('ı', 'i')
                    
                    html_content += f"""
                    <div class="result risk-{risk_class}">
                        <h3>📁 Dosya {i}: {file_name}</h3>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                            <div>
                                <h4 style="margin: 0; color: #2c3e50;">Risk Seviyesi: {pred['risk_level']}</h4>
                                <p style="margin: 5px 0; color: #7f8c8d;">Güven: {pred['confidence']:.1%} | Risk Skoru: {pred['risk_score']}/6</p>
                            </div>
                        </div>
                        
                        {f'''
                        <div style="margin: 15px 0;">
                            <h5>🚨 Tespit Edilen Risk Faktörleri:</h5>
                            <ul>
                                {''.join([f'<li>{factor}</li>' for factor in pred['risk_factors']])}
                            </ul>
                        </div>
                        ''' if pred['risk_factors'] else '<p style="color: #27ae60;">✅ Önemli risk faktörü tespit edilmedi.</p>'}
                        
                        <div class="metrics">
                            <div class="metric">
                                <h5>🗣️ Konuşma Hızı</h5>
                                <p>{pred['key_metrics']['speaking_rate']:.3f}</p>
                            </div>
                            <div class="metric">
                                <h5>⏸️ Ortalama Duraklama</h5>
                                <p>{pred['key_metrics']['avg_pause_duration']:.2f}s</p>
                            </div>
                            <div class="metric">
                                <h5>🎵 Pitch Varyasyonu</h5>
                                <p>{pred['key_metrics']['pitch_std']:.1f}</p>
                            </div>
                            <div class="metric">
                                <h5>🔊 Ses Varyasyonu</h5>
                                <p>{pred['key_metrics']['mfcc_0_std']:.2f}</p>
                            </div>
                        </div>
                        
                        <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 20px;">
                            📝 <strong>Not:</strong> {result.get('note', 'Bu analiz demo amaçlıdır.')}
                        </p>
                    </div>
                    """
        
        html_content += """
            <div class="summary" style="margin-top: 30px;">
                <h3>⚠️ Önemli Uyarı</h3>
                <p>Bu analiz sadece araştırma ve geliştirme amaçlıdır. Gerçek tıbbi teşhis için bir sağlık uzmanına başvurun.</p>
                <p>Sonuçlar, ses dosyalarının kalitesi ve kayıt koşullarından etkilenebilir.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML rapor oluşturuldu: {output_file}")

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Ses Tabanlı Demans Risk Analizi Pipeline'ı")
    
    subparsers = parser.add_subparsers(dest='command', help='Komutlar')
    
    # Config oluştur
    config_parser = subparsers.add_parser('create-config', help='Örnek konfigürasyon dosyası oluştur')
    config_parser.add_argument('--output', '-o', default='config.json', help='Çıktı dosyası')
    
    # Özellik çıkarımı
    extract_parser = subparsers.add_parser('extract', help='Özellik çıkarımı')
    extract_parser.add_argument('input', help='Ses dosyası veya dizini')
    extract_parser.add_argument('--output', '-o', default='features.csv', help='Çıktı CSV dosyası')
    extract_parser.add_argument('--config', '-c', help='Konfigürasyon dosyası')
    extract_parser.add_argument('--model', '-m', help='Eğitilmiş model dosyası (.pkl)')
    extract_parser.add_argument('--dataset', '-d', help='Training dataset dizini')
    
    # Analiz
    analyze_parser = subparsers.add_parser('analyze', help='Risk analizi')
    analyze_parser.add_argument('input', nargs='+', help='Ses dosyası/dosyaları veya dizini')
    analyze_parser.add_argument('--output', '-o', help='Sonuç dosyası')
    analyze_parser.add_argument('--config', '-c', help='Konfigürasyon dosyası')
    analyze_parser.add_argument('--model', '-m', help='Eğitilmiş model dosyası (.pkl)')
    analyze_parser.add_argument('--dataset', '-d', help='Training dataset dizini (feature alignment için)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Pipeline başlat
    pipeline = DemantiaDetectionPipeline(
        config_file=getattr(args, 'config', None),
        model_file=getattr(args, 'model', None)
    )
    
    if args.command == 'create-config':
        pipeline.create_sample_config(args.output)
    
    elif args.command == 'extract':
        # Ses verilerini hazırla
        audio_files = pipeline.prepare_audio_data(args.input)
        
        # Özellik çıkarımı
        features_df = pipeline.extract_features_batch(audio_files, args.output)
        
        print(f"\n✅ Özellik çıkarımı tamamlandı!")
        print(f"📊 {len(features_df)} dosya, {len(features_df.columns)} özellik")
    
    elif args.command == 'analyze':
        # Ses verilerini hazırla (birden fazla path'i handle et)
        all_audio_files = []
        for input_path in args.input:
            audio_files = pipeline.prepare_audio_data(input_path)
            all_audio_files.extend(audio_files)
        
        # Risk analizi
        results = pipeline.batch_predict(
            all_audio_files, 
            args.output or "risk_analysis.json",
            getattr(args, 'dataset', None)
        )
        
        # Rapor oluştur
        pipeline.generate_report(results)
        
        print(f"\n✅ Risk analizi tamamlandı! {len(results)} dosya işlendi.")
        print("📄 Detaylı rapor için 'dementia_analysis_report.html' dosyasını açın.")

if __name__ == "__main__":
    main() 