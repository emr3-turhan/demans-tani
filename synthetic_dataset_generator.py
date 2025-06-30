#!/usr/bin/env python3
"""
Sentetik Audio Dataset Üretici
Text-to-Speech ile Normal, MCI ve Demans sınıfları için ses örnekleri üretir
"""

import os
import random
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

# TTS için
try:
    import pyttsx3
except ImportError:
    print("⚠️ pyttsx3 yüklü değil. Kurulum: pip install pyttsx3")

try:
    import gtts
    from gtts import gTTS
except ImportError:
    print("⚠️ gtts yüklü değil. Kurulum: pip install gtts")

# Ses işleme için
import librosa
import soundfile as sf
from pydub import AudioSegment
# pydub için speedchange alternative
from pydub.silence import split_on_silence

class SyntheticDatasetGenerator:
    """Sentetik audio dataset üretici sınıfı"""
    
    def __init__(self, output_dir="synthetic_dataset"):
        """
        Args:
            output_dir (str): Çıktı dizini
        """
        self.output_dir = Path(output_dir)
        self.setup_directories()
        
        # Metin şablonları - çeşitli kognitif görevler
        self.base_texts = [
            "Bugün sabah kahvaltıda peynir ve zeytin yedim, sonra dışarı çıktım.",
            "Geçen hafta arkadaşımla sinemaya gittik, çok güzel bir film izledik.",
            "Yarın market alışverişi yapacağım, ekmek, süt ve meyve almalıyım.",
            "Akşam yemeğinde tavuk ve pilav yedik, çok lezzetliydi.",
            "Doktorla randevum saat üçte, hastaneye erken gitmeliyim.",
            "Kızım beni aradı, torunlarının nasıl olduğunu sordu.",
            "Bahçede çiçek ekiyorum, güller ve laleler çok güzel açtı.",
            "Televizyonda haberleri izledim, hava durumuna baktım.",
            "Komşumla konuştuk, onun kedisi çok sevimli.",
            "Kitap okumayı seviyorum, özellikle tarihi romanları."
        ]
        
        # Sınıf tanımları
        self.class_configs = {
            'normal': {
                'speech_rate': (180, 220),  # kelime/dakika
                'pause_probability': 0.1,   # cümle arası duraklama
                'repetition_probability': 0.0,  # tekrar olasılığı
                'hesitation_probability': 0.05,  # tereddüt
                'energy_level': (0.8, 1.0),
                'pitch_variation': (0.9, 1.1),
                'description': 'Akıcı, düzgün telaffuz, net duraksamalar'
            },
            'mci': {
                'speech_rate': (120, 180),
                'pause_probability': 0.25,
                'repetition_probability': 0.1,
                'hesitation_probability': 0.15,
                'energy_level': (0.6, 0.8),
                'pitch_variation': (0.8, 1.0),
                'description': 'Orta derecede duraksamalar, kararsızlık tonu'
            },
            'dementia': {
                'speech_rate': (80, 120),
                'pause_probability': 0.4,
                'repetition_probability': 0.25,
                'hesitation_probability': 0.3,
                'energy_level': (0.4, 0.6),
                'pitch_variation': (0.7, 0.9),
                'description': 'Bozuk cümle yapıları, sık tekrar, düşük enerji'
            }
        }
        
    def setup_directories(self):
        """Çıktı dizinlerini oluştur"""
        self.output_dir.mkdir(exist_ok=True)
        
        for class_name in ['normal', 'mci', 'dementia']:
            class_dir = self.output_dir / class_name
            class_dir.mkdir(exist_ok=True)
        
        print(f"📁 Dizinler oluşturuldu: {self.output_dir}")
    
    def modify_text_for_class(self, text, class_name):
        """Sınıfa göre metni değiştir"""
        config = self.class_configs[class_name]
        modified_text = text
        
        # MCI ve demans için metni değiştir
        if class_name == 'mci':
            # Hafif tekrarlar ve tereddütler ekle
            if random.random() < config['hesitation_probability']:
                modified_text = "Şey... " + modified_text
            
            if random.random() < config['repetition_probability']:
                words = modified_text.split()
                if len(words) > 3:
                    repeat_idx = random.randint(1, min(3, len(words)-1))
                    words.insert(repeat_idx, words[repeat_idx-1])
                    modified_text = " ".join(words)
        
        elif class_name == 'dementia':
            # Daha fazla tekrar ve bozuk yapı
            if random.random() < config['hesitation_probability']:
                hesitations = ["Şey...", "Hmm...", "Ne diyordum...", "Bir dakika..."]
                modified_text = random.choice(hesitations) + " " + modified_text
            
            if random.random() < config['repetition_probability']:
                words = modified_text.split()
                if len(words) > 2:
                    # Birkaç kelimeyi tekrarla
                    repeat_count = random.randint(1, 3)
                    for _ in range(repeat_count):
                        repeat_idx = random.randint(0, len(words)-1)
                        words.insert(repeat_idx+1, words[repeat_idx])
                    modified_text = " ".join(words)
            
            # Cümle yapısını boz
            if random.random() < 0.2:
                modified_text = modified_text.replace(".", "... neyse...")
        
        return modified_text
    
    def generate_speech_pyttsx3(self, text, output_file, class_name):
        """pyttsx3 ile ses üret"""
        try:
            engine = pyttsx3.init()
            config = self.class_configs[class_name]
            
            # Hız ayarı
            speech_rate = random.uniform(*config['speech_rate'])
            engine.setProperty('rate', speech_rate)
            
            # Ses seviyesi
            volume = random.uniform(*config['energy_level'])
            engine.setProperty('volume', volume)
            
            # Farklı sesler için
            voices = engine.getProperty('voices')
            if voices:
                voice_idx = random.randint(0, min(2, len(voices)-1))
                engine.setProperty('voice', voices[voice_idx].id)
            
            engine.save_to_file(text, str(output_file))
            engine.runAndWait()
            engine.stop()
            
            return True
            
        except Exception as e:
            print(f"❌ pyttsx3 hatası: {e}")
            return False
    
    def generate_speech_gtts(self, text, output_file, class_name):
        """Google TTS ile ses üret"""
        try:
            # GTTS ses üretimi
            tts = gTTS(text=text, lang='tr', slow=False)
            temp_file = output_file.with_suffix('.mp3')
            tts.save(str(temp_file))
            
            # MP3'ü WAV'a çevir ve modifikasyonları uygula
            audio = AudioSegment.from_mp3(str(temp_file))
            
            config = self.class_configs[class_name]
            
            # Hız değişikliği (playback speed ile)
            if class_name == 'mci':
                # MCI için %80 hız
                new_sample_rate = int(audio.frame_rate * 0.8)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                audio = audio.set_frame_rate(audio.frame_rate)
            elif class_name == 'dementia':
                # Dementia için %60 hız  
                new_sample_rate = int(audio.frame_rate * 0.6)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                audio = audio.set_frame_rate(audio.frame_rate)
            
            # Ses seviyesi
            volume_change = random.uniform(-10, 0) if class_name != 'normal' else random.uniform(-5, 5)
            audio = audio + volume_change
            
            # Duraksamalar ekle
            if random.random() < config['pause_probability']:
                silence = AudioSegment.silent(duration=random.randint(500, 2000))
                # Cümle ortasına sessizlik ekle
                mid_point = len(audio) // 2
                audio = audio[:mid_point] + silence + audio[mid_point:]
            
            # WAV olarak kaydet
            audio.export(str(output_file), format="wav")
            
            # Geçici MP3 dosyasını sil
            temp_file.unlink()
            
            return True
            
        except Exception as e:
            print(f"❌ GTTS hatası: {e}")
            return False
    
    def post_process_audio(self, audio_file, class_name):
        """Ses dosyasını son işle"""
        try:
            # Librosa ile yükle
            y, sr = librosa.load(audio_file, sr=22050)
            
            config = self.class_configs[class_name]
            
            # Pitch değişikliği (demans için daha düşük)
            if class_name in ['mci', 'dementia']:
                pitch_shift = random.uniform(-2, 0)  # Daha düşük pitch
                y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_shift)
            
            # Gürültü ekle (demans için daha fazla)
            if class_name == 'dementia':
                noise_level = random.uniform(0.01, 0.03)
                noise = np.random.normal(0, noise_level, len(y))
                y = y + noise
            elif class_name == 'mci':
                noise_level = random.uniform(0.005, 0.015)
                noise = np.random.normal(0, noise_level, len(y))
                y = y + noise
            
            # Normalizasyon
            y = y / np.max(np.abs(y))
            
            # Kaydet
            sf.write(audio_file, y, sr)
            
            return True
            
        except Exception as e:
            print(f"❌ Post-processing hatası: {e}")
            return False
    
    def generate_dataset(self, samples_per_class=100, use_gtts=True):
        """
        Sentetik dataset üret
        
        Args:
            samples_per_class (int): Her sınıf için örnek sayısı
            use_gtts (bool): Google TTS kullan (False ise pyttsx3)
        """
        print(f"🎵 Sentetik dataset üretimi başlıyor...")
        print(f"📊 Her sınıf için {samples_per_class} örnek üretilecek")
        
        dataset_info = {
            'creation_date': datetime.now().isoformat(),
            'samples_per_class': samples_per_class,
            'tts_engine': 'gtts' if use_gtts else 'pyttsx3',
            'classes': {},
            'texts_used': self.base_texts,
            'total_samples': 0
        }
        
        labels_data = []
        
        for class_name, config in self.class_configs.items():
            print(f"\n🎯 {class_name.upper()} sınıfı işleniyor...")
            print(f"   {config['description']}")
            
            class_dir = self.output_dir / class_name
            successful_samples = 0
            
            for i in range(samples_per_class):
                # Rastgele metin seç
                base_text = random.choice(self.base_texts)
                
                # Sınıfa göre modifiye et
                modified_text = self.modify_text_for_class(base_text, class_name)
                
                # Dosya adı
                output_file = class_dir / f"{i+1:03d}.wav"
                
                # Ses üret
                success = False
                if use_gtts:
                    success = self.generate_speech_gtts(modified_text, output_file, class_name)
                else:
                    success = self.generate_speech_pyttsx3(modified_text, output_file, class_name)
                
                if success:
                    # Post-processing
                    self.post_process_audio(output_file, class_name)
                    successful_samples += 1
                    
                    # Label bilgisini kaydet
                    labels_data.append({
                        'filename': f"{class_name}/{output_file.name}",
                        'class': class_name,
                        'text': modified_text,
                        'original_text': base_text
                    })
                    
                    if (i + 1) % 10 == 0:
                        print(f"   ✅ {i+1}/{samples_per_class} tamamlandı")
                else:
                    print(f"   ❌ {i+1}. örnek başarısız")
            
            dataset_info['classes'][class_name] = {
                'samples_generated': successful_samples,
                'config': config
            }
            
            print(f"   📊 {class_name}: {successful_samples}/{samples_per_class} başarılı")
        
        # Labels CSV'sini kaydet
        labels_df = pd.DataFrame(labels_data)
        labels_file = self.output_dir / "labels.csv"
        labels_df.to_csv(labels_file, index=False)
        
        # Dataset bilgilerini kaydet
        dataset_info['total_samples'] = len(labels_data)
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dataset üretimi tamamlandı!")
        print(f"📁 Dizin: {self.output_dir}")
        print(f"📊 Toplam örnek: {len(labels_data)}")
        print(f"📋 Labels: {labels_file}")
        print(f"📄 Bilgi: {info_file}")
        
        return labels_df, dataset_info

def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentetik Audio Dataset Üretici")
    parser.add_argument('--output', '-o', default='synthetic_dataset', 
                       help='Çıktı dizini (varsayılan: synthetic_dataset)')
    parser.add_argument('--samples', '-s', type=int, default=100,
                       help='Her sınıf için örnek sayısı (varsayılan: 100)')
    parser.add_argument('--engine', '-e', choices=['gtts', 'pyttsx3'], default='gtts',
                       help='TTS motoru (varsayılan: gtts)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test modu (her sınıftan 5 örnek)')
    
    args = parser.parse_args()
    
    if args.test:
        args.samples = 5
        print("🧪 Test modu aktif - her sınıftan 5 örnek üretilecek")
    
    # Dataset üretici
    generator = SyntheticDatasetGenerator(args.output)
    
    # Dataset üret
    try:
        labels_df, info = generator.generate_dataset(
            samples_per_class=args.samples,
            use_gtts=(args.engine == 'gtts')
        )
        
        print(f"\n📈 Dataset İstatistikleri:")
        print(f"   🎯 Normal: {len(labels_df[labels_df['class'] == 'normal'])} örnek")
        print(f"   🎯 MCI: {len(labels_df[labels_df['class'] == 'mci'])} örnek") 
        print(f"   🎯 Dementia: {len(labels_df[labels_df['class'] == 'dementia'])} örnek")
        
        print(f"\n🚀 Bir sonraki adım:")
        print(f"   python train_synthetic_model.py --dataset {args.output}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 