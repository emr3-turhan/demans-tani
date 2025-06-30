#!/usr/bin/env python3
"""
Ses dönüştürücü test scripti
"""

import os
import tempfile
from pathlib import Path

def test_pydub_converter():
    """Pydub dönüştürücüsünü test eder"""
    print("🧪 Pydub Dönüştürücü Testi")
    
    try:
        from audio_converter import convert_m4a_to_wav
        
        # Test dosyası oluştur (gerçek .m4a dosyası gerekli)
        test_file = "test.m4a"
        
        if os.path.exists(test_file):
            print(f"📁 Test dosyası bulundu: {test_file}")
            
            # Dönüştürme testi
            result = convert_m4a_to_wav(test_file)
            if result:
                print(f"✅ Dönüştürme başarılı: {result}")
                
                # Dosya boyutunu kontrol et
                if os.path.exists(result):
                    size = os.path.getsize(result)
                    print(f"📊 Çıkış dosyası boyutu: {size:,} bytes")
                else:
                    print("❌ Çıkış dosyası oluşturulamadı")
            else:
                print("❌ Dönüştürme başarısız")
        else:
            print(f"⚠️ Test dosyası bulunamadı: {test_file}")
            print("   Gerçek bir .m4a dosyası ile test edin")
            
    except ImportError as e:
        print(f"❌ Pydub import hatası: {e}")
        print("   pip install pydub komutunu çalıştırın")
    except Exception as e:
        print(f"❌ Test hatası: {e}")

def test_ffmpeg_converter():
    """FFmpeg dönüştürücüsünü test eder"""
    print("\n🧪 FFmpeg Dönüştürücü Testi")
    
    try:
        from audio_converter_ffmpeg import convert_m4a_to_wav_ffmpeg, get_audio_info
        
        # Test dosyası oluştur
        test_file = "test.m4a"
        
        if os.path.exists(test_file):
            print(f"📁 Test dosyası bulundu: {test_file}")
            
            # Dosya bilgilerini göster
            print("📊 Dosya bilgileri:")
            get_audio_info(test_file)
            
            # Dönüştürme testi
            result = convert_m4a_to_wav_ffmpeg(test_file, sample_rate=44100, bit_depth=16)
            if result:
                print(f"✅ Dönüştürme başarılı<: {result}")
                
                # Dosya boyutunu kontrol et
                if os.path.exists(result):
                    size = os.path.getsize(result)
                    print(f"📊 Çıkış dosyası boyutu: {size:,} bytes")
                else:
                    print("❌ Çıkış dosyası oluşturulamadı")
            else:
                print("❌ Dönüştürme başarısız")
        else:
            print(f"⚠️ Test dosyası bulunamadı: {test_file}")
            print("   Gerçek bir .m4a dosyası ile test edin")
            
    except ImportError as e:
        print(f"❌ FFmpeg import hatası: {e}")
        print("   pip install ffmpeg-python komutunu çalıştırın")
    except Exception as e:
        print(f"❌ Test hatası: {e}")

def create_sample_m4a():
    """Örnek .m4a dosyası oluşturur (test amaçlı)"""
    print("\n📝 Örnek .m4a Dosyası Oluşturma")
    
    # Bu fonksiyon gerçek bir .m4a dosyası oluşturmaz
    # Sadece test amaçlı bir placeholder dosyası oluşturur
    sample_content = b"# This is a test file\n# Not a real .m4a file"
    
    with open("test.m4a", "wb") as f:
        f.write(sample_content)
    
    print("📄 test.m4a dosyası oluşturuldu (test amaçlı)")
    print("⚠️ Bu gerçek bir .m4a dosyası değildir!")
    print("   Gerçek bir .m4a dosyası ile test edin")

def main():
    """Ana test fonksiyonu"""
    print("🎵 Ses Dönüştürücü Test Scripti")
    print("=" * 40)
    
    # Test dosyası kontrolü
    if not os.path.exists("test.m4a"):
        print("📝 Test dosyası bulunamadı, oluşturuluyor...")
        create_sample_m4a()
    
    # Pydub testi
    test_pydub_converter()
    
    # FFmpeg testi
    test_ffmpeg_converter()
    
    print("\n" + "=" * 40)
    print("✅ Test tamamlandı!")
    print("\n💡 Gerçek .m4a dosyaları ile test etmek için:")
    print("   python audio_converter.py gercek_dosya.m4a")
    print("   python audio_converter_ffmpeg.py gercek_dosya.m4a")

if __name__ == "__main__":
    main() 