#!/usr/bin/env python3
"""
Ses dosyalarını .m4a formatından .wav formatına çeviren Python scripti
"""

import os
import sys
from pathlib import Path
from pydub import AudioSegment

def convert_m4a_to_wav(input_file, output_file=None, sample_rate=44100):
    """
    .m4a dosyasını .wav formatına çevirir
    
    Args:
        input_file (str): Giriş .m4a dosyasının yolu
        output_file (str, optional): Çıkış .wav dosyasının yolu. Belirtilmezse aynı isimle .wav uzantısıyla kaydedilir
        sample_rate (int): Örnekleme hızı (varsayılan: 44100 Hz)
    
    Returns:
        str: Çıkış dosyasının yolu
    """
    try:
        # Giriş dosyasının varlığını kontrol et
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Dosya bulunamadı: {input_file}")
        
        # Çıkış dosyası belirtilmemişse otomatik oluştur
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.with_suffix('.wav'))
        
        print(f"Çeviriliyor: {input_file} -> {output_file}")
        
        # .m4a dosyasını yükle
        audio = AudioSegment.from_file(input_file, format="m4a")
        
        # .wav formatına çevir
        audio.export(output_file, format="wav", parameters=["-ar", str(sample_rate)])
        
        print(f"✅ Başarıyla çevrildi: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        return None

def batch_convert_m4a_to_wav(input_directory, output_directory=None, sample_rate=44100):
    """
    Bir dizindeki tüm .m4a dosyalarını .wav formatına çevirir
    
    Args:
        input_directory (str): Giriş dizini
        output_directory (str, optional): Çıkış dizini. Belirtilmezse aynı dizine kaydedilir
        sample_rate (int): Örnekleme hızı
    """
    input_path = Path(input_directory)
    
    if not input_path.exists():
        print(f"❌ Dizin bulunamadı: {input_directory}")
        return
    
    # .m4a dosyalarını bul
    m4a_files = list(input_path.glob("*.m4a"))
    
    if not m4a_files:
        print(f"❌ {input_directory} dizininde .m4a dosyası bulunamadı")
        return
    
    print(f"📁 {len(m4a_files)} adet .m4a dosyası bulundu")
    
    # Çıkış dizinini oluştur
    if output_directory:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Her dosyayı çevir
    success_count = 0
    for m4a_file in m4a_files:
        output_file = output_path / f"{m4a_file.stem}.wav"
        if convert_m4a_to_wav(str(m4a_file), str(output_file), sample_rate):
            success_count += 1
    
    print(f"\n📊 Toplam {len(m4a_files)} dosyadan {success_count} tanesi başarıyla çevrildi")

def main():
    """Ana fonksiyon - komut satırı argümanlarını işler"""
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  Tek dosya: python audio_converter.py dosya.m4a")
        print("  Dizin: python audio_converter.py /dizin/yolu")
        print("  Özel çıkış: python audio_converter.py dosya.m4a cikis.wav")
        return
    
    input_path = sys.argv[1]
    
    if os.path.isfile(input_path):
        # Tek dosya çevirme
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_m4a_to_wav(input_path, output_file)
    elif os.path.isdir(input_path):
        # Dizin çevirme
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        batch_convert_m4a_to_wav(input_path, output_dir)
    else:
        print(f"❌ Dosya veya dizin bulunamadı: {input_path}")

if __name__ == "__main__":
    main() 