#!/usr/bin/env python3
"""
FFmpeg kullanarak ses dosyalarını .m4a formatından .wav formatına çeviren Python scripti
Daha gelişmiş ses işleme seçenekleri sunar
"""

import os
import sys
import ffmpeg
from pathlib import Path

def convert_m4a_to_wav_ffmpeg(input_file, output_file=None, sample_rate=44100, bit_depth=16):
    """
    FFmpeg kullanarak .m4a dosyasını .wav formatına çevirir
    
    Args:
        input_file (str): Giriş .m4a dosyasının yolu
        output_file (str, optional): Çıkış .wav dosyasının yolu
        sample_rate (int): Örnekleme hızı (varsayılan: 44100 Hz)
        bit_depth (int): Bit derinliği (16 veya 24)
    
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
        
        print(f"FFmpeg ile çeviriliyor: {input_file} -> {output_file}")
        
        # FFmpeg pipeline oluştur
        stream = ffmpeg.input(input_file)
        
        # Ses parametrelerini ayarla
        stream = ffmpeg.output(
            stream,
            output_file,
            acodec='pcm_s16le' if bit_depth == 16 else 'pcm_s24le',
            ar=sample_rate,
            ac=2,  # Stereo
            loglevel='error'
        )
        
        # Dönüşümü gerçekleştir
        ffmpeg.run(stream, overwrite_output=True)
        
        print(f"✅ Başarıyla çevrildi: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        return None

def get_audio_info(input_file):
    """
    Ses dosyasının bilgilerini gösterir
    
    Args:
        input_file (str): Ses dosyasının yolu
    """
    try:
        probe = ffmpeg.probe(input_file)
        audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        
        print(f"📊 Ses Dosyası Bilgileri: {input_file}")
        print(f"   Codec: {audio_info.get('codec_name', 'Bilinmiyor')}")
        print(f"   Sample Rate: {audio_info.get('sample_rate', 'Bilinmiyor')} Hz")
        print(f"   Channels: {audio_info.get('channels', 'Bilinmiyor')}")
        print(f"   Duration: {float(probe['format']['duration']):.2f} saniye")
        print(f"   Bit Rate: {int(probe['format']['bit_rate'])/1000:.0f} kbps")
        
    except Exception as e:
        print(f"❌ Dosya bilgileri alınamadı: {str(e)}")

def batch_convert_with_ffmpeg(input_directory, output_directory=None, sample_rate=44100, bit_depth=16):
    """
    FFmpeg kullanarak bir dizindeki tüm .m4a dosyalarını .wav formatına çevirir
    
    Args:
        input_directory (str): Giriş dizini
        output_directory (str, optional): Çıkış dizini
        sample_rate (int): Örnekleme hızı
        bit_depth (int): Bit derinliği
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
        if convert_m4a_to_wav_ffmpeg(str(m4a_file), str(output_file), sample_rate, bit_depth):
            success_count += 1
    
    print(f"\n📊 Toplam {len(m4a_files)} dosyadan {success_count} tanesi başarıyla çevrildi")

def main():
    """Ana fonksiyon - komut satırı argümanlarını işler"""
    if len(sys.argv) < 2:
        print("FFmpeg Ses Dönüştürücü")
        print("Kullanım:")
        print("  Tek dosya: python audio_converter_ffmpeg.py dosya.m4a")
        print("  Dizin: python audio_converter_ffmpeg.py /dizin/yolu")
        print("  Özel çıkış: python audio_converter_ffmpeg.py dosya.m4a cikis.wav")
        print("  Bilgi göster: python audio_converter_ffmpeg.py --info dosya.m4a")
        return
    
    input_path = sys.argv[1]
    
    # Bilgi gösterme modu
    if input_path == "--info" and len(sys.argv) > 2:
        get_audio_info(sys.argv[2])
        return
    
    if os.path.isfile(input_path):
        # Tek dosya çevirme
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_m4a_to_wav_ffmpeg(input_path, output_file)
    elif os.path.isdir(input_path):
        # Dizin çevirme
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        batch_convert_with_ffmpeg(input_path, output_dir)
    else:
        print(f"❌ Dosya veya dizin bulunamadı: {input_path}")

if __name__ == "__main__":
    main() 