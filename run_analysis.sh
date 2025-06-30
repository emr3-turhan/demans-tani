#!/bin/bash

# Demans Analizi için Shell Script
# Bu script conda ortamını düzgün şekilde ayarlar ve analizi çalıştırır

# Conda'yı PATH'e ekle
export PATH="/opt/anaconda3/bin:$PATH"

# Kullanım talimatları
if [ $# -eq 0 ]; then
    echo "🧠 Demans Risk Analizi - Kullanım:"
    echo ""
    echo "Hızlı analiz:"
    echo "  ./run_analysis.sh analyze kayit.m4a"
    echo ""
    echo "Özellik çıkarımı:"
    echo "  ./run_analysis.sh extract kayit.wav --output features.csv"
    echo ""
    echo "Konfigürasyon oluştur:"
    echo "  ./run_analysis.sh create-config --output config.json"
    echo ""
    echo "Basit ses dönüştürme:"
    echo "  ./run_analysis.sh convert kayit.m4a output.wav"
    exit 0
fi

# Komut türüne göre çalıştır
case "$1" in
    "analyze"|"extract"|"create-config")
        echo "🚀 Pipeline çalıştırılıyor..."
        python dementia_detection_pipeline.py "$@"
        ;;
    "convert")
        if [ $# -lt 3 ]; then
            echo "❌ Kullanım: ./run_analysis.sh convert input.m4a output.wav"
            exit 1
        fi
        echo "🎵 Ses dosyası dönüştürülüyor..."
        python audio_converter.py "$2" "$3"
        ;;
    *)
        echo "❌ Bilinmeyen komut: $1"
        echo "Kullanım için: ./run_analysis.sh"
        exit 1
        ;;
esac 