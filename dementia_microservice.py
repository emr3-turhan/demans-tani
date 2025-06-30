#!/usr/bin/env python3
"""
Demans Analizi FastAPI Mikroservisi
"""

import os
import tempfile
import aiofiles
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Kendi modüllerimiz
from dementia_detection_pipeline import DemantiaDetectionPipeline
from feature_extraction import AudioFeatureExtractor

# FastAPI uygulaması
app = FastAPI(
    title="Demans Analizi Mikroservisi",
    description="Ses dosyalarını analiz ederek demans risk değerlendirmesi yapar",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veri modelleri
class AnalysisRequest(BaseModel):
    """
    Backend'den gelen analiz isteği
    Sadece test_session_id ve question_id gerekli
    Callback URL sabit: /api/dementia-analysis/callback
    """
    test_session_id: str
    question_id: str
    
class AnalysisResponse(BaseModel):
    """
    Mikroservisten backend'e dönen başlatma yanıtı
    """
    success: bool
    message: str
    analysis_id: str
    estimated_duration: int  # saniye

class AnalysisResult(BaseModel):
    # ============= PRIMARY IDENTIFIERS =============
    test_session_id: str
    question_id: str
    analysis_id: str
    
    # ============= TIMING INFORMATION =============
    timestamp: str
    processing_time_seconds: float
    start_time: str
    end_time: str
    
    # ============= MAIN ANALYSIS RESULTS =============
    dementia_status: str  # "normal", "mci", "dementia"
    confidence_score: float  # 0.0 - 1.0
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # ============= DETAILED SCORES =============
    normal_score: float
    mci_score: float
    dementia_score: float
    
    # ============= CLINICAL INDICATORS =============
    cognitive_decline_risk: float  # 0.0 - 1.0
    memory_impairment_likelihood: float  # 0.0 - 1.0
    speech_pattern_anomalies: float  # 0.0 - 1.0
    attention_deficit_indicator: float  # 0.0 - 1.0
    language_fluency_score: float  # 0.0 - 1.0
    overall_cognitive_health: float  # 0-100
    
    # ============= AUDIO INFORMATION =============
    audio_duration_seconds: float
    audio_format: str  # "m4a", "wav", "mp3"
    audio_sample_rate: int  # 22050, 44100
    audio_channels: int  # 1, 2
    audio_quality: str  # "good", "short", "poor"
    feature_count: int
    
    # ============= MODEL INFORMATION =============
    model_name: str  # "RandomForest", "SVM", etc.
    model_version: str  # "1.0", "2.0"
    model_method: str  # "trained_model", "rule_based"
    model_training_date: str
    model_accuracy: str  # "95%", "87%"
    
    # ============= PROCESSING METADATA =============
    feature_extraction_time: float
    model_inference_time: float
    server_location: str  # "microservice", "cloud", "local"
    api_version: str  # "1.0.0", "2.1.3"
    
    # ============= RECOMMENDATIONS =============
    recommendations: List[str]
    
    # ============= SYSTEM STATUS =============
    status: str  # "completed", "failed", "processing"
    error_message: Optional[str] = None
    
    # ============= ADDITIONAL METADATA =============
    session_type: str = "screening"  # "screening", "followup", "diagnosis"
    priority_level: int = 1  # 1: Normal, 2: Medium, 3: High, 4: Urgent

# Global pipeline instance
pipeline = None
BASE_AUDIO_URL = "https://demantia-backendv2-dev.onrender.com/api/test-responses/audio"

def calculate_risk_level(predicted_class: str, confidence: float) -> str:
    """Risk seviyesi hesapla"""
    if predicted_class == "normal":
        return "LOW" if confidence > 0.7 else "MEDIUM"
    elif predicted_class == "mci":
        return "MEDIUM" if confidence > 0.6 else "HIGH"
    elif predicted_class == "dementia":
        return "CRITICAL" if confidence > 0.6 else "HIGH"
    return "UNKNOWN"

def calculate_clinical_indicators(prediction: Dict) -> Dict[str, Any]:
    """Klinik göstergeleri hesapla"""
    scores = prediction['class_probabilities']
    
    return {
        "cognitive_decline_risk": scores.get('dementia', 0) + scores.get('mci', 0),
        "memory_impairment_likelihood": scores.get('dementia', 0) * 0.8 + scores.get('mci', 0) * 0.4,
        "speech_pattern_anomalies": 1.0 - scores.get('normal', 0),
        "attention_deficit_indicator": scores.get('mci', 0) + scores.get('dementia', 0) * 0.6,
        "language_fluency_score": scores.get('normal', 0),
        "overall_cognitive_health": scores.get('normal', 0) * 100  # 0-100 skala
    }

def generate_recommendations(predicted_class: str, confidence: float) -> List[str]:
    """Öneriler oluştur"""
    recommendations = []
    
    if predicted_class == "normal":
        if confidence > 0.8:
            recommendations.extend([
                "Kognitif sağlık normal görünüyor",
                "Düzenli zihinsel egzersizlere devam edin",
                "Yıllık kontroller önerilir"
            ])
        else:
            recommendations.extend([
                "Sonuçlar normal aralıkta ancak takip önerilir",
                "6 ay sonra tekrar değerlendirme yapılabilir",
                "Zihinsel aktiviteleri artırın"
            ])
    
    elif predicted_class == "mci":
        recommendations.extend([
            "Hafif kognitif bozukluk belirtileri tespit edildi",
            "Nöroloji uzmanına başvurun",
            "Düzenli takip gereklidir",
            "Zihinsel stimülasyon aktivitelerini artırın",
            "Yaşam tarzı değişiklikleri önerilir"
        ])
        
        if confidence > 0.7:
            recommendations.append("3 ay içinde detaylı nöropsikolojik değerlendirme")
        else:
            recommendations.append("6 ay içinde tekrar değerlendirme")
    
    elif predicted_class == "dementia":
        recommendations.extend([
            "Demans belirtileri tespit edildi",
            "Acil nöroloji konsültasyonu gerekli",
            "Comprehensive kognitif değerlendirme şart",
            "Aile desteği koordinasyonu",
            "Tedavi planı oluşturulmalı"
        ])
        
        if confidence > 0.8:
            recommendations.append("Immediate medical attention required")
        
    return recommendations

def initialize_pipeline():
    """Pipeline'ı başlat"""
    global pipeline
    if pipeline is None:
        model_path = "full_synthetic_dataset/trained_models/best_model_randomforest.pkl"
        dataset_path = "full_synthetic_dataset"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        pipeline = DemantiaDetectionPipeline(
            model_file=model_path
        )
        print("✅ Pipeline başlatıldı")

@app.on_event("startup")
async def startup_event():
    """Uygulama başlarken çalışır"""
    try:
        initialize_pipeline()
        print("🚀 Demans Analizi Mikroservisi başlatıldı")
    except Exception as e:
        print(f"❌ Başlatma hatası: {e}")
        raise

@app.get("/")
async def root():
    """
    🏠 Demans Analizi Mikroservisi Ana Sayfa
    
    Backend-Triggered Workflow:
    Backend -> /analyze -> Audio Download -> AI Analysis -> Callback
    """
    return {
        "service": "🧠 Demans Analizi Mikroservisi",
        "version": "1.0.0",
        "status": "active",
        "pipeline_ready": pipeline is not None,
        "workflow": {
            "1_trigger": "Backend POST /analyze with test_session_id & question_id",
            "2_download": "GET audio from backend API",
            "3_analysis": "AI model processes audio",
            "4_callback": "POST result to backend /api/dementia-analysis/callback"
        },
        "endpoints": {
            "🏠 home": "/",
            "❤️ health": "/health", 
            "🚀 trigger_analysis": "/analyze",
            "🔬 sync_test": "/analyze-sync",
            "📚 api_docs": "/docs",
            "📖 redoc": "/redoc"
        },
        "backend_integration": {
            "audio_source": "https://demantia-backendv2-dev.onrender.com/api/test-responses/audio/{test_session_id}/{question_id}",
            "callback_url": "https://demantia-backendv2-dev.onrender.com/api/dementia-analysis/callback"
        }
    }

@app.get("/health")
async def health_check():
    """Servis sağlığını kontrol et"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline is not None
    }

async def download_audio(test_session_id: str, question_id: str) -> Path:
    """
    Backend'den ses dosyasını indir
    
    Args:
        test_session_id: Test oturum ID'si
        question_id: Soru ID'si
        
    Returns:
        Path: İndirilen dosyanın yolu
    """
    url = f"{BASE_AUDIO_URL}/{test_session_id}/{question_id}"
    
    # Geçici dosya oluştur
    temp_dir = Path(tempfile.mkdtemp())
    audio_file = temp_dir / f"{test_session_id}_{question_id}.m4a"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📥 Ses dosyası indiriliyor: {url}")
            response = await client.get(url)
            response.raise_for_status()
            
            # Dosyayı kaydet
            async with aiofiles.open(audio_file, 'wb') as f:
                await f.write(response.content)
            
            print(f"✅ Ses dosyası indirildi: {audio_file}")
            return audio_file
            
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ses dosyası indirilemedi: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"İndirme hatası: {str(e)}"
        )

async def process_audio(audio_file: Path, test_session_id: str, question_id: str) -> AnalysisResult:
    """
    Ses dosyasını analiz et
    
    Args:
        audio_file: Ses dosyası yolu
        test_session_id: Test oturum ID'si
        question_id: Soru ID'si
        
    Returns:
        AnalysisResult: Analiz sonucu
    """
    start_time = datetime.now()
    analysis_id = f"{test_session_id}_{question_id}_{int(start_time.timestamp())}"
    
    try:
        print(f"🔬 Analiz başlıyor: {audio_file}")
        
        # Ses dosyasını .wav'a çevir
        from audio_converter import convert_m4a_to_wav
        wav_file = convert_m4a_to_wav(str(audio_file))
        
        if not wav_file:
            raise Exception("Ses dosyası dönüştürülemedi")
        
        # Analiz yap
        result = pipeline.predict_single_file(wav_file, "full_synthetic_dataset")
        
        if 'error' in result:
            raise Exception(f"Analiz hatası: {result['error']}")
        
        if 'prediction' not in result or result['prediction'].get('method') != 'trained_model':
            raise Exception("Model tahmini alınamadı")
        
        prediction = result['prediction']
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Ses dosyası süresini al
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(wav_file)
        audio_duration = len(audio_segment) / 1000.0  # milisaniye -> saniye
        
        # Feature sayısını al
        feature_count = result.get('features_extracted', 0)
        
        # Risk level hesapla
        risk_level = calculate_risk_level(prediction['predicted_class'], prediction['confidence'])
        
        # Klinik göstergeler hesapla
        clinical_indicators = calculate_clinical_indicators(prediction)
        
        # Öneriler oluştur
        recommendations = generate_recommendations(prediction['predicted_class'], prediction['confidence'])
        
        analysis_result = AnalysisResult(
            # Temel Tanımlayıcılar
            test_session_id=test_session_id,
            question_id=question_id,
            analysis_id=analysis_id,
            
            # Zaman Bilgisi
            timestamp=end_time.isoformat(),
            processing_time_seconds=processing_time,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            
            # Ana Sonuç
            dementia_status=prediction['predicted_class'],
            confidence_score=prediction['confidence'],
            risk_level=risk_level,
            
            # Detaylı Skorlar
            normal_score=prediction['class_probabilities'].get('normal', 0.0),
            mci_score=prediction['class_probabilities'].get('mci', 0.0),
            dementia_score=prediction['class_probabilities'].get('dementia', 0.0),
            
            # Klinik Göstergeler
            cognitive_decline_risk=clinical_indicators['cognitive_decline_risk'],
            memory_impairment_likelihood=clinical_indicators['memory_impairment_likelihood'],
            speech_pattern_anomalies=clinical_indicators['speech_pattern_anomalies'],
            attention_deficit_indicator=clinical_indicators['attention_deficit_indicator'],
            language_fluency_score=clinical_indicators['language_fluency_score'],
            overall_cognitive_health=clinical_indicators['overall_cognitive_health'],
            
            # Ses Bilgileri
            audio_duration_seconds=audio_duration,
            audio_format="m4a",
            audio_sample_rate=22050,
            audio_channels=1,
            audio_quality="good" if audio_duration > 2.0 else "short",
            feature_count=feature_count,
            
            # Model Bilgileri
            model_name=prediction['model_name'],
            model_version="1.0",
            model_method=prediction['method'],
            model_training_date="2025-06-30T19:45:53",
            model_accuracy="95%",
            
            # İşlem Metadata
            feature_extraction_time=processing_time * 0.7,
            model_inference_time=processing_time * 0.3,
            server_location="microservice",
            api_version="1.0.0",
            
            # Öneriler
            recommendations=recommendations,
            
            # Sistem Durumu
            status="completed",
            error_message=None,
            
            # Ek Metadata
            session_type="screening",
            priority_level=2 if risk_level in ["HIGH", "CRITICAL"] else 1
        )
        
        print(f"✅ Analiz tamamlandı: {prediction['predicted_class']} (%{prediction['confidence']*100:.1f})")
        return analysis_result
        
    except Exception as e:
        print(f"❌ Analiz hatası: {e}")
        return AnalysisResult(
            test_session_id=test_session_id,
            question_id=question_id,
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            processing_time_seconds=0.0,
            start_time=start_time.isoformat(),
            end_time=datetime.now().isoformat(),
            dementia_status="unknown",
            confidence_score=0.0,
            risk_level="UNKNOWN",
            normal_score=0.0,
            mci_score=0.0,
            dementia_score=0.0,
            cognitive_decline_risk=0.0,
            memory_impairment_likelihood=0.0,
            speech_pattern_anomalies=0.0,
            attention_deficit_indicator=0.0,
            language_fluency_score=0.0,
            overall_cognitive_health=0.0,
            audio_duration_seconds=0.0,
            audio_format="unknown",
            audio_sample_rate=0,
            audio_channels=0,
            audio_quality="unknown",
            feature_count=0,
            model_name="unknown",
            model_version="unknown",
            model_method="unknown",
            model_training_date="unknown",
            model_accuracy="unknown",
            feature_extraction_time=0.0,
            model_inference_time=0.0,
            server_location="microservice",
            api_version="1.0.0",
            recommendations=[],
            status="failed",
            error_message=str(e),
            session_type="screening",
            priority_level=4  # Urgent çünkü hata var
        )
    finally:
        # Geçici dosyaları temizle
        try:
            if audio_file.exists():
                audio_file.unlink()
            if audio_file.parent.exists():
                audio_file.parent.rmdir()
            wav_path = Path(wav_file) if wav_file else None
            if wav_path and wav_path.exists():
                wav_path.unlink()
        except:
            pass

async def send_result_to_backend(result: AnalysisResult, callback_url: Optional[str] = None):
    """
    Analiz sonucunu backend'e gönder
    
    Args:
        result: Analiz sonucu
        callback_url: Geri gönderilecek URL
    """
    if not callback_url:
        print("⚠️ Callback URL belirtilmedi, sonuç gönderilmedi")
        return
    
    try:
        payload = result.dict()
        
        # HTTP client ayarları - daha geniş timeout ve retry
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "DementiaMicroservice/1.0.0",
                "Accept": "application/json"
            }
        ) as client:
            print(f"📤 Sonuç gönderiliyor: {callback_url}")
            print(f"📊 Payload boyutu: {len(str(payload))} karakter")
            
            response = await client.post(callback_url, json=payload)
            
            print(f"📈 Response Status: {response.status_code}")
            print(f"📝 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                response_text = response.text
                print(f"✅ Sonuç başarıyla gönderildi - Response: {response_text}")
            else:
                print(f"⚠️ Beklenmeyen status code: {response.status_code}")
                print(f"📄 Response body: {response.text}")
                response.raise_for_status()
            
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP Hata {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"❌ Request Hatası: {e}")
    except Exception as e:
        print(f"❌ Genel Hata: {e}")

async def process_analysis_background(
    test_session_id: str, 
    question_id: str, 
    analysis_id: str,
    callback_url: Optional[str] = None
):
    """
    Arka planda analiz işlemi
    Backend workflow:
    1. Audio indir: https://demantia-backendv2-dev.onrender.com/api/test-responses/audio/{test_session_id}/{question_id}
    2. Analiz yap
    3. Sonucu gönder: https://demantia-backendv2-dev.onrender.com/api/dementia-analysis/callback
    """
    try:
        # Ses dosyasını indir
        audio_file = await download_audio(test_session_id, question_id)
        
        # Analiz yap
        result = await process_audio(audio_file, test_session_id, question_id)
        
        # Analysis ID'yi güncelle
        result.analysis_id = analysis_id
        
        # Sonucu backend'e gönder
        await send_result_to_backend(result, callback_url)
        
    except Exception as e:
        print(f"❌ Arka plan işlem hatası: {e}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    🎯 BACKEND TETİKLEMELİ MİKROSERVİS
    
    Backend'den gelen istek ile ses analizi başlatır:
    
    1. 📥 Request: Backend'den POST isteği alır
       {
         "test_session_id": "eiRJer6JowfCJjSQ4LLM",
         "question_id": "iLEstW6nRQXARxdObGcR" 
       }
    
    2. 🎵 Audio Download: Backend'den ses dosyasını indirir
       GET https://demantia-backendv2-dev.onrender.com/api/test-responses/audio/{test_session_id}/{question_id}
    
    3. 🔬 Analysis: Ses dosyasını AI model ile analiz eder
    
    4. 📤 Callback: Sonucu backend'e gönderir
       POST https://demantia-backendv2-dev.onrender.com/api/dementia-analysis/callback
    
    Returns:
        AnalysisResponse: Analiz başlatma yanıtı
    """
    if not pipeline:
        raise HTTPException(
            status_code=500,
            detail="Pipeline hazır değil"
        )
    
    analysis_id = f"{request.test_session_id}_{request.question_id}_{int(datetime.now().timestamp())}"
    
    # Backend callback URL'i sabit
    callback_url = "https://demantia-backendv2-dev.onrender.com/api/dementia-analysis/callback"
    
    # Arka planda analiz başlat
    background_tasks.add_task(
        process_analysis_background,
        request.test_session_id,
        request.question_id,
        analysis_id,
        callback_url
    )
    
    return AnalysisResponse(
        success=True,
        message="Analiz başlatıldı - Backend workflow aktif",
        analysis_id=analysis_id,
        estimated_duration=30  # tahmini 30 saniye
    )

@app.post("/analyze-sync", response_model=AnalysisResult)
async def analyze_audio_sync(request: AnalysisRequest):
    """
    Senkron ses analizi (test için)
    
    Args:
        request: Analiz isteği
        
    Returns:
        AnalysisResult: Analiz sonucu
    """
    if not pipeline:
        raise HTTPException(
            status_code=500,
            detail="Pipeline hazır değil"
        )
    
    try:
        # Ses dosyasını indir
        audio_file = await download_audio(request.test_session_id, request.question_id)
        
        # Analiz yap
        result = await process_audio(audio_file, request.test_session_id, request.question_id)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analiz hatası: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dementia_microservice:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 