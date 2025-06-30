# Demans Analizi Mikroservisi API Kılavuzu

## 🚀 Genel Bakış

Bu mikroservis, ses dosyalarını analiz ederek demans risk değerlendirmesi yapar. Spring Boot backend'iniz ile entegre çalışır.

## 🔗 API Endpoints

### Base URL

```
http://localhost:8000
```

### 1. Health Check

**Endpoint:** `GET /health`

**Açıklama:** Servis sağlık durumunu kontrol eder.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-06-30T20:08:31.526185",
  "pipeline_ready": true
}
```

### 2. Ana Bilgi

**Endpoint:** `GET /`

**Açıklama:** Servis bilgilerini ve mevcut endpoint'leri listeler.

**Response:**

```json
{
  "service": "Demans Analizi Mikroservisi",
  "version": "1.0.0",
  "status": "active",
  "endpoints": {
    "analyze": "/analyze",
    "status": "/status/{analysis_id}",
    "health": "/health"
  }
}
```

### 3. Asenkron Analiz (Ana Endpoint)

**Endpoint:** `POST /analyze`

**Açıklama:** Analizi arka planda başlatır ve hemen sonuç döner.

**Request Body:**

```json
{
  "test_session_id": "eiRJer6JowfCJjSQ4LLM",
  "question_id": "iLEstW6nRQXARxdObGcR",
  "callback_url": "https://your-backend.com/api/dementia/results"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Analiz başlatıldı",
  "analysis_id": "eiRJer6JowfCJjSQ4LLM_iLEstW6nRQXARxdObGcR_1751303507",
  "estimated_duration": 30
}
```

**Callback Payload (Backend'e gönderilen):**

```json
{
  "test_session_id": "eiRJer6JowfCJjSQ4LLM",
  "question_id": "iLEstW6nRQXARxdObGcR",
  "analysis_id": "eiRJer6JowfCJjSQ4LLM_iLEstW6nRQXARxdObGcR_1751303409",
  "timestamp": "2025-06-30T20:10:12.023185",
  "predicted_class": "normal",
  "confidence": 0.4,
  "class_probabilities": {
    "dementia": 0.257,
    "mci": 0.343,
    "normal": 0.4
  },
  "model_name": "RandomForest",
  "processing_time": 2.108846,
  "audio_duration": 3.413,
  "feature_count": 60
}
```

### 4. Senkron Analiz (Test İçin)

**Endpoint:** `POST /analyze-sync`

**Açıklama:** Analizi yapar ve sonucu direkt döner. Test amaçlı.

**Request Body:**

```json
{
  "test_session_id": "eiRJer6JowfCJjSQ4LLM",
  "question_id": "iLEstW6nRQXARxdObGcR"
}
```

**Response:** Yukarıdaki callback payload ile aynı format.

## 🔄 İş Akışı

1. **Mobil uygulama** → **Spring Boot backend'e** ses kaydı gönderir
2. **Spring Boot backend** → **Mikroservise** `POST /analyze` isteği gönderir
3. **Mikroservis** → Spring Boot backend'den ses dosyasını indirir
4. **Mikroservis** → Ses analizini yapar
5. **Mikroservis** → Sonucu callback URL'e gönderir
6. **Spring Boot backend** → Sonucu mobil uygulamaya iletir

## 📝 Entegrasyon Örnekleri

### Java (Spring Boot) Client

```java
@Service
public class DementiaAnalysisService {

    @Value("${dementia.microservice.url}")
    private String microserviceUrl;

    public void triggerAnalysis(String testSessionId, String questionId) {
        AnalysisRequest request = new AnalysisRequest();
        request.setTestSessionId(testSessionId);
        request.setQuestionId(questionId);
        request.setCallbackUrl("https://your-domain.com/api/dementia/callback");

        RestTemplate restTemplate = new RestTemplate();
        AnalysisResponse response = restTemplate.postForObject(
            microserviceUrl + "/analyze",
            request,
            AnalysisResponse.class
        );

        log.info("Analysis started: {}", response.getAnalysisId());
    }

    @PostMapping("/api/dementia/callback")
    public ResponseEntity<Void> receiveAnalysisResult(@RequestBody AnalysisResult result) {
        // Sonucu veritabanına kaydet
        analysisResultRepository.save(result);

        // Mobil uygulamaya bildirim gönder
        notificationService.sendToMobile(result.getTestSessionId(), result);

        return ResponseEntity.ok().build();
    }
}
```

### cURL Örnekleri

```bash
# Health check
curl http://localhost:8000/health

# Asenkron analiz başlat
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "test_session_id": "test123",
    "question_id": "q456",
    "callback_url": "https://httpbin.org/post"
  }'

# Senkron test
curl -X POST "http://localhost:8000/analyze-sync" \
  -H "Content-Type: application/json" \
  -d '{
    "test_session_id": "test123",
    "question_id": "q456"
  }'
```

### Python Client

```python
import requests
import asyncio

class DementiaClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    async def analyze_async(self, test_session_id, question_id, callback_url=None):
        payload = {
            "test_session_id": test_session_id,
            "question_id": question_id,
            "callback_url": callback_url
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/analyze", json=payload)
            return response.json()

    async def analyze_sync(self, test_session_id, question_id):
        payload = {
            "test_session_id": test_session_id,
            "question_id": question_id
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/analyze-sync", json=payload)
            return response.json()

# Kullanım
client = DementiaClient()
result = await client.analyze_sync("test123", "q456")
print(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
```

## 🐳 Deployment

### Development

```bash
./start_microservice.sh dev
```

### Production

```bash
./start_microservice.sh prod
```

### Docker

```bash
./start_microservice.sh docker
```

### Docker Compose (Nginx ile)

```bash
./start_microservice.sh compose
```

## 🔧 Konfigürasyon

### Environment Variables

- `PYTHONPATH`: Python modül yolu
- `PYTHONUNBUFFERED`: Log buffering kapatma
- `MODEL_PATH`: Model dosyası yolu (opsiyonel)
- `DATASET_PATH`: Dataset yolu (opsiyonel)

### Model Dosyası

Mikroservis şu model dosyasını arar:

```
full_synthetic_dataset/trained_models/best_model_randomforest.pkl
```

## 🚨 Hata Yönetimi

### Yaygın Hatalar

1. **Model bulunamadı (500)**

   ```json
   { "detail": "Pipeline hazır değil" }
   ```

   Çözüm: Model dosyasının varlığını kontrol edin.

2. **Ses dosyası indirilemedi (400)**

   ```json
   { "detail": "Ses dosyası indirilemedi: HTTP 404" }
   ```

   Çözüm: test_session_id ve question_id'yi kontrol edin.

3. **Analiz hatası (500)**
   ```json
   { "detail": "Analiz hatası: Feature extraction failed" }
   ```
   Çözüm: Ses dosyası formatını kontrol edin.

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Logs

```bash
docker logs -f dementia-api
```

### Metrics (Gelecekte)

- Analiz süreleri
- Hata oranları
- API kullanım istatistikleri

## 🔐 Güvenlik

- Rate limiting (nginx ile)
- CORS support
- Input validation
- Error sanitization

## 📈 Performans

- **Ortalama Analiz Süresi**: 2-5 saniye
- **Maksimum Dosya Boyutu**: 50MB
- **Eş Zamanlı İşlem**: 2 worker (production)
- **Memory Kullanımı**: ~1-2GB

## 🔗 API Dokümantasyonu

Mikroservis çalışırken:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 💡 İpuçları

1. **Callback URL** mutlaka HTTPS olmalı (production'da)
2. **Timeout** değerlerini 60 saniye olarak ayarlayın
3. **Retry logic** ekleyin (ağ hataları için)
4. **Rate limiting** kullanın
5. **Log monitoring** ekleyin
