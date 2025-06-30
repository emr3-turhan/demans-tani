# 🚀 Render.com Deployment Rehberi

## 📋 Ön Gereksinimler

✅ **GitHub Repository:** Kod GitHub'da olmalı  
✅ **Render.com Hesabı:** [render.com](https://render.com) hesabı oluşturun  
✅ **Docker Container:** Dockerfile hazır  
✅ **AI Model:** `full_synthetic_dataset/` klasörü mevcut

---

## 🐳 1. Docker Test (Yerel)

Önce yerel olarak container'ı test edelim:

### **Docker Build:**

```bash
# Container'ı build et
docker build -t dementia-microservice .

# Container'ı çalıştır
docker run -p 8000:8000 dementia-microservice

# Test et
curl http://localhost:8000/health
```

### **Başarılı Test Çıktısı:**

```json
{
  "status": "healthy",
  "timestamp": "2025-06-30T21:45:00.000000",
  "pipeline_ready": true
}
```

---

## 🌐 2. Render.com Deployment

### **Adım 1: Repository Hazırlığı**

1. **GitHub'a Push:** Tüm dosyaları GitHub repository'nize push edin
2. **Branch:** `main` branch'i kullanın
3. **Dosya Kontrolü:** Şu dosyaların olduğundan emin olun:
   ```
   ├── Dockerfile ✅
   ├── .dockerignore ✅
   ├── requirements.txt ✅
   ├── dementia_microservice.py ✅
   ├── full_synthetic_dataset/ ✅
   │   └── trained_models/
   │       └── best_model_randomforest.pkl ✅
   └── my_config.json ✅
   ```

### **Adım 2: Render.com'da Service Oluşturma**

1. **Render Dashboard'a Git:** https://dashboard.render.com/
2. **New +** butonuna tıkla
3. **Web Service** seç
4. **Build and deploy from a Git repository** seç

### **Adım 3: Repository Bağlantısı**

1. **GitHub hesabınızı bağlayın**
2. **Repository seçin:** `demans-tani` repository'sini seç
3. **Branch:** `main` branch'ini seç

### **Adım 4: Service Konfigürasyonu**

#### **Basic Settings:**

```
Name: dementia-microservice
Region: Frankfurt (EU Central) [Türkiye'ye en yakın]
Branch: main
Runtime: Docker
```

#### **Build Settings:**

```
Build Command: [Boş bırak - Docker kullanıyor]
Start Command: [Boş bırak - Dockerfile'da tanımlı]
```

#### **Advanced Settings:**

```
Port: 8000
Health Check Path: /health
```

### **Adım 5: Plan Seçimi**

- **Starter Plan:** $7/month - 512MB RAM, 0.1 CPU
- **Professional Plan:** $25/month - 2GB RAM, 1 CPU (Önerilen)

### **Adım 6: Environment Variables**

Environment Variables bölümünde şunları ekleyin:

```
PYTHON_VERSION=3.12
PORT=8000
```

### **Adım 7: Deploy**

**Create Web Service** butonuna tıklayın!

---

## 📊 3. Deployment İzleme

### **Build Logs:**

Deploy sırasında şu adımları göreceksiniz:

```
==> Cloning from https://github.com/YOUR_USERNAME/demans-tani...
==> Using Dockerfile to build image
==> Step 1/15 : FROM python:3.12-slim
==> Step 2/15 : LABEL name="dementia-microservice"...
...
==> Building the image completed successfully
==> Pushing the image completed successfully
==> Starting service with 'uvicorn dementia_microservice:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info'
```

### **Başarılı Deploy Log'ları:**

```
📋 Varsayılan konfigürasyon kullanılıyor
✅ Eğitilmiş model yüklendi: RandomForest
📅 Eğitim tarihi: 2025-06-30T19:45:53.986608
🏷️ Sınıflar: ['dementia', 'mci', 'normal']
🚀 Demans Tespit Pipeline Başlatıldı
✅ Pipeline başlatıldı
🚀 Demans Analizi Mikroservisi başlatıldı
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 🧪 4. Production Test

Deploy tamamlandığında Render size bir URL verecek:

```
https://dementia-microservice.onrender.com
```

### **Test Komutları:**

#### **1. Health Check:**

```bash
curl https://dementia-microservice.onrender.com/health
```

#### **2. Ana Sayfa:**

```bash
curl https://dementia-microservice.onrender.com/
```

#### **3. Analiz Tetikleme:**

```bash
curl -X POST "https://dementia-microservice.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "test_session_id": "eiRJer6JowfCJjSQ4LLM",
    "question_id": "iLEstW6nRQXARxdObGcR"
  }'
```

#### **4. API Docs:**

```
https://dementia-microservice.onrender.com/docs
```

---

## ⚙️ 5. Backend Entegrasyonu

Deploy edildikten sonra Spring Boot backend'inizde mikroservis URL'ini güncelleyin:

### **application.properties:**

```properties
# Mikroservis URL (Production)
microservice.url=https://dementia-microservice.onrender.com
microservice.analyze.endpoint=${microservice.url}/analyze
microservice.health.endpoint=${microservice.url}/health
```

### **Backend Service Update:**

```java
@Service
public class DementiaAnalysisService {

    @Value("${microservice.url}")
    private String microserviceUrl;

    public ResponseEntity<?> triggerAnalysis(String testSessionId, String questionId) {
        String analyzeUrl = microserviceUrl + "/analyze";

        Map<String, String> payload = Map.of(
            "test_session_id", testSessionId,
            "question_id", questionId
        );

        return restTemplate.postForEntity(analyzeUrl, payload, Map.class);
    }
}
```

---

## 🔧 6. Troubleshooting

### **Build Hatalar:**

#### **"Model dosyası bulunamadı":**

```bash
# Dosya varlığını kontrol et
ls -la full_synthetic_dataset/trained_models/
```

#### **"Memory limit exceeded":**

- Render plan'ınızı Professional'a yükseltin
- Model dosyasının boyutunu kontrol edin

#### **"Port already in use":**

- Render otomatik port atar, Dockerfile'da 8000 doğru

### **Runtime Hatalar:**

#### **Health check fail:**

```bash
# Health endpoint'i test et
curl https://YOUR_APP.onrender.com/health
```

#### **AI model yüklenmedi:**

- Build log'larında model yükleme mesajlarını kontrol edin
- `full_synthetic_dataset/` klasörünün kopyalandığından emin olun

### **Performance Issues:**

- **Cold start:** İlk istek 30+ saniye sürebilir
- **Memory:** Professional plan önerilir (2GB RAM)
- **CPU:** AI model inference için yeterli CPU gerekli

---

## 📈 7. Monitoring & Scaling

### **Render Dashboard:**

- **Metrics:** CPU, Memory, Request count
- **Logs:** Real-time application logs
- **Health:** Automated health checks

### **Custom Monitoring:**

```bash
# Sürekli health check
watch -n 30 curl -s https://YOUR_APP.onrender.com/health
```

### **Auto-scaling:**

Render Professional plan ile auto-scaling mevcut

---

## 🎯 8. Final Checklist

Deploy öncesi kontrol listesi:

- [ ] ✅ GitHub repository güncel
- [ ] ✅ Dockerfile syntax doğru
- [ ] ✅ Model dosyaları mevcut (~100MB)
- [ ] ✅ requirements.txt güncel
- [ ] ✅ Health endpoint çalışıyor
- [ ] ✅ Backend callback URL güncellendi
- [ ] ✅ Render plan seçimi yapıldı
- [ ] ✅ Environment variables ayarlandı

Deploy sonrası kontrol listesi:

- [ ] ✅ Health check başarılı
- [ ] ✅ API docs erişilebilir
- [ ] ✅ Analiz tetikleme testi geçti
- [ ] ✅ Backend callback çalışıyor
- [ ] ✅ Performance kabul edilebilir (~5-30s)

---

## 🎉 Deploy Başarılı!

Mikroservisiniz artık production'da çalışıyor:

```
🌐 URL: https://dementia-microservice.onrender.com
📚 Docs: https://dementia-microservice.onrender.com/docs
❤️ Health: https://dementia-microservice.onrender.com/health
🚀 Analyze: POST /analyze
```

**Backend integration için hazır!** 🎯
