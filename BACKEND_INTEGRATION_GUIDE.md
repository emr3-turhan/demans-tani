# 🔗 Backend Entegrasyon Rehberi

## 📋 Mikroservis Sonuç Formatı

Mikroservisiniz backend'inize **tam veri** gönderiyor. Backend'de istediğiniz alanları seçip mobil uygulamaya gönderebilirsiniz.

### 🚀 Mikroservisten Gelen TAM JSON Formatı

```json
{
  "test_session_id": "eiRJer6JowfCJjSQ4LLM",
  "question_id": "iLEstW6nRQXARxdObGcR",
  "analysis_id": "eiRJer6JowfCJjSQ4LLM_iLEstW6nRQXARxdObGcR_1751304599",

  // ============= ZAMAN BİLGİSİ =============
  "timestamp": "2025-06-30T20:30:01.957410",
  "processing_time_seconds": 2.058374,
  "start_time": "2025-06-30T20:29:59.899036",
  "end_time": "2025-06-30T20:30:01.957410",

  // ============= ANA SONUÇLAR =============
  "dementia_status": "normal", // "normal", "mci", "dementia"
  "confidence_score": 0.4, // 0.0 - 1.0
  "risk_level": "MEDIUM", // "LOW", "MEDIUM", "HIGH", "CRITICAL"

  // ============= DETAYLI SKORLAR =============
  "normal_score": 0.4,
  "mci_score": 0.343,
  "dementia_score": 0.257,

  // ============= KLİNİK GÖSTERGELERİ =============
  "cognitive_decline_risk": 0.6, // 0.0 - 1.0
  "memory_impairment_likelihood": 0.3428, // 0.0 - 1.0
  "speech_pattern_anomalies": 0.6, // 0.0 - 1.0
  "attention_deficit_indicator": 0.4972, // 0.0 - 1.0
  "language_fluency_score": 0.4, // 0.0 - 1.0
  "overall_cognitive_health": 40.0, // 0-100

  // ============= SES BİLGİLERİ =============
  "audio_duration_seconds": 3.413,
  "audio_format": "m4a",
  "audio_sample_rate": 22050,
  "audio_channels": 1,
  "audio_quality": "good", // "good", "short", "poor"
  "feature_count": 60,

  // ============= MODEL BİLGİLERİ =============
  "model_name": "RandomForest",
  "model_version": "1.0",
  "model_method": "trained_model", // "trained_model", "rule_based"
  "model_training_date": "2025-06-30T19:45:53",
  "model_accuracy": "95%",

  // ============= İŞLEM METADATA =============
  "feature_extraction_time": 1.4408618,
  "model_inference_time": 0.6175122,
  "server_location": "microservice",
  "api_version": "1.0.0",

  // ============= ÖNERİLER =============
  "recommendations": [
    "Sonuçlar normal aralıkta ancak takip önerilir",
    "6 ay sonra tekrar değerlendirme yapılabilir",
    "Zihinsel aktiviteleri artırın"
  ],

  // ============= SİSTEM DURUMU =============
  "status": "completed", // "completed", "failed", "processing"
  "error_message": null,

  // ============= EK METADATA =============
  "session_type": "screening", // "screening", "followup", "diagnosis"
  "priority_level": 1 // 1: Normal, 2: Medium, 3: High, 4: Urgent
}
```

---

## 🏥 Database Entity Yapısı

### JPA Entity Önerisi

```java
@Entity
@Table(name = "dementia_analysis_results")
public class DementiaAnalysisResult {

    @Id
    @Column(name = "analysis_id")
    private String analysisId;

    @Column(name = "test_session_id", nullable = false)
    private String testSessionId;

    @Column(name = "question_id", nullable = false)
    private String questionId;

    // Ana Sonuçlar
    @Enumerated(EnumType.STRING)
    @Column(name = "dementia_status")
    private DementiaStatus dementiaStatus; // NORMAL, MCI, DEMENTIA

    @Column(name = "confidence_score")
    private Double confidenceScore;

    @Enumerated(EnumType.STRING)
    @Column(name = "risk_level")
    private RiskLevel riskLevel; // LOW, MEDIUM, HIGH, CRITICAL

    // Detaylı Skorlar
    @Column(name = "normal_score")
    private Double normalScore;

    @Column(name = "mci_score")
    private Double mciScore;

    @Column(name = "dementia_score")
    private Double dementiaScore;

    // Klinik Göstergeler
    @Column(name = "cognitive_decline_risk")
    private Double cognitiveDeclineRisk;

    @Column(name = "memory_impairment_likelihood")
    private Double memoryImpairmentLikelihood;

    @Column(name = "speech_pattern_anomalies")
    private Double speechPatternAnomalies;

    @Column(name = "attention_deficit_indicator")
    private Double attentionDeficitIndicator;

    @Column(name = "language_fluency_score")
    private Double languageFluencyScore;

    @Column(name = "overall_cognitive_health")
    private Double overallCognitiveHealth;

    // Ses Bilgileri
    @Column(name = "audio_duration_seconds")
    private Double audioDurationSeconds;

    @Column(name = "audio_format")
    private String audioFormat;

    @Column(name = "audio_quality")
    private String audioQuality;

    @Column(name = "feature_count")
    private Integer featureCount;

    // Model Bilgileri
    @Column(name = "model_name")
    private String modelName;

    @Column(name = "model_accuracy")
    private String modelAccuracy;

    // İşlem Zamanları
    @Column(name = "timestamp")
    private LocalDateTime timestamp;

    @Column(name = "processing_time_seconds")
    private Double processingTimeSeconds;

    // Öneriler (JSON olarak saklanabilir)
    @Column(name = "recommendations", columnDefinition = "TEXT")
    private String recommendations; // JSON array as string

    // Sistem Durumu
    @Enumerated(EnumType.STRING)
    @Column(name = "status")
    private AnalysisStatus status; // COMPLETED, FAILED, PROCESSING

    @Column(name = "error_message")
    private String errorMessage;

    // Priority ve Type
    @Column(name = "priority_level")
    private Integer priorityLevel;

    @Column(name = "session_type")
    private String sessionType;

    // Audit fields
    @CreatedDate
    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}
```

---

## 📱 Mobil Uygulama İçin Simplified Format

Mobil uygulamaya gönderebileceğiniz **basitleştirilmiş format**:

```json
{
  "test_session_id": "eiRJer6JowfCJjSQ4LLM",
  "question_id": "iLEstW6nRQXARxdObGcR",
  "analysis_id": "eiRJer6JowfCJjSQ4LLM_iLEstW6nRQXARxdObGcR_1751304599",
  "result": {
    "status": "normal", // Kullanıcı dostu: "normal", "mci", "dementia"
    "risk_level": "MEDIUM", // "LOW", "MEDIUM", "HIGH", "CRITICAL"
    "confidence": 40, // Yüzde olarak: 0-100
    "recommendation": "Sonuçlar normal aralıkta ancak takip önerilir. 6 ay sonra tekrar değerlendirme yapılabilir.",
    "detailed_scores": {
      "normal": 40, // Yüzde olarak
      "mci": 34,
      "dementia": 26
    }
  },
  "timestamp": "2025-06-30T20:30:01",
  "success": true,
  "message": "Analiz başarıyla tamamlandı"
}
```

---

## 🔄 API Endpoints

### 1. Analiz Başlatma (Mobil → Backend → Mikroservis)

```http
POST /api/dementia/analyze/start
Content-Type: application/json

{
    "test_session_id": "eiRJer6JowfCJjSQ4LLM",
    "question_id": "iLEstW6nRQXARxdObGcR",
    "user_id": "user123"  // Optional
}
```

**Response:**

```json
{
  "success": true,
  "message": "Analiz başlatıldı",
  "analysis_id": "eiRJer6JowfCJjSQ4LLM_iLEstW6nRQXARxdObGcR_1751304599",
  "estimated_duration": 5 // seconds
}
```

### 2. Callback Endpoint (Mikroservis → Backend)

```http
POST /api/dementia/callback
Content-Type: application/json

{TAM JSON FORMATI YUKARIDA}
```

### 3. Sonuç Getirme (Mobil → Backend)

```http
GET /api/dementia/result/{testSessionId}/{questionId}
```

**Response:**

```json
{SIMPLIFIED JSON FORMATI YUKARIDA}
```

### 4. Detaylı Sonuç (Doktor Paneli)

```http
GET /api/dementia/result/detailed/{analysisId}
```

**Response:**

```json
{TAM JSON FORMATI YUKARIDA}
```

---

## 🎯 Backend Service Mantığı

### Veri Dönüştürme Örnekleri

```java
// Tam veriyi mobil için basitleştirme
public MobileAnalysisResponseDto convertToMobileResponse(DementiaAnalysisResult entity) {
    MobileAnalysisResponseDto response = new MobileAnalysisResponseDto();

    response.setTestSessionId(entity.getTestSessionId());
    response.setQuestionId(entity.getQuestionId());
    response.setAnalysisId(entity.getAnalysisId());
    response.setTimestamp(entity.getTimestamp().toString());
    response.setSuccess(entity.getStatus() == AnalysisStatus.COMPLETED);

    // Ana sonuç
    MobileAnalysisResponseDto.AnalysisResult result = new MobileAnalysisResponseDto.AnalysisResult();
    result.setStatus(entity.getDementiaStatus().name().toLowerCase());
    result.setRiskLevel(entity.getRiskLevel().name());
    result.setConfidence((int)(entity.getConfidenceScore() * 100));

    // İlk öneriyi alın
    if (entity.getRecommendations() != null && !entity.getRecommendations().isEmpty()) {
        result.setRecommendation(String.join(" ", entity.getRecommendations()));
    }

    // Skorları yüzdelere çevir
    MobileAnalysisResponseDto.AnalysisResult.DetailedScores scores =
        new MobileAnalysisResponseDto.AnalysisResult.DetailedScores();
    scores.setNormal((int)(entity.getNormalScore() * 100));
    scores.setMci((int)(entity.getMciScore() * 100));
    scores.setDementia((int)(entity.getDementiaScore() * 100));

    result.setDetailedScores(scores);
    response.setResult(result);

    return response;
}
```

---

## 📊 Field Seçimi Önerileri

### 🏥 **Doktor Paneli için gösterilecek alanlar:**

- Tüm klinik göstergeler
- Model accuracy ve training bilgileri
- Audio quality ve feature count
- Processing times
- Full recommendations list
- Priority level

### 📱 **Mobil Uygulama için gösterilecek alanlar:**

- dementia_status (simplified)
- confidence_score (percentage)
- risk_level
- Main recommendation (first one)
- detailed_scores (for charts)
- timestamp

### 📈 **Analytics için kullanılacak alanlar:**

- processing_time_seconds
- audio_duration_seconds
- model_name ve accuracy
- feature_extraction_time
- server_location

---

## 🚀 Hızlı Başlangıç

1. **Database tablosunu oluşturun** (yukarıdaki entity ile)
2. **Callback endpoint**'i oluşturun (mikroservisten veri alacak)
3. **Mobil API**'yi oluşturun (simplified data dönecek)
4. **Mikroservise** callback URL'i verin
5. **Test edin**

Bu yapı ile backend'e **her şey geliyor**, mobil uygulamaya **sadece gerekli olanlar** gidiyor! 🎯
