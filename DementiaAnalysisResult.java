package com.dementia.entity;

import jakarta.persistence.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "dementia_analysis_results")
@EntityListeners(AuditingEntityListener.class)
public class DementiaAnalysisResult {

    // ============= PRIMARY IDENTIFIERS =============
    @Id
    @Column(name = "analysis_id", length = 150)
    private String analysisId;

    @Column(name = "test_session_id", nullable = false, length = 50)
    private String testSessionId;

    @Column(name = "question_id", nullable = false, length = 50)
    private String questionId;

    // ============= TIMING INFORMATION =============
    @Column(name = "timestamp", nullable = false)
    private LocalDateTime timestamp;

    @Column(name = "processing_time_seconds")
    private Double processingTimeSeconds;

    @Column(name = "start_time")
    private LocalDateTime startTime;

    @Column(name = "end_time")
    private LocalDateTime endTime;

    // ============= MAIN ANALYSIS RESULTS =============
    @Enumerated(EnumType.STRING)
    @Column(name = "dementia_status", nullable = false)
    private DementiaStatus dementiaStatus; // NORMAL, MCI, DEMENTIA

    @Column(name = "confidence_score", nullable = false)
    private Double confidenceScore; // 0.0 - 1.0

    @Enumerated(EnumType.STRING)
    @Column(name = "risk_level", nullable = false)
    private RiskLevel riskLevel; // LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN

    // ============= DETAILED SCORES =============
    @Column(name = "normal_score")
    private Double normalScore;

    @Column(name = "mci_score")
    private Double mciScore;

    @Column(name = "dementia_score")
    private Double dementiaScore;

    // ============= CLINICAL INDICATORS =============
    @Column(name = "cognitive_decline_risk")
    private Double cognitiveDeclineRisk; // 0.0 - 1.0

    @Column(name = "memory_impairment_likelihood")
    private Double memoryImpairmentLikelihood; // 0.0 - 1.0

    @Column(name = "speech_pattern_anomalies")
    private Double speechPatternAnomalies; // 0.0 - 1.0

    @Column(name = "attention_deficit_indicator")
    private Double attentionDeficitIndicator; // 0.0 - 1.0

    @Column(name = "language_fluency_score")
    private Double languageFluencyScore; // 0.0 - 1.0

    @Column(name = "overall_cognitive_health")
    private Double overallCognitiveHealth; // 0-100

    // ============= AUDIO INFORMATION =============
    @Column(name = "audio_duration_seconds")
    private Double audioDurationSeconds;

    @Column(name = "audio_format", length = 10)
    private String audioFormat; // m4a, wav, mp3

    @Column(name = "audio_sample_rate")
    private Integer audioSampleRate; // 22050, 44100

    @Column(name = "audio_channels")
    private Integer audioChannels; // 1, 2

    @Column(name = "audio_quality", length = 20)
    private String audioQuality; // good, short, poor

    @Column(name = "feature_count")
    private Integer featureCount; // Number of extracted features

    // ============= MODEL INFORMATION =============
    @Column(name = "model_name", length = 50)
    private String modelName; // RandomForest, SVM, etc.

    @Column(name = "model_version", length = 20)
    private String modelVersion; // 1.0, 2.0

    @Column(name = "model_method", length = 30)
    private String modelMethod; // trained_model, rule_based

    @Column(name = "model_training_date")
    private LocalDateTime modelTrainingDate;

    @Column(name = "model_accuracy", length = 10)
    private String modelAccuracy; // 95%, 87%

    // ============= PROCESSING METADATA =============
    @Column(name = "feature_extraction_time")
    private Double featureExtractionTime;

    @Column(name = "model_inference_time")
    private Double modelInferenceTime;

    @Column(name = "server_location", length = 50)
    private String serverLocation; // microservice, cloud, local

    @Column(name = "api_version", length = 10)
    private String apiVersion; // 1.0.0, 2.1.3

    // ============= RECOMMENDATIONS =============
    @ElementCollection
    @CollectionTable(
        name = "dementia_recommendations", 
        joinColumns = @JoinColumn(name = "analysis_id")
    )
    @Column(name = "recommendation", length = 500)
    private List<String> recommendations;

    // ============= SYSTEM STATUS =============
    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    private AnalysisStatus status; // COMPLETED, FAILED, PROCESSING, PENDING

    @Column(name = "error_message", length = 1000)
    private String errorMessage;

    // ============= ADDITIONAL METADATA =============
    @Column(name = "user_id", length = 50)
    private String userId; // Optional: kullanıcı ID'si

    @Column(name = "session_type", length = 30)
    private String sessionType; // screening, followup, diagnosis

    @Column(name = "patient_age")
    private Integer patientAge;

    @Column(name = "patient_gender", length = 10)
    private String patientGender; // male, female, other

    @Column(name = "notes", length = 2000)
    private String notes; // Doktor notları

    @Column(name = "is_reviewed")
    private Boolean isReviewed = false; // Doktor tarafından incelendi mi?

    @Column(name = "reviewed_by", length = 50)
    private String reviewedBy; // İnceleyen doktor

    @Column(name = "reviewed_at")
    private LocalDateTime reviewedAt;

    @Column(name = "priority_level")
    private Integer priorityLevel; // 1: Normal, 2: Medium, 3: High, 4: Urgent

    // ============= AUDIT FIELDS =============
    @CreatedDate
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Column(name = "created_by", length = 50)
    private String createdBy;

    @Column(name = "updated_by", length = 50)
    private String updatedBy;

    // ============= ENUMS =============
    public enum DementiaStatus {
        NORMAL("Normal"),
        MCI("Mild Cognitive Impairment"),
        DEMENTIA("Dementia"),
        UNKNOWN("Unknown");

        private final String description;

        DementiaStatus(String description) {
            this.description = description;
        }

        public String getDescription() {
            return description;
        }
    }

    public enum RiskLevel {
        LOW("Düşük Risk"),
        MEDIUM("Orta Risk"),
        HIGH("Yüksek Risk"),
        CRITICAL("Kritik Risk"),
        UNKNOWN("Bilinmeyen");

        private final String description;

        RiskLevel(String description) {
            this.description = description;
        }

        public String getDescription() {
            return description;
        }
    }

    public enum AnalysisStatus {
        PENDING("Beklemede"),
        PROCESSING("İşleniyor"),
        COMPLETED("Tamamlandı"),
        FAILED("Başarısız"),
        REVIEWING("İnceleniyor"),
        APPROVED("Onaylandı");

        private final String description;

        AnalysisStatus(String description) {
            this.description = description;
        }

        public String getDescription() {
            return description;
        }
    }

    // ============= CONSTRUCTORS =============
    public DementiaAnalysisResult() {}

    public DementiaAnalysisResult(String analysisId, String testSessionId, String questionId) {
        this.analysisId = analysisId;
        this.testSessionId = testSessionId;
        this.questionId = questionId;
        this.timestamp = LocalDateTime.now();
        this.status = AnalysisStatus.PENDING;
    }

    // ============= GETTERS AND SETTERS =============
    public String getAnalysisId() { return analysisId; }
    public void setAnalysisId(String analysisId) { this.analysisId = analysisId; }

    public String getTestSessionId() { return testSessionId; }
    public void setTestSessionId(String testSessionId) { this.testSessionId = testSessionId; }

    public String getQuestionId() { return questionId; }
    public void setQuestionId(String questionId) { this.questionId = questionId; }

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

    public Double getProcessingTimeSeconds() { return processingTimeSeconds; }
    public void setProcessingTimeSeconds(Double processingTimeSeconds) { this.processingTimeSeconds = processingTimeSeconds; }

    public LocalDateTime getStartTime() { return startTime; }
    public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }

    public LocalDateTime getEndTime() { return endTime; }
    public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }

    public DementiaStatus getDementiaStatus() { return dementiaStatus; }
    public void setDementiaStatus(DementiaStatus dementiaStatus) { this.dementiaStatus = dementiaStatus; }

    public Double getConfidenceScore() { return confidenceScore; }
    public void setConfidenceScore(Double confidenceScore) { this.confidenceScore = confidenceScore; }

    public RiskLevel getRiskLevel() { return riskLevel; }
    public void setRiskLevel(RiskLevel riskLevel) { this.riskLevel = riskLevel; }

    public Double getNormalScore() { return normalScore; }
    public void setNormalScore(Double normalScore) { this.normalScore = normalScore; }

    public Double getMciScore() { return mciScore; }
    public void setMciScore(Double mciScore) { this.mciScore = mciScore; }

    public Double getDementiaScore() { return dementiaScore; }
    public void setDementiaScore(Double dementiaScore) { this.dementiaScore = dementiaScore; }

    public Double getCognitiveDeclineRisk() { return cognitiveDeclineRisk; }
    public void setCognitiveDeclineRisk(Double cognitiveDeclineRisk) { this.cognitiveDeclineRisk = cognitiveDeclineRisk; }

    public Double getMemoryImpairmentLikelihood() { return memoryImpairmentLikelihood; }
    public void setMemoryImpairmentLikelihood(Double memoryImpairmentLikelihood) { this.memoryImpairmentLikelihood = memoryImpairmentLikelihood; }

    public Double getSpeechPatternAnomalies() { return speechPatternAnomalies; }
    public void setSpeechPatternAnomalies(Double speechPatternAnomalies) { this.speechPatternAnomalies = speechPatternAnomalies; }

    public Double getAttentionDeficitIndicator() { return attentionDeficitIndicator; }
    public void setAttentionDeficitIndicator(Double attentionDeficitIndicator) { this.attentionDeficitIndicator = attentionDeficitIndicator; }

    public Double getLanguageFluencyScore() { return languageFluencyScore; }
    public void setLanguageFluencyScore(Double languageFluencyScore) { this.languageFluencyScore = languageFluencyScore; }

    public Double getOverallCognitiveHealth() { return overallCognitiveHealth; }
    public void setOverallCognitiveHealth(Double overallCognitiveHealth) { this.overallCognitiveHealth = overallCognitiveHealth; }

    public Double getAudioDurationSeconds() { return audioDurationSeconds; }
    public void setAudioDurationSeconds(Double audioDurationSeconds) { this.audioDurationSeconds = audioDurationSeconds; }

    public String getAudioFormat() { return audioFormat; }
    public void setAudioFormat(String audioFormat) { this.audioFormat = audioFormat; }

    public Integer getAudioSampleRate() { return audioSampleRate; }
    public void setAudioSampleRate(Integer audioSampleRate) { this.audioSampleRate = audioSampleRate; }

    public Integer getAudioChannels() { return audioChannels; }
    public void setAudioChannels(Integer audioChannels) { this.audioChannels = audioChannels; }

    public String getAudioQuality() { return audioQuality; }
    public void setAudioQuality(String audioQuality) { this.audioQuality = audioQuality; }

    public Integer getFeatureCount() { return featureCount; }
    public void setFeatureCount(Integer featureCount) { this.featureCount = featureCount; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public String getModelMethod() { return modelMethod; }
    public void setModelMethod(String modelMethod) { this.modelMethod = modelMethod; }

    public LocalDateTime getModelTrainingDate() { return modelTrainingDate; }
    public void setModelTrainingDate(LocalDateTime modelTrainingDate) { this.modelTrainingDate = modelTrainingDate; }

    public String getModelAccuracy() { return modelAccuracy; }
    public void setModelAccuracy(String modelAccuracy) { this.modelAccuracy = modelAccuracy; }

    public Double getFeatureExtractionTime() { return featureExtractionTime; }
    public void setFeatureExtractionTime(Double featureExtractionTime) { this.featureExtractionTime = featureExtractionTime; }

    public Double getModelInferenceTime() { return modelInferenceTime; }
    public void setModelInferenceTime(Double modelInferenceTime) { this.modelInferenceTime = modelInferenceTime; }

    public String getServerLocation() { return serverLocation; }
    public void setServerLocation(String serverLocation) { this.serverLocation = serverLocation; }

    public String getApiVersion() { return apiVersion; }
    public void setApiVersion(String apiVersion) { this.apiVersion = apiVersion; }

    public List<String> getRecommendations() { return recommendations; }
    public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }

    public AnalysisStatus getStatus() { return status; }
    public void setStatus(AnalysisStatus status) { this.status = status; }

    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }

    public String getSessionType() { return sessionType; }
    public void setSessionType(String sessionType) { this.sessionType = sessionType; }

    public Integer getPatientAge() { return patientAge; }
    public void setPatientAge(Integer patientAge) { this.patientAge = patientAge; }

    public String getPatientGender() { return patientGender; }
    public void setPatientGender(String patientGender) { this.patientGender = patientGender; }

    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }

    public Boolean getIsReviewed() { return isReviewed; }
    public void setIsReviewed(Boolean isReviewed) { this.isReviewed = isReviewed; }

    public String getReviewedBy() { return reviewedBy; }
    public void setReviewedBy(String reviewedBy) { this.reviewedBy = reviewedBy; }

    public LocalDateTime getReviewedAt() { return reviewedAt; }
    public void setReviewedAt(LocalDateTime reviewedAt) { this.reviewedAt = reviewedAt; }

    public Integer getPriorityLevel() { return priorityLevel; }
    public void setPriorityLevel(Integer priorityLevel) { this.priorityLevel = priorityLevel; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }

    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }

    // ============= UTILITY METHODS =============
    public boolean isHighRisk() {
        return this.riskLevel == RiskLevel.HIGH || this.riskLevel == RiskLevel.CRITICAL;
    }

    public boolean needsReview() {
        return isHighRisk() && !isReviewed;
    }

    public String getFormattedConfidence() {
        return String.format("%.1f%%", confidenceScore * 100);
    }

    public String getFormattedDuration() {
        if (audioDurationSeconds == null) return "N/A";
        return String.format("%.1f sn", audioDurationSeconds);
    }

    @Override
    public String toString() {
        return String.format("DementiaAnalysisResult{id='%s', session='%s', status='%s', risk='%s', confidence='%.2f'}", 
                analysisId, testSessionId, dementiaStatus, riskLevel, confidenceScore);
    }
} 