# 🚀 Dementia Analysis Microservice - Production Image
FROM python:3.11-slim

# 📋 Metadata
LABEL name="dementia-microservice" \
      version="1.0.0" \
      description="AI-powered dementia detection microservice for Render.com deployment"

# 🔧 System packages + cleanup in single layer
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 👤 Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 📁 Working directory
WORKDIR /app

# 📦 Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# 🐍 Install Python dependencies with more verbose output
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --verbose -r requirements.txt

# 📂 Copy application files (numba-free version)
COPY dementia_microservice.py .
COPY dementia_detection_pipeline.py .
COPY feature_extraction.py .
COPY feature_extraction_lite.py .
COPY audio_converter.py .
COPY my_config.json .

# 🤖 Copy model and dataset (required for AI functionality)
COPY full_synthetic_dataset/ ./full_synthetic_dataset/

# ✅ Verify critical files exist
RUN python -c "\
import os, sys; \
required_files = [ \
    'full_synthetic_dataset/trained_models/best_model_randomforest.pkl', \
    'dementia_microservice.py', \
    'dementia_detection_pipeline.py', \
    'feature_extraction_lite.py' \
]; \
missing = [f for f in required_files if not os.path.exists(f)]; \
print('Missing files:', missing) if missing else print('All required files present'); \
sys.exit(1) if missing else None \
"

# 🔒 Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# 🌐 Expose port for Render.com
EXPOSE 8000

# ❤️ Health check for monitoring
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# 🚀 Production-ready startup command with dynamic port for Render.com
CMD ["sh", "-c", "uvicorn dementia_microservice:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info"] 