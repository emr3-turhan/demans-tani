# 🧠 Dementia Detection Microservice

[![🐳 Docker Build & Publish](https://github.com/emr3turhan/demans-tani/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/emr3turhan/demans-tani/actions/workflows/docker-publish.yml)
[![Docker Hub](https://img.shields.io/docker/pulls/emr3turhan/dementia-microservice)](https://hub.docker.com/r/emr3turhan/dementia-microservice)

AI-powered microservice for dementia detection from speech audio analysis, designed for Spring Boot backend integration.

## 🚀 Quick Start

### Using Docker Hub

```bash
# Pull and run from Docker Hub
docker pull emr3turhan/dementia-microservice:latest
docker run -d -p 8000:8000 emr3turhan/dementia-microservice:latest

# Test the service
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Using Render.com

The service is configured for one-click deployment on Render.com using the provided `render.yaml` configuration.

## 📡 API Endpoints

- `GET /` - Service information and available endpoints
- `GET /health` - Health check with pipeline status
- `POST /analyze` - Asynchronous analysis (backend-triggered)
- `POST /analyze-sync` - Synchronous analysis (for testing)
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc API documentation

## 🔧 Development

### Local Development

```bash
# Clone the repository
git clone https://github.com/emr3turhan/demans-tani.git
cd demans-tani

# Install dependencies
pip install -r requirements.txt

# Run the microservice
python dementia_microservice.py
```

### Docker Development

```bash
# Build locally
docker build -t dementia-microservice:local .

# Run locally
docker run -p 8000:8000 dementia-microservice:local
```

## 🏗️ CI/CD Pipeline

This project uses GitHub Actions for automated:

- ✅ Docker image building
- ✅ Multi-platform support (linux/amd64, linux/arm64)
- ✅ Docker Hub publishing
- ✅ Automated testing
- ✅ Container health verification

### Setting up GitHub Secrets

To enable Docker Hub publishing, add these secrets to your GitHub repository:

- `DOCKER_USERNAME` - Your Docker Hub username
- `DOCKER_PASSWORD` - Your Docker Hub access token

## 🔬 AI Model

- **Model Type**: RandomForest Classifier
- **Features**: 60 audio features (MFCC, spectral, pitch, temporal, rhythm)
- **Classes**: Normal, MCI (Mild Cognitive Impairment), Dementia
- **Accuracy**: 95%
- **Processing Time**: ~2-3 seconds per audio file

## 📊 Integration

### Backend Integration

```json
POST /analyze
{
  "test_session_id": "eiRJer6JowfCJjSQ4LLM",
  "question_id": "iLEstW6nRQXARxdObGcR"
}
```

### Response Format

```json
{
  "test_session_id": "eiRJer6JowfCJjSQ4LLM",
  "question_id": "iLEstW6nRQXARxdObGcR",
  "dementia_status": "normal",
  "confidence_score": 0.4,
  "risk_level": "MEDIUM",
  "normal_score": 0.4,
  "mci_score": 0.343,
  "dementia_score": 0.257,
  "processing_time_seconds": 2.058374,
  "recommendations": ["Sonuçlar normal aralıkta ancak takip önerilir"]
}
```

## 🔐 Security & Production

- ✅ Non-root user execution
- ✅ Health checks configured
- ✅ Rate limiting support
- ✅ CORS configuration
- ✅ Error handling and logging
- ✅ Temporary file cleanup

## 📚 Documentation

- [API Guide](API_GUIDE.md) - Complete API documentation
- [Backend Integration Guide](BACKEND_INTEGRATION_GUIDE.md) - Spring Boot integration
- [Render Deployment Guide](RENDER_DEPLOYMENT_GUIDE.md) - Deployment instructions
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md) - Production deployment checklist

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Medical Disclaimer

This tool is for research and screening purposes only. It should not be used as a sole diagnostic tool for medical conditions. Always consult with qualified healthcare professionals for proper medical diagnosis and treatment.
