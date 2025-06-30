#!/bin/bash

# Demans Analizi Mikroservisi Başlatma Scripti

echo "🚀 Demans Analizi Mikroservisi Başlatılıyor..."

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Environment kontrolü
check_requirements() {
    echo -e "${BLUE}📋 Gereksinimler kontrol ediliyor...${NC}"
    
    # Python kontrolü
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 bulunamadı${NC}"
        exit 1
    fi
    
    # Model dosyası kontrolü
    if [ ! -f "full_synthetic_dataset/trained_models/best_model_randomforest.pkl" ]; then
        echo -e "${RED}❌ Model dosyası bulunamadı${NC}"
        echo "Lütfen önce modeli eğitin: python train_synthetic_model.py"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Gereksinimler tamam${NC}"
}

# Bağımlılık yüklemesi
install_dependencies() {
    echo -e "${BLUE}📦 Bağımlılıklar yükleniyor...${NC}"
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo -e "${GREEN}✅ Bağımlılıklar yüklendi${NC}"
    else
        echo -e "${YELLOW}⚠️ requirements.txt bulunamadı${NC}"
    fi
}

# Mikroservisi başlat
start_microservice() {
    echo -e "${BLUE}🚀 Mikroservis başlatılıyor...${NC}"
    echo -e "${YELLOW}Port: 8000${NC}"
    echo -e "${YELLOW}Swagger UI: http://localhost:8000/docs${NC}"
    echo -e "${YELLOW}API Docs: http://localhost:8000/redoc${NC}"
    echo ""
    
    # Uvicorn ile başlat
    python -m uvicorn dementia_microservice:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info
}

# Development modunda başlat
start_dev() {
    echo -e "${BLUE}🔧 Development mode başlatılıyor...${NC}"
    python -m uvicorn dementia_microservice:app \
        --host 127.0.0.1 \
        --port 8000 \
        --reload \
        --log-level debug
}

# Production modunda başlat
start_prod() {
    echo -e "${BLUE}🏭 Production mode başlatılıyor...${NC}"
    python -m uvicorn dementia_microservice:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 2 \
        --log-level info
}

# Docker ile başlat
start_docker() {
    echo -e "${BLUE}🐳 Docker ile başlatılıyor...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker bulunamadı${NC}"
        exit 1
    fi
    
    # Docker build
    echo "Building Docker image..."
    docker build -t dementia-microservice .
    
    # Docker run
    echo "Starting container..."
    docker run -d \
        --name dementia-api \
        -p 8000:8000 \
        -v $(pwd)/logs:/app/logs \
        dementia-microservice
    
    echo -e "${GREEN}✅ Docker container başlatıldı${NC}"
    echo "Container logs: docker logs -f dementia-api"
}

# Docker Compose ile başlat
start_compose() {
    echo -e "${BLUE}🐳 Docker Compose ile başlatılıyor...${NC}"
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose bulunamadı${NC}"
        exit 1
    fi
    
    docker-compose up -d
    echo -e "${GREEN}✅ Services başlatıldı${NC}"
}

# Test endpoint'leri
test_endpoints() {
    echo -e "${BLUE}🧪 Endpoint'ler test ediliyor...${NC}"
    
    # Health check
    echo "Testing health endpoint..."
    curl -s http://localhost:8000/health | jq .
    
    echo ""
    echo "Testing root endpoint..."
    curl -s http://localhost:8000/ | jq .
}

# Stop services
stop_services() {
    echo -e "${BLUE}🛑 Servisler durduruluyor...${NC}"
    
    # Docker containers'ı durdur
    docker stop dementia-api 2>/dev/null || true
    docker rm dementia-api 2>/dev/null || true
    
    # Docker Compose'u durdur
    docker-compose down 2>/dev/null || true
    
    # Process'leri öldür
    pkill -f "uvicorn dementia_microservice" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Servisler durduruldu${NC}"
}

# Kullanım bilgisi
show_usage() {
    echo "Kullanım: $0 [KOMUT]"
    echo ""
    echo "Komutlar:"
    echo "  start        Mikroservisi başlat (varsayılan)"
    echo "  dev          Development modunda başlat"
    echo "  prod         Production modunda başlat"
    echo "  docker       Docker ile başlat"
    echo "  compose      Docker Compose ile başlat"
    echo "  test         Endpoint'leri test et"
    echo "  stop         Tüm servisleri durdur"
    echo "  install      Bağımlılıkları yükle"
    echo "  help         Bu yardım mesajını göster"
}

# Ana fonksiyon
main() {
    case ${1:-start} in
        "start")
            check_requirements
            start_microservice
            ;;
        "dev")
            check_requirements
            start_dev
            ;;
        "prod")
            check_requirements
            start_prod
            ;;
        "docker")
            check_requirements
            start_docker
            ;;
        "compose")
            check_requirements
            start_compose
            ;;
        "test")
            test_endpoints
            ;;
        "stop")
            stop_services
            ;;
        "install")
            install_dependencies
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            echo -e "${RED}❌ Bilinmeyen komut: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
}

# Script'i çalıştır
main "$@" 