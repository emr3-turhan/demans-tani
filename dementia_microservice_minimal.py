#!/usr/bin/env python3
"""
🚀 Minimal Microservice for Docker Build Testing
"""

import os
from fastapi import FastAPI

app = FastAPI(
    title="🧠 Dementia Detection Microservice - Minimal",
    description="Minimal version for Docker build testing",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Ana endpoint"""
    return {
        "service": "Dementia Detection Microservice",
        "status": "running",
        "mode": "minimal",
        "message": "Docker build test successful! 🎉"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "dementia-microservice",
        "mode": "minimal"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 