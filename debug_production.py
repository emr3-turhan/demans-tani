#!/usr/bin/env python3
"""
🔍 Production Debugging Script for Render.com Deployment
"""

import httpx
import json
import time
import asyncio

# 🌐 Production URL (update with your actual Render.com URL)
PRODUCTION_URL = "https://your-app-name.onrender.com"  # Replace with actual URL

async def test_production_endpoints():
    """Test all production endpoints"""
    print("🔍 Production Debugging - Render.com Deployment")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # Test 1: Health Check
        print("\n1️⃣ Testing Health Endpoint...")
        try:
            response = await client.get(f"{PRODUCTION_URL}/health")
            print(f"✅ Status: {response.status_code}")
            print(f"📄 Response: {response.text[:200]}...")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        
        # Test 2: Root Endpoint
        print("\n2️⃣ Testing Root Endpoint...")
        try:
            response = await client.get(f"{PRODUCTION_URL}/")
            print(f"✅ Status: {response.status_code}")
            print(f"📄 Response: {response.text[:200]}...")
        except Exception as e:
            print(f"❌ Root endpoint failed: {e}")
        
        # Test 3: Docs Endpoint
        print("\n3️⃣ Testing Docs Endpoint...")
        try:
            response = await client.get(f"{PRODUCTION_URL}/docs")
            print(f"✅ Status: {response.status_code}")
            print(f"📄 Content-Type: {response.headers.get('content-type', 'Unknown')}")
        except Exception as e:
            print(f"❌ Docs endpoint failed: {e}")
        
        # Test 4: Analyze Sync (with known working data)
        print("\n4️⃣ Testing Analyze Sync Endpoint...")
        try:
            test_payload = {
                "test_session_id": "Iu0q6FkhMu3OHJeIPNUM",
                "question_id": "iLEstW6nRQXARxdObGcR"
            }
            
            response = await client.post(
                f"{PRODUCTION_URL}/analyze-sync",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"✅ Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"🧠 Dementia Status: {result.get('dementia_status', 'Unknown')}")
                print(f"📊 Confidence: {result.get('confidence_score', 0) * 100:.1f}%")
                print(f"⏱️ Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
            else:
                print(f"❌ Error Response: {response.text[:500]}...")
                
        except Exception as e:
            print(f"❌ Analyze sync failed: {e}")

def test_docker_locally():
    """Test local Docker container"""
    print("\n🐳 Testing Local Docker Container...")
    print("=" * 40)
    
    import subprocess
    
    try:
        # Pull latest image
        result = subprocess.run([
            "docker", "pull", "emr3turhan/dementia-microservice:latest"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker image pulled successfully")
        else:
            print(f"❌ Docker pull failed: {result.stderr}")
            return
        
        # Run container
        print("🚀 Starting container...")
        subprocess.run([
            "docker", "run", "-d", "-p", "9000:8000", 
            "--name", "debug-container", 
            "emr3turhan/dementia-microservice:latest"
        ], capture_output=True)
        
        # Wait for startup
        time.sleep(10)
        
        # Test health
        result = subprocess.run([
            "curl", "-s", "http://localhost:9000/health"
        ], capture_output=True, text=True)
        
        if "healthy" in result.stdout:
            print("✅ Local Docker container is healthy")
            print(f"📄 Response: {result.stdout}")
        else:
            print(f"❌ Local Docker health check failed: {result.stdout}")
        
        # Cleanup
        subprocess.run(["docker", "stop", "debug-container"], capture_output=True)
        subprocess.run(["docker", "rm", "debug-container"], capture_output=True)
        
    except Exception as e:
        print(f"❌ Docker test failed: {e}")

def print_common_issues():
    """Print common production deployment issues"""
    print("\n🔧 Common Production Issues & Solutions:")
    print("=" * 50)
    print("""
1️⃣ PORT Configuration:
   - Render.com expects app to bind to 0.0.0.0:$PORT
   - Check if Dockerfile exposes correct port
   
2️⃣ File Permissions:
   - Model files might not be readable
   - Check if full_synthetic_dataset/ is included
   
3️⃣ Memory Issues:
   - AI model loading might exceed memory limits
   - Consider using smaller model or more memory
   
4️⃣ Startup Time:
   - Render.com has startup timeout (usually 5-10 minutes)
   - Model loading might be taking too long
   
5️⃣ Dependencies:
   - Some packages might fail to install
   - Check build logs for pip install errors
   
6️⃣ Environment Variables:
   - Missing required environment variables
   - Check Render.com environment settings
""")

if __name__ == "__main__":
    print("🔍 Production Debugging Toolkit")
    print("Update PRODUCTION_URL with your Render.com URL")
    print("\nOptions:")
    print("1. Test production endpoints: python debug_production.py production")
    print("2. Test local Docker: python debug_production.py docker")
    print("3. Show common issues: python debug_production.py issues")
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "production":
            asyncio.run(test_production_endpoints())
        elif sys.argv[1] == "docker":
            test_docker_locally()
        elif sys.argv[1] == "issues":
            print_common_issues()
    else:
        print_common_issues() 