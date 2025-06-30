#!/usr/bin/env python3
"""
🧪 Production Deployment Test Script
Render.com deployment sonrası mikroservisi test eder
"""

import httpx
import asyncio
import json
from datetime import datetime

# 🌐 Production URL - Deploy sonrası güncelleyin
PRODUCTION_URL = "https://dementia-microservice.onrender.com"

async def test_health():
    """Health check testi"""
    print("❤️ Health Check Test...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{PRODUCTION_URL}/health")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Health Check: {result['status']}")
                print(f"🤖 Pipeline Ready: {result['pipeline_ready']}")
                return True
            else:
                print(f"❌ Health Check Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health Check Error: {e}")
            return False

async def test_home_page():
    """Ana sayfa testi"""
    print("\n🏠 Home Page Test...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(PRODUCTION_URL)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Service: {result['service']}")
                print(f"📝 Version: {result['version']}")
                print(f"🔗 Endpoints: {len(result['endpoints'])} available")
                return True
            else:
                print(f"❌ Home Page Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Home Page Error: {e}")
            return False

async def test_analyze_trigger():
    """Analiz tetikleme testi"""
    print("\n🚀 Analyze Trigger Test...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            payload = {
                "test_session_id": "eiRJer6JowfCJjSQ4LLM",
                "question_id": "iLEstW6nRQXARxdObGcR"
            }
            
            response = await client.post(
                f"{PRODUCTION_URL}/analyze",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Analysis Triggered: {result['success']}")
                print(f"🆔 Analysis ID: {result['analysis_id']}")
                print(f"⏱️ Estimated Duration: {result['estimated_duration']}s")
                return True
            else:
                print(f"❌ Analyze Failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Analyze Error: {e}")
            return False

async def test_sync_analysis():
    """Senkron analiz testi (full workflow)"""
    print("\n🔬 Synchronous Analysis Test...")
    print("⚠️ Bu test 30+ saniye sürebilir (cold start + AI processing)")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            payload = {
                "test_session_id": "eiRJer6JowfCJjSQ4LLM",
                "question_id": "iLEstW6nRQXARxdObGcR"
            }
            
            start_time = datetime.now()
            response = await client.post(
                f"{PRODUCTION_URL}/analyze-sync",
                json=payload
            )
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Sync Analysis Complete!")
                print(f"🧠 Result: {result['dementia_status']}")
                print(f"🎯 Confidence: {result['confidence_score']:.1%}")
                print(f"⚡ Processing Time: {duration:.1f}s")
                print(f"🎵 Audio Duration: {result['audio_duration_seconds']:.1f}s")
                return True
            else:
                print(f"❌ Sync Analysis Failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Sync Analysis Error: {e}")
            return False

async def test_api_docs():
    """API docs erişilebilirlik testi"""
    print("\n📚 API Documentation Test...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{PRODUCTION_URL}/docs")
            
            if response.status_code == 200:
                print("✅ API Docs accessible")
                print(f"📖 URL: {PRODUCTION_URL}/docs")
                return True
            else:
                print(f"❌ API Docs Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ API Docs Error: {e}")
            return False

async def main():
    """Ana test fonksiyonu"""
    print("🧪 PRODUCTION DEPLOYMENT TEST")
    print("=" * 50)
    print(f"🌐 Testing: {PRODUCTION_URL}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Health Check", test_health),
        ("Home Page", test_home_page),
        ("API Documentation", test_api_docs),
        ("Analyze Trigger", test_analyze_trigger),
        ("Sync Analysis", test_sync_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} Test Exception: {e}")
            results.append((test_name, False))
    
    # 📊 Sonuç Özeti
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Production deployment successful!")
        print(f"🚀 Microservice ready at: {PRODUCTION_URL}")
    else:
        print("⚠️ Some tests failed. Check logs above.")
    
    print(f"\n🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # URL'i güncelleme talimatı
    print("⚠️ IMPORTANT: Update PRODUCTION_URL variable with your actual Render.com URL")
    print("Example: https://dementia-microservice.onrender.com")
    print()
    
    asyncio.run(main()) 