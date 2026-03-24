#!/usr/bin/env python3
"""Test script for FastAPI application."""

import sys
import time
import requests
from pathlib import Path

def test_api_endpoints():
    """Test all API endpoints."""
    print("Testing FastAPI Endpoints")
    print("=" * 40)

    base_url = "http://localhost:8000"

    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check: {data['status']}")
            print(f"   Components: {data['components']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to {base_url}")
        print("   Make sure to start the server first:")
        print("   PYTHONPATH=. uvicorn src.main:app --reload")
        return False

    print("\n2. Testing system info...")
    response = requests.get(f"{base_url}/info")
    if response.status_code == 200:
        info = response.json()
        print(f"   Ingestion status: {info['ingestion_status']}")
        print(f"   Vector store available: {info['vector_store_available']}")

    print("\n3. Testing data ingestion...")
    ingest_data = {
        "document_limit": 3,
        "chunk_size": 500,
        "batch_size": 1,
        "overwrite": True
    }

    response = requests.post(f"{base_url}/ingest", json=ingest_data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Ingestion started: {result['status']}")

        # Wait for ingestion to complete
        print("   Waiting for ingestion to complete...")
        for i in range(30):  # Wait up to 3 minutes
            status_response = requests.get(f"{base_url}/ingest/status")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"   Status: {status['status']}")

                if status['status'] == 'completed':
                    print("   ✅ Ingestion completed!")
                    break
                elif status['status'] == 'failed':
                    print(f"   ❌ Ingestion failed: {status.get('progress', {})}")
                    return False

            time.sleep(6)  # Wait 6 seconds between checks
        else:
            print("   ⚠️  Ingestion timeout")
            return False
    else:
        print(f"   ❌ Ingestion failed: {response.status_code}")
        return False

    print("\n4. Testing similarity search...")
    query_data = {
        "query": "Uruguay capital city",
        "top_k": 2,
        "similarity_threshold": 0.0
    }

    response = requests.post(f"{base_url}/query", json=query_data)
    if response.status_code == 200:
        results = response.json()
        print(f"   ✅ Found {results['total_results']} results")
        for result in results['results']:
            score = result['similarity_score']
            text_preview = result['text'][:80] + "..."
            print(f"   Rank {result['rank']}: Score={score:.3f}")
            print(f"                     Text: {text_preview}")
    else:
        print(f"   ❌ Query failed: {response.status_code}")

    print("\n5. Testing RAG generation...")
    rag_data = {
        "query": "What is the capital of Uruguay?",
        "top_k": 2
    }

    response = requests.post(f"{base_url}/rag", json=rag_data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ RAG Answer: {result['answer'][:100]}...")
        print(f"   Sources used: {len(result['sources'])}")
    else:
        print(f"   ❌ RAG generation failed: {response.status_code}")
        print(f"   Error: {response.text}")

    print("\n✅ API testing complete!")
    return True

if __name__ == "__main__":
    print("FastAPI Test Script")
    print("\nBefore running this script, start the server:")
    print("uvicorn src.main:app --reload")
    print("\nPress Enter when server is running...")
    input()

    test_api_endpoints()