import requests
import json
import time

BASE_URL = "https://zufan-backend.onrender.com"

def test_health():
    print("Testing /health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

def test_upload_chunks():
    print("\nTesting /api/upload/chunks...")
    url = f"{BASE_URL}/api/upload/chunks"
    data = {
        "documentId": "test_doc_1",
        "chunks": [
            {
                "text": "The capital of Ethiopia is Addis Ababa. It is located in the Horn of Africa.",
                "metadata": {"source": "test_doc_1", "page_number": 1}
            }
        ]
    }
    try:
        response = requests.post(url, json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Upload chunks failed: {e}")

def test_chat():
    print("\nTesting /api/chat...")
    url = f"{BASE_URL}/api/chat"
    data = {
        "messages": [
            {"role": "user", "content": "What is the capital of Ethiopia?"}
        ]
    }
    try:
        response = requests.post(url, json=data, stream=True)
        print(f"Status: {response.status_code}")
        print("Response Stream:")
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        print("\n")
    except Exception as e:
        print(f"Chat failed: {e}")

if __name__ == "__main__":
    print(f"Checking API at {BASE_URL}")
    test_health()
    print("Waiting 40s to respect Rate Limits...")
    time.sleep(40) 
    test_upload_chunks()
    print("Waiting 40s due to indexing & rate limits...")
    time.sleep(40)
    test_chat()
