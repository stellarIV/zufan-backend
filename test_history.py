import requests
import time
import json

BASE_URL = "http://127.0.0.1:5000"
USER_ID = "test_user_123"

def test_chat_history_flow():
    print("--- Testing Chat History Flow ---")
    
    # 1. Create Session
    print("\n1. Creating Session...")
    res = requests.post(f"{BASE_URL}/api/chat/sessions", json={"userId": USER_ID, "title": "History Test Chat"})
    if res.status_code != 201:
        print(f"Failed to create session: {res.text}")
        return
    session_data = res.json()
    session_id = session_data["sessionId"]
    print(f"Session Created: {session_id}")

    # 2. Chat with Session ID
    print("\n2. Sending Message (Streamed)...")
    res = requests.post(
        f"{BASE_URL}/api/chat", 
        json={
            "messages": [{"role": "user", "content": "Hello, who are you? Answer in 1 short sentence."}],
            "sessionId": session_id,
            "userId": USER_ID
        },
        stream=True
    )
    print("Response Stream: ", end="")
    for chunk in res.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode('utf-8'), end="", flush=True)
    print("\n")
    
    # 3. List Sessions
    print("\n3. Listing Sessions...")
    res = requests.get(f"{BASE_URL}/api/chat/sessions?userId={USER_ID}")
    sessions = res.json()
    print(f"Found {len(sessions)} sessions.")
    found = any(s["id"] == session_id for s in sessions)
    print(f"Session {session_id} in list: {found}")

    # 4. Get History
    print("\n4. Getting History...")
    res = requests.get(f"{BASE_URL}/api/chat/sessions/{session_id}")
    history = res.json()
    print(f"History Length: {len(history)} messages")
    for msg in history:
        print(f"- [{msg['role']}]: {msg['content']}")

    # 5. Delete Session
    print("\n5. Deleting Session...")
    res = requests.delete(f"{BASE_URL}/api/chat/sessions/{session_id}")
    print(f"Delete Status: {res.status_code}")

if __name__ == "__main__":
    # Ensure app is running locally for this test or change BASE_URL
    # We will assume localhost for quick dev cycle verification
    try:
        test_chat_history_flow()
    except Exception as e:
        print(f"Test Failed: {e}")
