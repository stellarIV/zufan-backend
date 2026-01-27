import requests
import json

BASE_URL = "http://localhost:5000"

sample_chunks = [
    {
        'page_number': 1,
        'source': 'የኢትዮጵያ ፌዴራል ዴሞክራሲያዊ ሪፐብሊክ ሕገ-መንግሥት',
        'sentence_chunk': 'መ ግ ቢ ያ እኛ የኢትዮጵያ ብሔሮች፣ ብሔረሰቦች፣ ሕዝቦች...',
        'chunk_char_count': 1613,
        'chunk_word_count': 280,
        'chunk_token_count': 403.25
    },
    {
        'page_number': 2,
        'source': 'የኢትዮጵያ ፌዴራል ዴሞክራሲያዊ ሪፐብሊክ ሕገ-መንግሥት',
        'sentence_chunk': 'አንቀጽ 5:ስለ ቋንቋ 1. ማናቸውም የኢትዮጵያ ቋንቋዎች በእኩልነት የመንግሥት እውቅና ይኖራቸዋል...',
        'chunk_char_count': 920,
        'chunk_word_count': 171,
        'chunk_token_count': 230.0
    }
]

def test_chunk_upload():
    print("Testing chunk upload...")
    response = requests.post(f"{BASE_URL}/api/upload/chunks", json=sample_chunks)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_stats():
    print("Checking stats...")
    response = requests.get(f"{BASE_URL}/api/vector/stats")
    print(f"Stats: {response.json()}")

def test_search():
    print("Testing search...")
    query = {"query": "ቋንቋ", "k": 2}
    response = requests.post(f"{BASE_URL}/api/vector/search", json=query)
    print(f"Search results: {len(response.json())} items found")
    for res in response.json():
        print(f"- {res['metadata']['source']} (Page {res['metadata']['page']}): {res['content'][:50]}...")

if __name__ == "__main__":
    try:
        if test_chunk_upload():
            test_stats()
            test_search()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the server is running on http://localhost:5000")
