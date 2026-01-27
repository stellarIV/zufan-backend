# Zufan Amharic Legal RAG Backend

A high-performance Retrieval-Augmented Generation (RAG) backend tailored for Amharic legal documents. Powered by Flask, MongoDB Atlas Vector Search, and Google Gemini.

## 🚀 Quick Links
- **Interactive Documentation**: `/apidocs` (Swagger UI)
- **Knowledge Sync UI**: `/upload` (Direct PDF Upload)
- **Health Status**: `/health`

---

## 🛠 Features
- **Amharic Optimized**: specialized prompt engineering and tokenization for Ethiopian legal contexts.
- **Persistent Chat History**: Gemini-style multi-session support.
- **Vector Store Management**: Real-time stats, semantic search playground, and data purging.
- **Streaming Responses**: Real-time AI response delivery for a premium UX.

---

## 📡 API Reference

### 1. Chat & Conversation
#### `POST /api/chat`
The core endpoint for AI interaction.
- **Payload**:
  ```json
  {
    "messages": [{"role": "user", "content": "Your query here"}],
    "sessionId": "UUID",
    "userId": "User-123"
  }
  ```
- **Returns**: A plain-text stream of the AI response.

#### `POST /api/chat/sessions`
Create a new chat container.
- **Payload**: `{"userId": "string", "title": "Optional Title"}`
- **Returns**: `{"sessionId": "...", "title": "..."}`

#### `GET /api/chat/sessions`
List all historical sessions for a user.
- **Query Param**: `userId`

---

### 2. Knowledge Base (Admin)
#### `POST /api/upload/file`
Upload a PDF for indexing.
- **Format**: `multipart/form-data`
- **Key**: `file` (PDF)

#### `POST /api/upload/chunks`
Directly index pre-chunked data (useful for integrating with custom scrapers or existing databases).
- **Payload**: A list of chunk objects:
  ```json
  [
    {
      "page_number": 1,
      "source": "Document_Name",
      "sentence_chunk": "Amharic text content here..."
    }
  ]
  ```

#### `GET /api/documents`
Returns a list of all indexed documents with detailed stats:
- `type`: "PDF" or "Chunked"
- `chunks`: Total vector count
- `total_chars`: Total character count
- `page_count`: Number of unique pages

#### `DELETE /api/documents/<filename>`
Removes all vectors associated with a specific file or source name.

---

### 3. Vector Management
#### `GET /api/vector/stats`
Returns system health: Total vectors, Index size, and Model info.

#### `POST /api/vector/search`
**Playground**: Test the raw semantic search without AI generation.
- **Payload**: `{"query": "Amharic legal term", "k": 5}`

#### `DELETE /api/vector/clear`
**CAUTION**: Deletes the entire vector store (Factory Reset).

---

## 🔄 Integration Path (Next.js)

### Step 1: Centralized API Client
We recommend creating a `server/api.ts` file to wrap these calls. (See `frontend_api_implementation.md` in the artifacts for full code).

### Step 2: Streaming Implementation
Use the `ReadableStream` API from within your React hooks to read the `/api/chat` response chunk-by-chunk for a professional "typewriter" effect.

### Step 3: Direct Chunk Ingestion (Python Example)
If you have a custom pipeline or another AI project (e.g., using **Antigravity AI**) and want to sync pre-processed Amharic data:

```python
import requests

data = [
    {
        "page_number": 1,
        "source": "Legal_Proclamation_Draft",
        "sentence_chunk": "የኢትዮጵያ ፌዴራላዊ ዴሞክራሲያዊ ሪፐብሊክ..."
    }
]

response = requests.post("http://localhost:5000/api/upload/chunks", json=data)
print(response.json())
```

---

## ⚙️ Environment Setup
Create a `.env` file with the following:
```env
GOOGLE_API_KEY=your_gemini_key
MONGODB_URI=your_atlas_connection_string
MONGODB_DB_NAME=zufan
MONGODB_COLLECTION=embeddings
```

## 📦 Requirements
- Python 3.10+
- `pip install -r requirements.txt`

---
*Built with ❤️ for the Ethiopian Legal Tech Ecosystem*