# Legal RAG Backend Specification (Amharic)

This document serves as the technical blueprint for generating a production-ready backend for the 'Zufan' Amharic Legal RAG system.

## Performance Requirements
- **Language Support**: Native Amharic (UTF-8).
- **LLM**: Gemini 2.0 Flash (via Google Generative AI).
- **RAG Framework**: LangChain (Python).
- **Vector Database**: MongoDB Atlas Vector Search (Primary) or Chroma DB (Fallback).
- **API Framework**: Flask (Python) - *Chosen for seamless LangChain integration.*
- **Core Logic**: Extracted functions are available in [rag_utils.py](file:///c:/Users/biruk/OneDrive/Documents/GitHub/zufan/server/rag_utils.py).

---

## Environment Variables (.env)

| Variable | Description | Example |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API Key | `AIza...` |
| `MONGODB_URI` | MongoDB Atlas Connection String | `mongodb+srv://...` |
| `MONGODB_DB_NAME` | Database name for vector storage | `legal_rag` |
| `MONGODB_COLLECTION` | Collection name for vectors | `embeddings` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `true` |
| `LANGCHAIN_API_KEY` | LangSmith API Key | `lsv2_...` |
| `PORT` | Backend port | `5000` |

---

## Technical Logic (Notebook Extraction)

### 1. Amharic Text Processing
The backend must handle Amharic specific punctuation:
- **Full Stop**: `፡፡` (Used for sentence splitting).
- **Cleaning**: Remove excessive newlines and strip whitespace.

### 2. PDF Extraction (`PyMuPDF/fitz`)
- Extract text page-by-page.
- Track metadata: `page_number`, `page_char_count`, `page_word_count`, `page_token_count`.
- Token count estimate: `len(text) / 4`.

### 3. RAG Pipeline
- **Embedding Model**: `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`.
- **Vector Store**: MongoDB Atlas Vector Search.
- **Prompt**: Custom prompt tailored for Amharic legal queries.
- **Temperature**: Default `0.3` for high precision.

---

## API Endpoints

### 1. Chat & Assistant
`POST /api/chat`
- **Request**: `{ "messages": Message[] }`
- **Logic**: 
  1. Retrieve top-k (default 3-5) relevant context from MongoDB.
  2. Format context into the Amharic prompt.
  3. Invoke Gemini with context and query.
- **Response**: `{ "role": "assistant", "content": string, "citations": Citation[] }`

### 2. Document Management (Admin)
`POST /api/upload/file`
- **Request**: `FormData (file, metadata)`
- **Logic**: Process PDF, chunk by page, generate embeddings, store in MongoDB.

`POST /api/upload/chunks`
- **Request**: `{ "documentId": string, "chunks": { "text": string, "metadata": object }[] }`
- **Logic**: Direct insertion of pre-processed chunks into the vector store.

`GET /api/documents`
- **Response**: `Document[]` with processing status.

`DELETE /api/documents/:id`
- **Logic**: Remove document record and all associated vectors from MongoDB.

---

## Data Models

```typescript
interface Document {
  id: string;
  name: string;
  type: string;
  status: "Indexed" | "Processing" | "Error";
  date: string;
  size: string;
  jurisdiction?: string;
}

interface Citation {
  source: string;
  content: string;
  pageNumber?: number;
}
```

## Prompt for Backend Generation
> [!TIP]
> Use the following prompt in a new project to generate this backend:
> "Generate a Python Flask backend for an Amharic Legal RAG system based on the specifications in this README. Use the provided `rag_utils.py` for all core logic. Implement a clean API structure with error handling, logging, and CORS enabled. Ensure the `/api/chat` endpoint supports streaming if possible, and `/api/upload/chunks` handles direct JSON indexing."

## Setup Guide for Generated Project
1. Install dependencies: `pip install flask flask-cors langchain langchain-google-genai pymongo pymupdf tiktoken python-dotenv`.
2. Configure MongoDB Search Index: Create a vector index named `vector_index` on the embeddings field.
3. Use the `GeminiLLM` wrapper or LangChain's `ChatGoogleGenerativeAI`.
