import os
import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flasgger import Swagger, swag_from
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Import from rag_utils
from rag_utils import (
    setup_chroma_vector_store,
    get_rag_chain,
    open_and_read_pdf,
    index_text_chunks,
    delete_document_by_source,
    GeminiLLM,
    index_processed_chunks,
    AuditLogger
)

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify API Key
if not os.environ.get("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not found in environment variables!")
    # In some environments, it might be named GEMINI_API_KEY
    if os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")
        logger.info("Using GEMINI_API_KEY as GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})
swagger = Swagger(app)

# Global objects
vector_store = None
rag_chain = None
audit_logger = AuditLogger()

def init_app():
    global vector_store, rag_chain
    try:
        logger.info("Initializing ChromaDB Vector Store...")
        # Change path because E5 dimensions (1024) are different from Gemini (768)
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db_e5")
        collection_name = os.environ.get("CHROMA_COLLECTION", "embeddings")
        vector_store = setup_chroma_vector_store(persist_dir, collection_name)
        
        logger.info("Initializing LLM and RAG Chain...")
        # Using gemini-1.5-flash which is free-tier friendly
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            temperature=0.3,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        # Define Prompt Template (Amharic)
        # We need a prompt compatible with get_rag_chain which expects specific inputs
        # The rag_utils.get_rag_chain uses: 
        # {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        # So the prompt must accept 'context' and 'question' input variables.
        
        template = """ከታች ያለው መረጃን በመጠቀም፣ የተጠየቀውን ጥያቄ መልስ።\n\nማብራሪያ:\n{context}\n\nጥያቄ: {question}\nመልስ:"""
        prompt = PromptTemplate.from_template(template)
        
        rag_chain, _ = get_rag_chain(vector_store, llm, prompt)
        logger.info("Initialization Complete.")
    except Exception as e:
        logger.error(f"Failed to initialize app: {e}")
        raise e

# Initialize on module load (or create a factory, but simple global for this script is fine)
# We wrap in a try-catch to allow app to start even if DB connection fails (optional, but good for debugging)
try:
    init_app()
except Exception as e:
    logger.error("Application failed to initialize correctly.")

@app.route('/')
def index():
    """
    Landing page for the API.
    """
    return jsonify({
        "message": "Welcome to the Zufan Amharic Legal RAG API",
        "documentation": "/apidocs",
        "health": "/health",
        "upload_ui": "/upload",
        "status": "online"
    }), 200

@app.route('/upload')
def upload_page():
    """
    Serve the simple upload web app.
    """
    return render_template('upload.html')

@app.route('/test')
def test_page():
    """
    Serve the comprehensive API tester.
    """
    return render_template('test_api.html')


@app.route('/health', methods=['GET'])
def health_check():
    """
    Check the health of the application.
    ---
    responses:
      200:
        description: Returns the health status of the application.
        schema:
          properties:
            status:
              type: string
              example: healthy
            vector_store_initialized:
              type: boolean
            rag_chain_initialized:
              type: boolean
    """
    status = {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "rag_chain_initialized": rag_chain is not None
    }
    return jsonify(status), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat interaction with streaming support.
    """
    if not rag_chain:
        return jsonify({"error": "RAG pipeline not initialized"}), 503

    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing messages field"}), 400

    messages = data["messages"]
    if not messages:
        return jsonify({"error": "Empty messages list"}), 400
    
    # Extract latest query
    last_message = messages[-1]
    query = last_message.get("content", "")

    try:
        def generate():
            full_response = ""
            try:
                # Streaming with LangChain
                for chunk in rag_chain.stream(query):
                    full_response += chunk
                    yield chunk
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                audit_logger.log_event("CHAT_ERROR", str(e), level="ERROR")
                yield f"Error: {str(e)}"

        audit_logger.log_event("CHAT_QUERY", query[:100] + "..." if len(query) > 100 else query)
        return Response(stream_with_context(generate()), content_type='text/plain')

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Vector Management Endpoints ---

@app.route('/api/vector/stats', methods=['GET'])
def vector_stats():
    """
    Get statistics for the Vector Store.
    ---
    responses:
      200:
        description: Statistics about the vector store.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        # ChromaDB doesn't have a direct "stats" aggregate like Mongo without fetching all data
        # We can use the collection's count() method
        total_vectors = vector_store._collection.count()
        
        # For unique documents, we'd need to fetch metadatas and find unique sources
        # This is expensive in Chroma for large datasets, but fine for legal PDFs
        metadatas = vector_store._collection.get(include=["metadatas"])["metadatas"]
        total_docs = len(set(m.get("source") for m in metadatas if m))
        
        return jsonify({
            "total_vectors": total_vectors,
            "total_documents": total_docs,
            "index_size_mb": -1, 
            "dimensions": 1024, # multilingual-e5-large is 1024
            "model": "multilingual-e5-large"
        }), 200
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vector/recent', methods=['GET'])
def get_recent_embeddings():
    """
    Get recently indexed vector chunks.
    ---
    responses:
      200:
        description: List of the most recent embeddings.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        # Get most recent items. Chroma doesn't store timestamps by default,
        # but we can get the latest IDs if they are sequential or just the last few.
        res = vector_store._collection.get(limit=10, include=["metadatas", "documents"])
        
        recent = []
        for i in range(len(res["ids"])):
            text = res["documents"][i]
            metadata = res["metadatas"][i] or {}
            
            recent.append({
                "id": res["ids"][i],
                "source": metadata.get("source", "Unknown"),
                "text_snippet": text[:50] + "..." if len(text) > 50 else text,
                "tokens": int(len(text) / 4),
                "indexed_at": "Unknown" # Chroma doesn't store this by default
            })
            
        return jsonify(recent), 200
    except Exception as e:
        logger.error(f"Recent vectors error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vector/search', methods=['POST'])
def semantic_search_playground():
    """
    Test retrieval relevance with semantic search.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          properties:
            query:
              type: string
            k:
              type: integer
    responses:
      200:
        description: List of top semantic search results.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    data = request.get_json()
    query = data.get("query", "")
    k = data.get("k", 5)
    
    if not query:
        return jsonify({"error": "Query required"}), 400
        
    try:
        # Perform similarity search with scores
        results = vector_store.similarity_search_with_score(query, k=k)
        
        formatted_res = []
        for doc, score in results:
            formatted_res.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(score, 4) # Cosine similarity score (usually 0 to 1 or -1 to 1)
            })
            
        return jsonify(formatted_res), 200
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/vector/clear', methods=['DELETE'])
def clear_all_vectors():
    """
    Purge all vectors and documents from the collection.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        # Re-initialize or delete all
        ids = vector_store._collection.get()["ids"]
        if ids:
            vector_store.delete(ids=ids)
        audit_logger.log_event("VECTOR_STORE_CLEAR", f"Deleted {len(ids)} vectors", level="WARNING")
        return jsonify({
            "message": "Vector store purged successfully",
            "deleted_count": len(ids)
        }), 200
    except Exception as e:
        logger.error(f"Clear all error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/vectors/<vector_id>', methods=['DELETE'])
def delete_single_vector(vector_id):
    """
    Delete a single vector by its ID.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        vector_store.delete(ids=[vector_id])
        return jsonify({"message": f"Vector '{vector_id}' deleted"}), 200
    except Exception as e:
        logger.error(f"Delete vector error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload/file', methods=['POST'])
def upload_file():
    """
    Upload a PDF file, process it, and index its content.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: File processed and indexed successfully.
      400:
        description: Invalid request.
    """
    if not vector_store:
        return jsonify({"error": "Vector store not initialized"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save temp file
        filename = file.filename
        temp_path = os.path.join("/tmp", filename) if os.name != 'nt' else os.path.join(os.environ.get("TEMP", "."), filename)
        file.save(temp_path)
        
        logger.info(f"Processing file: {filename}")
        
        # Read PDF
        # rag_utils.open_and_read_pdf returns list of dicts with text and metadata stats
        chunks_data = open_and_read_pdf(temp_path)
        
        # We need to adapt chunks_data to format expected by index_text_chunks
        # open_and_read_pdf returns: 
        # { "page_number", "page_char_count", ..., "text": "..." }
        # index_text_chunks expects: { "text": "...", "metadata": {...} }
        
        formatted_chunks = []
        for page in chunks_data:
            text = page.pop("text")
            # remaining keys are metadata
            formatted_chunks.append({
                "text": text,
                "metadata": page # includes source, page_number, etc.
            })
            
        count = index_text_chunks(vector_store, formatted_chunks)
        
        audit_logger.log_event("DOCUMENT_UPLOAD", f"File: {filename}, Chunks: {count}", metadata={"filename": filename})
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            "message": "File processed and indexed successfully",
            "chunks_count": count,
            "filename": filename
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload/chunks', methods=['POST'])
def upload_chunks():
    """
    Directly index pre-processed text chunks.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          properties:
            documentId:
              type: string
            chunks:
              type: array
              items:
                properties:
                  text:
                    type: string
                  metadata:
                    type: object
    responses:
      200:
        description: Chunks indexed successfully.
      400:
        description: Invalid request.
    """
    if not vector_store:
        return jsonify({"error": "Vector store not initialized"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request format"}), 400

    try:
        if isinstance(data, list):
            # User provided a list directly (amh_pages_and_chunks format)
            count = index_processed_chunks(vector_store, data)
        elif isinstance(data, dict) and "chunks" in data:
            # Traditional format with "chunks" key
            chunks = data["chunks"]
            doc_id = data.get("documentId")
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                if doc_id:
                    chunk["metadata"]["documentId"] = doc_id
            count = index_text_chunks(vector_store, chunks)
        else:
            return jsonify({"error": "Invalid request format. Expected a list of chunks or a dict with 'chunks' key."}), 400
        
        audit_logger.log_event("CHUNK_INGESTION", f"Count: {count}")
        
        return jsonify({
            "message": "Chunks indexed successfully",
            "count": count
        }), 200

    except Exception as e:
        logger.error(f"Chunk upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """
    List unique documents and their statistics from the vector store.
    """
    if not vector_store:
        return jsonify({"error": "Vector store not initialized"}), 503
    try:
        # Fetch all metadatas to aggregate stats
        res = vector_store._collection.get(include=["metadatas"])
        metadatas = res["metadatas"]
        
        doc_stats = {}
        for m in metadatas:
            if not m: continue
            
            # Use source as the primary identifier
            source = m.get("source", "Unknown")
            
            if source not in doc_stats:
                doc_stats[source] = {
                    "count": 0, 
                    "chars": 0, 
                    "pages": set(),
                    "type": "Chunked" if "sentence_chunk" in m or "page" in m else "PDF"
                }
            
            doc_stats[source]["count"] += 1
            doc_stats[source]["chars"] += m.get("page_char_count", m.get("chunk_char_count", 0))
            
            # Track unique pages
            page = m.get("page_number") or m.get("page")
            if page:
                doc_stats[source]["pages"].add(page)
            
        documents = []
        for name, stats in doc_stats.items():
            documents.append({
                "id": name,
                "name": name,
                "type": stats["type"],
                "status": "Indexed",
                "chunks": stats["count"],
                "total_chars": stats["chars"],
                "page_count": len(stats["pages"])
            })
            
        # Optional: Sort by name
        documents.sort(key=lambda x: x["name"])
        
        return jsonify(documents), 200

    except Exception as e:
        logger.error(f"List documents error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/api/documents/<path:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """
    Delete a document and all its associated vectors from the vector store.
    """
    if not vector_store:
        return jsonify({"error": "Vector store not initialized"}), 503
    
    try:
        # rag_utils.delete_document_by_source uses generic delete
        # We assume doc_id is the source name
        success = delete_document_by_source(vector_store, doc_id)
        
        if success:
            audit_logger.log_event("DOCUMENT_DELETE", f"Source: {doc_id}")
            return jsonify({"message": f"Document '{doc_id}' deleted"}), 200
        else:
            return jsonify({"error": "Delete failed"}), 400

    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Audit Log Endpoints ---

@app.route('/api/audit/logs', methods=['GET'])
def get_audit_logs():
    """
    Retrieve recently system events.
    """
    limit = request.args.get('limit', 50, type=int)
    logs = audit_logger.get_logs(limit=limit)
    return jsonify(logs), 200

@app.route('/api/audit/logs', methods=['DELETE'])
def clear_audit_logs():
    """
    Clear the audit history.
    """
    success = audit_logger.clear_logs()
    if success:
        return jsonify({"message": "Audit logs cleared"}), 200
    else:
        return jsonify({"error": "Failed to clear logs"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
