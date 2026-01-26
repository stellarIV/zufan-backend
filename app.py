import os
import logging
import datetime
from bson import ObjectId
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flasgger import Swagger, swag_from
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Import from rag_utils
from rag_utils import (
    setup_mongodb_vector_store,
    get_rag_chain,
    open_and_read_pdf,
    index_text_chunks,
    delete_document_by_source,
    GeminiLLM
)

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})
swagger = Swagger(app)

# Global objects
vector_store = None
rag_chain = None

def init_app():
    global vector_store, rag_chain
    try:
        logger.info("Initializing Vector Store...")
        db_name = os.environ.get("MONGODB_DB_NAME", "zufan_legal")
        collection_name = os.environ.get("MONGODB_COLLECTION", "embeddings")
        vector_store = setup_mongodb_vector_store(db_name, collection_name)
        
        logger.info("Initializing LLM and RAG Chain...")
        # Using ChatGoogleGenerativeAI for better streaming support
        # Using gemini-flash-latest alias to avoid version compatibility issues
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.3,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        # Define Prompt Template (Amharic)
        # We need a prompt compatible with get_rag_chain which expects specific inputs
        # The rag_utils.get_rag_chain uses: 
        # {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        # So the prompt must accept 'context' and 'question' input variables.
        
        template = """You are an Amharic Legal Assistant aka 'Zufan'. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.
Always answer in Amharic unless requested otherwise.

Context:
{context}

Question:
{question}

Answer:"""
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
    Chat interaction with streaming support and history.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          properties:
            messages:
              type: array
              items:
                properties:
                  role:
                    type: string
                  content:
                    type: string
            sessionId:
              type: string
            userId:
              type: string
    responses:
      200:
        description: Streamed response from the chatbot.
      400:
        description: Missing required fields.
      503:
        description: RAG pipeline not initialized.
    """
    if not rag_chain:
        return jsonify({"error": "RAG pipeline not initialized"}), 503

    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing messages field"}), 400

    messages = data["messages"]
    if not messages:
        return jsonify({"error": "Empty messages list"}), 400
    
    session_id = data.get("sessionId")
    user_id = data.get("userId")

    # Extract latest query
    last_message = messages[-1]
    query = last_message.get("content", "")
    
    # Save User Message if session exists
    if session_id and vector_store:
        try:
            db_name = os.environ.get("MONGODB_DB_NAME", "zufan_legal")
            client = vector_store._collection.database.client
            db = client[db_name]
            
            # Ensure session exists or create if valid ID provided (or handle client side creation)
            # Here we assume client creates session first or we just log to the ID provided
            
            user_msg_doc = {
                "sessionId": session_id,
                "role": "user",
                "content": query,
                "createdAt": datetime.datetime.utcnow()
            }
            db.chat_messages.insert_one(user_msg_doc)
            
            # Update session timestamp
            db.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"updatedAt": datetime.datetime.utcnow()}}
            )
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")

    try:
        def generate():
            full_response = ""
            try:
                # Streaming with LangChain
                for chunk in rag_chain.stream(query):
                    full_response += chunk
                    yield chunk
                
                # Save AI Message after streaming completes
                if session_id and vector_store:
                    try:
                        db_name = os.environ.get("MONGODB_DB_NAME", "zufan_legal")
                        client = vector_store._collection.database.client
                        db = client[db_name]
                        
                        ai_msg_doc = {
                            "sessionId": session_id,
                            "role": "assistant",
                            "content": full_response,
                            "createdAt": datetime.datetime.utcnow()
                        }
                        db.chat_messages.insert_one(ai_msg_doc)
                    except Exception as e:
                        logger.error(f"Failed to save AI message: {e}")

            except Exception as e:
                logger.error(f"Error during generation: {e}")
                yield f"Error: {str(e)}"

        return Response(stream_with_context(generate()), content_type='text/plain')

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Chat History Endpoints ---

@app.route('/api/chat/sessions', methods=['POST'])
def create_session():
    """
    Create a new chat session.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          properties:
            userId:
              type: string
            title:
              type: string
    responses:
      201:
        description: Session created successfully.
      500:
        description: Error creating session.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
    
    data = request.get_json() or {}
    user_id = data.get("userId", "anonymous")
    title = data.get("title", "New Chat")
    
    try:
        db_name = os.environ.get("MONGODB_DB_NAME", "zufan_legal")
        client = vector_store._collection.database.client
        db = client[db_name]
        
        doc = {
            "userId": user_id,
            "title": title,
            "createdAt": datetime.datetime.utcnow(),
            "updatedAt": datetime.datetime.utcnow()
        }
        result = db.chat_sessions.insert_one(doc)
        
        return jsonify({
            "sessionId": str(result.inserted_id),
            "title": title,
            "createdAt": doc["createdAt"].isoformat()
        }), 201
    except Exception as e:
        logger.error(f"Create session error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/sessions', methods=['GET'])
def list_sessions():
    """
    List chat sessions for a specific user.
    ---
    parameters:
      - name: userId
        in: query
        type: string
        required: true
    responses:
      200:
        description: List of chat sessions.
      400:
        description: Missing userId.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
    
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "Missing userId param"}), 400
        
    try:
        db_name = os.environ.get("MONGODB_DB_NAME", "zufan_legal")
        client = vector_store._collection.database.client
        db = client[db_name]
        
        cursor = db.chat_sessions.find({"userId": user_id}).sort("updatedAt", -1)
        sessions = []
        for doc in cursor:
            sessions.append({
                "id": str(doc["_id"]),
                "title": doc.get("title", "Untitled"),
                "createdAt": doc["createdAt"].isoformat(),
                "updatedAt": doc["updatedAt"].isoformat()
            })
            
        return jsonify(sessions), 200
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/sessions/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """
    Get message history for a specific chat session.
    ---
    parameters:
      - name: session_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: List of messages in the session.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        db_name = os.environ.get("MONGODB_DB_NAME", "zufan_legal")
        client = vector_store._collection.database.client
        db = client[db_name]
        
        cursor = db.chat_messages.find({"sessionId": session_id}).sort("createdAt", 1)
        messages = []
        for doc in cursor:
            messages.append({
                "id": str(doc["_id"]),
                "role": doc["role"],
                "content": doc["content"],
                "createdAt": doc["createdAt"].isoformat()
            })
            
        return jsonify(messages), 200
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """
    Delete a specific chat session and its messages.
    ---
    parameters:
      - name: session_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Session deleted successfully.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        db_name = os.environ.get("MONGODB_DB_NAME", "zufan_legal")
        client = vector_store._collection.database.client
        db = client[db_name]
        
        db.chat_sessions.delete_one({"_id": ObjectId(session_id)})
        db.chat_messages.delete_many({"sessionId": session_id})
            
        return jsonify({"message": "Session deleted"}), 200
    except Exception as e:
        logger.error(f"Delete session error: {e}")
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
        collection = vector_store._collection
        
        # 1. Total Vectors (Chunks)
        total_vectors = collection.count_documents({})
        
        # 2. Total Documents (Unique Sources)
        pipeline = [{"$group": {"_id": "$metadata.source"}}, {"$count": "count"}]
        docs_res = list(collection.aggregate(pipeline))
        total_docs = docs_res[0]["count"] if docs_res else 0
        
        # 3. Index Size (Estimate or Real if available via command)
        # Using db.command("collStats") is standard but often restricted in Atlas free tier or requires specific permissions
        # We'll try a simple fallback: avg doc size * count or just raw collStats if allowed
        index_size_mb = 0
        try:
            stats = collection.database.command("collStats", collection.name)
            index_size_mb = stats.get("totalSize", 0) / (1024 * 1024) # Bytes to MB
        except:
            # Fallback invalid
            index_size_mb = -1
            
        return jsonify({
            "total_vectors": total_vectors,
            "total_documents": total_docs,
            "index_size_mb": round(index_size_mb, 2),
            "dimensions": 768, # ge-multilingual-e5-large is 768
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
        collection = vector_store._collection
        # Assuming we don't have a 'createdAt' on chunks unless we add it during indexing.
        # If standard chunks don't have it, we might be sorting by implicit insertion order or _id
        
        cursor = collection.find({}).sort("_id", -1).limit(10)
        recent = []
        for doc in cursor:
            # content field depends on LangChain Mongo schema usually 'text' or 'page_content'
            text = doc.get("text") or doc.get("page_content") or ""
            metadata = doc.get("metadata", {})
            
            recent.append({
                "id": str(doc["_id"]),
                "source": metadata.get("source", "Unknown"),
                "text_snippet": text[:50] + "..." if len(text) > 50 else text,
                "tokens": int(len(text) / 4), # rough est
                "indexed_at": doc["_id"].generation_time.isoformat() # ObjectId has timestamp
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
    ---
    responses:
      200:
        description: Vector store cleared successfully.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        collection = vector_store._collection
        res = collection.delete_many({})
        return jsonify({
            "message": "Vector store purged successfully",
            "deleted_count": res.deleted_count
        }), 200
    except Exception as e:
        logger.error(f"Clear all error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/vectors/<vector_id>', methods=['DELETE'])
def delete_single_vector(vector_id):
    """
    Delete a single vector by its MongoDB ID.
    ---
    parameters:
      - name: vector_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Vector deleted successfully.
      404:
        description: Vector not found.
    """
    if not vector_store:
        return jsonify({"error": "Database not initialized"}), 503
        
    try:
        collection = vector_store._collection
        res = collection.delete_one({"_id": ObjectId(vector_id)})
        
        if res.deleted_count > 0:
            return jsonify({"message": f"Vector '{vector_id}' deleted"}), 200
        else:
            return jsonify({"error": "Vector not found"}), 404
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
    if not data or "chunks" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    chunks = data["chunks"]
    doc_id = data.get("documentId") # Can be added to metadata

    try:
        # Ensure metadata has documentId if provided
        for chunk in chunks:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            if doc_id:
                chunk["metadata"]["documentId"] = doc_id
        
        count = index_text_chunks(vector_store, chunks)
        
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
    ---
    responses:
      200:
        description: List of indexed documents.
    """
    if not vector_store:
        return jsonify({"error": "Vector store not initialized"}), 503

    try:
        # We need to access the underlying collection to perform aggregation
        # vector_store._collection is available in LangChain MongoDB Atlas integration
        collection = vector_store._collection
        
        pipeline = [
            {"$group": {
                "_id": "$metadata.source",
                "count": {"$sum": 1},
                "total_chars": {"$sum": "$metadata.page_char_count"}
            }},
            {"$project": {
                "name": "$_id",
                "chunk_count": "$count",
                "size_estimate": "$total_chars",
                "_id": 0
            }}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Transform to match Data Model in README vaguely (Document interface)
        documents = []
        for res in results:
            documents.append({
                "id": res.get("name"), # using source name as ID for now
                "name": res.get("name"),
                "type": "PDF", # Assumed
                "status": "Indexed",
                "chunks": res.get("chunk_count")
            })

        return jsonify(documents), 200

    except Exception as e:
        logger.error(f"List documents error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<path:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """
    Delete a document and all its associated vectors from the vector store.
    ---
    parameters:
      - name: doc_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Document deleted successfully.
    """
    if not vector_store:
        return jsonify({"error": "Vector store not initialized"}), 503
    
    try:
        # rag_utils.delete_document_by_source uses generic delete
        # We assume doc_id is the source name
        success = delete_document_by_source(vector_store, doc_id)
        
        if success:
            return jsonify({"message": f"Document '{doc_id}' deleted"}), 200
        else:
            return jsonify({"error": "Delete failed"}), 400

    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
