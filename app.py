import os
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
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
        # using ChatGoogleGenerativeAI for better streaming support
        # Switching to gemini-1.5-flash for better Free Tier availability
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
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

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "rag_chain_initialized": rag_chain is not None
    }
    return jsonify(status), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint.
    Request: { "messages": [{"role": "user", "content": "..."}] }
    Supports streaming response.
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
    
    # We could use the history, but rag_utils' simple chain expects 'question' input.
    # For a simple RAG, we stick to the last query.
    # Todo: advanced history handling if needed.

    try:
        # Check if client accepts streaming
        # For simplicity, we default to streaming if possible or just stream always and let client handle
        
        def generate():
            try:
                # Streaming with LangChain
                for chunk in rag_chain.stream(query):
                    yield chunk
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                yield f"Error: {str(e)}"

        return Response(stream_with_context(generate()), content_type='text/plain')

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload/file', methods=['POST'])
def upload_file():
    """
    Upload PDF, extract text, and index.
    Request: Multipart Form (file, metadata as json string optional)
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
    Directly index chunks.
    Request: { "documentId": string, "chunks": { "text": string, "metadata": object }[] }
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
    List documents.
    Note: vector search doesn't natively support "list unique documents" efficiently without aggregation.
    We will attempt to aggregate by 'source' metadata field.
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
    Delete document by ID (source name).
    Using <path:doc_id> to capture filenames with extensions.
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

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)
