import os
import certifi
import fitz  # PyMuPDF
import tiktoken
from tqdm.auto import tqdm
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient

# --- Text Processing ---

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# --- PDF Reading ---

def open_and_read_pdf(pdf_path: str, source: str = None) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number + 1,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split("፡፡")), # Amharic specific
            "page_token_count": len(text) / 4, # Fallback estimate
            "source": source or os.path.basename(pdf_path),
            "text": text
        })
    return pages_and_texts

# --- Custom Gemini Wrapper (If needed, otherwise use ChatGoogleGenerativeAI) ---

class GeminiLLM(Runnable):
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.3):
        import google.generativeai as genai
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    def invoke(self, input, config=None):
        if isinstance(input, dict) and "messages" in input:
            prompt_str = "\n".join(m.content for m in input["messages"])
        else:
            prompt_str = str(input)

        response = self.model.generate_content(
            prompt_str,
            generation_config={"temperature": self.temperature}
        )
        return response.text

# --- RAG Pipeline ---

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_chroma_vector_store(persist_directory="./chroma_db", collection_name="embeddings"):
    """
    Initializes ChromaDB Vector Store with local persistence.
    """
    # Using local HuggingFace embeddings
    # Note: multilingual-e5-large expects "query: " and "passage: " prefixes
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def get_rag_chain(vector_store, llm, prompt):
    """
    Creates a LangChain RAG chain.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# --- Document & Chunk Management ---

def index_text_chunks(vector_store, chunks: list[dict]):
    """
    Directly indexes pre-processed text chunks.
    Expected chunk format: {"text": "...", "metadata": {...}}
    """
    from langchain_core.documents import Document as LcDocument
    documents = [
        LcDocument(page_content=c["text"], metadata=c.get("metadata", {}))
        for c in chunks
    ]
    vector_store.add_documents(documents)
    return len(documents)

def delete_document_by_source(vector_store, source_name: str):
    """
    Deletes all vectors associated with a specific source in ChromaDB.
    """
    # Chroma allows deleting by metadata filter
    try:
        vector_store.delete(where={"source": source_name})
        return True
    except Exception as e:
        return False

def index_processed_chunks(vector_store, amh_pages_and_chunks: list[dict]):
    """
    Directly indexes chunks in the format provided by the user.
    Format: [{'page_number': 1, 'source': '...', 'sentence_chunk': '...'}, ...]
    """
    from langchain_core.documents import Document as LcDocument
    
    documents = [
        LcDocument(
            page_content=doc["sentence_chunk"], 
            metadata={
                'source': doc['source'],
                'page': doc['page_number']
            }
        ) 
        for doc in amh_pages_and_chunks 
        if doc.get("sentence_chunk")
    ]
    
    if documents:
        vector_store.add_documents(documents)
    
    return len(documents)
