# server/app/services/rag_service.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Union # Import Union
import os
import re
import uuid
import logging
import io

from ..config import settings

logger = logging.getLogger(__name__)

# --- Initialize ChromaDB client and embedding function globally ---
# CORRECTED TYPE HINT for CHROMA_CLIENT
CHROMA_CLIENT: Optional[Union[chromadb.HttpClient, chromadb.PersistentClient]] = None
SENTENCE_TRANSFORMER_EF: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None

try:
    CHROMA_CLIENT = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
    logger.info(f"RAGService: ChromaDB PersistentClient initialized at '{settings.CHROMA_PERSIST_DIRECTORY}'. Available collections: {CHROMA_CLIENT.list_collections()}")
except Exception as e:
    logger.critical(f"RAGService: Failed to initialize ChromaDB PersistentClient at '{settings.CHROMA_PERSIST_DIRECTORY}': {e}", exc_info=True)
    CHROMA_CLIENT = None 

try:
    if CHROMA_CLIENT: 
        SENTENCE_TRANSFORMER_EF = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.DEFAULT_EMBEDDING_MODEL
        )
        logger.info(f"RAGService: SentenceTransformerEmbeddingFunction initialized with model '{settings.DEFAULT_EMBEDDING_MODEL}'.")
    else:
        SENTENCE_TRANSFORMER_EF = None
        logger.warning("RAGService: Chroma client not initialized, so SentenceTransformer EF was not loaded.")
except Exception as e:
    logger.critical(f"RAGService: Failed to initialize SentenceTransformerEmbeddingFunction with model '{settings.DEFAULT_EMBEDDING_MODEL}': {e}", exc_info=True)
    SENTENCE_TRANSFORMER_EF = None

def is_rag_service_ready() -> bool:
    """Checks if the core components of the RAG service are initialized."""
    ready = True
    if not CHROMA_CLIENT:
        logger.error("RAGService Check: ChromaDB client is NOT initialized.")
        ready = False
    if not SENTENCE_TRANSFORMER_EF:
        logger.error("RAGService Check: SentenceTransformer embedding function is NOT initialized.")
        ready = False
    if ready:
        logger.debug("RAGService Check: All components ready.")
    return ready

def _sanitize_collection_name(name: str, max_len: int = 63) -> str:
    if not name: name = f"col_{uuid.uuid4().hex[:max_len-4]}"
    name = name.lower()
    name = re.sub(r"[^a-z0-9_-]", "_", name) 
    name = re.sub(r"__+", "_", name) 
    name = re.sub(r"--+", "-", name)
    name = re.sub(r"[_.-]+$", "", name) 
    name = re.sub(r"^[_.-]+", "", name)  
    if len(name) < 3: name = f"{name}{uuid.uuid4().hex[:(3-len(name))]}" 
    if len(name) > max_len: name = name[:max_len]
    if not name or not name[0].isalnum(): name = f"c{name[1:]}" if len(name) > 1 else f"c{name}"
    if len(name) > max_len : name = name[:max_len] 
    if not name or not name[-1].isalnum(): name = name[:-1] + "e" if len(name) > 1 else f"{name}e"
    if len(name) > max_len : name = name[:max_len] 
    if len(name) < 3: name = f"col_{uuid.uuid4().hex[:max_len-4]}"
    return name[:max_len]

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Return text extracted from uploaded bytes based on file extension."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in {".txt", ".md"}:
        try:
            return file_bytes.decode("utf-8")
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")

    if ext == ".docx":
        try:
            from docx import Document  # type: ignore
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"RAGService: Failed to read DOCX '{filename}': {e}")
            return ""

    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(io.BytesIO(file_bytes))
            texts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    texts.append(page_text)
            return "\n".join(texts)
        except Exception as e:
            logger.error(f"RAGService: Failed to read PDF '{filename}': {e}")
            return ""

    if ext in {".xls", ".xlsx"}:
        try:
            import openpyxl  # type: ignore
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
            rows = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    values = [str(c) if c is not None else "" for c in row]
                    row_text = " ".join(values).strip()
                    if row_text:
                        rows.append(row_text)
            return "\n".join(rows)
        except Exception as e:
            logger.error(f"RAGService: Failed to read Excel '{filename}': {e}")
            return ""

    logger.warning(f"RAGService: Unsupported file extension '{ext}' for '{filename}'. Attempting text decode.")
    return file_bytes.decode("utf-8", errors="ignore")

def get_or_create_project_collection(project_id: str) -> Optional[chromadb.api.models.Collection.Collection]:
    if not is_rag_service_ready():
        logger.error(f"RAGService: Cannot get/create collection for project '{project_id}' - service not ready.")
        return None
    collection_name = _sanitize_collection_name(f"project_{project_id}")
    try:
        collection = CHROMA_CLIENT.get_or_create_collection(
            name=collection_name,
            embedding_function=SENTENCE_TRANSFORMER_EF 
        )
        logger.info(f"RAGService: Accessed/Created ChromaDB collection '{collection.name}' (ID: {collection.id}) for project_id '{project_id}'.")
        return collection
    except Exception as e:
        logger.error(f"RAGService: Failed to get/create ChromaDB collection '{collection_name}' for project '{project_id}': {e}", exc_info=True)
        return None

def add_document_to_project_collection(
    project_id: str, 
    document_text: str, 
    document_id: str, 
    metadata: Optional[Dict[str, Any]] = None
):
    collection = get_or_create_project_collection(project_id)
    if not collection:
        logger.error(f"RAGService: Could not get collection for project '{project_id}'. Document '{document_id}' not added.")
        return

    chunks = [chunk.strip() for chunk in document_text.split("\n\n") if chunk.strip()]
    if not chunks:
        logger.warning(f"RAGService: Document '{document_id}' for project '{project_id}' had no processable chunks.")
        return

    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    chunk_metadatas = []
    base_metadata = metadata.copy() if metadata else {}
    base_metadata["original_document_id"] = document_id 
    for i in range(len(chunks)):
        meta_for_chunk = base_metadata.copy()
        meta_for_chunk["chunk_index"] = i
        chunk_metadatas.append(meta_for_chunk)

    try:
        collection.upsert(ids=chunk_ids, documents=chunks, metadatas=chunk_metadatas)
        logger.info(f"RAGService: Upserted {len(chunks)} chunks from document '{document_id}' to project '{project_id}' collection '{collection.name}'.")
    except Exception as e:
        logger.error(f"RAGService: Failed to upsert document '{document_id}' for project '{project_id}': {e}", exc_info=True)

def query_project_collection(project_id: str, query_text: str, n_results: int = settings.RAG_TOP_K) -> List[str]:
    collection = get_or_create_project_collection(project_id)
    if not collection: return []
    
    collection_count = collection.count()
    if collection_count == 0: 
        logger.info(f"RAGService: Collection for project '{project_id}' is empty. No RAG results for query.")
        return []
    
    actual_n_results = min(n_results, collection_count)
    if actual_n_results <= 0: return []

    try:
        results = collection.query(query_texts=[query_text], n_results=actual_n_results)
        retrieved_chunks: List[str] = results["documents"][0] if results and results.get("documents") and results["documents"][0] else []
        logger.debug(f"RAGService: Retrieved {len(retrieved_chunks)} chunks for query in project '{project_id}'.")
        return retrieved_chunks
    except Exception as e:
        logger.error(f"RAGService: Error querying collection for project '{project_id}': {e}", exc_info=True)
        return []

def seed_documents_from_directory(project_id: str, seed_dir_path: str):
    if not is_rag_service_ready():
        logger.error(f"RAGService: Cannot seed documents because RAG service components are not ready.")
        return

    logger.info(f"RAGService: Attempting to seed documents for project '{project_id}' from directory '{seed_dir_path}'...")
    if not os.path.exists(seed_dir_path) or not os.path.isdir(seed_dir_path):
        logger.warning(f"RAGService: Seed directory '{seed_dir_path}' not found or is not a directory.")
        return

    seeded_count = 0
    for filename in os.listdir(seed_dir_path):
        file_path = os.path.join(seed_dir_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    bytes_data = f.read()
                content = extract_text_from_bytes(bytes_data, filename)
                if not content.strip():
                    continue
                doc_id_base = os.path.splitext(filename)[0]
                add_document_to_project_collection(
                    project_id=project_id,
                    document_text=content,
                    document_id=doc_id_base,
                    metadata={"source_file": filename}
                )
                seeded_count += 1
            except Exception as e:
                logger.error(
                    f"RAGService: Failed to seed document '{filename}' from '{file_path}': {e}",
                    exc_info=True,
                )
    
    if seeded_count > 0:
        logger.info(f"RAGService: Seeded {seeded_count} documents for project '{project_id}'.")
    else:
        logger.info(
            f"RAGService: No suitable documents found for project '{project_id}' in '{seed_dir_path}'."
        )

