# server/app/services/rag_service.py
import chromadb
from chromadb.utils import embedding_functions # For SentenceTransformer EF
from typing import List, Dict, Any, Optional
import os
import re # For basic collection name sanitization
import uuid # Added for fallback in _sanitize_collection_name
import logging
from ..config import settings

logger = logging.getLogger(__name__)

# Initialize ChromaDB client with persistence
# This client will be used by all functions in this service.
try:
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
    logger.info(f"RAGService: ChromaDB PersistentClient initialized at '{settings.CHROMA_PERSIST_DIRECTORY}'.")
except Exception as e:
    logger.error(f"RAGService: Failed to initialize ChromaDB PersistentClient at '{settings.CHROMA_PERSIST_DIRECTORY}': {e}")
    # You might want to raise the exception or handle it in a way that prevents app startup if Chroma is critical
    chroma_client = None 

# Initialize the embedding function using sentence-transformers
# This will download the model on first use if it's not already cached by sentence-transformers.
try:
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.DEFAULT_EMBEDDING_MODEL
    )
    logger.info(f"RAGService: SentenceTransformerEmbeddingFunction initialized with model '{settings.DEFAULT_EMBEDDING_MODEL}'.")
except Exception as e:
    logger.error(f"RAGService: Failed to initialize SentenceTransformerEmbeddingFunction with model '{settings.DEFAULT_EMBEDDING_MODEL}': {e}")
    sentence_transformer_ef = None 

def _sanitize_collection_name(name: str) -> str:
    """Basic sanitization for ChromaDB collection names."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9_.-]", "_", name) # Replace invalid chars with underscore
    name = re.sub(r"__+", "_", name)          # Replace multiple underscores
    name = re.sub(r"^\.+", "", name)           # Remove leading dots
    name = re.sub(r"\.+\Z", "", name)         # Remove trailing dots
    if not (2 < len(name) < 64):
        # If too short/long, hash it or create a more robust slug. For now, truncate/pad.
        # Ensure uuid is imported if this fallback is used.
        new_name = f"collection_{uuid.uuid4().hex[:8]}"
        logger.warning(f"Sanitized collection name '{name}' was invalid length, using fallback: {new_name}")
        name = new_name
    if name and (not name[0].isalnum() or not name[-1].isalnum()): # Check if name is not empty
        original_name = name
        if not name[0].isalnum() and len(name) > 1:
            name = "c" + name[1:]
        elif not name[0].isalnum() and len(name) == 1:
            name = "c" + name + "e" # handle single non-alphanum char
        
        if not name[-1].isalnum() and len(name) > 1:
            name = name[:-1] + "e"
        elif not name[-1].isalnum() and len(name) == 1 and original_name != name : # if it became single char and still not alnum
             name = name + "e" # e.g. was "_" -> "c_" -> "c_e"
        elif not name[-1].isalnum() and len(name) == 1: # e.g. was "c"
            name = name + "e"


        logger.warning(f"Sanitized collection name '{original_name}' had non-alphanumeric start/end, adjusted to: {name}")

    return name[:63] # Final length check


def get_or_create_project_collection(project_id: str) -> Optional[chromadb.api.models.Collection.Collection]:
    """
    Gets or creates a ChromaDB collection for a specific project.
    The collection name will be derived from the project_id.
    """
    if not chroma_client or not sentence_transformer_ef:
        logger.error("RAGService: ChromaDB client or embedding function not initialized.")
        return None

    collection_name = _sanitize_collection_name(f"project_{project_id}")
    
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        logger.info(f"RAGService: Accessed/Created ChromaDB collection '{collection.name}' (ID: {collection.id}) for project_id '{project_id}'.")
        return collection
    except Exception as e: # Catches various ChromaDB errors including invalid names after sanitization
        logger.error(f"RAGService: Failed to get/create ChromaDB collection '{collection_name}' for project '{project_id}': {e}")
        # Consider specific error handling for chromadb.errors.InvalidCollectionNameError if sanitization isn't perfect
        return None


def add_document_to_project_collection(
    project_id: str, 
    document_text: str, 
    document_id: str, # e.g., filename or a unique ID for the source document
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Chunks, embeds, and adds a document to the specified project's ChromaDB collection.
    """
    collection = get_or_create_project_collection(project_id)
    if not collection:
        logger.error(f"RAGService: Could not get collection for project '{project_id}'. Document '{document_id}' not added.")
        return

    # Basic chunking strategy: split by double newlines (paragraphs)
    # For more sophisticated chunking, use libraries like LangChain's text_splitters
    chunks = [chunk.strip() for chunk in document_text.split("\n\n") if chunk.strip()]
    
    if not chunks:
        logger.warning(f"RAGService: Document '{document_id}' for project '{project_id}' resulted in no processable chunks after splitting.")
        return

    # Create unique IDs for each chunk based on the document ID and chunk index
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    
    # Prepare metadata for each chunk
    chunk_metadatas = []
    base_metadata = metadata.copy() if metadata else {}
    base_metadata["original_document_id"] = document_id 
    for i in range(len(chunks)):
        meta_for_chunk = base_metadata.copy()
        meta_for_chunk["chunk_index"] = i
        chunk_metadatas.append(meta_for_chunk)

    try:
        # Add documents (will be embedded by the collection's embedding_function)
        collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=chunk_metadatas
        )
        logger.info(f"RAGService: Added {len(chunks)} chunks from document '{document_id}' to project '{project_id}' collection '{collection.name}'.")
    except chromadb.api.errors.IDAlreadyExistsError:
        logger.warning(f"RAGService: Document/chunk IDs from '{document_id}' already exist in project '{project_id}'. Skipping or consider update logic.")
        # For simplicity, we're not implementing upsert here. collection.upsert() could be used.
    except Exception as e:
        logger.error(f"RAGService: Failed to add document '{document_id}' to collection for project '{project_id}': {e}")


def query_project_collection(project_id: str, query_text: str, n_results: int = settings.RAG_TOP_K) -> List[str]:
    """
    Queries the project's ChromaDB collection and returns the text of the most relevant document chunks.
    """
    collection = get_or_create_project_collection(project_id)
    if not collection:
        return []
    
    # Ensure n_results is not greater than the number of items in the collection
    collection_count = collection.count()
    if collection_count == 0:
        logger.info(f"RAGService: Collection for project '{project_id}' is empty. No RAG results.")
        return []
    
    actual_n_results = min(n_results, collection_count)
    if actual_n_results <= 0: # Should only happen if n_results was 0 or negative
        return []

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=actual_n_results
            # You can add metadata filters here later: where={"source_type": "manual_upload"}
        )
        
        retrieved_chunks: List[str] = []
        if results and results.get("documents") and results["documents"][0]:
            retrieved_chunks = results["documents"][0]
            # logger.info(f"RAGService: Retrieved {len(retrieved_chunks)} chunks for query in project '{project_id}'.")
        return retrieved_chunks
    except Exception as e:
        logger.error(f"RAGService: Error querying collection for project '{project_id}': {e}")
        return []


def seed_documents_from_directory(db_session, project_id: str, seed_dir_path: str): # Added db_session if needed for logging to audit
    """
    Scans a directory for text-based files and adds them to the specified project's RAG.
    """
    if not (chroma_client and sentence_transformer_ef): # Check if Chroma components initialized
        logger.error("RAGService: ChromaDB client or embedding function not ready for seeding.")
        return

    logger.info(f"RAGService: Attempting to seed documents for project '{project_id}' from directory '{seed_dir_path}'...")
    if not os.path.exists(seed_dir_path):
        logger.warning(f"RAGService: Seed directory '{seed_dir_path}' not found.")
        return

    seeded_count = 0
    for filename in os.listdir(seed_dir_path):
        file_path = os.path.join(seed_dir_path, filename)
        if os.path.isfile(file_path) and (filename.endswith(".txt") or filename.endswith(".md")): # Process .txt and .md
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use filename (without extension) as a base for document_id
                doc_id_base = os.path.splitext(filename)[0]
                
                add_document_to_project_collection(
                    project_id=project_id,
                    document_text=content,
                    document_id=doc_id_base, # Using filename as document ID
                    metadata={"source_file": filename}
                )
                seeded_count +=1
            except Exception as e:
                logger.error(f"RAGService: Failed to seed document '{filename}' for project '{project_id}': {e}")
    
    if seeded_count > 0:
        logger.info(f"RAGService: Successfully seeded {seeded_count} documents for project '{project_id}'.")
    else:
        logger.info(f"RAGService: No new documents found or seeded for project '{project_id}' from '{seed_dir_path}'.")