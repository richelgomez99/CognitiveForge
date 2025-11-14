"""
ChromaDB vector store for semantic memory.

Provides:
- Paper embedding storage
- Semantic similarity search
- Session query embeddings
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper for ChromaDB operations.
    """

    def __init__(self, persist_directory: str = "./chroma_db", embedding_dim: int = 1024):
        """
        Initialize ChromaDB client with persistence.

        Args:
            persist_directory: Path to ChromaDB storage
            embedding_dim: Dimension of embeddings (1024 for E5-large-v2)
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_dim = embedding_dim

        # Delete existing collections if they have wrong dimensions
        try:
            existing = self.client.get_collection("papers")
            if existing.count() > 0:
                sample = existing.peek(limit=1)
                if sample['embeddings'] and len(sample['embeddings'][0]) != embedding_dim:
                    logger.warning(f"Deleting 'papers' collection (dimension mismatch)")
                    self.client.delete_collection("papers")
        except Exception:
            pass  # Collection doesn't exist

        try:
            existing = self.client.get_collection("sessions")
            if existing.count() > 0:
                sample = existing.peek(limit=1)
                if sample['embeddings'] and len(sample['embeddings'][0]) != embedding_dim:
                    logger.warning(f"Deleting 'sessions' collection (dimension mismatch)")
                    self.client.delete_collection("sessions")
        except Exception:
            pass

        # Create collections with manual embeddings (no default embedding function)
        # This allows us to use E5-large-v2 (1024-dim) instead of default MiniLM (384-dim)
        self.papers = self.client.get_or_create_collection(
            name="papers",
            metadata={
                "description": "Paper embeddings (title + abstract)",
                "embedding_dim": str(embedding_dim),
                "model": "intfloat/e5-large-v2"
            },
            embedding_function=None  # Manual embeddings
        )

        self.sessions = self.client.get_or_create_collection(
            name="sessions",
            metadata={
                "description": "Session query embeddings",
                "embedding_dim": str(embedding_dim),
                "model": "intfloat/e5-large-v2"
            },
            embedding_function=None  # Manual embeddings
        )

        logger.info(f"✅ ChromaDB initialized: {persist_directory}")
        logger.info(f"   Papers: {self.papers.count()} documents")
        logger.info(f"   Sessions: {self.sessions.count()} documents")

    def add_paper(self, paper_url: str, title: str, abstract: str, metadata: Dict = None):
        """
        Add paper embedding to ChromaDB using default embedding.

        DEPRECATED: This method uses ChromaDB's default embedding which is 384-dim.
        Use add_paper_with_embedding() with E5-large-v2 embeddings (1024-dim) instead.

        Args:
            paper_url: Unique identifier (URL)
            title: Paper title
            abstract: Paper abstract
            metadata: Additional metadata (authors, year, etc.)
        """
        raise NotImplementedError(
            "add_paper() is deprecated. Use add_paper_with_embedding() with E5-large-v2 "
            "embeddings (1024-dim). This ensures consistent embedding dimensions across "
            "the vector store."
        )

    def add_paper_with_embedding(self, paper_url: str, title: str, abstract: str,
                                 embedding: List[float], metadata: Dict = None):
        """
        Add paper with pre-computed embedding (e.g., from E5-large-v2).

        Args:
            paper_url: Unique identifier (URL)
            title: Paper title
            abstract: Paper abstract
            embedding: Pre-computed embedding vector
            metadata: Additional metadata
        """
        text = f"{title}. {abstract}"

        self.papers.add(
            ids=[paper_url],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )

        logger.debug(f"Added paper with custom embedding: {title[:50]}...")

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search papers by semantic similarity.

        DEPRECATED: This method requires ChromaDB's default embedding function.
        Use semantic_search_with_embedding() with E5-large-v2 embeddings instead.

        Args:
            query: Natural language query
            top_k: Number of results

        Returns:
            List of dicts with {id, distance, metadata}
        """
        raise NotImplementedError(
            "semantic_search() is deprecated. Use semantic_search_with_embedding() with "
            "pre-computed E5-large-v2 query embeddings (1024-dim)."
        )

    def semantic_search_with_embedding(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """
        Search papers using pre-computed query embedding.

        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of results

        Returns:
            List of dicts with {id, distance, metadata}
        """
        results = self.papers.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        papers = []
        for i in range(len(results['ids'][0])):
            papers.append({
                'url': results['ids'][0][i],
                'distance': results['distances'][0][i] if results.get('distances') else None,
                'metadata': results['metadatas'][0][i]
            })

        return papers


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
