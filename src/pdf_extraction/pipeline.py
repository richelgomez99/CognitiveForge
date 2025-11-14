"""
End-to-end PDF extraction pipeline.

Integrates:
- PDF extraction (PyMuPDF)
- Semantic chunking
- ChromaDB storage
- Error handling
- Batch processing
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import traceback

from .extractor import PDFExtractor
from .chunker import SemanticChunker

logger = logging.getLogger(__name__)


class PDFPipeline:
    """
    Complete PDF processing pipeline.

    Features:
    - Extract → Chunk → Embed → Store
    - Error handling for scanned/corrupted PDFs
    - Batch processing with progress tracking
    - OCR fallback (placeholder for future)
    """

    def __init__(
        self,
        extractor: Optional[PDFExtractor] = None,
        chunker: Optional[SemanticChunker] = None,
    ):
        """
        Initialize pipeline.

        Args:
            extractor: PDF extractor (default: PDFExtractor())
            chunker: Semantic chunker (default: SemanticChunker())
        """
        self.extractor = extractor or PDFExtractor()
        self.chunker = chunker or SemanticChunker()
        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'scanned_pdfs': 0,
            'total_chunks': 0,
        }

    def process_pdf(
        self,
        pdf_path: str,
        store_embeddings: bool = False,
        embedding_model = None
    ) -> Dict:
        """
        Process single PDF through full pipeline.

        Args:
            pdf_path: Path to PDF file
            store_embeddings: Whether to generate and store embeddings
            embedding_model: Optional embedding model (e.g., SentenceTransformer)

        Returns:
            Dict with:
                - success: bool
                - pdf_path: str
                - chunks: List of chunk dicts
                - metadata: PDF metadata
                - error: Optional error message
                - is_scanned: bool
        """
        pdf_path = Path(pdf_path)
        self.stats['total_pdfs'] += 1

        try:
            # Step 1: Extract text
            logger.info(f"Extracting: {pdf_path.name}")
            extraction = self.extractor.extract(str(pdf_path))

            # Check if scanned
            if extraction['is_scanned']:
                self.stats['scanned_pdfs'] += 1
                logger.warning(f"Scanned PDF detected: {pdf_path.name}")
                # TODO: Add OCR fallback in future
                return {
                    'success': False,
                    'pdf_path': str(pdf_path),
                    'chunks': [],
                    'metadata': extraction['metadata'],
                    'error': 'PDF appears to be scanned. OCR not yet implemented.',
                    'is_scanned': True,
                }

            # Step 2: Chunk text
            logger.info(f"Chunking: {pdf_path.name}")
            metadata = {
                'file_path': str(pdf_path),
                'file_name': pdf_path.name,
                'page_count': extraction['page_count'],
                **extraction['metadata'],
            }

            chunks = self.chunker.chunk_pages(extraction['pages'], metadata)
            self.stats['total_chunks'] += len(chunks)

            # Step 3: Generate embeddings (if requested)
            if store_embeddings and embedding_model:
                logger.info(f"Generating embeddings for {len(chunks)} chunks")
                for chunk in chunks:
                    # Prefix for E5 model: "passage: <text>"
                    text_with_prefix = f"passage: {chunk['text']}"
                    embedding = embedding_model.encode(text_with_prefix, normalize_embeddings=True)
                    chunk['embedding'] = embedding.tolist()

            self.stats['successful'] += 1

            return {
                'success': True,
                'pdf_path': str(pdf_path),
                'chunks': chunks,
                'metadata': metadata,
                'error': None,
                'is_scanned': False,
                'extraction_stats': {
                    'page_count': extraction['page_count'],
                    'time_per_page_ms': extraction['time_per_page_ms'],
                    'avg_chars_per_page': extraction['avg_chars_per_page'],
                },
            }

        except FileNotFoundError as e:
            self.stats['failed'] += 1
            logger.error(f"File not found: {pdf_path}")
            return {
                'success': False,
                'pdf_path': str(pdf_path),
                'chunks': [],
                'metadata': {},
                'error': str(e),
                'is_scanned': False,
            }

        except ValueError as e:
            self.stats['failed'] += 1
            logger.error(f"Invalid PDF: {pdf_path} - {e}")
            return {
                'success': False,
                'pdf_path': str(pdf_path),
                'chunks': [],
                'metadata': {},
                'error': f"Invalid PDF: {e}",
                'is_scanned': False,
            }

        except Exception as e:
            self.stats['failed'] += 1
            logger.error(f"Unexpected error processing {pdf_path}: {e}")
            logger.debug(traceback.format_exc())
            return {
                'success': False,
                'pdf_path': str(pdf_path),
                'chunks': [],
                'metadata': {},
                'error': f"Unexpected error: {e}",
                'is_scanned': False,
            }

    def process_batch(
        self,
        pdf_paths: List[str],
        store_embeddings: bool = False,
        embedding_model = None,
        progress_callback = None
    ) -> List[Dict]:
        """
        Process multiple PDFs in batch.

        Args:
            pdf_paths: List of PDF file paths
            store_embeddings: Whether to generate embeddings
            embedding_model: Optional embedding model
            progress_callback: Optional callback(current, total, pdf_name)

        Returns:
            List of processing results (one per PDF)
        """
        results = []
        total = len(pdf_paths)

        logger.info(f"Starting batch processing: {total} PDFs")

        for i, pdf_path in enumerate(pdf_paths, 1):
            if progress_callback:
                progress_callback(i, total, Path(pdf_path).name)

            result = self.process_pdf(pdf_path, store_embeddings, embedding_model)
            results.append(result)

            # Log progress
            if i % 10 == 0 or i == total:
                logger.info(
                    f"Progress: {i}/{total} PDFs "
                    f"({self.stats['successful']} successful, {self.stats['failed']} failed)"
                )

        logger.info(f"Batch complete: {self.get_stats()}")
        return results

    def store_chunks_to_chromadb(
        self,
        chunks: List[Dict],
        collection_name: str = "pdf_chunks",
        vector_store = None
    ) -> int:
        """
        Store chunks with embeddings to ChromaDB.

        Args:
            chunks: List of chunk dicts (must have 'embedding' field)
            collection_name: ChromaDB collection name
            vector_store: Optional VectorStore instance

        Returns:
            Number of chunks stored
        """
        if not vector_store:
            from src.memory.vector_store import get_vector_store
            vector_store = get_vector_store()

        # Get or create collection for PDF chunks
        collection = vector_store.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "PDF chunk embeddings",
                "embedding_dim": "1024",
                "model": "intfloat/e5-large-v2"
            },
            embedding_function=None
        )

        stored_count = 0
        for chunk in chunks:
            if 'embedding' not in chunk:
                logger.warning(f"Chunk {chunk.get('chunk_index')} missing embedding, skipping")
                continue

            # Create unique ID: file_path + chunk_index
            chunk_id = f"{chunk['metadata'].get('file_name', 'unknown')}__chunk_{chunk['chunk_index']}"

            try:
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk['text']],
                    embeddings=[chunk['embedding']],
                    metadatas=[{
                        'chunk_index': chunk['chunk_index'],
                        'token_count': chunk['token_count'],
                        'page_range': chunk['metadata'].get('page_range', ''),
                        'file_name': chunk['metadata'].get('file_name', ''),
                        'file_path': chunk['metadata'].get('file_path', ''),
                    }]
                )
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store chunk {chunk_id}: {e}")

        logger.info(f"Stored {stored_count}/{len(chunks)} chunks to ChromaDB")
        return stored_count

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful'] / self.stats['total_pdfs']
                if self.stats['total_pdfs'] > 0 else 0
            ),
            'avg_chunks_per_pdf': (
                self.stats['total_chunks'] / self.stats['successful']
                if self.stats['successful'] > 0 else 0
            ),
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'total_pdfs': 0,
            'successful': 0,
            'failed': 0,
            'scanned_pdfs': 0,
            'total_chunks': 0,
        }
