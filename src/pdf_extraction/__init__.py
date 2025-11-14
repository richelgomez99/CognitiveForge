"""
PDF Extraction Pipeline - Phase 2.

Provides:
- PyMuPDF-based text extraction (42ms/page)
- Semantic chunking (512-1024 tokens)
- Error handling for scanned PDFs
- Batch processing
"""

from .extractor import PDFExtractor
from .chunker import SemanticChunker
from .pipeline import PDFPipeline

__all__ = ['PDFExtractor', 'SemanticChunker', 'PDFPipeline']
