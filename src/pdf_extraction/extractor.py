"""
PyMuPDF-based PDF text extraction.

Target performance: 42ms/page (based on Phase 1 research)
F1 score: 0.97 (text extraction accuracy)
"""

import fitz  # PyMuPDF
import logging
from typing import Dict, List, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Wrapper for PyMuPDF text extraction with performance monitoring.

    Features:
    - Fast text extraction (42ms/page target)
    - Metadata extraction (title, authors, year)
    - Page-level extraction tracking
    - Error handling for corrupted/scanned PDFs
    """

    def __init__(self):
        self.total_pages_processed = 0
        self.total_time_ms = 0.0

    def extract(self, pdf_path: str) -> Dict:
        """
        Extract text and metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with:
                - text: Full extracted text
                - pages: List of page texts
                - metadata: PDF metadata (title, author, etc.)
                - page_count: Number of pages
                - extraction_time_ms: Time taken per page
                - is_scanned: Whether PDF appears to be scanned

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF is corrupted or unreadable
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        start_time = time.time()

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}")

        # Extract metadata
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'created': doc.metadata.get('creationDate', ''),
            'modified': doc.metadata.get('modDate', ''),
        }

        # Extract text page by page
        pages = []
        full_text = []
        total_chars = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append(text)
            full_text.append(text)
            total_chars += len(text)

        doc.close()

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000

        # Update stats
        page_count = len(pages)
        self.total_pages_processed += page_count
        self.total_time_ms += elapsed_ms

        # Heuristic: if average chars per page < 100, likely scanned
        avg_chars_per_page = total_chars / page_count if page_count > 0 else 0
        is_scanned = avg_chars_per_page < 100

        result = {
            'text': '\n\n'.join(full_text),
            'pages': pages,
            'metadata': metadata,
            'page_count': page_count,
            'extraction_time_ms': elapsed_ms,
            'time_per_page_ms': elapsed_ms / page_count if page_count > 0 else 0,
            'is_scanned': is_scanned,
            'avg_chars_per_page': avg_chars_per_page,
            'file_path': str(pdf_path),
        }

        if is_scanned:
            logger.warning(
                f"PDF appears to be scanned ({avg_chars_per_page:.0f} chars/page): {pdf_path.name}"
            )
        else:
            logger.info(
                f"Extracted {page_count} pages in {elapsed_ms:.1f}ms "
                f"({result['time_per_page_ms']:.1f}ms/page): {pdf_path.name}"
            )

        return result

    def get_stats(self) -> Dict:
        """
        Get extraction performance statistics.

        Returns:
            Dict with total_pages, total_time_ms, avg_time_per_page_ms
        """
        avg_time = (
            self.total_time_ms / self.total_pages_processed
            if self.total_pages_processed > 0 else 0
        )

        return {
            'total_pages_processed': self.total_pages_processed,
            'total_time_ms': self.total_time_ms,
            'avg_time_per_page_ms': avg_time,
        }
