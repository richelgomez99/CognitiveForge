"""
Semantic chunking for PDF text.

Target: 512-1024 tokens per chunk with semantic boundaries
Strategy: Prefer section/paragraph boundaries over hard token cuts
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Chunk text into semantically coherent segments.

    Features:
    - Respects paragraph/section boundaries
    - Target chunk size: 512-1024 tokens
    - Overlap: 50 tokens for context continuity
    - Metadata tracking (chunk index, page range)
    """

    def __init__(
        self,
        min_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        overlap: int = 50
    ):
        """
        Initialize chunker.

        Args:
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text into semantic segments.

        Args:
            text: Full text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dicts with:
                - text: Chunk text
                - token_count: Approximate token count
                - chunk_index: Sequential index
                - start_char: Character offset in original text
                - end_char: End character offset
                - metadata: Copy of input metadata
        """
        if not text or not text.strip():
            return []

        # Split into paragraphs (common semantic boundaries)
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        char_offset = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            # If single paragraph exceeds max_chunk_size, split it
            if para_tokens > self.max_chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        len(chunks),
                        char_offset,
                        char_offset + len(chunk_text),
                        metadata
                    ))
                    char_offset += len(chunk_text) + 2  # +2 for \n\n
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = self._split_sentences(para)
                for sent in sentences:
                    sent_tokens = self._estimate_tokens(sent)

                    if current_tokens + sent_tokens > self.max_chunk_size and current_chunk:
                        # Create chunk
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append(self._create_chunk(
                            chunk_text,
                            len(chunks),
                            char_offset,
                            char_offset + len(chunk_text),
                            metadata
                        ))
                        char_offset += len(chunk_text) + 2
                        current_chunk = []
                        current_tokens = 0

                    current_chunk.append(sent)
                    current_tokens += sent_tokens

            # Normal paragraph handling
            elif current_tokens + para_tokens > self.max_chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        len(chunks),
                        char_offset,
                        char_offset + len(chunk_text),
                        metadata
                    ))
                    char_offset += len(chunk_text) + 2
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens

            else:
                # Add to current chunk
                current_chunk.append(para)
                current_tokens += para_tokens

        # Flush remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text,
                len(chunks),
                char_offset,
                char_offset + len(chunk_text),
                metadata
            ))

        logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
        return chunks

    def chunk_pages(self, pages: List[str], metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text from multiple pages, tracking page boundaries.

        Args:
            pages: List of page texts
            metadata: Optional metadata

        Returns:
            List of chunks with page_range metadata
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        page_start = 0

        for page_idx, page_text in enumerate(pages):
            paragraphs = self._split_paragraphs(page_text)

            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)

                if current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                    # Create chunk with page range
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_meta = metadata.copy() if metadata else {}
                    chunk_meta['page_range'] = f"{page_start}-{page_idx}"
                    chunk_meta['page_start'] = page_start
                    chunk_meta['page_end'] = page_idx

                    chunks.append(self._create_chunk(
                        chunk_text,
                        len(chunks),
                        0,  # Char offset not meaningful across pages
                        len(chunk_text),
                        chunk_meta
                    ))

                    current_chunk = [para]
                    current_tokens = para_tokens
                    page_start = page_idx
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta['page_range'] = f"{page_start}-{len(pages)-1}"
            chunk_meta['page_start'] = page_start
            chunk_meta['page_end'] = len(pages) - 1

            chunks.append(self._create_chunk(
                chunk_text,
                len(chunks),
                0,
                len(chunk_text),
                chunk_meta
            ))

        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or section headers
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter (can be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Heuristic: ~4 characters per token (conservative for scientific text)
        """
        return len(text) // 4

    def _create_chunk(
        self,
        text: str,
        index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict]
    ) -> Dict:
        """Create chunk dict."""
        return {
            'text': text,
            'token_count': self._estimate_tokens(text),
            'chunk_index': index,
            'start_char': start_char,
            'end_char': end_char,
            'metadata': metadata or {},
        }
