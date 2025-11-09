"""
Test to reproduce URL truncation issue (T088).

Issue: Counter-research citations showing "http://arxi" instead of full arXiv URLs.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.paper_discovery import search_arxiv, normalize_arxiv_url
from src.tools.counter_research import papers_to_conflicting_evidence
from src.models import PaperMetadata


def test_arxiv_url_extraction():
    """Test that arXiv URLs are extracted correctly and not truncated."""
    # Search for a simple query
    papers = search_arxiv("neural networks", max_results=2)
    
    assert len(papers) > 0, "Should find at least one paper"
    
    for paper in papers:
        url = paper.url
        print(f"Paper URL: {url}")
        
        # Check that URL is complete
        assert url.startswith("http://arxiv.org/abs/") or url.startswith("https://arxiv.org/abs/"), \
            f"Invalid arXiv URL format: {url}"
        
        # Check that URL is not truncated
        assert len(url) > 20, f"URL appears truncated: {url} (length: {len(url)})"
        
        # Check that URL contains paper ID
        assert "/abs/" in url, f"URL missing /abs/ path: {url}"


def test_normalize_arxiv_url():
    """Test URL normalization doesn't truncate."""
    test_urls = [
        "http://arxiv.org/abs/2301.12345v1",
        "https://export.arxiv.org/abs/2301.12345",
        "http://arxiv.org/pdf/2301.12345.pdf",
    ]
    
    for test_url in test_urls:
        normalized = normalize_arxiv_url(test_url)
        print(f"Original: {test_url}")
        print(f"Normalized: {normalized}")
        
        # Should not truncate
        assert len(normalized) >= 20, f"Normalized URL appears truncated: {normalized}"
        assert "arxiv.org/abs/" in normalized, f"Normalized URL missing key parts: {normalized}"


def test_conflicting_evidence_url_storage():
    """Test that ConflictingEvidence stores full URLs."""
    # Create test papers with full arXiv URLs
    test_papers = [
        PaperMetadata(
            title="Test Paper on Neural Networks Architecture",
            url="http://arxiv.org/abs/2301.12345",
            abstract="This is a test abstract that meets the minimum length requirement of 50 characters for validation.",
            authors=["Test Author"],
            published="2023-01-15",
            source="arxiv",
            citation_count=10,
            fields_of_study=["Computer Science"]
        )
    ]
    
    # Convert to ConflictingEvidence
    conflicting = papers_to_conflicting_evidence(test_papers)
    
    assert len(conflicting) == 1, "Should create one ConflictingEvidence"
    
    evidence_url = conflicting[0].source_url
    print(f"ConflictingEvidence URL: {evidence_url}")
    
    # Check that URL is complete in ConflictingEvidence
    assert evidence_url == "http://arxiv.org/abs/2301.12345", \
        f"URL was altered: expected 'http://arxiv.org/abs/2301.12345', got '{evidence_url}'"
    
    assert len(evidence_url) > 20, f"URL in ConflictingEvidence appears truncated: {evidence_url}"


if __name__ == "__main__":
    print("=" * 80)
    print("T088: Testing URL Truncation Issue")
    print("=" * 80)
    print()
    
    print("Test 1: arXiv URL Extraction")
    print("-" * 80)
    try:
        test_arxiv_url_extraction()
        print("✅ PASSED: arXiv URLs extracted correctly")
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
    except Exception as e:
        print(f"⚠️  ERROR: {e}")
    print()
    
    print("Test 2: URL Normalization")
    print("-" * 80)
    try:
        test_normalize_arxiv_url()
        print("✅ PASSED: URL normalization works correctly")
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
    print()
    
    print("Test 3: ConflictingEvidence URL Storage")
    print("-" * 80)
    try:
        test_conflicting_evidence_url_storage()
        print("✅ PASSED: URLs stored correctly in ConflictingEvidence")
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
    except Exception as e:
        print(f"⚠️  ERROR: {e}")
    print()
    
    print("=" * 80)
    print("Test suite complete")
    print("=" * 80)

