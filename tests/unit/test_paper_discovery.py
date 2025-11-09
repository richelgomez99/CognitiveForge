"""
Unit tests for Tier 1 US1: Paper Discovery with claim_id and deduplication (T038)

Tests:
1. claim_id parameter is accepted and logged
2. Deduplication works across keywords
3. Papers found by multiple keywords are tracked
"""

import asyncio
import uuid
from src.tools.paper_discovery import discover_papers_for_query, discover_papers_for_keywords_parallel


def test_claim_id_parameter_accepted():
    """T038: Test that claim_id parameter is accepted"""
    query = "neural networks"
    claim_id = str(uuid.uuid4())
    
    # Should not raise exception
    papers = discover_papers_for_query(query, max_results_per_source=2, claim_id=claim_id)
    
    # Should return list of papers
    assert isinstance(papers, list)
    # (Papers may be empty if rate limited, but function should complete)


def test_deduplication_across_keywords():
    """T038: Test that deduplication works when same paper found by multiple keywords"""
    # Use related keywords that might find overlapping papers
    keywords = [
        "neural networks deep learning",
        "deep learning architectures",
        "convolutional neural networks"
    ]
    claim_id = str(uuid.uuid4())
    
    # Discover papers
    unique_papers, url_to_keywords = asyncio.run(
        discover_papers_for_keywords_parallel(
            keywords=keywords,
            max_results_per_keyword=3,
            claim_id=claim_id
        )
    )
    
    # Check that we got results
    assert isinstance(unique_papers, list)
    assert isinstance(url_to_keywords, dict)
    
    # Check that url_to_keywords maps URLs to keyword lists
    for url, kws in url_to_keywords.items():
        assert isinstance(kws, list)
        assert len(kws) >= 1
        assert all(kw in keywords for kw in kws)
    
    # Papers found by multiple keywords should appear once in unique_papers
    # but have multiple keywords in url_to_keywords
    multi_keyword_papers = [url for url, kws in url_to_keywords.items() if len(kws) > 1]
    
    # If any duplicates were found, verify they're only in unique_papers once
    if multi_keyword_papers:
        paper_urls = [p.url for p in unique_papers]
        for url in multi_keyword_papers:
            # Should appear exactly once in unique_papers
            assert paper_urls.count(url) == 1, f"Paper {url} appears multiple times in unique_papers"


def test_parallel_discovery_returns_correct_structure():
    """T038: Test that parallel discovery returns correct data structures"""
    keywords = ["quantum computing", "quantum algorithms"]
    claim_id = str(uuid.uuid4())
    
    unique_papers, url_to_keywords = asyncio.run(
        discover_papers_for_keywords_parallel(
            keywords=keywords,
            max_results_per_keyword=2,
            claim_id=claim_id
        )
    )
    
    # Check types
    assert isinstance(unique_papers, list)
    assert isinstance(url_to_keywords, dict)
    
    # Check that each paper in unique_papers has its URL in url_to_keywords
    for paper in unique_papers:
        assert paper.url in url_to_keywords
        
        # Check that the keywords for this paper are valid
        paper_keywords = url_to_keywords[paper.url]
        assert isinstance(paper_keywords, list)
        assert len(paper_keywords) >= 1
        assert all(kw in keywords for kw in paper_keywords)


def test_empty_keywords_list():
    """T038: Test graceful handling of empty keywords list"""
    keywords = []
    claim_id = str(uuid.uuid4())
    
    unique_papers, url_to_keywords = asyncio.run(
        discover_papers_for_keywords_parallel(
            keywords=keywords,
            max_results_per_keyword=3,
            claim_id=claim_id
        )
    )
    
    # Should return empty results gracefully
    assert unique_papers == []
    assert url_to_keywords == {}


def test_max_results_per_keyword_respected():
    """T038: Test that max_results_per_keyword limit is respected"""
    keywords = ["machine learning"]
    claim_id = str(uuid.uuid4())
    max_per_keyword = 2
    
    unique_papers, url_to_keywords = asyncio.run(
        discover_papers_for_keywords_parallel(
            keywords=keywords,
            max_results_per_keyword=max_per_keyword,
            claim_id=claim_id
        )
    )
    
    # Should not exceed max_per_keyword * len(keywords) * 2 (for 2 sources)
    # (May be less due to deduplication)
    max_possible = max_per_keyword * len(keywords) * 2
    assert len(unique_papers) <= max_possible


if __name__ == "__main__":
    print("=" * 70)
    print("Running Tier 1 US1 Paper Discovery Unit Tests (T038)")
    print("=" * 70)
    
    test_claim_id_parameter_accepted()
    print("✅ Test 1: claim_id parameter accepted")
    
    test_deduplication_across_keywords()
    print("✅ Test 2: Deduplication across keywords")
    
    test_parallel_discovery_returns_correct_structure()
    print("✅ Test 3: Parallel discovery returns correct structure")
    
    test_empty_keywords_list()
    print("✅ Test 4: Empty keywords list")
    
    test_max_results_per_keyword_respected()
    print("✅ Test 5: Max results per keyword respected")
    
    print()
    print("=" * 70)
    print("🎉 ALL PAPER DISCOVERY TESTS PASSED!")
    print("=" * 70)

