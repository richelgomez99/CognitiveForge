"""
Unit tests for Tier 1 US1: Keyword Extraction (T036)

Tests:
1. Valid claim returns 3-5 keywords
2. Keywords are multi-word phrases
3. LLM failure fallback works
"""

import os
import pytest
from src.tools.keyword_extraction import extract_keywords, _fallback_keywords
from src.models import KeywordExtractionOutput


def test_valid_claim_returns_3_to_5_keywords():
    """T036: Test that valid claim returns 3-5 keywords"""
    claim = "Consciousness emerges from integrated information in neural networks"
    reasoning = "IIT proposes consciousness arises from phi"
    
    result = extract_keywords(claim, reasoning)
    
    # Should be KeywordExtractionOutput
    assert isinstance(result, KeywordExtractionOutput)
    
    # Should have 3-5 keywords
    assert 3 <= len(result.keywords) <= 5, f"Expected 3-5 keywords, got {len(result.keywords)}"
    
    # Should have reasoning
    assert len(result.reasoning) > 0


def test_keywords_are_multi_word():
    """T036: Test that keywords are multi-word phrases (not single words)"""
    claim = "Quantum computing enables exponential speedup for certain algorithms"
    reasoning = "Shor's algorithm demonstrates quantum advantage"
    
    result = extract_keywords(claim, reasoning)
    
    # All keywords should be multi-word (at least 2 words)
    for keyword in result.keywords:
        word_count = len(keyword.split())
        assert word_count >= 2, f"Keyword '{keyword}' is not multi-word (only {word_count} words)"


def test_llm_failure_fallback_works():
    """T036: Test that LLM failure fallback returns valid KeywordExtractionOutput"""
    # Test fallback directly
    claim = "This is a test claim about neural networks"
    result = _fallback_keywords(claim)
    
    # Should be KeywordExtractionOutput
    assert isinstance(result, KeywordExtractionOutput)
    
    # Should have exactly 3 keywords (fallback behavior)
    assert len(result.keywords) == 3, f"Fallback should return 3 keywords, got {len(result.keywords)}"
    
    # Should have reasoning explaining fallback
    assert "Fallback" in result.reasoning or "fallback" in result.reasoning
    
    # Keywords should include the original claim
    assert any(claim in kw for kw in result.keywords)


def test_empty_claim_graceful_handling():
    """T036: Test graceful handling of edge cases"""
    # Very short claim
    short_claim = "AI"
    result = extract_keywords(short_claim, "minimal reasoning")
    
    # Should still return valid output (fallback)
    assert isinstance(result, KeywordExtractionOutput)
    assert 3 <= len(result.keywords) <= 5


def test_pydantic_validation():
    """T036: Test that Pydantic validation works correctly"""
    from pydantic import ValidationError
    
    # Valid data
    valid_data = {
        "keywords": ["neural networks consciousness", "integrated information theory", "emergent properties"],
        "reasoning": "These keywords target specific theoretical frameworks"
    }
    result = KeywordExtractionOutput(**valid_data)
    assert len(result.keywords) == 3
    
    # Invalid data: too few keywords
    with pytest.raises(ValidationError):
        KeywordExtractionOutput(
            keywords=["one", "two"],  # Only 2 keywords, need min 3
            reasoning="test"
        )
    
    # Invalid data: single-word keywords
    with pytest.raises(ValidationError):
        KeywordExtractionOutput(
            keywords=["consciousness", "neural", "information"],  # Single words
            reasoning="test"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Running Tier 1 US1 Keyword Extraction Unit Tests (T036)")
    print("=" * 70)
    
    test_valid_claim_returns_3_to_5_keywords()
    print("✅ Test 1: Valid claim returns 3-5 keywords")
    
    test_keywords_are_multi_word()
    print("✅ Test 2: Keywords are multi-word phrases")
    
    test_llm_failure_fallback_works()
    print("✅ Test 3: LLM failure fallback works")
    
    test_empty_claim_graceful_handling()
    print("✅ Test 4: Empty claim graceful handling")
    
    test_pydantic_validation()
    print("✅ Test 5: Pydantic validation")
    
    print()
    print("=" * 70)
    print("🎉 ALL KEYWORD EXTRACTION TESTS PASSED!")
    print("=" * 70)

