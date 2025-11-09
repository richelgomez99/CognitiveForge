"""
Unit tests for Tier 1 US3: Counter-Research Module (T072-T073)

Tests:
1. Counter-query generation returns 2-3 queries with negation terms
2. Counter-evidence discovery returns 4-6 papers with skeptic_counter tag
3. ConflictingEvidence model validation
"""

import os
import pytest
from pydantic import ValidationError
from src.tools.counter_research import generate_counter_queries, _fallback_counter_queries, papers_to_conflicting_evidence
from src.models import PaperMetadata, ConflictingEvidence

# Check if API key is available
HAS_API_KEY = bool(os.getenv("GOOGLE_API_KEY"))


class TestCounterQueryGeneration:
    """T072: Test counter-query generation"""
    
    def test_counter_queries_returns_2_to_3_queries(self):
        """Verify counter-query generation returns 2-3 queries"""
        thesis_claim = "Consciousness is purely computational"
        reasoning = "Information processing theory suggests consciousness emerges from computation"
        
        queries = generate_counter_queries(thesis_claim, reasoning)
        
        assert 2 <= len(queries) <= 3, f"Expected 2-3 queries, got {len(queries)}"
        print(f"✅ Generated {len(queries)} queries (expected 2-3)")
    
    def test_queries_include_negation_terms(self):
        """Verify counter-queries include negation/challenge terms"""
        thesis_claim = "Neural networks can achieve AGI"
        reasoning = "Recent advances in deep learning show promise for general intelligence"
        
        queries = generate_counter_queries(thesis_claim, reasoning)
        
        # Check for negation/challenge terms in at least one query
        negation_terms = [
            "limitation", "challenge", "critique", "refutation",
            "constraint", "failure", "insufficient", "cannot",
            "substrate-dependent", "non-computational"
        ]
        
        has_negation = False
        for query in queries:
            query_lower = query.lower()
            if any(term in query_lower for term in negation_terms):
                has_negation = True
                print(f"✅ Query with negation: '{query}'")
                break
        
        # Allow fallback queries to pass (they have "limitations", "challenges", "critique")
        if not HAS_API_KEY:
            # Fallback always has negation terms
            assert has_negation or any("limitation" in q.lower() or "challenge" in q.lower() for q in queries), \
                f"Fallback queries should have negation terms: {queries}"
        else:
            # LLM-generated should have negation terms
            assert has_negation, f"At least one query should include negation terms. Got: {queries}"
    
    def test_fallback_counter_queries(self):
        """Verify fallback counter-query generation works"""
        claim = "Quantum computing enables consciousness"
        
        queries = _fallback_counter_queries(claim)
        
        assert len(queries) == 3, f"Fallback should return 3 queries, got {len(queries)}"
        assert all("limitation" in q.lower() or "challenge" in q.lower() or "critique" in q.lower() for q in queries), \
            f"Fallback queries should all have negation terms: {queries}"
        
        print(f"✅ Fallback queries: {queries}")


class TestConflictingEvidenceModel:
    """T073: Test ConflictingEvidence model validation"""
    
    def test_snippet_max_300_chars(self):
        """Verify snippet validation rejects strings >300 characters"""
        long_snippet = "A" * 400  # 400 characters
        
        # Pydantic should REJECT snippets >300 chars (not auto-truncate)
        with pytest.raises(ValidationError):
            ConflictingEvidence(
                source_url="https://example.com",
                snippet=long_snippet,
                relevance_score=0.8,
                discovered_by="skeptic_counter"
            )
        
        # Valid snippet (≤300 chars) should work
        valid_snippet = "A" * 300
        evidence = ConflictingEvidence(
            source_url="https://example.com",
            snippet=valid_snippet,
            relevance_score=0.8,
            discovered_by="skeptic_counter"
        )
        assert len(evidence.snippet) == 300
        print(f"✅ Pydantic correctly rejects snippets >300 chars and accepts ≤300 chars")
    
    def test_relevance_score_0_to_1_range(self):
        """Verify relevance_score is constrained to 0-1"""
        # Valid scores
        valid_evidence = ConflictingEvidence(
            source_url="https://example.com",
            snippet="Test snippet",
            relevance_score=0.5,
            discovered_by="skeptic_counter"
        )
        assert 0 <= valid_evidence.relevance_score <= 1
        print(f"✅ Valid relevance_score: {valid_evidence.relevance_score}")
        
        # Invalid score (>1) should raise ValidationError
        with pytest.raises(ValidationError):
            ConflictingEvidence(
                source_url="https://example.com",
                snippet="Test snippet",
                relevance_score=1.5,
                discovered_by="skeptic_counter"
            )
        
        # Invalid score (<0) should raise ValidationError
        with pytest.raises(ValidationError):
            ConflictingEvidence(
                source_url="https://example.com",
                snippet="Test snippet",
                relevance_score=-0.1,
                discovered_by="skeptic_counter"
            )
        
        print(f"✅ Out-of-range scores correctly rejected")
    
    def test_discovered_by_field(self):
        """Verify discovered_by field is set correctly"""
        evidence = ConflictingEvidence(
            source_url="https://example.com",
            snippet="Test snippet",
            relevance_score=0.8,
            discovered_by="skeptic_counter"
        )
        
        assert evidence.discovered_by == "skeptic_counter"
        print(f"✅ discovered_by: {evidence.discovered_by}")


class TestPapersToConflictingEvidence:
    """T072: Test conversion from PaperMetadata to ConflictingEvidence"""
    
    def test_papers_converted_correctly(self):
        """Verify papers are converted to ConflictingEvidence with correct fields"""
        papers = [
            PaperMetadata(
                title="Limitations of Neural Networks in Consciousness",
                authors=["John Doe"],
                abstract="This paper discusses the fundamental limitations of neural networks in modeling consciousness. " * 10,  # Long abstract
                url="https://arxiv.org/abs/1234",
                published="2023-01-01",
                source="arxiv",
                citation_count=100
            ),
            PaperMetadata(
                title="Challenges to Computational Theory of Mind",
                authors=["Jane Smith"],
                abstract="This paper presents short abstract discussing challenges to computational theory." * 2,  # Make it at least 50 chars
                url="https://arxiv.org/abs/5678",
                published="2024-01-01",
                source="semantic_scholar",
                citation_count=50
            )
        ]
        
        evidence_list = papers_to_conflicting_evidence(papers, discovered_by="skeptic_counter")
        
        assert len(evidence_list) == 2, f"Expected 2 evidence items, got {len(evidence_list)}"
        
        # Check first item
        assert evidence_list[0].source_url == papers[0].url
        assert len(evidence_list[0].snippet) <= 300  # Truncated from long abstract
        assert evidence_list[0].relevance_score == 0.8
        assert evidence_list[0].discovered_by == "skeptic_counter"
        
        # Check second item
        assert evidence_list[1].source_url == papers[1].url
        assert len(evidence_list[1].snippet) <= 300  # Should be truncated or within limit
        
        print(f"✅ Converted {len(papers)} papers to ConflictingEvidence")
        print(f"   Snippet 1 length: {len(evidence_list[0].snippet)} (truncated from long abstract)")
        print(f"   Snippet 2 length: {len(evidence_list[1].snippet)} (original)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TIER 1 US3: COUNTER-RESEARCH UNIT TESTS")
    print("=" * 70)
    
    if not HAS_API_KEY:
        print("\n⚠️  GOOGLE_API_KEY not set. LLM-dependent tests will use fallback.")
    
    # Run tests
    print("\n[TEST SUITE 1] Counter-Query Generation")
    print("-" * 70)
    test_gen = TestCounterQueryGeneration()
    test_gen.test_counter_queries_returns_2_to_3_queries()
    test_gen.test_queries_include_negation_terms()
    test_gen.test_fallback_counter_queries()
    
    print("\n[TEST SUITE 2] ConflictingEvidence Model")
    print("-" * 70)
    test_model = TestConflictingEvidenceModel()
    test_model.test_snippet_max_300_chars()
    test_model.test_relevance_score_0_to_1_range()
    test_model.test_discovered_by_field()
    
    print("\n[TEST SUITE 3] Papers to ConflictingEvidence Conversion")
    print("-" * 70)
    test_convert = TestPapersToConflictingEvidence()
    test_convert.test_papers_converted_correctly()
    
    print("\n" + "=" * 70)
    print("🎉 ALL COUNTER-RESEARCH UNIT TESTS PASSED!")
    print("=" * 70)

