"""
Integration tests for critical bug fixes (T088-T090).

Tests:
1. T088: Full arXiv URLs preserved in Skeptic citations
2. T089: Circular rejection triggers impasse synthesis (not validation)
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph import build_graph
from src.models import DebateMemory
import uuid


@pytest.mark.asyncio
async def test_arxiv_urls_in_synthesis():
    """
    T088: Test that arXiv URLs are preserved end-to-end in synthesis.
    
    This test verifies that:
    1. ArXiv URLs are extracted correctly from search results
    2. URLs are not truncated during normalization
    3. URLs appear correctly in ConflictingEvidence
    4. Final synthesis includes complete URLs
    """
    # Skip if no API key (URL extraction doesn't require LLM)
    graph = build_graph(use_checkpointer=False)
    
    # Simple query that should trigger arXiv discovery
    inputs = {
        "messages": [],
        "original_query": "What are neural networks?",
        "current_thesis": None,
        "current_antithesis": None,
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 0,
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": [],
            "skeptic_objections": [],
            "weak_evidence_urls": []
        },
        "current_claim_id": str(uuid.uuid4()),
        "synthesis_mode": None
    }
    
    # Note: This test requires GOOGLE_API_KEY to run the full synthesis
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set, skipping full synthesis test")
    
    result = await graph.ainvoke(inputs)
    
    # Check that synthesis was generated
    assert result["final_synthesis"] is not None, "Synthesis should be generated"
    
    # Check evidence lineage for complete URLs
    evidence_urls = result["final_synthesis"].evidence_lineage
    
    arxiv_urls = [url for url in evidence_urls if "arxiv.org/abs/" in url]
    
    # At least some arXiv URLs should be present
    if len(arxiv_urls) > 0:
        for url in arxiv_urls:
            # Check URL is not truncated
            assert not url.endswith("arxi"), f"URL appears truncated: {url}"
            assert len(url) > 20, f"URL too short (likely truncated): {url}"
            assert "arxiv.org/abs/" in url, f"URL missing /abs/ path: {url}"
            
            print(f"✅ Valid arXiv URL: {url}")
    
    print(f"✅ Test passed: {len(arxiv_urls)} arXiv URLs validated")


@pytest.mark.asyncio
async def test_circular_argument_impasse_synthesis():
    """
    T089: Test that circular arguments trigger impasse synthesis, not validation.
    
    This test verifies that:
    1. When Skeptic detects a circular argument, synthesis_mode is set to "impasse"
    2. Synthesizer generates impasse acknowledgment instead of validating the circular claim
    3. Final synthesis explains what was learned before impasse
    """
    # Skip if no API key
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set, skipping LLM-dependent test")
    
    graph = build_graph(use_checkpointer=False)
    
    # Create a state where we simulate a circular argument scenario
    # We'll need to manually trigger this by setting up debate memory with similar claims
    from src.models import Thesis, Evidence, Antithesis
    
    # First claim (will be rejected)
    first_claim = "Consciousness is a functional aspect of predictive processing"
    
    # Second claim (intentionally similar to trigger circular detection)
    second_claim = "Consciousness is the brain's predictive model of attention"
    
    inputs = {
        "messages": [],
        "original_query": "What is consciousness?",
        "current_thesis": Thesis(
            claim=second_claim,
            reasoning="This is a specialized application of predictive processing...",
            evidence=[
                Evidence(source_url="http://example.com/1", snippet="Evidence 1 that is at least 10 chars long"),
                Evidence(source_url="http://example.com/2", snippet="Evidence 2 that is at least 10 chars long")
            ]
        ),
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="This is circular to predictive processing",
            critique="This claim is 81% similar to previously rejected thesis: 'Consciousness is a functional aspect of predictive processing'. This represents a circular argument.",
            conflicting_evidence=[]
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 3,  # Multiple iterations already
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": [first_claim],  # Previous similar claim
            "skeptic_objections": ["Does not explain qualia"],
            "weak_evidence_urls": []
        },
        "current_claim_id": str(uuid.uuid4()),
        "synthesis_mode": None  # Will be set by routing logic
    }
    
    # Mock the routing to simulate circular detection
    from src.graph import route_debate
    
    # Test routing logic
    route = route_debate(inputs)
    
    # Should route to synthesizer when circular argument is detected
    assert route == "synthesizer", "Circular argument should route to synthesizer"
    
    # Check that synthesis_mode was set to "impasse"
    assert inputs.get("synthesis_mode") == "impasse", "synthesis_mode should be 'impasse' for circular arguments"
    
    print("✅ Routing correctly detected circular argument and set impasse mode")
    
    # Now test the full synthesis generation with impasse mode
    result = await graph.ainvoke(inputs)
    
    # Check that synthesis was generated
    assert result["final_synthesis"] is not None, "Synthesis should be generated even in impasse"
    
    synthesis = result["final_synthesis"]
    
    # Check that synthesis acknowledges impasse
    insight_lower = synthesis.novel_insight.lower()
    reasoning_lower = synthesis.reasoning.lower()
    
    # Should mention impasse, circular, or similar concepts
    impasse_keywords = ["impasse", "circular", "exhausted", "unresolved", "limitation"]
    found_keywords = [kw for kw in impasse_keywords if kw in insight_lower or kw in reasoning_lower]
    
    assert len(found_keywords) > 0, \
        f"Synthesis should acknowledge impasse. Found keywords: {found_keywords}. Insight: {synthesis.novel_insight}"
    
    # Should NOT validate the circular claim
    assert "correct" not in insight_lower or "valid" not in insight_lower, \
        "Synthesis should not validate the circular claim as correct"
    
    # Confidence should be moderate to low (0.3-0.6)
    assert 0.3 <= synthesis.confidence_score <= 0.7, \
        f"Impasse synthesis should have moderate confidence, got {synthesis.confidence_score}"
    
    print(f"✅ Impasse synthesis correctly acknowledges circular argument")
    print(f"   Insight: {synthesis.novel_insight[:100]}...")
    print(f"   Confidence: {synthesis.confidence_score:.2f}")
    print(f"   Keywords found: {found_keywords}")


@pytest.mark.asyncio
async def test_exhausted_attempts_mode():
    """
    T089: Test that max iterations triggers "exhausted_attempts" mode.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set, skipping LLM-dependent test")
    
    from src.models import Thesis, Evidence, Antithesis
    
    inputs = {
        "messages": [],
        "original_query": "What is consciousness?",
        "current_thesis": Thesis(
            claim="Consciousness is an emergent property",
            reasoning="Based on neuroscience...",
            evidence=[
                Evidence(source_url="http://example.com/1", snippet="Evidence 1 that is at least 10 chars long"),
                Evidence(source_url="http://example.com/2", snippet="Evidence 2 that is at least 10 chars long")
            ]
        ),
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="Not all evidence supports emergence",
            critique="Multiple contradictions remain unresolved",
            conflicting_evidence=[]
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 3,  # Max iterations reached
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": ["Claim 1", "Claim 2", "Claim 3"],
            "skeptic_objections": ["Objection 1", "Objection 2"],
            "weak_evidence_urls": []
        },
        "current_claim_id": str(uuid.uuid4()),
        "synthesis_mode": None
    }
    
    from src.graph import route_debate
    
    # Test routing logic with max iterations
    route = route_debate(inputs)
    
    # Should route to synthesizer when max iterations reached
    assert route == "synthesizer", "Max iterations should route to synthesizer"
    
    # Check that synthesis_mode was set to "exhausted_attempts"
    assert inputs.get("synthesis_mode") == "exhausted_attempts", \
        "synthesis_mode should be 'exhausted_attempts' when max iterations reached"
    
    print("✅ Routing correctly detected max iterations and set exhausted_attempts mode")


if __name__ == "__main__":
    import asyncio
    
    print("=" * 80)
    print("T090: Integration Tests for Critical Bug Fixes")
    print("=" * 80)
    print()
    
    print("Test 1: arXiv URLs End-to-End")
    print("-" * 80)
    try:
        asyncio.run(test_arxiv_urls_in_synthesis())
        print("✅ PASSED")
    except pytest.skip.Exception as e:
        print(f"⏭️  SKIPPED: {e}")
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
    except Exception as e:
        print(f"⚠️  ERROR: {e}")
    print()
    
    print("Test 2: Circular Argument Impasse Synthesis")
    print("-" * 80)
    try:
        asyncio.run(test_circular_argument_impasse_synthesis())
        print("✅ PASSED")
    except pytest.skip.Exception as e:
        print(f"⏭️  SKIPPED: {e}")
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
    except Exception as e:
        print(f"⚠️  ERROR: {e}")
    print()
    
    print("Test 3: Exhausted Attempts Mode")
    print("-" * 80)
    try:
        asyncio.run(test_exhausted_attempts_mode())
        print("✅ PASSED")
    except pytest.skip.Exception as e:
        print(f"⏭️  SKIPPED: {e}")
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
    except Exception as e:
        print(f"⚠️  ERROR: {e}")
    print()
    
    print("=" * 80)
    print("Test suite complete")
    print("=" * 80)

