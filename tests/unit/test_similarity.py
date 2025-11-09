"""
Unit tests for Tier 1 US2: Similarity utilities (T052)

Tests:
1. Identical claims score >0.95
2. Different claims score <0.50
3. Near-duplicates score 0.75-0.90
4. Threshold 0.80 works correctly for circular argument detection
"""

from src.utils.similarity import compute_similarity, is_circular_argument, SIMILARITY_THRESHOLD


def test_identical_claims_high_similarity():
    """T052: Test that identical claims have similarity >0.95"""
    claim = "Consciousness emerges from integrated information in neural networks"
    
    similarity = compute_similarity(claim, claim)
    
    assert similarity > 0.95, f"Identical claims should have similarity >0.95, got {similarity:.3f}"
    print(f"✅ Identical claims: {similarity:.3f} (>0.95)")


def test_different_claims_low_similarity():
    """T052: Test that completely different claims have similarity <0.50"""
    claim1 = "Consciousness emerges from integrated information in neural networks"
    claim2 = "Quantum mechanics explains the double-slit experiment through wave-particle duality"
    
    similarity = compute_similarity(claim1, claim2)
    
    assert similarity < 0.50, f"Different claims should have similarity <0.50, got {similarity:.3f}"
    print(f"✅ Different claims: {similarity:.3f} (<0.50)")


def test_near_duplicate_claims_medium_similarity():
    """T052: Test that near-duplicate claims have similarity 0.75-0.95"""
    claim1 = "Consciousness emerges from integrated information in neural networks"
    claim2 = "Neural networks exhibit consciousness through integrated information processing"
    
    similarity = compute_similarity(claim1, claim2)
    
    assert 0.75 <= similarity <= 0.95, f"Near-duplicates should have similarity 0.75-0.95, got {similarity:.3f}"
    print(f"✅ Near-duplicates: {similarity:.3f} (0.75-0.95)")


def test_threshold_works_correctly():
    """T052: Test that SIMILARITY_THRESHOLD (0.80) correctly identifies circular arguments"""
    # Highly similar claim (should be flagged as circular)
    high_similarity_claim = "Neural networks exhibit consciousness through integrated information"
    rejected_claims = ["Consciousness emerges from integrated information in neural networks"]
    
    is_circular, most_similar_claim, similarity = is_circular_argument(high_similarity_claim, rejected_claims)
    
    assert is_circular, f"High similarity ({similarity:.3f}) should be flagged as circular (threshold: {SIMILARITY_THRESHOLD})"
    assert similarity >= SIMILARITY_THRESHOLD, f"Flagged claim should exceed threshold: {similarity:.3f} >= {SIMILARITY_THRESHOLD}"
    assert most_similar_claim is not None, "Should return the most similar claim"
    print(f"✅ Circular argument detected: {similarity:.3f} (>={SIMILARITY_THRESHOLD})")
    
    # Moderately similar claim (should NOT be flagged)
    moderate_similarity_claim = "Consciousness might arise from complex information processing"
    
    is_circular2, most_similar_claim2, similarity2 = is_circular_argument(moderate_similarity_claim, rejected_claims)
    
    sim_str = f"{similarity2:.3f}" if similarity2 is not None else "N/A"
    assert not is_circular2, f"Moderate similarity ({sim_str}) should NOT be flagged as circular (threshold: {SIMILARITY_THRESHOLD})"
    print(f"✅ Non-circular argument: {sim_str} (<{SIMILARITY_THRESHOLD})")


def test_empty_rejected_claims():
    """T052: Test that empty rejected claims list returns False"""
    claim = "Any claim"
    rejected_claims = []
    
    is_circular, most_similar_claim, similarity = is_circular_argument(claim, rejected_claims)
    
    assert not is_circular, "Empty rejected claims should return False"
    assert most_similar_claim is None, "Empty rejected claims should return None"
    assert similarity is None, "Empty rejected claims should return None similarity"
    print(f"✅ Empty rejected claims: is_circular={is_circular}, similarity={similarity}")


def test_multiple_rejected_claims_finds_max():
    """T052: Test that multiple rejected claims returns max similarity"""
    new_claim = "Neural networks process information to generate consciousness"
    rejected_claims = [
        "Quantum mechanics explains wave-particle duality",  # Low similarity
        "Consciousness emerges from neural network information processing",  # High similarity
        "Machine learning uses gradient descent optimization"  # Low similarity
    ]
    
    is_circular, most_similar_claim, max_similarity = is_circular_argument(new_claim, rejected_claims)
    
    # Should find the high similarity with the second claim
    assert max_similarity > 0.75, f"Should find high similarity claim, got {max_similarity:.3f}"
    assert "Consciousness emerges from neural network information processing" in most_similar_claim
    print(f"✅ Max similarity from multiple claims: {max_similarity:.3f}")


def test_paraphrased_claims():
    """T052: Test that paraphrased claims are correctly identified"""
    claim1 = "AI systems cannot truly understand language"
    claim2 = "Artificial intelligence lacks genuine comprehension of linguistic meaning"
    
    similarity = compute_similarity(claim1, claim2)
    
    # Paraphrased claims should have high similarity (typically 0.75-0.85)
    assert 0.70 <= similarity <= 0.90, f"Paraphrased claims should have similarity 0.70-0.90, got {similarity:.3f}"
    print(f"✅ Paraphrased claims: {similarity:.3f} (0.70-0.90)")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Tier 1 US2 Similarity Unit Tests (T052)")
    print("=" * 70)
    
    test_identical_claims_high_similarity()
    test_different_claims_low_similarity()
    test_near_duplicate_claims_medium_similarity()
    test_threshold_works_correctly()
    test_empty_rejected_claims()
    test_multiple_rejected_claims_finds_max()
    test_paraphrased_claims()
    
    print()
    print("=" * 70)
    print("🎉 ALL SIMILARITY TESTS PASSED!")
    print("=" * 70)

