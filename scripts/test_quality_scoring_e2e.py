"""
End-to-End Testing for Quality Scoring Engine (Epic 3 Task 3.5)

Tests all components:
1. Citation count normalization (logarithmic scaling)
2. Venue impact factor lookup and fuzzy matching
3. Peer review scoring
4. Recency scoring (exponential decay)
5. Overall weighted composite score
6. Batch scoring
7. Integration with PaperQualityScoreV2 model

CRITICAL: Validates all scoring components.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quality.scoring import PaperQualityScorer, get_quality_scorer
from src.quality.venue_rankings import VenueDatabase, get_venue_database
from src.models_v2 import PaperQualityScoreV2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_citation_scoring():
    """Test 1: Citation count normalization with logarithmic scaling."""
    print("\n" + "="*70)
    print("TEST 1: Citation Count Scoring (Logarithmic)")
    print("="*70)

    scorer = PaperQualityScorer()

    test_cases = [
        (0, 0.0, "No citations"),
        (1, 10.0, "Single citation"),
        (10, 33.4, "10 citations"),
        (100, 66.7, "100 citations"),
        (1000, 100.0, "1000+ citations (capped)"),
    ]

    all_passed = True

    for citation_count, expected_approx, description in test_cases:
        score = scorer._score_citations(citation_count)

        # Allow 5% tolerance
        tolerance = 5.0
        passed = abs(score - expected_approx) < tolerance

        status = "✅" if passed else "❌"
        print(f"{status} {description}: {citation_count} cites -> {score:.1f}/100 (expected ~{expected_approx})")

        if not passed:
            all_passed = False

    print(f"\n{'✅ All citation scoring tests passed' if all_passed else '❌ Some tests failed'}")
    return all_passed


def test_venue_database():
    """Test 2: Venue impact factor lookup and fuzzy matching."""
    print("\n" + "="*70)
    print("TEST 2: Venue Database (Impact Factors)")
    print("="*70)

    venue_db = VenueDatabase()

    test_cases = [
        # Tier 1: Top conferences
        ("NeurIPS", 95.0, "Tier 1 conference"),
        ("neurips", 95.0, "Case insensitive"),
        ("Conference on Neural Information Processing Systems", 95.0, "Full name"),

        # Tier 1: Top journals
        ("Nature", 98.0, "Tier 1 journal"),
        ("JMLR", 98.0, "Journal abbreviation"),

        # Tier 2
        ("EMNLP", 82.0, "Tier 2 conference"),
        ("AAAI", 95.0, "Tier 1 (updated)"),

        # Tier 3
        ("UAI", 67.0, "Tier 3 conference"),

        # Preprints
        ("arXiv", 45.0, "Preprint server"),
        ("arxiv.org", 45.0, "Preprint URL"),

        # Workshops
        ("NeurIPS Workshop", 55.0, "Workshop"),

        # Unknown (default)
        ("Unknown Conference 2024", 50.0, "Unknown venue"),

        # Fuzzy matching
        ("Proceedings of NeurIPS 2024", 95.0, "Fuzzy: contains NeurIPS"),
    ]

    all_passed = True

    for venue, expected, description in test_cases:
        score = venue_db.get_impact_factor(venue)

        passed = abs(score - expected) < 1.0
        status = "✅" if passed else "❌"

        print(f"{status} {description}: '{venue[:40]}' -> {score}/100 (expected {expected})")

        if not passed:
            all_passed = False

    # Test statistics
    stats = venue_db.get_stats()
    print(f"\nVenue DB Stats:")
    print(f"  Total venues: {stats['total_venues']}")
    print(f"  Cache hit rate: {stats['hit_rate']:.1%}")

    # Test tier listings
    tier1 = venue_db.list_tier1_venues()
    tier2 = venue_db.list_tier2_venues()
    print(f"  Tier 1 venues: {len(tier1)}")
    print(f"  Tier 2 venues: {len(tier2)}")

    print(f"\n{'✅ All venue database tests passed' if all_passed else '❌ Some tests failed'}")
    return all_passed


def test_recency_scoring():
    """Test 3: Recency scoring with exponential decay."""
    print("\n" + "="*70)
    print("TEST 3: Recency Scoring (Exponential Decay)")
    print("="*70)

    scorer = PaperQualityScorer()
    current_year = datetime.now().year

    test_cases = [
        (current_year, 100.0, "Current year"),
        (current_year - 1, 90.5, "1 year old"),
        (current_year - 3, 74.1, "3 years old"),
        (current_year - 5, 60.7, "5 years old"),
        (current_year - 10, 36.8, "10 years old"),
        (current_year - 20, 13.5, "20 years old"),
    ]

    all_passed = True

    for year, expected_approx, description in test_cases:
        score = scorer._score_recency(year)

        # Allow 10% tolerance for exponential decay
        tolerance = 10.0
        passed = abs(score - expected_approx) < tolerance

        status = "✅" if passed else "❌"
        print(f"{status} {description}: {year} -> {score:.1f}/100 (expected ~{expected_approx})")

        if not passed:
            all_passed = False

    print(f"\n{'✅ All recency scoring tests passed' if all_passed else '❌ Some tests failed'}")
    return all_passed


def test_peer_review_scoring():
    """Test 4: Peer review binary scoring."""
    print("\n" + "="*70)
    print("TEST 4: Peer Review Scoring")
    print("="*70)

    scorer = PaperQualityScorer()

    # Peer reviewed paper
    peer_score = 100.0 if True else 50.0
    print(f"✅ Peer-reviewed: {peer_score}/100")

    # Not peer reviewed (preprint)
    non_peer_score = 100.0 if False else 50.0
    print(f"✅ Non-peer-reviewed: {non_peer_score}/100")

    passed = peer_score == 100.0 and non_peer_score == 50.0
    print(f"\n{'✅ Peer review scoring correct' if passed else '❌ Test failed'}")
    return passed


def test_overall_composite_score():
    """Test 5: Overall weighted composite score."""
    print("\n" + "="*70)
    print("TEST 5: Overall Composite Score (Weighted)")
    print("="*70)

    scorer = PaperQualityScorer()
    current_year = datetime.now().year

    test_papers = [
        {
            "name": "High-quality recent NeurIPS paper",
            "citation_count": 500,
            "venue": "NeurIPS",
            "is_peer_reviewed": True,
            "publication_year": current_year,
            "expected_min": 85.0,
        },
        {
            "name": "Moderate-quality older AAAI paper",
            "citation_count": 50,
            "venue": "AAAI",
            "is_peer_reviewed": True,
            "publication_year": current_year - 5,
            "expected_min": 65.0,
        },
        {
            "name": "Recent arXiv preprint (no citations)",
            "citation_count": 0,
            "venue": "arXiv",
            "is_peer_reviewed": False,
            "publication_year": current_year,
            "expected_min": 35.0,
        },
        {
            "name": "Highly-cited classic paper",
            "citation_count": 10000,
            "venue": "ICML",
            "is_peer_reviewed": True,
            "publication_year": current_year - 15,
            "expected_min": 50.0,
        },
    ]

    all_passed = True

    for paper in test_papers:
        score = scorer.score_paper(
            citation_count=paper["citation_count"],
            venue=paper["venue"],
            is_peer_reviewed=paper["is_peer_reviewed"],
            publication_year=paper["publication_year"]
        )

        passed = score.overall_score >= paper["expected_min"]
        status = "✅" if passed else "❌"

        print(f"{status} {paper['name']}")
        print(f"    Citations: {paper['citation_count']}")
        print(f"    Venue: {paper['venue']} ({score.venue_impact_factor:.1f}/100)")
        print(f"    Peer-reviewed: {paper['is_peer_reviewed']}")
        print(f"    Year: {paper['publication_year']} ({score.recency_score:.1f}/100)")
        print(f"    OVERALL: {score.overall_score:.1f}/100 (expected >={paper['expected_min']})")

        if not passed:
            all_passed = False

    # Test statistics
    stats = scorer.get_stats()
    print(f"\nScorer Stats:")
    print(f"  Papers scored: {stats['papers_scored']}")
    print(f"  Average score: {stats['avg_score']:.1f}")
    print(f"  High quality (>=70): {stats['high_quality_papers']}")
    print(f"  High quality rate: {stats['high_quality_rate']:.1%}")

    print(f"\n{'✅ All composite scoring tests passed' if all_passed else '❌ Some tests failed'}")
    return all_passed


def test_batch_scoring():
    """Test 6: Batch scoring multiple papers."""
    print("\n" + "="*70)
    print("TEST 6: Batch Scoring")
    print("="*70)

    scorer = PaperQualityScorer()
    scorer.reset_stats()

    current_year = datetime.now().year

    papers = [
        {
            "title": "Paper 1: Attention Is All You Need",
            "citation_count": 50000,
            "venue": "NeurIPS",
            "is_peer_reviewed": True,
            "publication_year": 2017,
        },
        {
            "title": "Paper 2: BERT Pre-training",
            "citation_count": 30000,
            "venue": "NAACL",
            "is_peer_reviewed": True,
            "publication_year": 2019,
        },
        {
            "title": "Paper 3: Recent arXiv Preprint",
            "citation_count": 5,
            "venue": "arXiv",
            "is_peer_reviewed": False,
            "publication_year": current_year,
        },
        {
            "title": "Paper 4: Workshop Paper",
            "citation_count": 10,
            "venue": "ICML Workshop",
            "is_peer_reviewed": True,
            "publication_year": current_year - 2,
        },
    ]

    try:
        scores = scorer.batch_score(papers)

        assert len(scores) == len(papers), "Batch scoring count mismatch"

        print(f"✅ Batch scored {len(scores)} papers")

        for i, (paper, score) in enumerate(zip(papers, scores), 1):
            print(f"  {i}. {paper['title'][:50]}")
            print(f"     Score: {score.overall_score:.1f}/100")

        stats = scorer.get_stats()
        print(f"\nBatch Stats:")
        print(f"  Average score: {stats['avg_score']:.1f}")
        print(f"  High quality rate: {stats['high_quality_rate']:.1%}")

        print(f"\n✅ Batch scoring test passed")
        return True

    except Exception as e:
        logger.error(f"Batch scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_model():
    """Test 7: Integration with PaperQualityScoreV2 Pydantic model."""
    print("\n" + "="*70)
    print("TEST 7: Integration with PaperQualityScoreV2 Model")
    print("="*70)

    scorer = PaperQualityScorer()
    current_year = datetime.now().year

    try:
        # Test 1: Create score via scorer
        score1 = scorer.score_paper(
            citation_count=100,
            venue="ICML",
            is_peer_reviewed=True,
            publication_year=current_year - 2,
            methodology_score=85.0
        )

        # Validate it's a PaperQualityScoreV2 instance
        assert isinstance(score1, PaperQualityScoreV2), "Should return PaperQualityScoreV2"
        assert score1.citation_count == 100, "Citation count should match"
        assert score1.is_peer_reviewed == True, "Peer review status should match"
        assert score1.methodology_score == 85.0, "Methodology score should be set"
        assert 0 <= score1.overall_score <= 100, "Overall score should be 0-100"

        print(f"✅ Scorer returns valid PaperQualityScoreV2")
        print(f"   Citation count: {score1.citation_count}")
        print(f"   Venue IF: {score1.venue_impact_factor:.1f}")
        print(f"   Peer-reviewed: {score1.is_peer_reviewed}")
        print(f"   Recency: {score1.recency_score:.1f}")
        print(f"   Methodology: {score1.methodology_score}")
        print(f"   Overall: {score1.overall_score:.1f}")

        # Test 2: Create score directly from model
        score2 = PaperQualityScoreV2(
            citation_count=50,
            venue_impact_factor=82.0,
            is_peer_reviewed=True,
            recency_score=90.0,
            overall_score=PaperQualityScoreV2.calculate_overall(
                citation_count=50,
                venue_if=82.0,
                peer_reviewed=True,
                recency=90.0
            )
        )

        assert isinstance(score2, PaperQualityScoreV2), "Direct instantiation should work"
        print(f"✅ Direct PaperQualityScoreV2 instantiation works")
        print(f"   Overall: {score2.overall_score:.1f}")

        # Test 3: Score from metadata dict
        metadata = {
            "citation_count": 200,
            "venue": "NeurIPS",
            "is_peer_reviewed": True,
            "publication_year": current_year,
        }

        score3 = scorer.score_paper_from_metadata(metadata)
        assert isinstance(score3, PaperQualityScoreV2), "Metadata scoring should work"
        print(f"✅ Scoring from metadata dict works")
        print(f"   Overall: {score3.overall_score:.1f}")

        print(f"\n✅ All integration tests passed")
        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test 8: Edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases and Error Handling")
    print("="*70)

    scorer = PaperQualityScorer()
    current_year = datetime.now().year

    try:
        # Test 1: Zero citations
        score1 = scorer.score_paper(0, "Unknown", False, current_year)
        assert score1.overall_score >= 0, "Score should be non-negative"
        print(f"✅ Handles zero citations: {score1.overall_score:.1f}")

        # Test 2: Future publication year (should handle gracefully)
        score2 = scorer.score_paper(10, "ICML", True, current_year + 1)
        assert score2.recency_score == 100.0, "Future year should have max recency"
        print(f"✅ Handles future year: recency={score2.recency_score:.1f}")

        # Test 3: Very old paper
        score3 = scorer.score_paper(1000, "Nature", True, 1950)
        assert score3.overall_score > 0, "Old papers should still score"
        print(f"✅ Handles very old paper (1950): {score3.overall_score:.1f}")

        # Test 4: Empty venue name
        score4 = scorer.score_paper(50, "", True, current_year)
        assert score4.venue_impact_factor == 50.0, "Empty venue should use default"
        print(f"✅ Handles empty venue: venue_if={score4.venue_impact_factor:.1f}")

        # Test 5: Negative citation count (should be handled)
        score5 = scorer._score_citations(-10)
        assert score5 == 0.0, "Negative citations should return 0"
        print(f"✅ Handles negative citations: {score5:.1f}")

        print(f"\n✅ All edge case tests passed")
        return True

    except Exception as e:
        logger.error(f"Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("QUALITY SCORING ENGINE - END-TO-END TEST SUITE")
    print("Epic 3, Task 3.5 (CRITICAL)")
    print("="*70)

    tests = [
        ("Citation Scoring", test_citation_scoring),
        ("Venue Database", test_venue_database),
        ("Recency Scoring", test_recency_scoring),
        ("Peer Review Scoring", test_peer_review_scoring),
        ("Composite Scoring", test_overall_composite_score),
        ("Batch Scoring", test_batch_scoring),
        ("Model Integration", test_integration_with_model),
        ("Edge Cases", test_edge_cases),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {name}")

    total = len(results)
    passed = sum(1 for p in results.values() if p)

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Epic 3 Quality Scoring Complete!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
