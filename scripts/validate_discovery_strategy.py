"""
Validation script for discovery strategy (Tier 1: T078).

Tests 20 diverse claims (5 simple, 10 complex, 5 novel) and outputs strategy statistics.

Usage:
    python scripts/validate_discovery_strategy.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.discovery_strategy import determine_discovery_strategy
from src.tools.telemetry import get_telemetry_summary

# Test claims categorized by complexity
TEST_CLAIMS = {
    "simple": [
        "Water is H2O",
        "DNA encodes genetic information",
        "Gravity is described by general relativity",
        "The Earth orbits the Sun",
        "Cells are the basic unit of life"
    ],
    "complex": [
        "Consciousness emerges from integrated information processing",
        "Dark matter is self-interacting",
        "Quantum entanglement enables faster-than-light communication",
        "The brain computes using quantum mechanics",
        "Free will is compatible with determinism",
        "The hard problem of consciousness is irreducible",
        "Artificial general intelligence requires symbolic reasoning",
        "Life arose through RNA world hypothesis",
        "The universe is a hologram",
        "String theory is testable"
    ],
    "novel": [
        "Consciousness is a fundamental property of all matter",
        "Time emerges from quantum entanglement",
        "Dark energy is variable across cosmic epochs",
        "The universe is computationally irreducible",
        "Quantum gravity requires loop quantum theory"
    ]
}

EXPECTED_RANGES = {
    "simple": (2, 3),
    "complex": (5, 8),
    "novel": (8, 10)
}


def main():
    print("\n" + "=" * 70)
    print("DISCOVERY STRATEGY VALIDATION SCRIPT (T078)")
    print("=" * 70)
    print(f"\nTesting 20 diverse claims:")
    print(f"  - 5 simple claims (expected: 2-3 papers)")
    print(f"  - 10 complex claims (expected: 5-8 papers)")
    print(f"  - 5 novel claims (expected: 8-10 papers)")
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ GOOGLE_API_KEY not set. Using fallback strategy (3 papers).")
        print("   Set API key for accurate testing.")
    
    results = {
        "simple": [],
        "complex": [],
        "novel": []
    }
    
    # Test each category
    for category, claims in TEST_CLAIMS.items():
        print(f"\n\n[{category.upper()} CLAIMS]")
        print("-" * 70)
        
        for i, claim in enumerate(claims, 1):
            print(f"\n{i}. {claim[:60]}...")
            
            strategy = determine_discovery_strategy(claim, existing_evidence=[])
            
            expected_min, expected_max = EXPECTED_RANGES[category]
            within_range = expected_min <= strategy.initial_papers <= expected_max
            
            status = "✅" if within_range else "⚠️"
            print(f"   {status} Recommended: {strategy.initial_papers} papers (expected: {expected_min}-{expected_max})")
            print(f"   Follow-up: {strategy.follow_up_needed}")
            
            results[category].append({
                "claim": claim,
                "recommended": strategy.initial_papers,
                "follow_up": strategy.follow_up_needed,
                "within_range": within_range
            })
    
    # Summary statistics
    print("\n\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for category in ["simple", "complex", "novel"]:
        category_results = results[category]
        avg_papers = sum(r["recommended"] for r in category_results) / len(category_results)
        follow_up_rate = sum(1 for r in category_results if r["follow_up"]) / len(category_results) * 100
        within_range_count = sum(1 for r in category_results if r["within_range"])
        accuracy = (within_range_count / len(category_results)) * 100
        
        expected_min, expected_max = EXPECTED_RANGES[category]
        
        print(f"\n{category.upper()}:")
        print(f"   Avg papers: {avg_papers:.1f} (expected: {expected_min}-{expected_max})")
        print(f"   Follow-up rate: {follow_up_rate:.0f}%")
        print(f"   Accuracy: {accuracy:.0f}% within expected range")
    
    # Telemetry summary (if available)
    print("\n\n" + "=" * 70)
    print("TELEMETRY SUMMARY (Historical)")
    print("=" * 70)
    
    telemetry = get_telemetry_summary()
    if telemetry["total_discoveries"] > 0:
        print(f"\nTotal discoveries logged: {telemetry['total_discoveries']}")
        print(f"Avg delta (actual - recommended): {telemetry['avg_delta']:.2f} papers")
        print(f"Follow-up rate: {telemetry['follow_up_rate']:.1f}%")
        print(f"\nComplexity breakdown:")
        for level, count in telemetry["complexity_breakdown"].items():
            print(f"   {level}: {count} discoveries")
    else:
        print("\nNo telemetry data available yet. Run synthesis with telemetry enabled.")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

