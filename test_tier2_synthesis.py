#!/usr/bin/env python3
"""
Tier 2 E2E Test: Comprehensive Synthesis Generation

Tests that the system generates 800-1500 word comprehensive research reports
with all required sections populated.
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "dev_test_key")  # Will use X-API-Key header
THREAD_ID = f"tier2-test-{int(time.time())}"

# Test query (simple enough to complete quickly but substantive enough for synthesis)
TEST_QUERY = "What is the relationship between neural networks and consciousness?"

print("=" * 80)
print("Tier 2 E2E Test: Comprehensive Synthesis Generation")
print("=" * 80)
print(f"\n📝 Query: {TEST_QUERY}")
print(f"🔑 Thread ID: {THREAD_ID}")
print(f"🌐 API: {API_BASE}")
print("\n" + "=" * 80)

# Stream the synthesis
url = f"{API_BASE}/stream_dialectics/{THREAD_ID}?query={requests.utils.quote(TEST_QUERY)}&auto_discover=true"
headers = {
    "X-API-Key": API_KEY
}

print("\n🚀 Starting dialectical synthesis...")
print("-" * 80)

try:
    response = requests.get(url, headers=headers, stream=True, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ Error: HTTP {response.status_code}")
        print(response.text)
        exit(1)
    
    # Track events
    event_count = 0
    synthesis_found = False
    synthesis_data = None
    
    # Stream events
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            
            # SSE format: "event: event_name" or "data: {json}"
            if line.startswith('event:'):
                event_name = line.split(':', 1)[1].strip()
                event_count += 1
                
                # Show progress for key events
                if event_name == 'keyword_extraction_complete':
                    print("  ✓ Keywords extracted")
                elif event_name == 'discovery_complete':
                    print("  ✓ Papers discovered")
                elif event_name == 'thesis_generation_complete':
                    print("  ✓ Thesis generated")
                elif event_name == 'critique_complete':
                    print("  ✓ Critique complete")
                elif event_name == 'synthesis_complete':
                    print("  ✓ Synthesis complete!")
                    synthesis_found = True
            
            elif line.startswith('data:'):
                data_str = line.split(':', 1)[1].strip()
                try:
                    data = json.loads(data_str)
                    
                    # Capture final synthesis data
                    if 'final_synthesis' in data_str and not synthesis_data:
                        synthesis_data = data
                
                except json.JSONDecodeError:
                    pass
    
    print("-" * 80)
    print(f"\n✅ Synthesis complete! ({event_count} events processed)")
    
    # Analyze synthesis
    if synthesis_data and 'final_synthesis' in synthesis_data:
        synthesis = synthesis_data['final_synthesis']
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE SYNTHESIS ANALYSIS")
        print("=" * 80)
        
        # Check all required sections
        sections = {
            'novel_insight': synthesis.get('novel_insight', ''),
            'dialectical_summary': synthesis.get('dialectical_summary', ''),
            'rounds_explored': synthesis.get('rounds_explored', []),
            'supporting_evidence': synthesis.get('supporting_evidence', []),
            'contradicting_evidence_addressed': synthesis.get('contradicting_evidence_addressed', []),
            'synthesis_reasoning': synthesis.get('synthesis_reasoning', ''),
            'confidence_score': synthesis.get('confidence_score', 0),
            'confidence_justification': synthesis.get('confidence_justification', ''),
            'novelty_score': synthesis.get('novelty_score', 0),
            'novelty_justification': synthesis.get('novelty_justification', ''),
            'practical_implications': synthesis.get('practical_implications', []),
            'testable_predictions': synthesis.get('testable_predictions', []),
            'open_questions': synthesis.get('open_questions', []),
            'evidence_lineage': synthesis.get('evidence_lineage', []),
            'key_papers': synthesis.get('key_papers', [])
        }
        
        # Calculate word count
        word_counts = {
            'novel_insight': len(sections['novel_insight'].split()),
            'dialectical_summary': len(sections['dialectical_summary'].split()),
            'synthesis_reasoning': len(sections['synthesis_reasoning'].split())
        }
        total_words = sum(word_counts.values())
        
        print(f"\n📄 WORD COUNT")
        print(f"  • Novel Insight: {word_counts['novel_insight']} words")
        print(f"  • Dialectical Summary: {word_counts['dialectical_summary']} words")
        print(f"  • Synthesis Reasoning: {word_counts['synthesis_reasoning']} words")
        print(f"  • TOTAL: {total_words} words")
        
        if 800 <= total_words <= 1500:
            print(f"  ✅ Within target range (800-1500 words)")
        else:
            print(f"  ⚠️  Outside target range (800-1500 words)")
        
        print(f"\n📊 QUALITY METRICS")
        print(f"  • Confidence: {sections['confidence_score']}/100")
        print(f"    Justification: {sections['confidence_justification'][:80]}...")
        print(f"  • Novelty: {sections['novelty_score']}/100")
        print(f"    Justification: {sections['novelty_justification'][:80]}...")
        
        print(f"\n📚 EVIDENCE SYNTHESIS")
        print(f"  • Supporting Evidence: {len(sections['supporting_evidence'])} papers")
        print(f"  • Counter-Evidence: {len(sections['contradicting_evidence_addressed'])} papers")
        print(f"  • Total Evidence: {len(sections['evidence_lineage'])} sources")
        print(f"  • Key Papers: {len(sections['key_papers'])} papers")
        
        print(f"\n🎯 IMPLICATIONS & PREDICTIONS")
        print(f"  • Practical Implications: {len(sections['practical_implications'])}")
        print(f"  • Testable Predictions: {len(sections['testable_predictions'])}")
        print(f"  • Open Questions: {len(sections['open_questions'])}")
        
        print(f"\n🔄 DIALECTICAL JOURNEY")
        print(f"  • Rounds Explored: {len(sections['rounds_explored'])}")
        
        # Validation
        print("\n" + "=" * 80)
        print("✅ VALIDATION RESULTS")
        print("=" * 80)
        
        checks = []
        checks.append(("Word count 800-1500", 800 <= total_words <= 1500))
        checks.append(("Novel insight present", len(sections['novel_insight']) >= 50))
        checks.append(("Dialectical summary present", len(sections['dialectical_summary']) >= 300))
        checks.append(("Synthesis reasoning present", len(sections['synthesis_reasoning']) >= 400))
        checks.append(("Rounds explored present", len(sections['rounds_explored']) >= 1))
        checks.append(("Supporting evidence present", len(sections['supporting_evidence']) >= 2))
        checks.append(("Confidence justification present", len(sections['confidence_justification']) >= 100))
        checks.append(("Novelty justification present", len(sections['novelty_justification']) >= 100))
        checks.append(("Practical implications present", len(sections['practical_implications']) >= 2))
        checks.append(("Testable predictions present", len(sections['testable_predictions']) >= 2))
        checks.append(("Open questions present", len(sections['open_questions']) >= 2))
        checks.append(("Key papers present", len(sections['key_papers']) >= 5))
        
        passed = sum(1 for _, result in checks if result)
        total = len(checks)
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"{status} {check_name}")
        
        print(f"\n📊 OVERALL: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 ALL VALIDATION CHECKS PASSED!")
            print("✅ Tier 2 Comprehensive Synthesis is working correctly!")
        else:
            print(f"\n⚠️  {total - passed} validation check(s) failed")
        
        # Sample output
        print("\n" + "=" * 80)
        print("📝 SAMPLE OUTPUT")
        print("=" * 80)
        print(f"\nNovel Insight:\n{sections['novel_insight'][:300]}...")
        print(f"\nDialectical Summary (first 200 words):\n{' '.join(sections['dialectical_summary'].split()[:200])}...")
        
    else:
        print("\n❌ No synthesis data found in response")
        print(f"Response data: {synthesis_data}")

except requests.exceptions.Timeout:
    print("\n❌ Request timed out (>300s)")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)

