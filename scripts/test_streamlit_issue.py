#!/usr/bin/env python3
"""
Diagnostic script to test the full dialectical flow and identify synthesis issues.

This script simulates what happens in the Streamlit UI to help debug
"Stream ended without final synthesis" errors.
"""

import os
import uuid
import time
from src.graph import build_graph

def test_dialectical_flow(query: str):
    """
    Test the full dialectical flow and identify where it might be failing.
    
    Args:
        query: Research question to test
    """
    print("=" * 80)
    print("DIALECTICAL FLOW DIAGNOSTIC TEST")
    print("=" * 80)
    print(f"\nQuery: {query}")
    print(f"MAX_ITERATIONS: {os.getenv('MAX_ITERATIONS', '3')}")
    print(f"SEQUENTIAL_SEMANTIC_SCHOLAR: {os.getenv('SEQUENTIAL_SEMANTIC_SCHOLAR', 'true')}")
    print()
    
    # Build graph with checkpointer
    print("Building graph...")
    graph = build_graph(use_checkpointer=True, db_path="test_diagnostic.db")
    print("✅ Graph built successfully\n")
    
    # Prepare initial state
    thread_id = f"test_{uuid.uuid4()}"
    inputs = {
        "messages": [],
        "original_query": query,
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
        "current_claim_id": str(uuid.uuid4())
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Starting execution...")
    print("-" * 80)
    
    start_time = time.time()
    step_count = 0
    synthesis_found = False
    
    try:
        for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
            step_count += 1
            elapsed = time.time() - start_time
            
            for node_name, node_output in chunk.items():
                print(f"\n[{step_count}] NODE: {node_name.upper()} (t={elapsed:.1f}s)")
                
                if node_name == "analyst":
                    thesis = node_output.get("current_thesis")
                    if thesis:
                        print(f"  ✅ Thesis: {thesis.claim[:80]}...")
                        print(f"  📄 Evidence: {len(thesis.evidence)} papers")
                    else:
                        print("  ❌ No thesis generated!")
                
                elif node_name == "skeptic":
                    antithesis = node_output.get("current_antithesis")
                    if antithesis:
                        contradiction = antithesis.contradiction_found
                        print(f"  {'⚠️  Contradiction found' if contradiction else '✅ No contradiction'}")
                        print(f"  🔍 Counter-evidence: {len(antithesis.conflicting_evidence)} papers")
                        
                        if contradiction:
                            print(f"  🔄 System will loop back to Analyst (iteration {inputs['iteration_count'] + 1})")
                    else:
                        print("  ❌ No antithesis generated!")
                
                elif node_name == "increment_iteration":
                    new_count = node_output.get("iteration_count", 0)
                    print(f"  🔄 Iteration count: {new_count}")
                
                elif node_name == "synthesizer":
                    synthesis = node_output.get("final_synthesis")
                    if synthesis:
                        synthesis_found = True
                        print(f"  ✨ SYNTHESIS COMPLETE!")
                        print(f"  💡 Insight: {synthesis.novel_insight[:100]}...")
                        print(f"  📊 Confidence: {synthesis.confidence_score:.2%}")
                        print(f"  🆕 Novelty: {synthesis.novelty_score:.2%}")
                    else:
                        print("  ❌ Synthesizer executed but no synthesis generated!")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("EXECUTION COMPLETE")
        print("=" * 80)
        print(f"Total time: {total_time:.1f}s")
        print(f"Total steps: {step_count}")
        print(f"Synthesis found: {'✅ YES' if synthesis_found else '❌ NO - This is the bug!'}")
        
        if not synthesis_found:
            print("\n🔴 DIAGNOSIS: Stream ended without synthesis!")
            print("\nPossible causes:")
            print("1. Synthesis node failed with an exception")
            print("2. Gemini API rate limit or timeout")
            print("3. Pydantic validation error in Synthesis")
            print("4. Neo4j connection issue")
            print("\nCheck the logs above for ERROR messages.")
        
        return synthesis_found
    
    except Exception as e:
        print(f"\n❌ EXECUTION FAILED: {e}")
        print(f"\nThis is why Streamlit shows 'Stream ended without final synthesis'")
        raise


if __name__ == "__main__":
    import sys
    
    # Default test query (simple to avoid rate limits)
    query = sys.argv[1] if len(sys.argv) > 1 else "What is machine learning?"
    
    try:
        success = test_dialectical_flow(query)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

