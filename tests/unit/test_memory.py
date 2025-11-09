"""
Unit tests for Tier 1 US2: Memory Management (T053)

Tests:
1. FIFO eviction works (add 15 items, only last 10 remain)
2. Memory updates correctly after rejection
3. Helper functions work as expected
"""

from src.utils.memory import update_debate_memory, format_memory_for_prompt, get_memory_summary, MAX_MEMORY_ITEMS
from src.models import DebateMemory


def test_fifo_eviction_rejected_claims():
    """T053: Test that FIFO eviction works for rejected claims (add 15, keep last 10)"""
    memory = DebateMemory(rejected_claims=[], skeptic_objections=[], weak_evidence_urls=[])
    
    # Add 15 claims
    for i in range(15):
        memory = update_debate_memory(memory, rejected_claim=f"Claim {i}")
    
    # Should only have last 10 claims (5-14)
    assert len(memory["rejected_claims"]) == MAX_MEMORY_ITEMS, f"Expected {MAX_MEMORY_ITEMS} claims, got {len(memory['rejected_claims'])}"
    assert "Claim 5" in memory["rejected_claims"], "Should have Claim 5 (oldest kept)"
    assert "Claim 14" in memory["rejected_claims"], "Should have Claim 14 (newest)"
    assert "Claim 0" not in memory["rejected_claims"], "Should NOT have Claim 0 (evicted)"
    assert "Claim 4" not in memory["rejected_claims"], "Should NOT have Claim 4 (evicted)"
    
    print(f"✅ FIFO eviction: Added 15 claims, retained last {MAX_MEMORY_ITEMS}")
    print(f"   Oldest kept: {memory['rejected_claims'][0]}")
    print(f"   Newest: {memory['rejected_claims'][-1]}")


def test_fifo_eviction_skeptic_objections():
    """T053: Test that FIFO eviction works for skeptic objections"""
    memory = DebateMemory(rejected_claims=[], skeptic_objections=[], weak_evidence_urls=[])
    
    # Add 12 objections
    for i in range(12):
        memory = update_debate_memory(memory, skeptic_objection=f"Objection {i}")
    
    # Should only have last 10 objections (2-11)
    assert len(memory["skeptic_objections"]) == MAX_MEMORY_ITEMS
    assert "Objection 2" in memory["skeptic_objections"]
    assert "Objection 11" in memory["skeptic_objections"]
    assert "Objection 0" not in memory["skeptic_objections"]
    
    print(f"✅ FIFO eviction: Added 12 objections, retained last {MAX_MEMORY_ITEMS}")


def test_fifo_eviction_weak_evidence_urls():
    """T053: Test that FIFO eviction works for weak evidence URLs"""
    memory = DebateMemory(rejected_claims=[], skeptic_objections=[], weak_evidence_urls=[])
    
    # Add 13 URLs
    for i in range(13):
        memory = update_debate_memory(memory, weak_evidence_url=f"https://example.com/paper{i}")
    
    # Should only have last 10 URLs (3-12)
    assert len(memory["weak_evidence_urls"]) == MAX_MEMORY_ITEMS
    assert "https://example.com/paper3" in memory["weak_evidence_urls"]
    assert "https://example.com/paper12" in memory["weak_evidence_urls"]
    assert "https://example.com/paper0" not in memory["weak_evidence_urls"]
    
    print(f"✅ FIFO eviction: Added 13 URLs, retained last {MAX_MEMORY_ITEMS}")


def test_memory_update_after_rejection():
    """T053: Test that memory updates correctly after a thesis rejection"""
    memory = DebateMemory(rejected_claims=[], skeptic_objections=[], weak_evidence_urls=[])
    
    # Simulate a rejection
    rejected_claim = "Consciousness is purely computational"
    objection = "This claim fails to account for qualia and subjective experience, which cannot be reduced to computation"
    
    updated_memory = update_debate_memory(
        memory,
        rejected_claim=rejected_claim,
        skeptic_objection=objection
    )
    
    # Verify updates
    assert len(updated_memory["rejected_claims"]) == 1
    assert len(updated_memory["skeptic_objections"]) == 1
    assert updated_memory["rejected_claims"][0] == rejected_claim
    assert updated_memory["skeptic_objections"][0] == objection
    
    print(f"✅ Memory updated after rejection:")
    print(f"   Rejected claims: {len(updated_memory['rejected_claims'])}")
    print(f"   Skeptic objections: {len(updated_memory['skeptic_objections'])}")


def test_format_memory_for_prompt():
    """T053: Test that memory formatting for prompts works correctly"""
    memory = DebateMemory(
        rejected_claims=["Claim 1", "Claim 2"],
        skeptic_objections=["Objection 1"],
        weak_evidence_urls=["https://example.com/paper1"]
    )
    
    formatted = format_memory_for_prompt(memory)
    
    # Should contain all sections
    assert "REJECTED CLAIMS" in formatted
    assert "PAST OBJECTIONS" in formatted
    assert "WEAK EVIDENCE SOURCES" in formatted
    assert "Claim 1" in formatted
    assert "Objection 1" in formatted
    assert "https://example.com/paper1" in formatted
    
    print(f"✅ Memory formatted for prompt (length: {len(formatted)} chars)")
    
    # Test empty memory
    empty_memory = DebateMemory(rejected_claims=[], skeptic_objections=[], weak_evidence_urls=[])
    empty_formatted = format_memory_for_prompt(empty_memory)
    
    assert empty_formatted == "", "Empty memory should return empty string"
    print(f"✅ Empty memory formatted correctly (empty string)")


def test_get_memory_summary():
    """T053: Test that memory summary generation works correctly"""
    # Non-empty memory
    memory = DebateMemory(
        rejected_claims=["Claim 1", "Claim 2"],
        skeptic_objections=["Objection 1", "Objection 2", "Objection 3"],
        weak_evidence_urls=["https://example.com/paper1"]
    )
    
    summary = get_memory_summary(memory)
    
    assert "2 rejected claims" in summary
    assert "3 objections" in summary
    assert "1 weak URLs" in summary
    
    print(f"✅ Memory summary: {summary}")
    
    # Empty memory
    empty_memory = DebateMemory(rejected_claims=[], skeptic_objections=[], weak_evidence_urls=[])
    empty_summary = get_memory_summary(empty_memory)
    
    assert empty_summary == "Memory: empty"
    print(f"✅ Empty memory summary: {empty_summary}")


def test_partial_memory_update():
    """T053: Test that partial updates (only some fields) work correctly"""
    memory = DebateMemory(
        rejected_claims=["Existing claim"],
        skeptic_objections=[],
        weak_evidence_urls=[]
    )
    
    # Update only objections
    updated = update_debate_memory(memory, skeptic_objection="New objection")
    
    assert len(updated["rejected_claims"]) == 1  # Should preserve existing
    assert len(updated["skeptic_objections"]) == 1  # Should add new
    assert updated["rejected_claims"][0] == "Existing claim"
    assert updated["skeptic_objections"][0] == "New objection"
    
    print(f"✅ Partial memory update works correctly")


def test_multiple_updates_in_sequence():
    """T053: Test that multiple sequential updates work correctly"""
    memory = DebateMemory(rejected_claims=[], skeptic_objections=[], weak_evidence_urls=[])
    
    # Add items sequentially
    memory = update_debate_memory(memory, rejected_claim="Claim 1")
    memory = update_debate_memory(memory, skeptic_objection="Objection 1")
    memory = update_debate_memory(memory, rejected_claim="Claim 2")
    memory = update_debate_memory(memory, weak_evidence_url="https://example.com/paper1")
    
    # Verify final state
    assert len(memory["rejected_claims"]) == 2
    assert len(memory["skeptic_objections"]) == 1
    assert len(memory["weak_evidence_urls"]) == 1
    
    print(f"✅ Multiple sequential updates work correctly")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Tier 1 US2 Memory Management Unit Tests (T053)")
    print("=" * 70)
    
    test_fifo_eviction_rejected_claims()
    test_fifo_eviction_skeptic_objections()
    test_fifo_eviction_weak_evidence_urls()
    test_memory_update_after_rejection()
    test_format_memory_for_prompt()
    test_get_memory_summary()
    test_partial_memory_update()
    test_multiple_updates_in_sequence()
    
    print()
    print("=" * 70)
    print("🎉 ALL MEMORY TESTS PASSED!")
    print("=" * 70)

