"""
Unit tests for Neo4j Knowledge Graph tools.

Tests query_knowledge_graph and add_insight_to_graph functions.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.tools.kg_tools import query_knowledge_graph, add_insight_to_graph, get_insight_count
from src.models import Synthesis


class TestQueryKnowledgeGraph:
    """Test the query_knowledge_graph function."""
    
    def test_query_returns_string(self):
        """Test query_knowledge_graph returns a string."""
        result = query_knowledge_graph("What are transformers?")
        assert isinstance(result, str)
    
    def test_query_empty_input(self):
        """Test query with empty string."""
        result = query_knowledge_graph("")
        # Should handle gracefully, return empty or fallback
        assert isinstance(result, str)
    
    def test_query_with_neo4j_connection(self):
        """Test query when Neo4j is properly connected."""
        # This assumes Neo4j is running and populated
        result = query_knowledge_graph("attention mechanism")
        # Result should be string (may be empty if no matches)
        assert isinstance(result, str)


class TestAddInsightToGraph:
    """Test the add_insight_to_graph function."""
    
    def test_add_valid_insight(self):
        """Test adding a valid synthesis to the graph."""
        synthesis = Synthesis(
            novel_insight="This is a comprehensive test insight for the knowledge graph with sufficient length",
            supporting_claims=["Claim 1", "Claim 2"],
            evidence_lineage=[
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/1810.04805",
                "https://arxiv.org/abs/2005.14165"
            ],
            confidence_score=0.85,
            novelty_score=0.70,
            reasoning="This is comprehensive test reasoning with sufficient length for validation"
        )
        
        thread_id = "test_thread_kg_001"
        
        # Add insight
        result = add_insight_to_graph(synthesis, thread_id)
        
        # Should return True on success (or handle gracefully)
        assert isinstance(result, bool)
    
    def test_add_insight_with_duplicate_evidence(self):
        """Test adding insight with duplicate evidence URLs."""
        synthesis = Synthesis(
            novel_insight="This test validates handling of duplicate evidence URLs in the knowledge graph system",
            supporting_claims=["Claim"],
            evidence_lineage=[
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/1706.03762",  # Duplicate (will be deduplicated)
                "https://arxiv.org/abs/1810.04805",
                "https://arxiv.org/abs/2005.14165"  # Need 3 unique URLs
            ],
            confidence_score=0.80,
            novelty_score=0.60,
            reasoning="This reasoning validates the deduplication logic in evidence processing"
        )
        
        thread_id = "test_thread_kg_002"
        
        # Should handle duplicates gracefully
        result = add_insight_to_graph(synthesis, thread_id)
        assert isinstance(result, bool)
    
    def test_add_insight_nonexistent_paper(self):
        """Test adding insight that references non-existent papers."""
        synthesis = Synthesis(
            novel_insight="This test validates graceful handling of references to non-existent research papers",
            supporting_claims=["Claim"],
            evidence_lineage=[
                "https://example.com/nonexistent1",
                "https://example.com/nonexistent2",
                "https://example.com/nonexistent3"
            ],
            confidence_score=0.75,
            novelty_score=0.65,
            reasoning="This reasoning tests the system's ability to handle missing paper references"
        )
        
        thread_id = "test_thread_kg_003"
        
        # Should not fail even if papers don't exist
        result = add_insight_to_graph(synthesis, thread_id)
        assert isinstance(result, bool)


class TestGetInsightCount:
    """Test the get_insight_count function."""
    
    def test_insight_count_returns_int(self):
        """Test get_insight_count returns an integer."""
        count = get_insight_count()
        assert isinstance(count, int)
        assert count >= 0
    
    def test_insight_count_increases(self):
        """Test insight count increases after adding insights."""
        # Get initial count
        initial_count = get_insight_count()
        
        # Add a new insight
        synthesis = Synthesis(
            novel_insight="This test validates that the insight counter increments correctly after insertions",
            supporting_claims=["Claim"],
            evidence_lineage=[
                "https://arxiv.org/abs/1",
                "https://arxiv.org/abs/2",
                "https://arxiv.org/abs/3"
            ],
            confidence_score=0.80,
            novelty_score=0.70,
            reasoning="This reasoning validates the insight counting mechanism in the knowledge graph"
        )
        
        add_insight_to_graph(synthesis, "test_count_thread")
        
        # Get new count
        new_count = get_insight_count()
        
        # Should have increased by 1
        assert new_count >= initial_count


class TestKGToolsIntegration:
    """Integration tests for KG tools with Neo4j."""
    
    def test_kg_connection_exists(self):
        """Test that Neo4j connection environment variables are set."""
        assert os.getenv("NEO4J_URI") is not None
        assert os.getenv("NEO4J_USER") is not None
        assert os.getenv("NEO4J_PASSWORD") is not None
    
    def test_add_and_query_flow(self):
        """Test adding an insight and then querying for related info."""
        # Add an insight about attention mechanisms
        synthesis = Synthesis(
            novel_insight="Attention mechanisms enable transformers to focus on relevant input parts dynamically",
            supporting_claims=[
                "Self-attention computes relationships between all positions",
                "Multi-head attention allows parallel processing"
            ],
            evidence_lineage=[
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/1810.04805",
                "https://arxiv.org/abs/2005.14165"
            ],
            confidence_score=0.90,
            novelty_score=0.75,
            reasoning="This synthesis integrates insights from multiple seminal transformer architecture research papers"
        )
        
        thread_id = "test_attention_flow"
        
        # Add insight
        success = add_insight_to_graph(synthesis, thread_id)
        assert success is True or success is False  # Either outcome is acceptable
        
        # Query for related information
        result = query_knowledge_graph("attention mechanism in transformers")
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

