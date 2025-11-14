"""
Neo4j migration script for Phase 2.

Adds:
1. `embedding` property to Paper nodes (List[Float])
2. `quality_score` property to Paper nodes (Float)
3. `last_accessed` property for memory decay
4. New node types: EpisodicMemory, SemanticFact
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def run_migration():
    """Execute Phase 2 schema migrations."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("🔧 Starting Phase 2 Neo4j migration...")

        # Migration 1: Add embedding property to Paper nodes
        print("  [1/5] Adding embedding property to Paper nodes...")
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.embedding IS NULL
            SET p.embedding = []
            RETURN count(p) as count
        """)
        count = result.single()["count"]
        print(f"    ✅ Added embedding to {count} papers")

        # Migration 2: Add quality_score property
        print("  [2/5] Adding quality_score property...")
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.quality_score IS NULL
            SET p.quality_score = 50.0
            RETURN count(p) as count
        """)
        count = result.single()["count"]
        print(f"    ✅ Added quality_score to {count} papers")

        # Migration 3: Add last_accessed property
        print("  [3/5] Adding last_accessed property...")
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.last_accessed IS NULL
            SET p.last_accessed = datetime()
            RETURN count(p) as count
        """)
        count = result.single()["count"]
        print(f"    ✅ Added last_accessed to {count} papers")

        # Migration 4: Create EpisodicMemory nodes (example)
        print("  [4/5] Creating EpisodicMemory node type...")
        session.run("""
            MERGE (em:EpisodicMemory {memory_id: 'migration_test'})
            SET em.timestamp = datetime(),
                em.query = 'Migration test query',
                em.session_id = 'test_session'
            RETURN em
        """)
        print("    ✅ EpisodicMemory node type created")

        # Migration 5: Create SemanticFact nodes (example)
        print("  [5/5] Creating SemanticFact node type...")
        session.run("""
            MERGE (sf:SemanticFact {fact_id: 'migration_test'})
            SET sf.claim = 'Migration test fact',
                sf.confidence = 1.0,
                sf.created_at = datetime()
            RETURN sf
        """)
        print("    ✅ SemanticFact node type created")

        print("\n✅ Phase 2 migration complete!")
        print("   Run this script again if you add more papers to backfill properties.")

    driver.close()


if __name__ == "__main__":
    run_migration()
