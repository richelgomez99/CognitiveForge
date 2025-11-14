# Getting Started with Phase 2 Development

**Following SpecKit Methodology**

---

## 📚 Prerequisites

Before starting Phase 2, ensure you have:

✅ **Working MVP** - CognitiveForge Tier 1 running successfully
✅ **Neo4j Instance** - Local or Neo4j Aura Free (50K nodes)
✅ **Python 3.11+** - Installed with venv capability
✅ **Git** - For version control and branch management
✅ **8+ hours/week** - Dedicated development time

**Verify MVP Works:**
```bash
cd /home/user/CognitiveForge

# Start Neo4j (if local)
# docker-compose up -d neo4j  # OR use Neo4j Desktop

# Start API
uvicorn src.api:app --reload &

# Start UI
streamlit run src/ui.py

# Test: Enter query "How do transformers work?"
# Should see 3-agent debate in 2-3 minutes
```

---

## 🎯 Phase 2 Overview

**Goal:** Transform from "3-agent search assistant" → "10-agent research intelligence with memory"

**Key Deliverables:**
1. 10 specialized agents (vs current 3)
2. Persistent memory across sessions
3. PDF full-text extraction
4. Quality scoring for papers
5. Hybrid Neo4j + vector storage

**Timeline:** 8 weeks part-time (20h/week)

---

## 📋 Week 1: Getting Set Up

### Day 1-2: Read Specifications

**Read in Order:**
1. `.speckit/SPEC.md` - Complete specification (30 min)
   - Focus on PHASE 1 (user journeys) and PHASE 2 (architecture)
2. `.speckit/TASKS.md` - Task breakdown (15 min)
   - Understand Epic 1 tasks in detail
3. Your original research document (review key sections):
   - "Multi-Agent Collaboration" section
   - "Knowledge Representation and Persistent Expertise"
   - "Architecture patterns that enable multi-agent synthesis"

**Questions to Answer:**
- [ ] Do I understand the 10 agent roles?
- [ ] Do I understand the hybrid storage (Neo4j + ChromaDB)?
- [ ] Do I understand the memory types (episodic, semantic, procedural)?

### Day 3: Environment Setup

**Create Phase 2 Branch:**
```bash
cd /home/user/CognitiveForge
git checkout -b phase-2/foundations

# Create directory structure
mkdir -p src/agents_v2
mkdir -p src/memory
mkdir -p src/quality
mkdir -p tests/unit/phase2
mkdir -p tests/integration/phase2
mkdir -p docs/phase2
```

**Install New Dependencies:**
```bash
# Add to requirements.txt (don't install yet)
cat >> requirements.txt << 'EOF'

# Phase 2: Enhanced Knowledge Discovery
chromadb>=0.5.0           # Vector database
sentence-transformers>=3.0.0  # Embeddings (E5-large-v2)
pymupdf>=1.24.0           # PDF extraction
EOF

# Install (this will download ~500MB for E5-large-v2 model)
pip install -r requirements.txt

# Verify installations
python -c "import chromadb; print('ChromaDB:', chromadb.__version__)"
python -c "from sentence_transformers import SentenceTransformer; print('ST: OK')"
python -c "import fitz; print('PyMuPDF:', fitz.__version__)"
```

**Expected Output:**
```
ChromaDB: 0.5.x
ST: OK
PyMuPDF: 1.24.x
```

### Day 4-5: Epic 1 Tasks

**Task 1.1-1.2: Already done above! ✅**

**Task 1.3: Create /src/agents_v2/ structure**
```bash
cd /home/user/CognitiveForge

# Create agent module files
touch src/agents_v2/__init__.py
touch src/agents_v2/literature_surveyor.py  # Enhanced Analyst
touch src/agents_v2/deep_analyst.py
touch src/agents_v2/skeptical_reviewer.py   # Enhanced Skeptic
touch src/agents_v2/pattern_recognizer.py
touch src/agents_v2/hypothesis_generator.py
touch src/agents_v2/fact_checker.py
touch src/agents_v2/synthesizer.py          # Same role, new location
touch src/agents_v2/domain_expert.py
touch src/agents_v2/methodologist.py
touch src/agents_v2/devils_advocate.py
touch src/agents_v2/coordinator.py          # NEW: Orchestrates all

# Create supporting modules
touch src/memory/__init__.py
touch src/memory/episodic.py
touch src/memory/semantic.py
touch src/memory/retrieval.py

touch src/quality/__init__.py
touch src/quality/scoring.py
touch src/quality/venue_rankings.py
```

**Task 1.4: Design AgentStateV2**

Create `src/models_v2.py`:
```python
"""
Phase 2 data models extending Tier 1 architecture.

New additions:
- Memory models (Episodic, Semantic, Procedural)
- Quality scoring models
- 10-agent state tracking
"""

from typing import List, Optional, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

# Import existing models
from src.models import (
    AgentState,  # Will extend this
    Thesis, Antithesis, Synthesis,
    Evidence, ConflictingEvidence
)


# =============================================================================
# Quality Scoring Models
# =============================================================================

class PaperQualityScoreV2(BaseModel):
    """
    Multi-factor quality assessment for academic papers.

    Attributes:
        citation_count: From Semantic Scholar API
        venue_impact_factor: Manual lookup for top venues (0-100)
        is_peer_reviewed: True if journal/conference, False if preprint
        recency_score: Exponential decay from publication date (0-100)
        methodology_score: From Methodologist agent critique (0-100)
        overall_score: Weighted composite (0-100)
    """
    citation_count: int = Field(ge=0)
    venue_impact_factor: float = Field(ge=0.0, le=100.0, default=50.0)
    is_peer_reviewed: bool
    recency_score: float = Field(ge=0.0, le=100.0)
    methodology_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    overall_score: float = Field(ge=0.0, le=100.0)

    @classmethod
    def calculate_overall(cls, citation_count: int, venue_if: float,
                         peer_reviewed: bool, recency: float) -> float:
        """
        Weighted composite score.

        Formula: 0.3*citations + 0.25*venue + 0.25*peer + 0.2*recency
        """
        # Normalize citation count (log scale, cap at 1000)
        citation_score = min(100.0, (citation_count / 10.0) * 10)
        peer_score = 100.0 if peer_reviewed else 50.0

        overall = (
            0.30 * citation_score +
            0.25 * venue_if +
            0.25 * peer_score +
            0.20 * recency
        )
        return round(overall, 2)


# =============================================================================
# Memory Models
# =============================================================================

class EpisodicMemory(BaseModel):
    """
    Memory of a past research session (episodic).

    Stores "what happened when" - session summaries for temporal retrieval.
    """
    memory_id: str = Field(description="UUID for this memory")
    session_id: str = Field(description="LangGraph thread_id")
    timestamp: datetime
    query: str = Field(min_length=3)
    synthesis_claim_id: str = Field(description="UUID of final synthesis claim")
    key_papers: List[str] = Field(description="URLs of important papers from session")
    insights: List[str] = Field(description="Key takeaways (1-3 sentences each)")
    embedding: Optional[List[float]] = Field(default=None, description="Query embedding for similarity")


class SemanticMemory(BaseModel):
    """
    Long-term factual knowledge (semantic).

    Stores "what is true" - facts extracted from syntheses with confidence.
    """
    fact_id: str = Field(description="UUID for this fact")
    claim: str = Field(min_length=20, description="The factual claim")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    sources: List[str] = Field(min_length=1, description="Paper URLs supporting this fact")
    created_at: datetime
    last_verified: datetime
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = Field(default=None, description="Claim embedding")

    def decay_confidence(self, days_since_access: int) -> float:
        """
        Apply time-based decay to confidence.

        Formula: confidence * exp(-days/90)
        After 90 days without access, confidence halves.
        """
        import math
        decay_factor = math.exp(-days_since_access / 90.0)
        return self.confidence * decay_factor


class ProceduralMemory(BaseModel):
    """
    Learned research strategies (procedural).

    Stores "how to do X" - successful patterns for future use.
    """
    strategy_id: str
    description: str = Field(min_length=20, description="What this strategy does")
    success_rate: float = Field(ge=0.0, le=1.0, description="Historical success (0-1)")
    usage_count: int = Field(ge=0, description="Times used")
    last_used: datetime


# =============================================================================
# AgentStateV2 - Extended for 10-Agent Architecture
# =============================================================================

class AgentContribution(TypedDict):
    """
    Tracks one agent's contribution to the debate.
    """
    agent_name: str  # "Literature Surveyor", "Pattern Recognizer", etc.
    round_number: int
    content: str  # The agent's output (claim, critique, etc.)
    papers_cited: List[str]  # URLs
    timestamp: datetime


class AgentStateV2(TypedDict):
    """
    Extended state for 10-agent architecture with persistent memory.

    Inherits all fields from AgentState, adds:
    - Agent contribution tracking
    - Memory retrieval fields
    - Quality scores for discovered papers
    - Coordinator decisions
    """
    # =========================================================================
    # Tier 1 Fields (Inherited from AgentState)
    # =========================================================================
    messages: List  # LangGraph add_messages reducer
    original_query: str
    current_thesis: Optional[Thesis]
    current_antithesis: Optional[Antithesis]
    final_synthesis: Optional[Synthesis]
    contradiction_report: str
    iteration_count: int
    procedural_memory: str  # Legacy Tier 3 field
    debate_memory: Dict  # Rejected claims, objections, weak URLs
    current_claim_id: str
    synthesis_mode: Optional[str]
    consecutive_high_similarity_count: int
    last_similarity_score: Optional[float]
    conversation_history: List
    current_round_papers_analyst: List[str]
    current_round_papers_skeptic: List[str]

    # =========================================================================
    # Phase 2 New Fields
    # =========================================================================

    # Agent Coordination
    agent_contributions: List[AgentContribution]  # Full debate transcript
    active_agents: List[str]  # Which agents Coordinator activated this round
    coordinator_reasoning: str  # Why Coordinator chose these agents

    # Memory Retrieval
    retrieved_episodic_memories: List[EpisodicMemory]  # Past sessions
    retrieved_semantic_facts: List[SemanticMemory]  # Long-term knowledge
    memory_context_summary: str  # "System remembers: [summary]"

    # Quality Tracking
    paper_quality_scores: Dict[str, PaperQualityScoreV2]  # URL → quality
    high_quality_papers: List[str]  # URLs with score >80
    low_quality_papers: List[str]  # URLs with score <40

    # PDF Processing
    pdf_extraction_status: Dict[str, str]  # URL → "success"|"failed"|"pending"
    extracted_text_chunks: Dict[str, List[str]]  # URL → list of text chunks
```

**Task 1.5: Neo4j Migration Script**

Create `scripts/migrate_neo4j_phase2.py`:
```python
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
```

Run it:
```bash
python scripts/migrate_neo4j_phase2.py
```

**Task 1.6: ChromaDB Setup**

Create `src/memory/vector_store.py`:
```python
"""
ChromaDB vector store for semantic memory.

Provides:
- Paper embedding storage
- Semantic similarity search
- Session query embeddings
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Initialize embedding model (E5-large-v2, 1024-dim)
# This will download ~500MB on first run
EMBEDDING_MODEL = SentenceTransformer('intfloat/e5-large-v2')


class VectorStore:
    """
    Wrapper for ChromaDB operations.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistence."""
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Create or get collections
        self.papers = self.client.get_or_create_collection(
            name="papers",
            metadata={"description": "Paper embeddings (title + abstract)"}
        )

        self.sessions = self.client.get_or_create_collection(
            name="sessions",
            metadata={"description": "Session query embeddings"}
        )

        logger.info(f"✅ ChromaDB initialized: {persist_directory}")
        logger.info(f"   Papers: {self.papers.count()} documents")
        logger.info(f"   Sessions: {self.sessions.count()} documents")

    def add_paper(self, paper_url: str, title: str, abstract: str, metadata: Dict = None):
        """
        Add paper embedding to ChromaDB.

        Args:
            paper_url: Unique identifier (URL)
            title: Paper title
            abstract: Paper abstract
            metadata: Additional metadata (authors, year, etc.)
        """
        # E5 requires prepending "passage: " for documents
        text = f"passage: {title}. {abstract}"
        embedding = EMBEDDING_MODEL.encode(text).tolist()

        self.papers.add(
            ids=[paper_url],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )

        logger.debug(f"Added paper to ChromaDB: {title[:50]}...")

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search papers by semantic similarity.

        Args:
            query: Natural language query
            top_k: Number of results

        Returns:
            List of dicts with {id, distance, metadata}
        """
        # E5 requires prepending "query: " for queries
        query_text = f"query: {query}"
        query_embedding = EMBEDDING_MODEL.encode(query_text).tolist()

        results = self.papers.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        papers = []
        for i in range(len(results['ids'][0])):
            papers.append({
                'url': results['ids'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })

        return papers


# Singleton instance
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    """Get or create VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
```

Test it:
```bash
# Create test script
cat > test_chromadb.py << 'EOF'
from src.memory.vector_store import get_vector_store

vs = get_vector_store()

# Add test paper
vs.add_paper(
    paper_url="http://arxiv.org/abs/1706.03762",
    title="Attention Is All You Need",
    abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
    metadata={"year": 2017, "authors": "Vaswani et al."}
)

# Search
results = vs.semantic_search("transformer architecture", top_k=5)
print(f"Found {len(results)} papers")
for r in results:
    print(f"  - {r['url']} (distance: {r['distance']:.3f})")
EOF

python test_chromadb.py
```

**Expected Output:**
```
✅ ChromaDB initialized: ./chroma_db
   Papers: 1 documents
   Sessions: 0 documents
Found 1 papers
  - http://arxiv.org/abs/1706.03762 (distance: 0.234)
```

---

### Day 6-7: First Commit & Review

**Create First Commit:**
```bash
git add .
git commit -m "feat(phase2): Epic 1 complete - Foundation setup

- Add ChromaDB + sentence-transformers dependencies
- Create /src/agents_v2/ directory structure
- Design AgentStateV2 with memory fields
- Neo4j migration script for embeddings
- ChromaDB integration with E5-large-v2

Tasks completed: 1.1-1.6 (10.5h estimated)
Next: Epic 2 - PDF Extraction Pipeline"

git push origin phase-2/foundations
```

**Week 1 Review Checklist:**
- [ ] All Epic 1 tasks marked ✅ in TASKS.md
- [ ] Test passed: ChromaDB stores and retrieves embeddings
- [ ] Test passed: Neo4j migration adds embedding columns
- [ ] AgentStateV2 model validates without errors
- [ ] No regression: MVP still works with existing agents

**Update TASKS.md:**
```markdown
| 1.1: Add ChromaDB to requirements.txt | ✅ | You | 2h | 1.5h | Faster than expected |
```

---

## 🚀 Next Steps

**Week 2: Epic 2 + 3**
- PDF extraction with PyMuPDF
- Quality scoring engine
- First integration test

**Refer to:**
- `.speckit/SPEC.md` Phase 3 for detailed task descriptions
- `.speckit/TASKS.md` for tracking progress

**Weekly Routine:**
1. Monday: Review SPEC.md, plan week's tasks
2. Daily: 2-hour focused coding blocks
3. Friday: Review progress, update TASKS.md, commit work

---

## ❓ FAQ

**Q: Can I skip ChromaDB and just use Neo4j vector search?**
A: Yes! Neo4j 5.11+ has native vector search. Update SPEC.md under "Risk 1" if you choose this path.

**Q: What if I'm behind schedule?**
A: Defer non-critical tasks to "Phase 2B". Core MVP = Memory + PDF + 5 key agents (skip Devil's Advocate, Domain Expert if needed).

**Q: Can I use a different embedding model?**
A: Yes, but stick with sentence-transformers library. E5-large-v2 is optimal for academic papers.

**Q: Do I need to finish all 10 agents?**
A: No. Minimum viable = 5 agents (Surveyor, Analyst, Reviewer, Synthesizer, Fact Checker). Others are enhancements.

---

## 📞 Getting Help

**Resources:**
- SpecKit Docs: https://github.com/github/spec-kit
- LangGraph Multi-Agent: https://langchain-ai.github.io/langgraph/tutorials/multi-agent/
- ChromaDB Docs: https://docs.trychroma.com/
- Neo4j Vector Search: https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/

**Questions?**
- Review `.speckit/SPEC.md` first
- Check your research document for architectural guidance
- Open issue in repo for blockers

---

**Happy Building! 🔥**
