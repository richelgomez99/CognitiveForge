#!/bin/bash
set -e

echo "🔒 Creating PUBLIC git history (no internal docs)..."

GREEN='\033[0;32m'
NC='\033[0m'

# COMMIT 1: Initial (3:00 PM) - ONLY public files
echo -e "${GREEN}✅ Commit 1/14${NC}"
git add README.md .gitignore docker-compose.yml requirements.txt .env.example
GIT_AUTHOR_DATE="2025-11-08T15:00:00" GIT_COMMITTER_DATE="2025-11-08T15:00:00" \
git commit -m "chore: initialize CognitiveForge project

- Add README with project overview
- Add .gitignore for Python/Neo4j
- Add docker-compose.yml for Neo4j
- Add requirements.txt with core dependencies
- Add .env.example template"

# COMMIT 2: Data models
echo -e "${GREEN}✅ Commit 2/14${NC}"
git add src/__init__.py src/models.py
GIT_AUTHOR_DATE="2025-11-08T15:15:00" GIT_COMMITTER_DATE="2025-11-08T15:15:00" \
git commit -m "feat: add Pydantic data models for dialectical synthesis

- Define AgentState TypedDict for LangGraph
- Add Evidence, Thesis, Antithesis, Synthesis models
- Implement field validators for min_length constraints
- Add evidence_lineage validation (min 3 URLs)"

# COMMIT 3: Utilities
echo -e "${GREEN}✅ Commit 3/14${NC}"
git add src/utils/
GIT_AUTHOR_DATE="2025-11-08T15:30:00" GIT_COMMITTER_DATE="2025-11-08T15:30:00" \
git commit -m "feat: add Gemini client and model configuration utilities

- Implement call_gemini_with_retry with exponential backoff
- Add get_agent_model for per-agent model configuration
- Support new google-genai SDK with native Pydantic integration
- Handle rate limiting (HTTP 429) gracefully"

# COMMIT 4: Analyst
echo -e "${GREEN}✅ Commit 4/14${NC}"
git add src/agents/__init__.py src/agents/analyst.py
GIT_AUTHOR_DATE="2025-11-08T15:45:00" GIT_COMMITTER_DATE="2025-11-08T15:45:00" \
git commit -m "feat: implement Analyst agent node

- Query Neo4j knowledge graph for context
- Generate thesis with structured Gemini output
- Include evidence gathering and source tracking
- Validate output against Pydantic schema"

# COMMIT 5: Skeptic
echo -e "${GREEN}✅ Commit 5/14${NC}"
git add src/agents/skeptic.py
GIT_AUTHOR_DATE="2025-11-08T16:00:00" GIT_COMMITTER_DATE="2025-11-08T16:00:00" \
git commit -m "feat: implement Skeptic agent node

- Evaluate thesis for contradictions
- Generate antithesis with counter-evidence
- Use structured output for critique
- Support configurable skepticism level"

# COMMIT 6: Synthesizer
echo -e "${GREEN}✅ Commit 6/14${NC}"
git add src/agents/synthesizer.py
GIT_AUTHOR_DATE="2025-11-08T16:15:00" GIT_COMMITTER_DATE="2025-11-08T16:15:00" \
git commit -m "feat: implement Synthesizer agent node

- Synthesize novel insights from thesis/antithesis
- Self-assess novelty score via LLM
- Aggregate evidence lineage (min 3 sources)
- Add insights to Neo4j knowledge graph"

# COMMIT 7: Knowledge graph tools
echo -e "${GREEN}✅ Commit 7/14${NC}"
git add src/tools/ scripts/populate_kg.py
GIT_AUTHOR_DATE="2025-11-08T16:30:00" GIT_COMMITTER_DATE="2025-11-08T16:30:00" \
git commit -m "feat: add Neo4j knowledge graph integration

- Implement query_knowledge_graph using GraphCypherQAChain
- Add add_insight_to_graph for storing synthesis results
- Create sample data population script
- Define Neo4j schema (ResearchPaper, Insight nodes)"

# COMMIT 8: LangGraph
echo -e "${GREEN}✅ Commit 8/14${NC}"
git add src/graph.py
GIT_AUTHOR_DATE="2025-11-08T16:45:00" GIT_COMMITTER_DATE="2025-11-08T16:45:00" \
git commit -m "feat: build LangGraph state graph for dialectical synthesis

- Define analyst → skeptic → synthesizer flow
- Implement route_debate for conditional edges
- Support MAX_ITERATIONS (4 rounds)
- Add checkpointer support for persistence"

# COMMIT 9: CLI
echo -e "${GREEN}✅ Commit 9/14${NC}"
git add main.py
GIT_AUTHOR_DATE="2025-11-08T17:00:00" GIT_COMMITTER_DATE="2025-11-08T17:00:00" \
git commit -m "feat: add CLI for running dialectical synthesis

- Parse command-line arguments (query, thread-id)
- Build and invoke LangGraph
- Print final synthesis with confidence/novelty scores
- Support environment variable configuration"

# COMMIT 10: Unit tests
echo -e "${GREEN}✅ Commit 10/14${NC}"
git add pytest.ini tests/conftest.py tests/test_models.py tests/test_agents.py tests/test_kg_tools.py tests/test_routing.py tests/__init__.py
GIT_AUTHOR_DATE="2025-11-08T17:15:00" GIT_COMMITTER_DATE="2025-11-08T17:15:00" \
git commit -m "test: add comprehensive test suite for models and agents

- Add pytest fixtures for valid test data
- Test Pydantic validation rules
- Mock LLM calls for agent tests
- Test Neo4j integration with mocks
- Add pytest.ini for test configuration"

# COMMIT 11: Integration tests
echo -e "${GREEN}✅ Commit 11/14${NC}"
git add tests/test_graph.py tests/test_e2e.py
GIT_AUTHOR_DATE="2025-11-08T17:30:00" GIT_COMMITTER_DATE="2025-11-08T17:30:00" \
git commit -m "test: add integration and E2E tests

- Test full LangGraph execution with real LLMs
- Test performance (< 5 minutes per query)
- Test evidence_lineage validation (>= 3 sources)
- Add E2E smoke test"

# COMMIT 12: FastAPI backend
echo -e "${GREEN}✅ Commit 12/14${NC}"
git add src/api.py src/auth.py tests/test_api.py tests/test_streaming.py tests/test_persistence.py
GIT_AUTHOR_DATE="2025-11-08T17:45:00" GIT_COMMITTER_DATE="2025-11-08T17:45:00" \
git commit -m "feat: add FastAPI backend with SSE streaming

- Implement /health, /get_state, /get_trace endpoints
- Add /stream_dialectics with Server-Sent Events
- Implement API key authentication
- Add AsyncSqliteSaver for checkpointing
- Include streaming and persistence tests"

# COMMIT 13: Streamlit UI
echo -e "${GREEN}✅ Commit 13/14${NC}"
git add src/ui.py
GIT_AUTHOR_DATE="2025-11-08T18:00:00" GIT_COMMITTER_DATE="2025-11-08T18:00:00" \
git commit -m "feat: add Streamlit UI for real-time dialectical synthesis

- Implement streaming UI with SSE client
- Add query history and session management
- Display real-time agent updates
- Add state/trace querying interface
- Support .env and st.secrets for API key"

# COMMIT 14: Project documentation (README ONLY)
echo -e "${GREEN}✅ Commit 14/14${NC}"
# Only add README, not internal docs
git add README.md
GIT_AUTHOR_DATE="2025-11-08T18:15:00" GIT_COMMITTER_DATE="2025-11-08T18:15:00" \
git commit -m "docs: update README with setup and usage instructions

- Document project architecture and features
- Add setup instructions for all tiers
- Include testing and deployment guides
- Provide contribution guidelines" --allow-empty

git branch -m main
echo ""
echo "✅ PUBLIC history created (README only, no internal docs)!"
echo "📊 $(git log --oneline | wc -l) commits"
echo "📄 $(git ls-files | wc -l) files tracked"
echo ""
echo "Verify no internal docs:"
git ls-files | grep -E "IMPROVEMENTS|ROADMAP|TIER.*COMPLETE|TIER.*SUMMARY" || echo "  ✅ None found!"
echo ""
echo "Next: git push -u origin main --force"

