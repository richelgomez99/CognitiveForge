# CognitiveForge Phase 2: Task Tracker

**Status Legend:** ⬜ Not Started | 🟦 In Progress | ✅ Complete | ⏸️ Blocked | ❌ Cancelled

---

## Epic 1: Foundation Setup (Week 1)
**Goal:** Prepare codebase and dependencies for 10-agent architecture

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 1.1: Add ChromaDB to requirements.txt | ⬜ | - | 2h | - | Test local install |
| 1.2: Add sentence-transformers | ⬜ | - | 1h | - | ~500MB model download |
| 1.3: Create /src/agents_v2/ structure | ⬜ | - | 0.5h | - | Parallel development |
| 1.4: Design AgentStateV2 TypedDict | ⬜ | - | 2h | - | Add memory fields |
| 1.5: Neo4j migration script | ⬜ | - | 3h | - | Add embedding columns |
| 1.6: ChromaDB collection setup | ⬜ | - | 2h | - | Test insert/query |

**Epic Total:** 10.5h estimated

---

## Epic 2: PDF Extraction Pipeline (Week 2)
**Goal:** Enable full-text processing of discovered papers

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 2.1: Install PyMuPDF | ⬜ | - | 2h | - | Write download_pdf() |
| 2.2: extract_text_from_pdf() | ⬜ | - | 2h | - | PyMuPDF 3-line API |
| 2.3: Semantic chunking | ⬜ | - | 3h | - | RecursiveCharacterTextSplitter |
| 2.4: embed_and_store_chunks() | ⬜ | - | 2h | - | ChromaDB integration |
| 2.5: PDF extraction LangGraph node | ⬜ | - | 2h | - | Add to workflow |
| 2.6: Error handling | ⬜ | - | 1h | - | Skip scanned PDFs |

**Epic Total:** 12h estimated

---

## Epic 3: Quality Scoring Engine (Week 2)
**Goal:** Rank papers by multi-factor quality

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 3.1: PaperQualityScoreV2 model | ⬜ | - | 1h | - | Pydantic schema |
| 3.2: calculate_quality_score() | ⬜ | - | 3h | - | Fetch citations, venue map |
| 3.3: Add quality_score to Neo4j | ⬜ | - | 1h | - | Schema migration |
| 3.4: Update discovery pipeline | ⬜ | - | 2h | - | Score post-retrieval |
| 3.5: Sort by quality in KG queries | ⬜ | - | 1h | - | ORDER BY quality_score DESC |

**Epic Total:** 8h estimated

---

## Epic 4: Persistent Memory System (Week 3-4)
**Goal:** Agents remember knowledge across sessions

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 4.1: Memory Pydantic models | ⬜ | - | 2h | - | Semantic + Episodic |
| 4.2: store_episodic_memory() | ⬜ | - | 4h | - | Session summaries to Neo4j |
| 4.3: store_semantic_facts() | ⬜ | - | 4h | - | Extract facts from synthesis |
| 4.4: get_relevant_memories() | ⬜ | - | 6h | - | Semantic + graph retrieval |
| 4.5: Memory decay function | ⬜ | - | 2h | - | Reduce confidence over time |
| 4.6: Integrate into Analyst prompt | ⬜ | - | 2h | - | "System remembers..." context |

**Epic Total:** 20h estimated

---

## Epic 5: 10-Agent Architecture (Week 4-5)
**Goal:** Implement specialized agent roles

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 5.1: Deep Analyst agent | ⬜ | - | 6h | - | PDF methodology extraction |
| 5.2: Pattern Recognizer agent | ⬜ | - | 6h | - | Semantic similarity |
| 5.3: Hypothesis Generator agent | ⬜ | - | 5h | - | Combinatorial reasoning |
| 5.4: Fact Checker agent | ⬜ | - | 6h | - | Claim verification |
| 5.5: Domain Expert agent | ⬜ | - | 5h | - | Field-specific memory |
| 5.6: Methodologist agent | ⬜ | - | 5h | - | Study quality critique |
| 5.7: Devil's Advocate agent | ⬜ | - | 4h | - | Contrarian prompting |
| 5.8: Enhance Analyst → Surveyor | ⬜ | - | 3h | - | Add quality scoring |
| 5.9: Enhance Skeptic → Reviewer | ⬜ | - | 3h | - | Methodology critique |

**Epic Total:** 43h estimated

---

## Epic 6: Orchestrator & Graph Workflow (Week 5-6)
**Goal:** Coordinate 10-agent debate

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 6.1: Design build_graph_v2() | ⬜ | - | 4h | - | 10 agent nodes |
| 6.2: Coordinator agent | ⬜ | - | 8h | - | Determine activation |
| 6.3: Parallel execution | ⬜ | - | 6h | - | Independent agents |
| 6.4: Debate transcript structure | ⬜ | - | 3h | - | Track contributions |
| 6.5: AgentStateV2 fields | ⬜ | - | 2h | - | Agent-specific state |
| 6.6: Conditional routing | ⬜ | - | 5h | - | Simple vs complex queries |

**Epic Total:** 28h estimated

---

## Epic 7: UI Enhancement (Week 6-7)
**Goal:** Display 10-agent conversation intuitively

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 7.1: Accordion view for 10 agents | ⬜ | - | 4h | - | Streamlit expanders |
| 7.2: Memory Panel sidebar | ⬜ | - | 3h | - | Show past sessions |
| 7.3: Quality score badges | ⬜ | - | 2h | - | 🟢🟡🔴 indicators |
| 7.4: Agent avatar icons | ⬜ | - | 2h | - | 📚🔍🧪💡 emojis |
| 7.5: PDF extraction progress | ⬜ | - | 1h | - | st.progress() |
| 7.6: "Explain insight" button | ⬜ | - | 3h | - | Expand reasoning |

**Epic Total:** 15h estimated

---

## Epic 8: Testing & Documentation (Week 7-8)
**Goal:** Ensure reliability and usability

| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 8.1: E2E test 10-agent flow | ⬜ | - | 4h | - | Full workflow test |
| 8.2: Integration test memory | ⬜ | - | 3h | - | Retrieval accuracy |
| 8.3: Unit test quality scoring | ⬜ | - | 2h | - | Mock citations |
| 8.4: Unit test PDF extraction | ⬜ | - | 2h | - | Sample papers |
| 8.5: Update README | ⬜ | - | 2h | - | Phase 2 features |
| 8.6: Migration guide | ⬜ | - | 3h | - | MVP → Phase 2 |
| 8.7: Agent runbook | ⬜ | - | 4h | - | Prompt templates |
| 8.8: Performance benchmarking | ⬜ | - | 3h | - | 3-agent vs 10-agent |

**Epic Total:** 23h estimated

---

## Summary

| Epic | Status | Estimated | Actual | Progress |
|------|--------|-----------|--------|----------|
| 1. Foundation | ⬜ | 10.5h | - | 0% |
| 2. PDF Extraction | ⬜ | 12h | - | 0% |
| 3. Quality Scoring | ⬜ | 8h | - | 0% |
| 4. Persistent Memory | ⬜ | 20h | - | 0% |
| 5. 10-Agent Architecture | ⬜ | 43h | - | 0% |
| 6. Orchestrator | ⬜ | 28h | - | 0% |
| 7. UI Enhancement | ⬜ | 15h | - | 0% |
| 8. Testing & Docs | ⬜ | 23h | - | 0% |
| **TOTAL** | | **159.5h** | **0h** | **0%** |

**Timeline:** 8 weeks × 20h/week = 160h available (matches estimate!)

---

## Usage

**Update this file as you work:**

```bash
# Mark task in progress
🟦 Task 1.1: Add ChromaDB to requirements.txt

# Mark task complete
✅ Task 1.1: Add ChromaDB to requirements.txt | Actual: 1.5h

# Mark task blocked
⏸️ Task 4.4: get_relevant_memories() | Blocked: Need ChromaDB setup first
```

**Weekly Review Checklist:**
- [ ] Update status for all tasks worked on
- [ ] Record actual time vs estimate
- [ ] Adjust upcoming week tasks if behind schedule
- [ ] Identify blockers and mitigation plans
