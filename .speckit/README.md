# SpecKit Workflow for CognitiveForge Phase 2

**Following spec-driven development methodology**

---

## 📐 The SpecKit Philosophy

> "AI models are exceptional at pattern completion, but not at mind reading. A vague prompt forces the model to guess at potentially thousands of unstated requirements."

**SpecKit solves this by:**
1. **Specify** first - write down what you're building and why
2. **Plan** the technical approach - architecture, tools, constraints
3. **Tasks** break it into actionable chunks with dependencies
4. **Implement** with AI assistance using clear, focused prompts

---

## 📁 Document Structure

```
.speckit/
├── README.md              # This file - workflow overview
├── SPEC.md               # Complete specification (30 pages)
├── TASKS.md              # Task tracker with estimates
└── GETTING_STARTED.md    # Week 1 setup guide
```

**Read in this order:**
1. `README.md` (5 min) - Understand the workflow
2. `SPEC.md` (30 min) - Read PHASE 1 (user journeys) and PHASE 2 (architecture)
3. `GETTING_STARTED.md` (15 min) - Set up Week 1
4. `TASKS.md` (ongoing) - Track daily progress

---

## 🔄 The 4-Phase Workflow

### Phase 1: SPECIFY (Already Done ✅)

**What:** Define user journeys, success criteria, constraints

**Key Outputs:**
- 3 user journeys (PhD researcher, interdisciplinary team, research lab)
- Measurable success targets (50-100 papers/query, <10 min, 70%+ coverage)
- Explicit non-goals (no advanced reasoning, no commercial deployment in Phase 2)

**Your Role:** Review SPEC.md Phase 1, ensure you understand the "why"

---

### Phase 2: PLAN (Already Done ✅)

**What:** Technical architecture, component breakdown, migration strategy

**Key Outputs:**
- Architecture diagram (10 agents + hybrid storage)
- Technology stack (ChromaDB + PyMuPDF + Sentence-Transformers)
- 10 agent role definitions
- Data models (AgentStateV2, Memory models)
- 8-week timeline

**Your Role:** Review SPEC.md Phase 2, understand the "how"

---

### Phase 3: TASKS (Ready to Execute 🚀)

**What:** Break specification into actionable tasks with estimates

**8 Epics:**
1. Foundation Setup (10.5h) - ChromaDB, Neo4j migration, AgentStateV2
2. PDF Extraction (12h) - PyMuPDF pipeline, chunking, embedding
3. Quality Scoring (8h) - Multi-factor rubric, citation fetching
4. Persistent Memory (20h) - Episodic/semantic storage, retrieval
5. 10-Agent Architecture (43h) - Implement specialized agents
6. Orchestrator (28h) - Coordinator, parallel execution
7. UI Enhancement (15h) - 10-agent view, memory panel
8. Testing & Docs (23h) - E2E tests, migration guide

**Total: 159.5 hours = 8 weeks × 20h/week**

**Your Role:** Pick tasks from TASKS.md, update status as you work

---

### Phase 4: IMPLEMENT (Your Work Starts Here 💻)

**What:** Code with AI assistance using focused prompts

**Daily Workflow:**
```
Morning:
1. Open TASKS.md
2. Pick next ⬜ task
3. Mark as 🟦 In Progress

Implementation:
4. Read task description in SPEC.md
5. Use Claude/ChatGPT with context:
   "I'm implementing Task X.Y from .speckit/SPEC.md.
    Current codebase: [describe]
    Task goal: [paste task description]
    Write the implementation."

Testing:
6. Write unit test for component
7. Run test, iterate until ✅

Commit:
8. Update TASKS.md: ⬜ → ✅
9. Git commit with task number
10. Push to phase-2/foundations branch

Weekly:
11. Friday: Review week vs TASKS.md
12. Adjust next week if behind
```

---

## 🎯 How to Use This with AI Coding Assistants

### When Working with Claude Code / GitHub Copilot

**Good Prompt (Spec-Driven):**
```
I'm implementing Task 2.2: PDF text extraction with PyMuPDF.

Context from .speckit/SPEC.md:
- Use PyMuPDF for 42ms/page speed
- Extract text with 3-line API: open, get_text(), close
- Accept 10-20% failures on scanned PDFs
- Store chunks in ChromaDB

Current codebase:
- src/tools/paper_discovery.py has download logic
- src/memory/vector_store.py has ChromaDB wrapper

Write src/tools/pdf_extraction.py with:
- extract_text_from_pdf(file_path: str) -> str
- Error handling for scanned PDFs
- Integration with existing discovery pipeline
```

**Bad Prompt (Vague):**
```
Write a PDF extractor
```

### Providing Context

**Always include:**
1. Task number from TASKS.md
2. Goal from SPEC.md
3. Relevant existing code paths
4. Success criteria

**Claude Code will:**
- Follow architecture decisions from SPEC.md
- Integrate with existing patterns
- Generate tests matching your standards

---

## 📊 Tracking Progress

### Update TASKS.md Daily

**Status Symbols:**
- ⬜ Not Started
- 🟦 In Progress
- ✅ Complete
- ⏸️ Blocked
- ❌ Cancelled

**Example Updates:**
```markdown
| Task | Status | Assignee | Est. | Actual | Notes |
|------|--------|----------|------|--------|-------|
| 1.1: Add ChromaDB | ✅ | You | 2h | 1.5h | Faster! |
| 1.2: Add sentence-transformers | 🟦 | You | 1h | - | Downloading model... |
| 1.3: Create agents_v2/ | ⬜ | - | 0.5h | - | Next |
```

### Weekly Review

**Every Friday:**
1. Count completed tasks: `grep "✅" .speckit/TASKS.md | wc -l`
2. Calculate % complete: `completed / total_tasks * 100`
3. Adjust next week if behind:
   - Defer non-critical agents (Devil's Advocate, Domain Expert)
   - Focus on core: Memory + PDF + 5 key agents
4. Update SPEC.md if architecture changed

---

## 🚨 When Things Go Wrong

### If Falling Behind Schedule

**Week 3 checkpoint:**
- [ ] Epic 1 done? (Foundation)
- [ ] Epic 2 done? (PDF)
- [ ] Epic 3 done? (Quality)

**If NO to any:**
1. Review SPEC.md "Migration Strategy" - use flag-based activation
2. Implement core first, enhancements later
3. Update TASKS.md: defer low-priority tasks
4. Adjust Week 4-8 plan

### If Architecture Doesn't Work

**Example: ChromaDB too slow**

1. Update SPEC.md Phase 2 under "Risk 1":
   ```markdown
   ## Risk 1: ChromaDB Performance (UPDATED)
   - **Status:** TRIGGERED - 200ms+ query time at 5K papers
   - **Decision:** Switch to Neo4j native vector search (5.11+)
   - **Changes:** Remove ChromaDB, use Neo4j VECTOR INDEX
   ```

2. Update TASKS.md:
   ```markdown
   | 1.1: Add ChromaDB | ❌ | - | - | - | Cancelled: Using Neo4j vectors |
   | 1.1b: Neo4j vector index | 🟦 | - | 3h | - | Replacement |
   ```

3. Document decision in commit:
   ```bash
   git commit -m "arch(phase2): Switch from ChromaDB to Neo4j vector search

   ChromaDB query time exceeded 200ms at 5K papers.
   Neo4j native vector index provides <50ms with same quality.

   Updated SPEC.md Risk 1 and TASKS.md Epic 1."
   ```

### If Task Takes 2× Longer Than Estimated

**Normal and expected!** First-time implementations often overshoot.

**Update TASKS.md:**
```markdown
| 4.4: get_relevant_memories() | ✅ | You | 6h | 12h | Complex query logic |
```

**Adjust future estimates:**
- If multiple tasks overshoot by 2×, multiply remaining estimates by 1.5×
- Extend timeline OR defer features

---

## 🎓 Learning from SpecKit

### What Makes This Different from "Just Coding"

**Without SpecKit:**
```
User: "Add memory to agents"
→ Build something
→ Realize it doesn't fit existing architecture
→ Rewrite
→ Repeat 3× until it works
```

**With SpecKit:**
```
User: "Add memory to agents"
→ SPECIFY: What memory types? (episodic, semantic, procedural)
→ PLAN: How to store? (Neo4j + ChromaDB)
→ TASKS: Break into 6 tasks with dependencies
→ IMPLEMENT: Each task builds on previous, no rewrites
```

**Result:** 3× faster development, higher quality, less frustration

### The "Contract" Concept

> "The specification becomes the primary artifact, and code becomes its expression."

**Your SPEC.md is the contract:**
- AI assistants implement against this contract
- Tests validate against this contract
- User expectations match this contract

**Benefits:**
- Fewer surprises ("I thought it would do X")
- Clearer scope ("That's explicitly a non-goal")
- Better collaboration ("Read the spec first")

---

## 🔗 Resources

**SpecKit Official:**
- GitHub Repo: https://github.com/github/spec-kit
- Blog Post: https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/
- Medium Guide: https://steviee.medium.com/from-prd-to-production-my-spec-kit-workflow-for-structured-development-d9bf6631d647

**CognitiveForge Phase 2:**
- Complete Spec: `.speckit/SPEC.md`
- Task Tracker: `.speckit/TASKS.md`
- Setup Guide: `.speckit/GETTING_STARTED.md`

**Related Methodologies:**
- Behavior-Driven Development (BDD) - similar "specify first" approach
- Test-Driven Development (TDD) - write tests from spec, then code
- README-Driven Development - spec is the README

---

## ✅ Next Steps

**Right Now:**
1. ✅ Read this README (done!)
2. ⬜ Read SPEC.md Phase 1 & 2 (30 min)
3. ⬜ Read GETTING_STARTED.md (15 min)
4. ⬜ Set up Week 1 environment (4 hours)
5. ⬜ Start Epic 1: Task 1.1 (2 hours)

**This Week:**
- Complete Epic 1: Foundation Setup (10.5h)
- First commit by Day 7

**This Month:**
- Complete Epic 1-4 (Memory system working)
- Mid-phase review and adjustment

**By Week 8:**
- All 8 epics complete
- 10-agent system with persistent memory
- Migration guide for users

---

**Welcome to Phase 2! 🚀**

The journey from "smart search" to "research intelligence" starts with a solid specification.
You've got the spec. Now go build something exceptional.

*Remember: The spec is your friend. When in doubt, read the spec.*
