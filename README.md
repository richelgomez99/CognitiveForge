# CognitiveForge

**A dialectical synthesis multi-agent AI system using LangGraph and Google Gemini.**

Generates novel insights through structured adversarial debate between Analyst, Skeptic, and Synthesizer agents.

---

## 📁 Project Structure

```
CognitiveForge/
├── README.md          # You are here
├── .specify/          # ⚠️ HIDDEN FOLDER - Spec Kit framework files
│   ├── memory/
│   │   └── constitution.md
│   ├── templates/
│   │   ├── spec-template.md
│   │   ├── plan-template.md
│   │   ├── tasks-template.md
│   │   ├── checklist-template.md
│   │   └── agent-file-template.md
│   └── scripts/
│       └── bash/      # Helper scripts
└── .cursor/           # ⚠️ HIDDEN FOLDER - Cursor IDE config
```

---

## 🚀 Spec Kit Commands (Use in Cursor Chat)

### Core Workflow (in order):
1. **`/speckit.constitution`** - Define project principles and engineering values
2. **`/speckit.specify`** - Create the baseline specification (requirements & user stories)
3. **`/speckit.plan`** - Develop the technical implementation plan
4. **`/speckit.tasks`** - Generate actionable task lists
5. **`/speckit.implement`** - Execute implementation

### Optional Enhancement Commands:
- **`/speckit.clarify`** - Ask structured questions to de-risk ambiguous areas (before planning)
- **`/speckit.analyze`** - Cross-artifact consistency & alignment report
- **`/speckit.checklist`** - Generate quality checklists for validation

---

## 🔍 To View Hidden Folders in Cursor:

**Press `Ctrl+H` in the file explorer** or:
1. Press `Ctrl+Shift+P`
2. Type: "Files: Show Hidden Files"
3. Or add this to your settings:
   ```json
   "files.exclude": {
     "**/.specify": false,
     "**/.cursor": false
   }
   ```

---

## 🏗️ Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Google Gemini 1.5/2.5 Pro
- **Backend**: FastAPI
- **Streaming**: Server-Sent Events (SSE)
- **Episodic Memory**: LangGraph Checkpointer (AsyncSqliteSaver)
- **Semantic Memory**: Neo4j
- **Frontend**: Streamlit → React + shadcn/magic-ui
- **Deployment**: Docker + Google Cloud Run

---

## 📖 Architecture

### 3-Agent System:
1. **Analyst (Thesis)** - Generates initial claims with evidence
2. **Skeptic (Antithesis)** - Identifies contradictions and weaknesses
3. **Synthesizer (Synthesis)** - Resolves conflicts into novel insights

### Dialectical Loop:
```
Analyst → Skeptic → [Contradiction?] 
    ↓ Yes              ↓ No
Refine Thesis    →  Synthesizer → Novel Insight
```

---

**Ready to begin? Start with `/speckit.constitution` in Cursor chat!** 🔥

