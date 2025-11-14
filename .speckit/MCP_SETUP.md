# SpecKit MCP Server Setup

**Status:** ✅ Installed and Configured

---

## What is the SpecKit MCP Server?

The **Model Context Protocol (MCP) server for SpecKit** provides AI coding assistants (like Claude Code) with direct access to your project specifications, enabling:

- **Automated specification management** - Create and update specs via natural language
- **Implementation planning** - Generate step-by-step plans from specifications
- **Code generation** - Generate code that follows your specs
- **Project analysis** - Analyze existing code against specifications
- **Task breakdown** - Decompose work into actionable tasks

---

## Installation Summary

✅ **Completed on 2025-11-14**

### What Was Installed:

1. **spec-kit-mcp-go** (v0.0.0-20251002042128)
   - Location: `~/.local/bin/spec-kit-mcp`
   - Size: 7.6MB
   - Language: Go
   - Source: https://github.com/ahanoff/spec-kit-mcp-go

2. **specify-cli** (v0.0.22)
   - Installed via: `uv tool install`
   - Source: https://github.com/github/spec-kit
   - Executable: `specify` (in PATH)

3. **MCP Configuration**
   - Location: `~/.claude/mcp.json`
   - Working directory: `/home/user/CognitiveForge`
   - Timeout: 120 seconds

---

## Available MCP Tools

The SpecKit MCP server provides these tools to Claude Code:

### 1. `init_project`
**Initialize a new spec-kit project**

```
Use the init_project tool to create a new spec-kit project called "my-feature"
```

Parameters:
- `project_name` - Name of the project
- `ai_assistant` - AI assistant to use (default: "claude")

### 2. `specify`
**Create or update feature specifications from natural language**

```
Use the specify tool to create a specification for user authentication with
email and password, including password reset functionality
```

Parameters:
- `description` - Natural language description of the feature
- `output_file` - Optional file path (default: auto-generated)

### 3. `plan`
**Generate implementation plans from specifications**

```
Use the plan tool to create an implementation plan for the authentication spec
```

Parameters:
- `spec_file` - Path to the specification file
- `output_format` - Format (markdown, json, yaml)

### 4. `implement`
**Generate code from specifications**

```
Use the implement tool to generate Python code for the authentication module
from the spec
```

Parameters:
- `spec_file` - Path to specification
- `language` - Target language (python, typescript, go, etc.)
- `framework` - Framework to use (optional)

### 5. `analyze`
**Analyze project state and provide insights**

```
Use the analyze tool to check which specifications have been implemented
```

Parameters:
- `analysis_type` - Type of analysis (coverage, compliance, quality)
- `scope` - Scope to analyze (all, specific module, etc.)

### 6. `tasks`
**Break down specifications into actionable tasks**

```
Use the tasks tool to break down the authentication feature into tasks with
time estimates
```

Parameters:
- `spec_file` - Path to specification
- `granularity` - Task granularity (high, medium, low)
- `format` - Output format (markdown, jira, github)

---

## How to Use with Claude Code

### Method 1: Direct Tool Invocation

Claude Code will automatically detect the MCP server and make these tools available. Simply ask:

```
Use the specify tool to create a specification for the PDF extraction pipeline
described in .speckit/SPEC.md Epic 2
```

Claude Code will:
1. Connect to the MCP server
2. Call the `specify` tool with your description
3. Generate a formal specification file
4. Save it to your project

### Method 2: Natural Language (Implicit)

```
I want to create a specification for the memory system with episodic and
semantic storage. Can you help?
```

Claude Code will:
1. Recognize this as a spec-creation task
2. Use the `specify` MCP tool automatically
3. Generate the specification
4. Show you the result

### Method 3: Workflow Integration

```
Following .speckit/SPEC.md Epic 4, use the specify tool to create formal
specifications for:
1. Episodic memory storage
2. Semantic memory retrieval
3. Memory decay functions

Then use the plan tool to generate implementation plans for each.
```

Claude Code will:
1. Create 3 specification files
2. Generate implementation plans for each
3. Track dependencies between specifications

---

## Configuration Details

### MCP Server Configuration
**File:** `~/.claude/mcp.json`

```json
{
  "mcpServers": {
    "spec-kit": {
      "type": "stdio",
      "command": "/root/.local/bin/spec-kit-mcp",
      "env": {
        "SPEC_KIT_WORKING_DIR": "/home/user/CognitiveForge"
      },
      "timeout": 120000
    }
  }
}
```

**Key settings:**
- **type**: `stdio` - Communication via standard input/output
- **command**: Path to MCP server binary
- **SPEC_KIT_WORKING_DIR**: Your project root directory
- **timeout**: 120 seconds (2 minutes) for long-running operations

### Verify Installation

Test the MCP server manually:

```bash
# Test that the server responds to MCP protocol
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | ~/.local/bin/spec-kit-mcp
```

Expected output: JSON listing available tools.

---

## Integration with Phase 2 Development

### How This Helps Your SpecKit Workflow

**Before MCP Server:**
```
You: "Create Epic 1 specification"
→ Manually write specification in .speckit/ directory
→ Manually track in TASKS.md
→ Hope you didn't miss anything
```

**With MCP Server:**
```
You: "Use specify tool for Epic 1: Foundation Setup from .speckit/SPEC.md"
→ MCP server reads SPEC.md Epic 1
→ Generates formal specification file
→ Validates against SpecKit best practices
→ Links to TASKS.md automatically
```

### Workflow Example: Implementing Epic 2 (PDF Extraction)

**Step 1: Create Specification**
```
Use the specify tool to create a detailed specification for Epic 2:
PDF Extraction Pipeline, including:
- PyMuPDF integration requirements
- Chunking strategy (512-1024 tokens)
- ChromaDB storage interface
- Error handling for scanned PDFs

Reference .speckit/SPEC.md Epic 2 for context.
```

**Step 2: Generate Implementation Plan**
```
Use the plan tool to create a step-by-step implementation plan for the
PDF extraction specification, breaking it into 2-hour tasks.
```

**Step 3: Implement with Code Generation**
```
Use the implement tool to generate the pdf_extraction.py module following
the specification, using PyMuPDF and integrating with our existing
src/memory/vector_store.py ChromaDB wrapper.
```

**Step 4: Analyze Compliance**
```
Use the analyze tool to verify that src/tools/pdf_extraction.py meets
the specification requirements and identify any gaps.
```

---

## Troubleshooting

### MCP Server Not Responding

**Symptom:** Claude Code doesn't recognize `specify` tool

**Solutions:**
1. Verify MCP server is in PATH:
   ```bash
   ls -lh ~/.local/bin/spec-kit-mcp
   ```

2. Check MCP configuration:
   ```bash
   cat ~/.claude/mcp.json
   ```

3. Test server manually:
   ```bash
   echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | ~/.local/bin/spec-kit-mcp
   ```

4. Check Claude Code logs (if available)

### "specify command not found"

**Symptom:** MCP server can't find `specify` CLI

**Solution:**
```bash
# Verify specify is installed
specify --help

# If not, reinstall:
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
```

### Permission Denied

**Symptom:** Cannot execute MCP server

**Solution:**
```bash
chmod +x ~/.local/bin/spec-kit-mcp
```

### Working Directory Not Found

**Symptom:** MCP server can't access project files

**Solution:** Update `SPEC_KIT_WORKING_DIR` in `~/.claude/mcp.json`:
```json
{
  "env": {
    "SPEC_KIT_WORKING_DIR": "/home/user/CognitiveForge"
  }
}
```

---

## Advanced Usage

### Custom Specification Templates

Create custom templates in `.speckit/templates/`:

```bash
mkdir -p .speckit/templates
cat > .speckit/templates/agent_spec.md << 'EOF'
# Agent Specification: {agent_name}

## Purpose
{purpose}

## Inputs
{inputs}

## Outputs
{outputs}

## Tools Available
{tools}

## Prompt Template
{prompt}
EOF
```

Then use with MCP:
```
Use the specify tool with template .speckit/templates/agent_spec.md to create
a specification for the Pattern Recognizer agent from Epic 5
```

### Batch Operations

Process multiple specifications:
```
For each agent in Epic 5 (Deep Analyst, Pattern Recognizer, Hypothesis Generator,
Fact Checker, Domain Expert, Methodologist, Devil's Advocate):
1. Use specify tool to create agent specification
2. Use plan tool to generate implementation plan
3. Use tasks tool to break into 2-hour chunks

Save all outputs to .speckit/agents/
```

### Integration with Git Workflow

Commit specifications as you create them:
```
After using the specify tool to create the memory system specification:
1. Review the generated spec file
2. Commit with message: "spec: Add episodic memory specification (Epic 4)"
3. Use the plan tool to generate implementation plan
4. Commit plan: "plan: Add episodic memory implementation tasks"
```

---

## Benefits of MCP Integration

### 1. **Consistency**
- All specifications follow SpecKit format automatically
- No manual formatting errors
- Validated against best practices

### 2. **Speed**
- Generate specs in seconds vs minutes manually
- Automated task breakdown
- Instant implementation plans

### 3. **Traceability**
- Every specification linked to implementation
- Git history tracks spec evolution
- Easy to see "what changed and why"

### 4. **Intelligence**
- MCP server understands SpecKit methodology
- Suggests improvements based on patterns
- Detects inconsistencies across specifications

### 5. **Workflow Integration**
- Works seamlessly with Claude Code
- No context switching
- Natural language interface

---

## Next Steps

### Start Using MCP Server Now

**For Epic 1 (Foundation Setup):**
```
Use the specify tool to create a formal specification for Epic 1: Foundation Setup,
including:
- ChromaDB integration requirements
- Neo4j migration schema
- AgentStateV2 data model
- Vector store interface

Reference .speckit/SPEC.md Epic 1 and .speckit/GETTING_STARTED.md for details.
```

**For Epic 2 (PDF Extraction):**
```
Use the specify tool to create a specification for the PDF extraction pipeline:
- PyMuPDF integration (target: 42ms/page)
- Semantic chunking strategy
- ChromaDB storage
- Error handling for scanned PDFs
- Integration with discovery pipeline

Then use the plan tool to generate a step-by-step implementation plan.
```

**For the Full Phase 2:**
```
For each of the 8 epics in .speckit/SPEC.md Phase 3:
1. Use specify tool to create formal specification
2. Use plan tool to generate implementation plan
3. Use tasks tool to break into 2-hour chunks
4. Track in .speckit/TASKS.md

This will give you a complete, validated specification set for Phase 2.
```

---

## Resources

**SpecKit MCP Server:**
- GitHub: https://github.com/ahanoff/spec-kit-mcp-go
- Documentation: https://ahanoff.dev/blog/bridging-spec-kit-to-amazon-q-developer-with-mcp/

**SpecKit CLI:**
- GitHub: https://github.com/github/spec-kit
- Blog: https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/

**Model Context Protocol:**
- Official Docs: https://modelcontextprotocol.io
- GitHub: https://github.com/modelcontextprotocol

**Your Phase 2 Docs:**
- `.speckit/README.md` - Workflow overview
- `.speckit/SPEC.md` - Complete specification
- `.speckit/TASKS.md` - Task tracker
- `.speckit/GETTING_STARTED.md` - Week 1 guide

---

## Summary

✅ **MCP Server:** Installed at `~/.local/bin/spec-kit-mcp`
✅ **CLI Tool:** `specify` available in PATH
✅ **Configuration:** `~/.claude/mcp.json` created
✅ **Working Dir:** `/home/user/CognitiveForge`
✅ **Tools Available:** init_project, specify, plan, implement, analyze, tasks

**You can now use natural language with Claude Code to:**
- Create specifications from your .speckit/SPEC.md epics
- Generate implementation plans automatically
- Break down work into tasks with estimates
- Generate code following specifications
- Analyze code compliance with specs

**Try it now:**
```
Use the specify tool to create a specification for Epic 1: Foundation Setup
```

Happy spec-driven development! 🚀
