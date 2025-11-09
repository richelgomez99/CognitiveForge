"""
Streamlit UI for CognitiveForge - Dialectical Synthesis System

Real-time visualization of the multi-agent research process.
"""

import streamlit as st
import requests
import json
import os
import uuid
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Page configuration
st.set_page_config(
    page_title="CognitiveForge - Dialectical Synthesis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration - Load from .env or Streamlit secrets
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, skip

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Try environment variable first, then Streamlit secrets
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    try:
        API_KEY = st.secrets.get("API_KEY", "")
    except:
        pass

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "current_synthesis" not in st.session_state:
    st.session_state.current_synthesis = None


def check_backend_health() -> Dict[str, Any]:
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}


def extract_citations(text: str) -> Tuple[str, List[str]]:
    """
    Extract [CITE: url] citations from text and return cleaned text with list of URLs.
    
    Implements T055-T056: Parse citations from agent outputs.
    
    Args:
        text: Text containing [CITE: url] markers
    
    Returns:
        Tuple of (cleaned_text, list_of_urls)
    
    Example:
        >>> extract_citations("Research shows [CITE: http://arxiv.org/abs/123] that...")
        ("Research shows that...", ["http://arxiv.org/abs/123"])
    """
    citation_pattern = r'\[CITE:\s*(https?://[^\]]+)\]'
    citations = re.findall(citation_pattern, text)
    cleaned_text = re.sub(citation_pattern, '', text)
    return cleaned_text, citations


def display_round(round_data: Dict[str, Any], round_num: int) -> None:
    """
    Display one round of Analyst-Skeptic exchange in VERTICAL chat format.
    
    Implements T042-T047: Conversational thread view (Tier 2: US1).
    Following CONVERSATIONAL_THREAD_RESEARCH.md Approach B design.
    
    Args:
        round_data: ConversationRound dict with thesis, antithesis, and papers
        round_num: Round number (1-indexed)
    
    Layout: Messages stack vertically with natural left/right alignment
        🧠 Round N
        [🔬 ANALYST - user role, naturally left-aligned]
        [🔍 SKEPTIC - assistant role, naturally right-aligned]
    """
    # T043: Round header
    with st.container():
        st.markdown(f"### 🧠 Round {round_num}")
        
        thesis = round_data.get("thesis", {})
        papers_analyst = round_data.get("papers_analyst", [])
        antithesis = round_data.get("antithesis", {})
        papers_skeptic = round_data.get("papers_skeptic", [])
        contradiction_found = antithesis.get("contradiction_found", False)
        
        # T044: ANALYST message (user role - naturally left-aligned)
        with st.chat_message("user", avatar="🔬"):
            st.markdown("**ANALYST**")
            st.write(f"**Claim:** {thesis.get('claim', 'N/A')}")
            
            # Display papers discovered inline with proper citations
            if papers_analyst:
                st.caption(f"📚 **{len(papers_analyst)} papers** discovered for this claim")
                with st.expander("View papers", expanded=False):
                    for idx, paper in enumerate(papers_analyst, 1):
                        # Handle both old format (str) and new format (dict)
                        if isinstance(paper, str):
                            st.markdown(f"{idx}. [{paper}]({paper})")
                        else:
                            # Format citation: Authors. Title. [Link]
                            authors_str = ", ".join(paper.get("authors", ["Unknown"]))
                            title = paper.get("title", "Untitled")
                            url = paper.get("url", "#")
                            st.markdown(f"**{idx}. {authors_str}**")
                            st.markdown(f"   *{title}*")
                            st.markdown(f"   🔗 [{url}]({url})")
                            if idx < len(papers_analyst):
                                st.divider()
            else:
                st.caption("📚 No new papers discovered this round")
        
        # T045: SKEPTIC message (assistant role - naturally right-aligned)
        with st.chat_message("assistant", avatar="🔍"):
            st.markdown("**SKEPTIC**")
            
            if contradiction_found:
                st.warning(f"⚠️ **Contradictions Found**")
                if antithesis.get("counter_claim"):
                    st.write(f"**Counter-claim:** {antithesis.get('counter_claim')}")
            else:
                st.success(f"✅ **No Major Contradictions** - Claim accepted!")
            
            st.write(f"**Critique:** {antithesis.get('critique', 'N/A')[:300]}...")
            
            # Display counter-papers discovered inline with proper citations
            if papers_skeptic:
                st.caption(f"📖 **{len(papers_skeptic)} counter-papers** discovered")
                with st.expander("View counter-papers", expanded=False):
                    for idx, paper in enumerate(papers_skeptic, 1):
                        # Handle both old format (str) and new format (dict)
                        if isinstance(paper, str):
                            st.markdown(f"{idx}. [{paper}]({paper})")
                        else:
                            # Format citation: Authors. Title. [Link]
                            authors_str = ", ".join(paper.get("authors", ["Unknown"]))
                            title = paper.get("title", "Untitled")
                            url = paper.get("url", "#")
                            st.markdown(f"**{idx}. {authors_str}**")
                            st.markdown(f"   *{title}*")
                            st.markdown(f"   🔗 [{url}]({url})")
                            if idx < len(papers_skeptic):
                                st.divider()
            else:
                st.caption("📖 No counter-papers discovered")


def stream_dialectics(thread_id: str, query: str, auto_discover: bool = True) -> None:
    """
    Stream dialectical synthesis process via SSE.
    
    Implements T049-T052:
    - Passes auto_discover parameter to backend
    - Handles discovery events (discovery_start, paper_found, etc.)
    - Displays discovery progress in real-time
    - Creates collapsible section for discovered papers
    """
    if not API_KEY:
        st.error("❌ API_KEY not set. Please configure it in environment variables or Streamlit secrets.")
        return
    
    headers = {
        "X-API-Key": API_KEY,
        "Accept": "text/event-stream"
    }
    
    url = f"{API_BASE_URL}/stream_dialectics/{thread_id}"
    params = {"query": query, "auto_discover": auto_discover}  # T049: Pass auto_discover
    
    # Initialize session state for discovered papers
    if "current_discovered_papers" not in st.session_state:
        st.session_state.current_discovered_papers = []
    st.session_state.current_discovered_papers = []  # Reset for new session
    
    # Create containers for streaming content
    status_container = st.container()
    discovery_container = st.container()  # T050: Container for discovery events
    progress_bar = st.progress(0)
    agent_containers = {
        "analyst": st.container(),
        "skeptic": st.container(),
        "synthesizer": st.container()
    }
    
    with status_container:
        if auto_discover:
            st.info("🔬 Auto-discovery enabled: Searching for relevant papers...")
        else:
            st.info("🚀 Starting dialectical synthesis...")
    
    try:
        with requests.get(url, params=params, headers=headers, stream=True, timeout=300) as response:
            if response.status_code == 401:
                st.error("❌ Authentication failed. Invalid API key.")
                return
            elif response.status_code != 200:
                st.error(f"❌ Server error: {response.status_code}")
                return
            
            event_count = 0
            synthesis_complete = False
            
            discovery_status_placeholder = None
            discovery_papers_list = []
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                # Decode SSE event
                line_str = line.decode('utf-8')
                
                # T050: Parse SSE events with 'event:' prefix (for discovery events)
                event_type = None
                if line_str.startswith('event: '):
                    event_type = line_str[7:].strip()
                    continue  # Next line will have data
                
                if not line_str.startswith('data: '):
                    continue
                
                event_data_str = line_str[6:]  # Remove "data: " prefix
                
                try:
                    event_data = json.loads(event_data_str)
                    
                    # T050-T051: Handle discovery events
                    if event_type:
                        with discovery_container:
                            if event_type == "discovery_start":
                                discovery_status_placeholder = st.empty()
                                discovery_status_placeholder.info(f"🔍 Searching for papers...")
                            
                            elif event_type == "source_searching":
                                source = event_data.get("source", "unknown")
                                if discovery_status_placeholder:
                                    discovery_status_placeholder.info(f"🔍 Searching {source}...")
                            
                            elif event_type == "paper_found":
                                # T051: Display each paper as it's found
                                title = event_data.get("title", "Unknown")
                                authors = event_data.get("authors", [])
                                source = event_data.get("source", "")
                                url = event_data.get("url", "#")
                                
                                discovery_papers_list.append(event_data)
                                st.session_state.current_discovered_papers.append(event_data)
                                
                                if discovery_status_placeholder:
                                    discovery_status_placeholder.success(f"📄 Found: {title[:60]}... by {', '.join(authors)}")
                            
                            elif event_type == "papers_ingesting":
                                count = event_data.get("count", 0)
                                if discovery_status_placeholder:
                                    discovery_status_placeholder.info(f"💾 Adding {count} papers to knowledge graph...")
                            
                            elif event_type == "discovery_complete":
                                added = event_data.get("added", 0)
                                skipped = event_data.get("skipped", 0)
                                duration = event_data.get("duration", 0)
                                
                                if discovery_status_placeholder:
                                    if added > 0:
                                        discovery_status_placeholder.success(
                                            f"✅ Discovery complete: {added} papers added, {skipped} skipped ({duration}s)"
                                        )
                                    else:
                                        discovery_status_placeholder.warning(
                                            f"⚠️ No new papers added ({skipped} duplicates, {duration}s)"
                                        )
                                
                                # T052: Create collapsible section for discovered papers
                                if discovery_papers_list:
                                    with st.expander(f"📚 Discovered Papers ({len(discovery_papers_list)})", expanded=False):
                                        for idx, paper in enumerate(discovery_papers_list, 1):
                                            st.markdown(f"**{idx}. [{paper['title']}]({paper['url']})**")
                                            st.caption(f"👤 {', '.join(paper['authors'][:3])} | 📚 {paper['source']}")
                                            st.divider()
                            
                            elif event_type == "discovery_error":
                                error = event_data.get("error", "Unknown error")
                                source = event_data.get("source", "unknown")
                                st.warning(f"⚠️ Discovery error ({source}): {error}")
                            
                            elif event_type == "discovery_timeout":
                                st.warning("⏱️ Discovery timeout - proceeding with synthesis")
                        
                        continue  # Don't process as agent event
                    
                    # Standard agent events (existing code)
                    node = event_data.get("node", "unknown")
                    data = event_data.get("data", {})
                    event_count += 1
                    
                    # Update progress
                    if node == "analyst":
                        progress_bar.progress(0.33)
                    elif node == "skeptic":
                        progress_bar.progress(0.66)
                    elif node == "synthesizer":
                        progress_bar.progress(1.0)
                        synthesis_complete = True
                    
                    # Display agent activity
                    container = agent_containers.get(node)
                    if container:
                        with container:
                            with st.chat_message("assistant", avatar="🤖"):
                                st.markdown(f"### **{node.upper()}** Agent")
                                
                                if node == "analyst" and "current_thesis" in data:
                                    thesis = data["current_thesis"]
                                    st.success("✅ Thesis Generated")
                                    st.markdown(f"**Claim:** {thesis.get('claim', 'N/A')}")
                                    with st.expander("📋 View Reasoning"):
                                        # T055: Extract and display citations inline
                                        reasoning_text = thesis.get('reasoning', 'N/A')
                                        cleaned_reasoning, inline_citations = extract_citations(reasoning_text)
                                        st.write(cleaned_reasoning)
                                        if inline_citations:
                                            st.caption("📖 Cited in reasoning:")
                                            for cite_url in inline_citations:
                                                st.caption(f"  • [{cite_url}]({cite_url})")
                                    with st.expander("📚 Evidence Sources"):
                                        for idx, ev in enumerate(thesis.get('evidence', []), 1):
                                            st.write(f"{idx}. [{ev.get('source_url', 'N/A')}]({ev.get('source_url', '#')})")
                                
                                elif node == "skeptic" and "current_antithesis" in data:
                                    antithesis = data["current_antithesis"]
                                    contradiction = antithesis.get('contradiction_found', False)
                                    
                                    if contradiction:
                                        st.warning("⚠️ Contradictions Found")
                                        st.markdown(f"**Counter-claim:** {antithesis.get('counter_claim', 'N/A')}")
                                    else:
                                        st.success("✅ No Major Contradictions")
                                    
                                    with st.expander("🔍 Critique"):
                                        # T055: Extract and display citations inline
                                        critique_text = antithesis.get('critique', 'N/A')
                                        cleaned_critique, inline_citations = extract_citations(critique_text)
                                        st.write(cleaned_critique)
                                        if inline_citations:
                                            st.caption("📖 Cited in critique:")
                                            for cite_url in inline_citations:
                                                st.caption(f"  • [{cite_url}]({cite_url})")
                                
                                elif node == "synthesizer" and "final_synthesis" in data:
                                    synthesis = data["final_synthesis"]
                                    st.session_state.current_synthesis = synthesis
                                    
                                    # Tier 2: Comprehensive Synthesis Display (T029-T034)
                                    st.success("✨ Comprehensive Research Synthesis Generated!")
                                    
                                    # T029: Key Insight Section
                                    st.markdown("### 📋 Key Insight")
                                    st.markdown(synthesis.get('novel_insight', 'N/A'))
                                    
                                    # T032: Confidence & Novelty Section (with justifications)
                                    st.markdown("### 📊 Quality Metrics")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        confidence = synthesis.get('confidence_score', 0.0)
                                        st.metric("Confidence", f"{confidence:.1f}/100")
                                        st.caption(synthesis.get('confidence_justification', 'No justification provided'))
                                    with col2:
                                        novelty = synthesis.get('novelty_score', 0.0)
                                        st.metric("Novelty", f"{novelty:.1f}/100")
                                        st.caption(synthesis.get('novelty_justification', 'No justification provided'))
                                    
                                    # T030: Dialectical Journey Section
                                    with st.expander("🧠 Dialectical Journey", expanded=True):
                                        st.markdown(synthesis.get('dialectical_summary', 'No dialectical summary available'))
                                        
                                        # Display rounds explored
                                        rounds_explored = synthesis.get('rounds_explored', [])
                                        if rounds_explored:
                                            st.markdown("#### Rounds Explored:")
                                            for round_data in rounds_explored:
                                                round_num = round_data.get('round_number', '?')
                                                thesis_claim = round_data.get('thesis_claim', 'No claim')
                                                rejection = round_data.get('rejection_reason')
                                                insights = round_data.get('key_insights', [])
                                                
                                                with st.container():
                                                    st.markdown(f"**Round {round_num}:**")
                                                    st.write(f"• Thesis: {thesis_claim}")
                                                    if rejection:
                                                        st.write(f"• ❌ Rejected: {rejection}")
                                                    else:
                                                        st.write(f"• ✅ Accepted")
                                                    if insights:
                                                        st.write(f"• Key Insights: {', '.join(insights)}")
                                                    st.divider()
                                    
                                    # T031: Evidence Base Section
                                    with st.expander("📚 Evidence Base", expanded=True):
                                        # Supporting Evidence
                                        supporting_evidence = synthesis.get('supporting_evidence', [])
                                        if supporting_evidence:
                                            st.markdown("#### Supporting Evidence:")
                                            for i, evidence in enumerate(supporting_evidence, 1):
                                                st.markdown(f"**{i}. {evidence.get('title', 'Unknown Title')}**")
                                                st.write(f"   🔗 [{evidence.get('url', '#')}]({evidence.get('url', '#')})")
                                                st.write(f"   💡 {evidence.get('how_it_supports', 'No explanation')}")
                                                if i < len(supporting_evidence):
                                                    st.divider()
                                        
                                        # Counter-Evidence Addressed
                                        counter_evidence = synthesis.get('contradicting_evidence_addressed', [])
                                        if counter_evidence:
                                            st.markdown("#### Counter-Evidence Addressed:")
                                            for i, evidence in enumerate(counter_evidence, 1):
                                                st.markdown(f"**{i}. {evidence.get('title', 'Unknown Title')}**")
                                                st.write(f"   🔗 [{evidence.get('url', '#')}]({evidence.get('url', '#')})")
                                                st.write(f"   ⚠️ Contradiction: {evidence.get('contradiction', 'No explanation')}")
                                                st.write(f"   ✅ Resolution: {evidence.get('resolution', 'No resolution')}")
                                                if i < len(counter_evidence):
                                                    st.divider()
                                    
                                    # T032: Comprehensive Reasoning (already included in Confidence & Novelty, moving here)
                                    with st.expander("🔬 Comprehensive Reasoning"):
                                        reasoning_text = synthesis.get('synthesis_reasoning', synthesis.get('reasoning', 'N/A'))
                                        cleaned_reasoning, inline_citations = extract_citations(reasoning_text)
                                        st.write(cleaned_reasoning)
                                        if inline_citations:
                                            st.caption("📖 Cited in reasoning:")
                                            for cite_url in inline_citations:
                                                st.caption(f"  • [{cite_url}]({cite_url})")
                                    
                                    # T033: Implications Section
                                    with st.expander("💡 Implications & Predictions"):
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.markdown("**Practical Implications:**")
                                            implications = synthesis.get('practical_implications', [])
                                            if implications:
                                                for i, impl in enumerate(implications, 1):
                                                    st.write(f"{i}. {impl}")
                                            else:
                                                st.caption("No implications provided")
                                        
                                        with col2:
                                            st.markdown("**Testable Predictions:**")
                                            predictions = synthesis.get('testable_predictions', [])
                                            if predictions:
                                                for i, pred in enumerate(predictions, 1):
                                                    st.write(f"{i}. {pred}")
                                            else:
                                                st.caption("No predictions provided")
                                        
                                        with col3:
                                            st.markdown("**Open Questions:**")
                                            questions = synthesis.get('open_questions', [])
                                            if questions:
                                                for i, q in enumerate(questions, 1):
                                                    st.write(f"{i}. {q}")
                                            else:
                                                st.caption("No open questions")
                                    
                                    # T034: References Section (Key Papers + All Citations)
                                    with st.expander("📖 References", expanded=True):
                                        # Key Papers with full context
                                        key_papers = synthesis.get('key_papers', [])
                                        if key_papers:
                                            st.markdown("#### Key Papers:")
                                            for i, paper in enumerate(key_papers, 1):
                                                st.markdown(f"**{i}. {paper.get('title', 'Unknown Title')}**")
                                                authors = paper.get('authors', [])
                                                year = paper.get('year')
                                                venue = paper.get('venue')
                                                
                                                # Paper metadata
                                                metadata_parts = []
                                                if authors:
                                                    metadata_parts.append(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
                                                if year:
                                                    metadata_parts.append(f"({year})")
                                                if venue:
                                                    metadata_parts.append(venue)
                                                
                                                if metadata_parts:
                                                    st.caption(" • ".join(metadata_parts))
                                                
                                                st.write(f"   🔗 [{paper.get('url', '#')}]({paper.get('url', '#')})")
                                                st.write(f"   📝 {paper.get('role_in_synthesis', 'No role specified')}")
                                                
                                                if i < len(key_papers):
                                                    st.divider()
                                        
                                        # All References (evidence lineage)
                                        st.markdown("#### All References:")
                                        all_refs = set(synthesis.get('evidence_lineage', []))
                                        # Add inline citations from reasoning
                                        if inline_citations:
                                            all_refs.update(inline_citations)
                                        
                                        if all_refs:
                                            st.caption(f"**{len(all_refs)} references cited in this synthesis:**")
                                            for idx, url in enumerate(sorted(all_refs), 1):
                                                st.write(f"{idx}. [{url}]({url})")
                                        else:
                                            st.caption("No external references cited.")
                
                except json.JSONDecodeError:
                    st.warning(f"Failed to parse event: {event_data_str[:100]}")
            
            # Completion message
            with status_container:
                if synthesis_complete:
                    st.success(f"🎉 Dialectical synthesis complete! ({event_count} events processed)")
                    
                    # Tier 2 (T041-T047): Display conversational thread view
                    try:
                        # Query final state to get conversation_history
                        state_response = requests.get(
                            f"{API_BASE_URL}/get_state/{thread_id}",
                            headers={"X-API-Key": API_KEY},
                            timeout=5
                        )
                        
                        if state_response.status_code == 200:
                            state = state_response.json()
                            conversation_history = state.get("conversation_history", [])
                            
                            if conversation_history:
                                # T041: Display all rounds in conversational thread format
                                st.markdown("---")
                                st.markdown("## 🔄 Dialectical Process")
                                st.caption(f"**{len(conversation_history)} rounds** of Analyst-Skeptic debate")
                                
                                # T042: Loop through all rounds
                                for i, round_data in enumerate(conversation_history, 1):
                                    display_round(round_data, i)
                                    
                                    # T046: Add visual separator between rounds (not after last)
                                    if i < len(conversation_history):
                                        st.divider()
                                
                                st.markdown("---")
                    except Exception as e:
                        # Don't fail the entire UI if thread view fails
                        st.warning(f"⚠️ Could not load conversation history: {str(e)}")
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "thread_id": thread_id,
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "synthesis": st.session_state.current_synthesis
                    })
                else:
                    st.warning("⚠️ Stream ended without final synthesis")
    
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The synthesis may be taking longer than expected.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {str(e)}")


def query_state(thread_id: str) -> None:
    """Query and display the current state of a thread."""
    if not API_KEY:
        st.error("❌ API_KEY not set.")
        return
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/get_state/{thread_id}",
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 404:
            st.warning("⚠️ Thread not found. It may not have been started yet.")
            return
        elif response.status_code == 401:
            st.error("❌ Authentication failed.")
            return
        
        response.raise_for_status()
        state = response.json()
        
        st.success("✅ State Retrieved")
        
        with st.expander("📊 Full Agent State", expanded=True):
            st.json(state)
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to query state: {str(e)}")


def query_trace(thread_id: str) -> None:
    """Query and display the checkpoint history of a thread."""
    if not API_KEY:
        st.error("❌ API_KEY not set.")
        return
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/get_trace/{thread_id}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 404:
            st.warning("⚠️ Thread not found or has no checkpoints.")
            return
        elif response.status_code == 401:
            st.error("❌ Authentication failed.")
            return
        
        response.raise_for_status()
        trace_data = response.json()
        checkpoints = trace_data.get("checkpoints", [])
        
        if not checkpoints:
            st.info("ℹ️ No checkpoints available yet.")
            return
        
        st.success(f"✅ Retrieved {len(checkpoints)} checkpoints")
        
        st.markdown("### 🔄 Reasoning Trace Evolution")
        
        for idx, checkpoint in enumerate(checkpoints):
            iteration = checkpoint.get("iteration_count", 0)
            
            with st.expander(f"Checkpoint {idx + 1}: Iteration {iteration}", expanded=(idx == 0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Thesis**")
                    thesis = checkpoint.get("current_thesis")
                    if thesis:
                        st.write(thesis.get("claim", "N/A"))
                    else:
                        st.write("_Not yet generated_")
                
                with col2:
                    st.markdown("**🔍 Antithesis**")
                    antithesis = checkpoint.get("current_antithesis")
                    if antithesis:
                        st.write(antithesis.get("critique", "N/A")[:100] + "...")
                    else:
                        st.write("_Not yet generated_")
                
                if checkpoint.get("final_synthesis"):
                    st.markdown("**✨ Final Synthesis**")
                    synthesis = checkpoint["final_synthesis"]
                    st.success(synthesis.get("novel_insight", "N/A"))
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to query trace: {str(e)}")


# Main UI
def main():
    # Header
    st.title("🧠 CognitiveForge")
    st.caption("Dialectical Synthesis System - Multi-Agent AI Research")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Check API key configuration
        if not API_KEY:
            st.error("⚠️ API_KEY not configured!")
            st.info("Set API_KEY in .env file or environment variables")
        
        # Backend health check
        health = check_backend_health()
        if health.get("status") == "healthy":
            st.success("✅ Backend Healthy")
            col1, col2, col3 = st.columns(3)
            with col1:
                neo4j_status = "🟢" if health.get("neo4j") else "🔴"
                st.metric("Neo4j", neo4j_status)
            with col2:
                gemini_status = "🟢" if health.get("gemini") else "🔴"
                st.metric("Gemini", gemini_status)
            with col3:
                db_status = "🟢" if health.get("checkpointer") else "🔴"
                st.metric("DB", db_status)
        else:
            st.error("❌ Backend Unavailable")
            st.stop()
        
        st.divider()
        
        # Query history
        st.header("📚 Query History")
        if st.session_state.query_history:
            for idx, item in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(
                    f"🔄 {item['query'][:30]}...",
                    key=f"history_{idx}",
                    help=f"Thread: {item['thread_id']}"
                ):
                    st.session_state.current_thread_id = item['thread_id']
                    st.rerun()
        else:
            st.info("No queries yet")
        
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
        
        st.divider()
        
        # Knowledge Discovery Section
        st.header("📚 Knowledge Discovery")
        st.caption("Search and add academic papers to the knowledge graph")
        
        # Initialize discovery session state
        if "discovered_papers" not in st.session_state:
            st.session_state.discovered_papers = []
        if "selected_paper_indices" not in st.session_state:
            st.session_state.selected_paper_indices = []
        if "last_search_source" not in st.session_state:
            st.session_state.last_search_source = None
        
        # Search input
        discovery_query = st.text_input(
            "Search Query",
            placeholder="e.g., transformer architecture",
            key="discovery_query",
            help="Enter keywords to search for papers"
        )
        
        # Search buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 arXiv", disabled=not discovery_query):
                with st.spinner("Searching arXiv..."):
                    try:
                        headers = {"X-API-Key": API_KEY}
                        response = requests.post(
                            f"{API_BASE_URL}/discover",
                            json={
                                "query": discovery_query,
                                "source": "arxiv",
                                "max_results": 10
                            },
                            headers=headers,
                            timeout=10
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.discovered_papers = data["papers"]
                            st.session_state.last_search_source = "arXiv"
                            st.success(f"✅ Found {data['count']} papers from arXiv")
                        else:
                            st.error(f"❌ Search failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🔍 Semantic Scholar", disabled=not discovery_query):
                with st.spinner("Searching Semantic Scholar..."):
                    try:
                        headers = {"X-API-Key": API_KEY}
                        response = requests.post(
                            f"{API_BASE_URL}/discover",
                            json={
                                "query": discovery_query,
                                "source": "semantic_scholar",
                                "max_results": 10
                            },
                            headers=headers,
                            timeout=10
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.discovered_papers = data["papers"]
                            st.session_state.last_search_source = "Semantic Scholar"
                            st.success(f"✅ Found {data['count']} papers from Semantic Scholar")
                        else:
                            st.error(f"❌ Search failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Display discovered papers
        if st.session_state.discovered_papers:
            st.write(f"**{len(st.session_state.discovered_papers)} papers from {st.session_state.last_search_source}**")
            
            # Paper selection with previews
            selected_indices = []
            for idx, paper in enumerate(st.session_state.discovered_papers):
                with st.expander(f"📄 {paper['title'][:60]}..."):
                    st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}" + ("..." if len(paper['authors']) > 3 else ""))
                    st.markdown(f"**Abstract:** {paper['abstract'][:200]}...")
                    st.markdown(f"**Citation Count:** {paper.get('citation_count', 0)}")
                    st.markdown(f"**Published:** {paper['published']}")
                    st.markdown(f"**Source:** {paper['source']}")
                    
                    if st.checkbox(f"Select for adding", key=f"paper_{idx}"):
                        selected_indices.append(idx)
            
            # Add selected papers button
            if selected_indices:
                if st.button(f"➕ Add {len(selected_indices)} Papers to Knowledge Graph"):
                    with st.spinner(f"Adding {len(selected_indices)} papers..."):
                        try:
                            # Send full paper objects (preferred approach)
                            selected_papers = [st.session_state.discovered_papers[idx] for idx in selected_indices]
                            headers = {"X-API-Key": API_KEY}
                            response = requests.post(
                                f"{API_BASE_URL}/add_papers",
                                json={
                                    "papers": selected_papers,
                                    "discovered_by": "manual"
                                },
                                headers=headers,
                                timeout=15
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.success(f"✅ Added {data['added']} papers, skipped {data['skipped']}")
                                # Clear selection after successful add
                                st.session_state.discovered_papers = []
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to add papers: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🚀 New Research", "📊 Query State", "🔄 View Trace"])
    
    # Tab 1: New Research Session
    with tab1:
        st.header("Start New Research Session")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            thread_id = st.text_input(
                "Thread ID",
                value=st.session_state.current_thread_id or "",
                placeholder="Leave empty for auto-generated UUID",
                help="Thread ID for session persistence. Leave empty to create a new session."
            )
        
        with col2:
            if st.button("🔄 Generate New", help="Generate a new UUID"):
                st.session_state.current_thread_id = str(uuid.uuid4())
                st.rerun()
        
        query = st.text_area(
            "Research Query",
            placeholder="e.g., What are the key limitations of transformer architectures?",
            height=100,
            help="Your research question for the dialectical synthesis process"
        )
        
        # T048: Auto-discovery checkbox
        auto_discover = st.checkbox(
            "🔬 Auto-discover papers before synthesis",
            value=True,
            help="Automatically search arXiv for relevant papers and add them to the knowledge graph before starting synthesis"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            start_button = st.button(
                "🚀 Start Dialectical Synthesis",
                type="primary",
                use_container_width=True,
                disabled=not query
            )
        
        if start_button and query:
            # Use provided thread_id or generate new one
            actual_thread_id = thread_id.strip() or str(uuid.uuid4())
            st.session_state.current_thread_id = actual_thread_id
            
            st.divider()
            st.markdown(f"**Session:** `{actual_thread_id}`")
            
            # Stream the dialectics (T049: Pass auto_discover parameter)
            stream_dialectics(actual_thread_id, query, auto_discover=auto_discover)
    
    # Tab 2: Query State
    with tab2:
        st.header("Query Current State")
        
        query_thread_id = st.text_input(
            "Thread ID to Query",
            value=st.session_state.current_thread_id or "",
            key="query_state_thread_id"
        )
        
        if st.button("🔍 Query State", disabled=not query_thread_id):
            query_state(query_thread_id)
    
    # Tab 3: View Trace
    with tab3:
        st.header("View Reasoning Trace")
        
        trace_thread_id = st.text_input(
            "Thread ID",
            value=st.session_state.current_thread_id or "",
            key="trace_thread_id"
        )
        
        if st.button("📜 View Trace", disabled=not trace_thread_id):
            query_trace(trace_thread_id)


if __name__ == "__main__":
    main()

