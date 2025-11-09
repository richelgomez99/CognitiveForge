"""
Streamlit UI for CognitiveForge - Dialectical Synthesis System

Real-time visualization of the multi-agent research process.
Feature 007-ui-polish: Professional UI with warm academic color palette
"""

import streamlit as st
import requests
import json
import os
import uuid
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

# Page configuration
st.set_page_config(
    page_title="CognitiveForge - Dialectical Synthesis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed from expanded for minimalist design
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

# ============================================================================
# T004-T012: FOUNDATIONAL DESIGN SYSTEM
# Warm Academic Color Palette & Typography (Feature 007-ui-polish)
# ============================================================================

def inject_custom_css():
    """
    Inject custom CSS with warm academic color palette and professional typography.
    
    Implements:
    - T004-T008: Color palette (warm backgrounds, agent colors, anti-AI-stereotype)
    - T009-T012: Typography & spacing (system fonts, 8px grid, 3-size hierarchy)
    
    Design Philosophy:
    - Warm backgrounds (#FFFBF5) not sterile white
    - Multiple accent colors (amber, red, green, blue) not purple everywhere
    - System fonts (NOT Inter - AI default)
    - Clear visual hierarchy with max 3 font sizes
    """
    st.markdown("""
    <style>
    /* ===== T004-T007: Color Palette (CSS Variables) ===== */
    :root {
        /* Core brand - Use sparingly */
        --color-brand: #3B82F6;           /* Blue (trust, knowledge) - NOT indigo */
        
        /* Agent-specific colors (T005) */
        --color-analyst: #F59E0B;         /* Amber (warmth, curiosity) */
        --color-skeptic: #EF4444;         /* Red (challenge, critical thinking) */
        --color-synthesizer: #10B981;     /* Green (growth, synthesis) */
        
        /* Supporting accents */
        --color-discovery: #8B5CF6;       /* Purple - ONLY for paper discovery */
        --color-success: #10B981;         /* Green (complete) */
        --color-in-progress: #F59E0B;     /* Amber (thinking) */
        --color-error: #EF4444;           /* Red (error) */
        
        /* Backgrounds - WARM (T006) */
        --bg-page: #FFFBF5;               /* Warm cream (not #FFFFFF) */
        --bg-card: #FFFFFF;               /* Pure white for contrast */
        --bg-subtle: #F7F3ED;             /* Subtle cream for sections */
        
        /* Text (T007) */
        --text-primary: #1F2937;          /* Near-black */
        --text-secondary: #6B7280;        /* Gray */
        --text-muted: #9CA3AF;            /* Light gray */
        
        /* ===== T010: Typography Scale ===== */
        --font-size-xl: 2rem;             /* 32px - Hero headlines */
        --font-size-lg: 1.5rem;           /* 24px - Section titles */
        --font-size-base: 1rem;           /* 16px - Body text */
        --font-size-sm: 0.875rem;         /* 14px - Captions */
        
        /* ===== T012: Font Weights ===== */
        --font-weight-bold: 700;
        --font-weight-semibold: 600;
        --font-weight-regular: 400;
        
        /* ===== T011: Spacing Scale (8px Grid) ===== */
        --space-1: 0.5rem;                /* 8px */
        --space-2: 1rem;                  /* 16px */
        --space-3: 1.5rem;                /* 24px */
        --space-4: 2rem;                  /* 32px */
        --space-6: 3rem;                  /* 48px */
    }
    
    /* ===== T009: System Font Stack (NOT Inter) ===== */
    body, .stMarkdown, .stText, .stCaption, .stHeader {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 
                     'Helvetica Neue', 'Arial', sans-serif !important;
    }
    
    /* ===== Apply Warm Background ===== */
    .stApp {
        background-color: var(--bg-page);
    }
    
    /* ===== Agent Message Styling (More Distinct Visuals) ===== */
    /* Analyst = Amber border + warm background */
    .stChatMessage[data-testid="chatAvatarIcon-user"] {
        border-left: 5px solid var(--color-analyst);
        background-color: #FEF3C7;  /* Light amber background */
        padding: var(--space-2);
        border-radius: 8px;
        margin-bottom: var(--space-2);
    }
    
    /* Skeptic = Red border + cool background */
    .stChatMessage[data-testid="chatAvatarIcon-assistant"] {
        border-left: 5px solid var(--color-skeptic);
        background-color: #FEE2E2;  /* Light red background */
        padding: var(--space-2);
        border-radius: 8px;
        margin-bottom: var(--space-2);
    }
    
    /* ===== Spacing Grid Application ===== */
    .stContainer {
        padding: var(--space-2);
    }
    
    .stExpander {
        margin-top: var(--space-2);
        margin-bottom: var(--space-2);
    }
    
    /* ===== Typography Hierarchy ===== */
    h1 {
        font-size: var(--font-size-xl);
        font-weight: var(--font-weight-bold);
        color: var(--text-primary);
    }
    
    h2, h3 {
        font-size: var(--font-size-lg);
        font-weight: var(--font-weight-semibold);
        color: var(--text-primary);
    }
    
    p, .stMarkdown {
        font-size: var(--font-size-base);
        font-weight: var(--font-weight-regular);
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    .stCaption {
        font-size: var(--font-size-sm);
        color: var(--text-secondary);
    }
    
    /* ===== Synthesis Section Styling (Green Accent) ===== */
    .synthesis-section {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--color-synthesizer);
        padding: var(--space-3);
        border-radius: 8px;
        margin-top: var(--space-3);
    }
    
    /* ===== Card Styling ===== */
    .stCard {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: var(--space-3);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* ===== Button Styling (Blue Brand, NOT Purple) ===== */
    .stButton > button[kind="primary"] {
        background-color: var(--color-brand);
        border-color: var(--color-brand);
    }
    
    /* ===== Activity Event Styling ===== */
    .activity-event {
        display: flex;
        align-items: center;
        padding: var(--space-1);
        margin-bottom: var(--space-1);
    }
    
    .activity-event-icon {
        font-size: 1.5rem;
        margin-right: var(--space-2);
    }
    
    .activity-event-complete {
        color: var(--color-success);
    }
    
    .activity-event-in-progress {
        color: var(--color-in-progress);
    }
    
    .activity-event-error {
        color: var(--color-error);
    }
    
    /* ===== Animations (Subtle, Not Overwhelming) ===== */
    @keyframes thinking-pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1.0; }
    }
    
    .agent-thinking {
        animation: thinking-pulse 2s ease-in-out infinite;
    }
    
    @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stChatMessage {
        animation: fade-in 300ms ease-in;
    }
    
    @keyframes scale-bounce {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .event-complete-icon {
        animation: scale-bounce 400ms ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "current_synthesis" not in st.session_state:
    st.session_state.current_synthesis = None
# T017: Additional UI state initialization
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "activity_events" not in st.session_state:
    st.session_state.activity_events = []
if "current_phase" not in st.session_state:
    st.session_state.current_phase = "idle"  # idle | discovery | debate | synthesis
if "is_activity_feed_collapsed" not in st.session_state:
    st.session_state.is_activity_feed_collapsed = False


# ============================================================================
# T013-T016: HELPER FUNCTIONS & DATA TRANSFORMATIONS
# ============================================================================

# T016: EVENT_ICONS - Map event types to emoji icons
EVENT_ICONS = {
    "keyword_extraction": "🔍",
    "discovery_start": "📚",
    "source_searching": "🔎",
    "paper_found": "📄",
    "papers_ingesting": "💾",
    "discovery_complete": "✅",
    "discovery_error": "⚠️",
    "discovery_timeout": "⏱️",
    "analyst_start": "🔬",
    "analyst_complete": "✅",
    "skeptic_start": "🔍",
    "skeptic_complete": "✅",
    "round_complete": "🔄",
    "synthesis_start": "💡",
    "synthesis_complete": "🎉",
    "error": "❌",
    "warning": "⚠️",
}


# T013: Transform paper dict to PaperCitationUI format
def transform_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform backend paper dict to PaperCitationUI format for display.
    
    Implements T013: Data transformation for consistent citation display.
    
    Args:
        paper: Backend paper dict with keys: title, authors, url, venue, year
    
    Returns:
        PaperCitationUI dict with:
        - author_display: "Smith et al." (max 3 authors)
        - title: Paper title
        - url: Full URL
        - venue: Optional journal/conference name
        - year: Optional publication year
    
    Example:
        >>> transform_paper({"title": "Paper", "authors": ["A", "B", "C", "D"], "url": "..."})
        {"author_display": "A et al.", "title": "Paper", "url": "...", ...}
    """
    authors = paper.get("authors", ["Unknown"])
    
    # Limit to 3 authors, add "et al." if more
    if len(authors) > 3:
        author_display = f"{', '.join(authors[:3])} et al."
    elif len(authors) > 1:
        author_display = ', '.join(authors)
    else:
        author_display = authors[0] if authors else "Unknown"
    
    return {
        "author_display": author_display,
        "title": paper.get("title", "Untitled"),
        "url": paper.get("url", "#"),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
    }


# T014: Transform conversation round from backend to UI format
def transform_conversation_round(round_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform backend ConversationRound to ConversationRoundUI for display.
    
    Implements T014: Flatten and enrich conversation round data.
    
    Args:
        round_data: Backend ConversationRound with thesis, antithesis, papers
    
    Returns:
        ConversationRoundUI dict with:
        - round_number: Round index (1-indexed)
        - analyst_claim: Thesis claim text
        - analyst_reasoning: Thesis reasoning text
        - analyst_papers: List of transformed papers
        - skeptic_critique: Antithesis critique text
        - skeptic_contradiction_found: Boolean
        - skeptic_counter_claim: Optional counter-claim text
        - skeptic_papers: List of transformed counter-papers
    """
    thesis = round_data.get("thesis", {})
    antithesis = round_data.get("antithesis", {})
    
    # Transform papers to UI format
    analyst_papers = [
        transform_paper(p) for p in round_data.get("papers_analyst", [])
    ]
    skeptic_papers = [
        transform_paper(p) for p in round_data.get("papers_skeptic", [])
    ]
    
    return {
        "round_number": round_data.get("round_number", 1),
        "analyst_claim": thesis.get("claim", "N/A"),
        "analyst_reasoning": thesis.get("reasoning", "N/A"),
        "analyst_papers": analyst_papers,
        "skeptic_critique": antithesis.get("critique", "N/A"),
        "skeptic_contradiction_found": antithesis.get("contradiction_found", False),
        "skeptic_counter_claim": antithesis.get("counter_claim"),
        "skeptic_papers": skeptic_papers,
    }


# T015: Transform SSE event to ActivityEvent object
def transform_sse_event(event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse SSE event and create ActivityEvent object for activity feed.
    
    Implements T015: Real-time event stream transformation.
    
    Args:
        event_type: Event type string (e.g., "analyst_complete", "paper_found")
        event_data: Event payload from backend
    
    Returns:
        ActivityEvent dict with:
        - event_type: Original event type
        - icon: Emoji icon from EVENT_ICONS
        - title: Human-readable event title
        - description: Optional event description
        - status: "pending" | "in-progress" | "complete" | "error"
        - duration: Optional duration in seconds
        - timestamp: Event timestamp
        - data: Raw event data
    """
    icon = EVENT_ICONS.get(event_type, "ℹ️")
    
    # Determine status from event type
    if event_type.endswith("_error"):
        status = "error"
    elif event_type.endswith("_complete"):
        status = "complete"
    elif event_type.endswith("_start"):
        status = "in-progress"
    else:
        status = "in-progress"
    
    # Extract title and description from event data
    title = event_data.get("message", event_type.replace("_", " ").title())
    description = event_data.get("description", "")
    duration = event_data.get("duration")
    
    return {
        "event_type": event_type,
        "icon": icon,
        "title": title,
        "description": description,
        "status": status,
        "duration": duration,
        "timestamp": datetime.now(),
        "data": event_data,
    }


def check_backend_health() -> Dict[str, Any]:
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}


# ============================================================================
# T019-T027: CONVERSATIONAL DEBATE VIEW (User Story 1 - P1)
# ============================================================================

def render_paper_citation(paper: Dict[str, Any]) -> None:
    """
    Render a single paper citation in academic format: Authors. Title. 🔗 Link
    
    Implements T024: Academic citation format for paper display.
    
    Args:
        paper: Paper dict with keys: author_display, title, url, venue, year
    
    Format:
        **Authors**
        *Title*
        📄 Venue (Year)
        🔗 [URL](URL)
    """
    # Authors
    author_display = paper.get('author_display', 'Unknown')
    if author_display and author_display != 'Unknown':
        st.markdown(f"**{author_display}**")
    
    # Title
    title = paper.get('title', 'Untitled')
    st.markdown(f"*{title}*")
    
    # Optional venue and year
    venue = paper.get('venue')
    year = paper.get('year')
    if venue or year:
        venue_str = venue if venue else "Unknown venue"
        year_str = f"({year})" if year else ""
        st.caption(f"📄 {venue_str} {year_str}")
    
    # URL
    url = paper.get('url', '#')
    st.markdown(f"🔗 [{url}]({url})")
    st.divider()  # Separator between papers


def render_analyst_message(msg: Dict[str, Any]) -> None:
    """
    Render individual Analyst message in left-aligned chat format.
    
    Implements T021, T024-T025: Analyst message display.
    
    Args:
        msg: Analyst message dict with keys: claim, reasoning, papers
    """
    # T021: ANALYST MESSAGE (LEFT-ALIGNED)
    with st.chat_message("user", avatar="🔬"):
        st.markdown("**ANALYST**")
        st.markdown(f"**Claim:** {msg.get('claim', 'N/A')}")
        
        # Reasoning in collapsible expander (user feedback: bring back expandable sections)
        reasoning = msg.get('reasoning', 'N/A')
        if reasoning and reasoning != 'N/A':
            with st.expander("📋 View Reasoning"):
                st.write(reasoning)
        
        # T025: Collapsible papers section
        papers = msg.get('papers', [])
        if papers:
            with st.expander(f"📚 Papers Cited ({len(papers)})"):
                for paper in papers:
                    render_paper_citation(paper)
        else:
            st.caption("📚 No new papers discovered")


def render_skeptic_message(msg: Dict[str, Any]) -> None:
    """
    Render individual Skeptic message in right-aligned chat format.
    
    Implements T022, T024, T026: Skeptic message display.
    
    Args:
        msg: Skeptic message dict with keys: critique, contradiction_found, counter_claim, papers
    """
    # T022: SKEPTIC MESSAGE (RIGHT-ALIGNED)
    with st.chat_message("assistant", avatar="🔍"):
        st.markdown("**SKEPTIC**")
        
        # Show contradiction status
        if msg.get('contradiction_found', False):
            st.warning("⚠️ **Contradictions Found**")
            counter_claim = msg.get('counter_claim')
            if counter_claim:
                st.markdown(f"**Counter-claim:** {counter_claim}")
        else:
            st.success("✅ **No Major Contradictions** - Claim accepted!")
        
        # Critique in collapsible expander (user feedback: bring back expandable sections)
        critique = msg.get('critique', 'N/A')
        if critique and critique != 'N/A':
            with st.expander("🔍 View Critique"):
                st.write(critique)
        
        # T026: Collapsible counter-papers section
        papers = msg.get('papers', [])
        if papers:
            with st.expander(f"📖 Counter-Papers ({len(papers)})"):
                for paper in papers:
                    render_paper_citation(paper)
        else:
            st.caption("📖 No counter-papers discovered")


def render_conversational_debate(conversation_history: List[Dict[str, Any]]) -> None:
    """
    Display all debate messages in vertical chat format.
    
    Implements T019 + T031: Main function for conversational debate view.
    Handles individual messages, groups them visually by round number.
    
    Updates in real-time as each message arrives (via SSE stream).
    
    Args:
        conversation_history: List of individual message dicts with keys:
            - type: "analyst" or "skeptic"
            - round_number: int
            - claim/critique: str
            - reasoning: str (analyst only)
            - papers: List[Dict]
            - contradiction_found: bool (skeptic only)
            - counter_claim: str (skeptic only)
    
    Layout:
        ### 🔄 Dialectical Debate
        [Message 1: Analyst]
        [Message 2: Skeptic]
        ---
        [Message 3: Analyst]
        [Message 4: Skeptic]
        ---
    """
    if not conversation_history:
        st.info("🔄 Debate messages will appear here as the dialectical process unfolds...")
        return
    
    # Count rounds (number of Analyst messages)
    num_rounds = sum(1 for msg in conversation_history if msg.get("type") == "analyst")
    st.markdown("### 🔄 Dialectical Debate")
    st.caption(f"**{num_rounds} rounds** • {len(conversation_history)} messages")
    
    # T031: Group messages visually by round_number
    current_round = None
    for msg in conversation_history:
        msg_round = msg.get("round_number")
        
        # Show round header when entering new round
        if msg_round != current_round:
            if current_round is not None:
                st.divider()  # Separator between rounds
            st.markdown(f"#### 🧠 Round {msg_round}")
            current_round = msg_round
        
        # Render individual message
        if msg.get("type") == "analyst":
            render_analyst_message(msg)
        elif msg.get("type") == "skeptic":
            render_skeptic_message(msg)


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
    
    # T028: Initialize conversation history for real-time updates
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    st.session_state.conversation_history = []  # Reset for new session
    
    # Create containers for streaming content
    status_container = st.container()
    discovery_container = st.container()  # T050: Container for discovery events
    progress_bar = st.progress(0)
    
    # T031: Create debate container for real-time conversational view updates
    debate_container = st.empty()
    
    # Progress indicator container (minimal display during streaming)
    progress_container = st.container()
    
    # Synthesis container (for final output)
    synthesis_container = st.container()
    
    # T028: Track message counter and current round number
    message_counter = 0
    current_round_number = 0
    
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
                    
                    # T028-T032: Real-time individual message display
                    # Each message appears IMMEDIATELY, not waiting for pairs
                    
                    # T029: Analyst message - append immediately
                    if node == "analyst" and "current_thesis" in data:
                        current_round_number += 1  # New round starts with Analyst
                        thesis = data["current_thesis"]
                        
                        # Create individual Analyst message
                        # FIXED: Extract paper metadata from current_round_papers_analyst (full Paper objects)
                        # NOT from evidence (which only has source_url + snippet)
                        round_papers = data.get("current_round_papers_analyst", [])
                        analyst_message = {
                            "type": "analyst",
                            "round_number": current_round_number,
                            "claim": thesis.get("claim", "N/A"),
                            "reasoning": thesis.get("reasoning", "N/A"),
                            "papers": [
                                {
                                    "title": paper.get("title", "Unknown Title"),
                                    "authors": paper.get("authors", []),
                                    "author_display": ", ".join((paper.get("authors", []) or ["Unknown"])[:3]) + (" et al." if len(paper.get("authors", []) or []) > 3 else ""),
                                    "url": paper.get("url", "#"),
                                    "venue": paper.get("venue"),
                                    "year": paper.get("year")
                                }
                                for paper in round_papers
                            ]
                        }
                        
                        # Immediately append and re-render
                        st.session_state.conversation_history.append(analyst_message)
                        message_counter += 1
                        
                        # T032: Re-render debate view with new message
                        with debate_container.container():
                            render_conversational_debate(st.session_state.conversation_history)
                    
                    # T030: Skeptic message - append immediately
                    elif node == "skeptic" and "current_antithesis" in data:
                        antithesis = data["current_antithesis"]
                        
                        # Create individual Skeptic message
                        # FIXED: Extract paper metadata from current_round_papers_skeptic (full Paper objects)
                        # NOT from counter_evidence (which only has source_url + snippet)
                        round_papers = data.get("current_round_papers_skeptic", [])
                        skeptic_message = {
                            "type": "skeptic",
                            "round_number": current_round_number,
                            "critique": antithesis.get("critique", "N/A"),
                            "contradiction_found": antithesis.get("contradiction_found", False),
                            "counter_claim": antithesis.get("counter_claim"),
                            "papers": [
                                {
                                    "title": paper.get("title", "Unknown Title"),
                                    "authors": paper.get("authors", []),
                                    "author_display": ", ".join((paper.get("authors", []) or ["Unknown"])[:3]) + (" et al." if len(paper.get("authors", []) or []) > 3 else ""),
                                    "url": paper.get("url", "#"),
                                    "venue": paper.get("venue"),
                                    "year": paper.get("year")
                                }
                                for paper in round_papers
                            ]
                        }
                        
                        # Immediately append and re-render
                        st.session_state.conversation_history.append(skeptic_message)
                        message_counter += 1
                        
                        # T032: Re-render debate view with new message
                        with debate_container.container():
                            render_conversational_debate(st.session_state.conversation_history)
                    
                    # Synthesizer: Display final synthesis
                    elif node == "synthesizer" and "final_synthesis" in data:
                        synthesis = data["final_synthesis"]
                        st.session_state.current_synthesis = synthesis
                        
                        # Display in dedicated synthesis container
                        with synthesis_container:
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
                    
                    # Note: Conversational debate view is already rendered in real-time
                    # during SSE streaming (no need to query /get_state)
                    
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
    # T008: Inject custom CSS at the top of main function
    inject_custom_css()
    
    # Header - User-friendly messaging
    st.title("🧠 CognitiveForge")
    st.caption("AI-powered research assistant that finds evidence, debates claims, and synthesizes insights")
    
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
    
    # Main content - Simplified, no tabs (user feedback: tabs are confusing)
    st.markdown("---")
    
    # Research Query - The star of the show
    query = st.text_area(
        "🔍 What would you like to research?",
        placeholder="e.g., What are the key limitations of transformer architectures in natural language processing?",
        height=100,
        help="Enter your research question. The AI will find evidence, debate different perspectives, and synthesize insights.",
        label_visibility="visible"
    )
    
    # Advanced options - Collapsed by default
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            thread_id = st.text_input(
                "Session ID (optional)",
                value=st.session_state.current_thread_id or "",
                placeholder="Auto-generated if empty",
                help="For resuming previous sessions. Leave empty for new research."
            )
        
        with col2:
            if st.button("🔄 New ID"):
                st.session_state.current_thread_id = str(uuid.uuid4())
                st.rerun()
        
        # T048: Auto-discovery checkbox
        auto_discover = st.checkbox(
            "🔬 Automatically search for papers",
            value=True,
            help="Search arXiv for relevant papers before starting the debate"
        )
    
    # Primary action - Use container width for prominence
    start_button = st.button(
        "🚀 Start Research",
        type="primary",
        use_container_width=True,
        disabled=not query,
        help="Begin the AI research process: discover papers → debate perspectives → synthesize insights"
    )
    
    if start_button and query:
        # Use provided thread_id or generate new one
        actual_thread_id = thread_id.strip() or str(uuid.uuid4())
        st.session_state.current_thread_id = actual_thread_id
        
        st.divider()
        st.caption(f"Session: `{actual_thread_id}`")
        
        # Stream the dialectics (T049: Pass auto_discover parameter)
        stream_dialectics(actual_thread_id, query, auto_discover=auto_discover)


if __name__ == "__main__":
    main()

