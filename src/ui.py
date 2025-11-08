"""
Streamlit UI for CognitiveForge - Dialectical Synthesis System

Real-time visualization of the multi-agent research process.
"""

import streamlit as st
import requests
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List

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


def stream_dialectics(thread_id: str, query: str) -> None:
    """Stream dialectical synthesis process via SSE."""
    if not API_KEY:
        st.error("❌ API_KEY not set. Please configure it in environment variables or Streamlit secrets.")
        return
    
    headers = {
        "X-API-Key": API_KEY,
        "Accept": "text/event-stream"
    }
    
    url = f"{API_BASE_URL}/stream_dialectics/{thread_id}"
    params = {"query": query}
    
    # Create containers for streaming content
    status_container = st.container()
    progress_bar = st.progress(0)
    agent_containers = {
        "analyst": st.container(),
        "skeptic": st.container(),
        "synthesizer": st.container()
    }
    
    with status_container:
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
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                # Decode SSE event
                line_str = line.decode('utf-8')
                if not line_str.startswith('data: '):
                    continue
                
                event_data_str = line_str[6:]  # Remove "data: " prefix
                
                try:
                    event_data = json.loads(event_data_str)
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
                                        st.write(thesis.get('reasoning', 'N/A'))
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
                                        st.write(antithesis.get('critique', 'N/A'))
                                
                                elif node == "synthesizer" and "final_synthesis" in data:
                                    synthesis = data["final_synthesis"]
                                    st.session_state.current_synthesis = synthesis
                                    
                                    st.success("✨ Novel Insight Generated!")
                                    st.markdown(f"**Insight:** {synthesis.get('novel_insight', 'N/A')}")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        confidence = synthesis.get('confidence_score', 0.0)
                                        st.metric("Confidence", f"{confidence:.2%}")
                                    with col2:
                                        novelty = synthesis.get('novelty_score', 0.0)
                                        st.metric("Novelty", f"{novelty:.2%}")
                                    
                                    with st.expander("🧠 Synthesis Reasoning"):
                                        st.write(synthesis.get('reasoning', 'N/A'))
                                    
                                    with st.expander("📚 Evidence Lineage"):
                                        for idx, url in enumerate(synthesis.get('evidence_lineage', []), 1):
                                            st.write(f"{idx}. [{url}]({url})")
                
                except json.JSONDecodeError:
                    st.warning(f"Failed to parse event: {event_data_str[:100]}")
            
            # Completion message
            with status_container:
                if synthesis_complete:
                    st.success(f"🎉 Dialectical synthesis complete! ({event_count} events processed)")
                    
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
            
            # Stream the dialectics
            stream_dialectics(actual_thread_id, query)
    
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

