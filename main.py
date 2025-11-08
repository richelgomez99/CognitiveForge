#!/usr/bin/env python3
"""
CognitiveForge: Multi-Agent Dialectical Synthesis System

CLI entry point for running the dialectical research engine.
Tier 1 MVP: Single-shot CLI execution with 3-agent debate.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from src.graph import build_graph
from src.models import AgentState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_environment():
    """
    Validate that all required environment variables are set.
    
    Raises:
        ValueError: If required env vars are missing
    """
    required_vars = [
        "GOOGLE_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please set them in your .env file or environment."
        )
    
    logger.info("✅ Environment validation passed")


def print_banner():
    """Print the CognitiveForge banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ████████╗  ██████╗  ██████╗  ███╗   ██╗██╗████████╗  ║
║   ██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝  ║
║   ██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║     ║
║   ██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║     ║
║   ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║     ║
║    ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝     ║
║                                                          ║
║              COGNITIVE FORGE v1.0 (Tier 1)              ║
║          Multi-Agent Dialectical Synthesis Engine        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def format_output(state: AgentState, verbose: bool = False):
    """
    Format and display the final output from the dialectical synthesis.
    
    Args:
        state: Final AgentState after synthesis
        verbose: If True, show intermediate steps
    """
    print("\n" + "="*70)
    print("📊 DIALECTICAL SYNTHESIS COMPLETE")
    print("="*70)
    
    # Show query
    print(f"\n🔍 Research Query:")
    print(f"   {state['original_query']}")
    
    # Show iterations
    print(f"\n🔄 Debate Iterations: {state['iteration_count']}")
    
    if verbose:
        # Show thesis
        thesis = state.get("current_thesis")
        if thesis:
            print(f"\n📋 Analyst's Thesis:")
            print(f"   Claim: {thesis.claim}")
            print(f"   Evidence Sources: {len(thesis.evidence)}")
        
        # Show antithesis
        antithesis = state.get("current_antithesis")
        if antithesis:
            print(f"\n🔬 Skeptic's Evaluation:")
            print(f"   Contradiction Found: {antithesis.contradiction_found}")
            if antithesis.counter_claim:
                print(f"   Counter-claim: {antithesis.counter_claim[:150]}...")
    
    # Show final synthesis
    synthesis = state.get("final_synthesis")
    if synthesis:
        print(f"\n💡 NOVEL INSIGHT:")
        print(f"   {synthesis.novel_insight}")
        print(f"\n📈 Metrics:")
        print(f"   - Confidence Score: {synthesis.confidence_score:.2f}")
        print(f"   - Novelty Score: {synthesis.novelty_score:.2f}")
        print(f"   - Evidence Sources: {len(synthesis.evidence_lineage)}")
        
        if verbose:
            print(f"\n🔗 Evidence Lineage:")
            for i, url in enumerate(synthesis.evidence_lineage, 1):
                print(f"   {i}. {url}")
            
            print(f"\n📝 Supporting Claims:")
            for i, claim in enumerate(synthesis.supporting_claims, 1):
                print(f"   {i}. {claim}")
            
            print(f"\n💭 Reasoning:")
            print(f"   {synthesis.reasoning}")
    
    print("\n" + "="*70)


def save_results(state: AgentState, output_dir: str = "outputs"):
    """
    Save synthesis results to JSON file.
    
    Args:
        state: Final AgentState
        output_dir: Directory to save outputs
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"synthesis_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    # Prepare output data
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "query": state["original_query"],
        "iterations": state["iteration_count"],
        "thesis": state.get("current_thesis").model_dump() if state.get("current_thesis") else None,
        "antithesis": state.get("current_antithesis").model_dump() if state.get("current_antithesis") else None,
        "synthesis": state.get("final_synthesis").model_dump() if state.get("final_synthesis") else None,
        "messages": state.get("messages", [])
    }
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"💾 Results saved to: {filepath}")
    return filepath


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CognitiveForge: Multi-Agent Dialectical Synthesis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple query
  python main.py --query "What are the limitations of transformer models?"
  
  # With verbose output
  python main.py --query "How does attention mechanism work?" --verbose
  
  # Save results to file
  python main.py --query "Compare BERT and GPT architectures" --save
  
  # Custom thread ID (for KG tracing)
  python main.py --query "Explain retrieval-augmented generation" --thread-id rag-session-1
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Research query for dialectical synthesis"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed intermediate steps (thesis, antithesis)"
    )
    
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save results to JSON file in outputs/ directory"
    )
    
    parser.add_argument(
        "--thread-id", "-t",
        default=None,
        help="Custom thread ID for session tracking (default: auto-generated)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override MAX_ITERATIONS from .env (default: 3)"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress banner display"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Override max iterations if specified
    if args.max_iterations:
        os.environ["MAX_ITERATIONS"] = str(args.max_iterations)
    
    # Print banner
    if not args.no_banner:
        print_banner()
    
    try:
        # Validate environment
        validate_environment()
        
        # Generate thread ID if not provided
        thread_id = args.thread_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build graph
        logger.info("🏗️  Building dialectical synthesis graph...")
        graph = build_graph()
        
        # Prepare initial state
        initial_state = {
            "messages": [],
            "original_query": args.query,
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",  # For Tier 3 compatibility
            "thread_id": thread_id
        }
        
        # Execute graph
        print(f"\n🚀 Initiating dialectical synthesis...")
        print(f"📝 Query: {args.query}")
        print(f"🆔 Thread ID: {thread_id}")
        print(f"🔄 Max Iterations: {os.getenv('MAX_ITERATIONS', '3')}")
        print(f"\n{'='*70}\n")
        
        logger.info("▶️  Executing graph...")
        final_state = graph.invoke(initial_state)
        
        # Display results
        format_output(final_state, verbose=args.verbose)
        
        # Save results if requested
        if args.save:
            save_results(final_state)
        
        logger.info("✅ Dialectical synthesis complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Synthesis interrupted by user.")
        return 130
    
    except Exception as e:
        logger.error(f"❌ Error during synthesis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

