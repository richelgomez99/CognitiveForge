"""
Skeptic agent node for the dialectical synthesis system.

The Skeptic evaluates the Analyst's thesis and identifies contradictions or weaknesses.

Tier 1: Enhanced with similarity-based auto-rejection (US2).
"""

import logging
from typing import Dict, Any
from pydantic import ValidationError

from src.models import AgentState, Antithesis
from src.utils.model_config import get_agent_model
from src.utils.gemini_client import call_gemini_with_retry
from src.utils.similarity import is_circular_argument  # T046
from src.utils.memory import update_debate_memory  # T050
from src.tools.counter_research import generate_counter_queries, discover_counter_evidence, papers_to_conflicting_evidence  # T066
from src.tools.kg_tools import add_papers_to_neo4j  # T067
import asyncio  # T067

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def skeptic_node(state: AgentState) -> Dict[str, Any]:
    """
    Skeptic agent node: Evaluate thesis and identify contradictions or weaknesses.
    
    This node:
    1. Extracts the current thesis from state
    2. Constructs a critical evaluation prompt
    3. Calls Gemini with structured output to generate an Antithesis
    4. Returns updated state with the antithesis and contradiction report
    
    Implements FR-T1-004: Skeptic generates Antithesis with contradiction detection
    
    Args:
        state: Current AgentState with current_thesis
    
    Returns:
        Dict with updated current_antithesis, contradiction_report, and messages
    
    Raises:
        Exception: If antithesis generation fails after retries
    """
    logger.info("---EXECUTING SKEPTIC NODE (Tier 1: Similarity-Based Auto-Rejection)---")
    
    # Extract thesis and memory
    thesis = state["current_thesis"]
    if not thesis:
        raise ValueError("Skeptic requires a thesis from Analyst")
    
    debate_memory = state.get("debate_memory", {"rejected_claims": [], "skeptic_objections": [], "weak_evidence_urls": []})
    rejected_claims = debate_memory.get("rejected_claims", [])
    
    # T047-T048: Similarity-based auto-rejection BEFORE full LLM evaluation
    if rejected_claims:
        logger.info(f"🔍 Checking similarity with {len(rejected_claims)} rejected claims...")
        is_circular, most_similar_claim, max_similarity = is_circular_argument(thesis.claim, rejected_claims)
        
        if is_circular:
            # T048: Auto-reject with high similarity (most_similar_claim already identified by is_circular_argument)
            logger.warning(f"⛔ AUTO-REJECTION: Similarity {max_similarity:.2f} with rejected claim")  # T049
            logger.warning(f"   Rejected claim: {most_similar_claim[:80]}...")  # T049
            logger.warning(f"   New claim: {thesis.claim[:80]}...")  # T049
            logger.info(f"⚡ Skipping full LLM evaluation (circular argument detected)")  # T049
            
            # Create auto-rejection antithesis
            antithesis = Antithesis(
                contradiction_found=True,
                counter_claim=f"This argument is circular and nearly identical to a previously rejected claim (similarity: {max_similarity:.2f}).",
                conflicting_evidence=[],
                critique=f"This claim is {max_similarity*100:.0f}% similar to previously rejected thesis: '{most_similar_claim}'. This represents a circular argument that does not advance the debate. Please formulate a genuinely different perspective or approach."
            )
            
            # T050-T051: Update memory with rejected claim
            updated_memory = update_debate_memory(
                debate_memory,
                rejected_claim=thesis.claim,
                skeptic_objection=antithesis.critique[:200]
            )
            
            return {
                "current_antithesis": antithesis,
                "contradiction_report": "CIRCULAR ARGUMENT (auto-rejected based on similarity)",
                "messages": [f"Skeptic: {antithesis.critique[:100]}..."],
                "debate_memory": updated_memory,
                "last_similarity_score": max_similarity,  # Natural termination: track similarity
                "current_round_papers_skeptic": []  # Tier 2 (T040): No counter-papers for circular arguments
            }
    
    # No circular argument detected, proceed with full LLM evaluation
    logger.info("✅ No circular argument detected, proceeding with full evaluation")
    
    # T066-T071: Active Counter-Research (Tier 1 US3)
    logger.info("🔍 Starting counter-research to find contradicting evidence...")
    
    # T066: Generate counter-queries
    counter_queries = generate_counter_queries(thesis.claim, thesis.reasoning)
    logger.info(f"✅ Generated {len(counter_queries)} counter-queries")  # T071
    
    # T067: Discover counter-evidence
    claim_id = state.get("current_claim_id", "unknown")
    counter_papers = asyncio.run(discover_counter_evidence(counter_queries, claim_id, max_papers_per_query=2))
    logger.info(f"✅ Discovered {len(counter_papers)} counter-papers")  # T071
    
    # T067: Add counter-papers to Neo4j with skeptic_counter tag
    if counter_papers:
        result = add_papers_to_neo4j(
            counter_papers,
            discovered_by="skeptic_counter",
            claim_id=claim_id,
            iteration_number=state.get("iteration_count", 0)
        )
        logger.info(f"✅ Neo4j: added={result['added']}, skipped={result['skipped']}, failed={result['failed']}")
    
    # T068-T069: Convert to ConflictingEvidence and populate antithesis field
    conflicting_evidence_list = papers_to_conflicting_evidence(counter_papers, discovered_by="skeptic_counter")
    logger.info(f"✅ Added {len(conflicting_evidence_list)} to conflicting_evidence")  # T071
    
    # Get model ID for skeptic
    model_id = get_agent_model("skeptic")
    
    # Format evidence for prompt
    evidence_text = "\n".join([
        f"- Source: {ev.source_url}\n  Snippet: {ev.snippet}"
        for ev in thesis.evidence
    ])
    
    # T070: Format counter-evidence for prompt
    counter_evidence_text = ""
    if conflicting_evidence_list:
        counter_evidence_text = "\n\nCONTRADICTING EVIDENCE FOUND (from counter-research):\n" + "\n".join([
            f"- Source: {ev.source_url}\n  Snippet: {ev.snippet}\n  Relevance: {ev.relevance_score:.2f}"
            for ev in conflicting_evidence_list
        ])
    
    # Construct critical evaluation prompt with improved dialectical guidance
    prompt = f"""You are an expert skeptical analyst tasked with critically evaluating a research thesis. Your role is to identify SIGNIFICANT logical weaknesses, not to nitpick minor issues.

Thesis Claim: {thesis.claim}

Thesis Reasoning: {thesis.reasoning}

Thesis Evidence:
{evidence_text}{counter_evidence_text}

Your task as a constructive skeptic:
1. Critically analyze the thesis for MAJOR logical weaknesses, contradictions, or unsupported claims
2. Apply the "Substantial Contradiction Test":
   - Would accepting this thesis lead to incorrect conclusions or actions?
   - Are there alternative interpretations that fundamentally challenge the core claim?
   - Is key evidence missing or misinterpreted in ways that undermine the conclusion?
3. **Set contradiction_found to TRUE only if you find SIGNIFICANT issues that:**
   - Directly contradict the thesis claim with strong counter-evidence
   - Reveal fundamental logical flaws that invalidate the reasoning
   - Identify critical missing considerations that change the conclusion
4. **Set contradiction_found to FALSE if:**
   - The thesis is well-reasoned and supported by evidence (even if not perfect)
   - Only minor clarifications or additional nuance would strengthen it
   - Alternative interpretations exist but don't fundamentally invalidate the claim

When contradiction_found is TRUE:
- Provide a counter_claim that offers a fundamentally different interpretation
- Cite conflicting_evidence with strong sources (especially from the CONTRADICTING EVIDENCE section above)
- Explain in critique why this is a SIGNIFICANT weakness (not just a minor gap)
- T070: When referencing papers in your critique (including counter-evidence), use format: [CITE: paper_url]

When contradiction_found is FALSE:
- Acknowledge the thesis strengths in your critique
- Note any minor areas for improvement (without labeling them as contradictions)
- Be intellectually honest: a good thesis should pass scrutiny

Generate an antithesis with:
- contradiction_found: boolean (TRUE only for SIGNIFICANT issues)
- counter_claim: string (required only if contradiction_found is true)
- conflicting_evidence: list of Evidence objects (supporting your critique)
- critique: detailed explanation focusing on major weaknesses if found, or acknowledgment of thesis quality if sound (minimum 50 characters)

Remember: Your goal is dialectical progress, not endless debate. If the thesis is sound, say so."""

    # Call Gemini with retry logic
    max_validation_retries = 2
    for attempt in range(max_validation_retries + 1):
        try:
            logger.info(f"Calling Gemini ({model_id}) for antithesis generation (attempt {attempt + 1})")
            
            # Get structured output from Gemini
            response_text = call_gemini_with_retry(
                model_id=model_id,
                prompt=prompt,
                response_schema=Antithesis.model_json_schema()
            )
            
            # Parse and validate response
            antithesis = Antithesis.model_validate_json(response_text)
            
            # T069: Ensure conflicting_evidence from counter-research is included
            # Merge LLM-generated conflicting_evidence with our discovered counter-evidence
            if conflicting_evidence_list:
                # If LLM didn't populate conflicting_evidence, use ours
                if not antithesis.conflicting_evidence:
                    antithesis.conflicting_evidence = conflicting_evidence_list
                    logger.info(f"📋 Populated conflicting_evidence with {len(conflicting_evidence_list)} counter-papers")
                else:
                    # Merge (avoid duplicates by URL)
                    existing_urls = {ev.source_url for ev in antithesis.conflicting_evidence}
                    new_evidence = [ev for ev in conflicting_evidence_list if ev.source_url not in existing_urls]
                    if new_evidence:
                        antithesis.conflicting_evidence.extend(new_evidence)
                        logger.info(f"📋 Merged {len(new_evidence)} new counter-papers into conflicting_evidence")
            
            logger.info(f"Antithesis generated successfully")
            logger.info(f"Contradiction found: {antithesis.contradiction_found}")
            if antithesis.counter_claim:
                logger.info(f"Counter-claim: {antithesis.counter_claim[:100]}...")
            
            # Prepare contradiction report
            contradiction_report = antithesis.critique if antithesis.contradiction_found else ""
            
            # T050-T051: Update memory if thesis was rejected
            updated_memory = debate_memory
            if antithesis.contradiction_found:
                logger.info("📝 Updating debate memory with rejected claim and objection")
                updated_memory = update_debate_memory(
                    debate_memory,
                    rejected_claim=thesis.claim,
                    skeptic_objection=antithesis.critique[:200]
                )
            
            # Tier 2 (T040): Track counter-papers discovered in this round for conversational thread view
            # Store full metadata (title, authors, url) for proper citation formatting
            counter_papers_metadata = [
                {
                    "title": paper.title,
                    "authors": paper.authors[:3] if paper.authors else ["Unknown"],  # Limit to 3 authors
                    "url": paper.url
                }
                for paper in counter_papers
            ]
            logger.info(f"📖 Tracked {len(counter_papers_metadata)} counter-papers for round visualization")
            
            # Return updated state
            return {
                "current_antithesis": antithesis,
                "contradiction_report": contradiction_report,
                "messages": [f"Skeptic: {antithesis.critique[:150]}..."],
                "debate_memory": updated_memory,
                "last_similarity_score": None,  # Natural termination: no circular argument detected
                "current_round_papers_skeptic": counter_papers_metadata  # Tier 2: US1
            }
            
        except ValidationError as e:
            logger.warning(f"Pydantic validation error (attempt {attempt + 1}): {e}")
            
            if attempt < max_validation_retries:
                # Add explicit schema guidance to prompt
                prompt += f"\n\nIMPORTANT: Your previous response had validation errors: {e}\nPlease ensure you follow the exact schema requirements."
                continue
            else:
                logger.error(f"Antithesis validation failed after {max_validation_retries + 1} attempts")
                raise
        
        except Exception as e:
            logger.error(f"Skeptic node error: {e}")
            raise
    
    # Should never reach here
    raise Exception("Skeptic node failed to generate valid antithesis")

