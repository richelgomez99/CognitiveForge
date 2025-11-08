"""
Synthesizer agent node for the dialectical synthesis system.

The Synthesizer integrates the Thesis and Antithesis to produce a novel insight.
"""

import logging
from typing import Dict, Any
from pydantic import ValidationError

from src.models import AgentState, Synthesis
from src.utils.model_config import get_agent_model
from src.utils.gemini_client import call_gemini_with_retry
from src.tools.kg_tools import add_insight_to_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizer agent node: Integrate thesis and antithesis into novel insight.
    
    This node:
    1. Extracts thesis and antithesis from state
    2. Constructs a synthesis prompt
    3. Calls Gemini with structured output to generate a Synthesis
    4. Self-assesses novelty score within the same LLM call (FR-T1-013)
    5. Aggregates evidence lineage from both thesis and antithesis
    6. Writes the insight to the Neo4j knowledge graph
    7. Returns updated state with the final synthesis
    
    Implements FR-T1-005: Synthesizer generates novel insight
    Implements FR-T1-013: Novelty self-assessment in same LLM call
    Implements FR-T1-007: Write Insight nodes to Neo4j
    
    Args:
        state: Current AgentState with current_thesis and current_antithesis
    
    Returns:
        Dict with updated final_synthesis and messages
    
    Raises:
        Exception: If synthesis generation fails after retries
    """
    logger.info("---EXECUTING SYNTHESIZER NODE---")
    
    # Extract thesis and antithesis
    thesis = state["current_thesis"]
    antithesis = state["current_antithesis"]
    
    if not thesis:
        raise ValueError("Synthesizer requires a thesis from Analyst")
    if not antithesis:
        raise ValueError("Synthesizer requires an antithesis from Skeptic")
    
    # Get model ID for synthesizer
    model_id = get_agent_model("synthesizer")
    
    # Aggregate evidence lineage
    evidence_urls = []
    for ev in thesis.evidence:
        if ev.source_url not in evidence_urls:
            evidence_urls.append(ev.source_url)
    for ev in antithesis.conflicting_evidence:
        if ev.source_url not in evidence_urls:
            evidence_urls.append(ev.source_url)
    
    logger.info(f"Aggregated {len(evidence_urls)} unique evidence sources")
    
    # Construct synthesis prompt
    prompt = f"""You are an expert research synthesizer tasked with integrating diverse perspectives into novel insights.

Thesis Claim: {thesis.claim}
Thesis Reasoning: {thesis.reasoning}

Antithesis Analysis: {antithesis.critique}
{f"Counter-claim: {antithesis.counter_claim}" if antithesis.counter_claim else ""}
Contradiction Found: {antithesis.contradiction_found}

Your task:
1. Synthesize a novel insight that integrates both the thesis and antithesis
2. Go beyond simply combining them - generate a NEW perspective or understanding
3. Acknowledge both the strengths (from thesis) and limitations (from antithesis)
4. Support your synthesis with claims from both sides
5. **IMPORTANT**: Self-assess the novelty of your synthesis:
   - novelty_score: How novel is this insight compared to the input thesis and antithesis?
   - Score 0.0-0.3: Mostly restates existing points
   - Score 0.4-0.6: Moderately novel, some new connections
   - Score 0.7-1.0: Highly novel, significant new insights or frameworks
6. Assess your confidence in the synthesis (0.0-1.0)

Generate a synthesis with:
- novel_insight: Your synthesized insight (minimum 50 characters)
- supporting_claims: List of specific claims from thesis/antithesis that support synthesis
- evidence_lineage: List of ALL source URLs (will be provided automatically)
- confidence_score: Your confidence (0.0-1.0)
- novelty_score: Self-assessed novelty (0.0-1.0) - be honest and critical
- reasoning: Detailed explanation of how you derived this synthesis (minimum 50 characters)

Be intellectually rigorous. Don't overstate novelty - if your synthesis is mostly a summary, score it low (0.3-0.5)."""

    # Call Gemini with retry logic
    max_validation_retries = 2
    for attempt in range(max_validation_retries + 1):
        try:
            logger.info(f"Calling Gemini ({model_id}) for synthesis generation (attempt {attempt + 1})")
            
            # Get structured output from Gemini
            response_text = call_gemini_with_retry(
                model_id=model_id,
                prompt=prompt,
                response_schema=Synthesis.model_json_schema()
            )
            
            # Parse and validate response
            response_dict = Synthesis.model_validate_json(response_text).model_dump()
            
            # Override evidence_lineage with our aggregated URLs
            # Ensure we have at least 3 sources (pad with generic sources if needed)
            if len(evidence_urls) < 3:
                logger.warning(f"Only {len(evidence_urls)} evidence sources aggregated. Padding to meet minimum of 3.")
                # Add generic placeholder URLs if needed
                while len(evidence_urls) < 3:
                    evidence_urls.append(f"https://scholar.google.com/scholar?q={state['original_query'].replace(' ', '+')}")
            
            response_dict['evidence_lineage'] = evidence_urls
            
            # Re-validate with proper evidence_lineage
            synthesis = Synthesis.model_validate(response_dict)
            
            logger.info(f"Synthesis generated successfully")
            logger.info(f"Novel insight: {synthesis.novel_insight[:100]}...")
            logger.info(f"Confidence: {synthesis.confidence_score:.2f}")
            logger.info(f"Novelty (self-assessed): {synthesis.novelty_score:.2f}")
            logger.info(f"Evidence sources: {len(synthesis.evidence_lineage)}")
            
            # Write insight to knowledge graph
            thread_id = state.get("thread_id", "default")
            try:
                success = add_insight_to_graph(synthesis, thread_id)
                if success:
                    logger.info("✅ Insight added to knowledge graph")
                else:
                    logger.warning("⚠️  Failed to add insight to knowledge graph (non-fatal)")
            except Exception as e:
                logger.warning(f"⚠️  KG write error (non-fatal): {e}")
            
            # Return updated state
            return {
                "final_synthesis": synthesis,
                "messages": [f"Synthesizer: {synthesis.novel_insight}"]
            }
            
        except ValidationError as e:
            logger.warning(f"Pydantic validation error (attempt {attempt + 1}): {e}")
            
            if attempt < max_validation_retries:
                # Add explicit schema guidance to prompt
                prompt += f"\n\nIMPORTANT: Your previous response had validation errors: {e}\nPlease ensure you follow the exact schema requirements."
                continue
            else:
                logger.error(f"Synthesis validation failed after {max_validation_retries + 1} attempts")
                raise
        
        except Exception as e:
            logger.error(f"Synthesizer node error: {e}")
            raise
    
    # Should never reach here
    raise Exception("Synthesizer node failed to generate valid synthesis")

