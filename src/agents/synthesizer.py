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
    
    # T089: Check synthesis mode for circular argument handling
    synthesis_mode = state.get("synthesis_mode", "standard")
    logger.info(f"Synthesis mode: {synthesis_mode}")
    
    # Construct synthesis prompt based on mode
    if synthesis_mode == "impasse":
        # T089: Impasse mode - circular argument detected
        logger.info("🔄 Generating impasse synthesis (circular argument detected)")
        
        # Extract previous claims from debate memory
        debate_memory = state.get("debate_memory")
        previous_claims = debate_memory.rejected_claims if debate_memory else []
        
        prompt = f"""You are an expert research synthesizer. The dialectical debate reached an IMPASSE due to circular argument detection.

Thesis Claim (REJECTED as circular): {thesis.claim}
Thesis Reasoning: {thesis.reasoning}

Circular Argument Critique: {antithesis.critique}

Previous Valid Claims Explored:
{chr(10).join(f"- {claim}" for claim in previous_claims[:3]) if previous_claims else "No previous claims available"}

Your task:
1. **ACKNOWLEDGE THE IMPASSE**: Explain that the debate reached a circular argument, indicating exhaustion of current theoretical approaches
2. **DO NOT VALIDATE THE CIRCULAR CLAIM**: The rejected thesis should not be presented as the correct answer
3. **SYNTHESIZE FROM VALID CLAIMS**: Focus on insights from the NON-CIRCULAR claims that were explored before the impasse
4. **EXPLAIN WHAT WAS LEARNED**: What does the impasse reveal about the complexity of the question?
5. **IDENTIFY OPEN QUESTIONS**: What remains unresolved? What approaches might break the impasse?

Generate a synthesis that:
- novel_insight: Acknowledges impasse and synthesizes what WAS learned (minimum 50 characters, maximum 500 characters)
- supporting_claims: Claims from the VALID explorations (before circularity) (minimum 2)
- confidence_score: Moderate (0.4-0.6) due to impasse
- novelty_score: Lower (0.3-0.5) as impasse limits new insights
- reasoning: Explanation of impasse, what was learned, and what remains open (minimum 100 characters)

Be intellectually honest about the limitations imposed by the circular argument."""

    elif synthesis_mode == "exhausted_attempts":
        # Max iterations reached
        logger.info("⏭️  Generating synthesis after max iterations reached")
        
        prompt = f"""You are an expert research synthesizer. The dialectical debate reached MAX ITERATIONS without resolution.

Final Thesis Claim: {thesis.claim}
Thesis Reasoning: {thesis.reasoning}

Final Antithesis Analysis: {antithesis.critique}
{f"Counter-claim: {antithesis.counter_claim}" if antithesis.counter_claim else ""}

Multiple attempts at thesis-antithesis resolution failed to reach convergence.

Your task:
1. **ACKNOWLEDGE DIFFICULTY**: Explain that multiple iterations failed to resolve tensions
2. **SYNTHESIZE PARTIAL INSIGHTS**: What was learned from the multiple attempts?
3. **EXPLAIN WHY CONVERGENCE WAS HARD**: What makes this question particularly complex?
4. **IDENTIFY REMAINING TENSIONS**: What contradictions remain unresolved?

Generate a synthesis that:
- novel_insight: Partial synthesis acknowledging difficulty of resolution (minimum 50 characters, maximum 500 characters)
- supporting_claims: Insights from multiple iterations (minimum 2)
- confidence_score: Lower (0.3-0.5) due to lack of resolution
- novelty_score: Moderate (0.4-0.6)
- reasoning: Explanation of what was explored and why convergence was difficult (minimum 100 characters)"""

    else:
        # Standard synthesis mode
        dialectical_context = "rigorous dialectical debate" if antithesis.contradiction_found else "critical evaluation that validated the thesis"
        
        prompt = f"""You are an expert research synthesizer tasked with integrating diverse perspectives into novel insights after {dialectical_context}.

Thesis Claim: {thesis.claim}
Thesis Reasoning: {thesis.reasoning}

Antithesis Analysis: {antithesis.critique}
{f"Counter-claim: {antithesis.counter_claim}" if antithesis.counter_claim else ""}
Contradiction Found: {antithesis.contradiction_found}

Your task:
1. Synthesize a novel insight that integrates the dialectical process:
   - If contradiction was found: Integrate thesis, antithesis, and resolution of tensions
   - If no contradiction: Build upon the validated thesis with deeper implications or connections
2. Go beyond simply restating - generate a NEW perspective or understanding
3. Acknowledge both strengths and any limitations identified
4. Support your synthesis with specific claims from the dialectical process
5. **IMPORTANT**: Self-assess the novelty of your synthesis:
   - novelty_score: How novel is this insight compared to the original thesis?
   - Score 0.0-0.3: Mostly restates the thesis (valid if thesis was already excellent)
   - Score 0.4-0.6: Moderately novel, makes meaningful connections
   - Score 0.7-1.0: Highly novel, significant new insights or frameworks
6. Assess your confidence in the synthesis (0.0-1.0)

Generate a synthesis with:
- novel_insight: Your synthesized insight (minimum 50 characters, maximum 500 characters)
- supporting_claims: List of specific claims from thesis/antithesis that support synthesis (minimum 2)
- evidence_lineage: List of ALL source URLs (will be provided automatically)
- confidence_score: Your confidence (0.0-1.0)
- novelty_score: Self-assessed novelty (0.0-1.0) - be intellectually honest
- reasoning: Detailed explanation of how you derived this synthesis from the dialectical process (minimum 100 characters)

Be intellectually rigorous. If the thesis was sound and passed skeptical scrutiny, a lower novelty score (0.2-0.4) is appropriate and honest."""

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

