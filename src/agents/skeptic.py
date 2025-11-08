"""
Skeptic agent node for the dialectical synthesis system.

The Skeptic evaluates the Analyst's thesis and identifies contradictions or weaknesses.
"""

import logging
from typing import Dict, Any
from pydantic import ValidationError

from src.models import AgentState, Antithesis
from src.utils.model_config import get_agent_model
from src.utils.gemini_client import call_gemini_with_retry

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
    logger.info("---EXECUTING SKEPTIC NODE---")
    
    # Extract thesis
    thesis = state["current_thesis"]
    if not thesis:
        raise ValueError("Skeptic requires a thesis from Analyst")
    
    # Get model ID for skeptic
    model_id = get_agent_model("skeptic")
    
    # Format evidence for prompt
    evidence_text = "\n".join([
        f"- Source: {ev.source_url}\n  Snippet: {ev.snippet}"
        for ev in thesis.evidence
    ])
    
    # Construct critical evaluation prompt
    prompt = f"""You are an expert skeptical analyst tasked with critically evaluating a research thesis.

Thesis Claim: {thesis.claim}

Thesis Reasoning: {thesis.reasoning}

Thesis Evidence:
{evidence_text}

Your task:
1. Critically analyze the thesis for logical weaknesses, contradictions, or unsupported claims
2. Identify if there are ANY contradictions, gaps in logic, or questionable assumptions
3. If contradictions exist:
   - Set contradiction_found to true
   - Provide a counter_claim that offers an alternative interpretation
   - Cite conflicting_evidence (can reference same sources with different interpretation)
   - Provide detailed critique explaining the weaknesses
4. If the thesis is sound:
   - Set contradiction_found to false
   - Still provide a critique acknowledging its strengths while noting any minor limitations

Generate an antithesis with:
- contradiction_found: boolean (true if you found significant issues)
- counter_claim: string (required if contradiction_found is true, alternative interpretation)
- conflicting_evidence: list of Evidence objects (supporting your critique)
- critique: detailed explanation of weaknesses or contradictions (minimum 30 characters)

Be thorough and intellectually honest. Don't invent contradictions that don't exist, but don't be overly lenient either."""

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
            
            logger.info(f"Antithesis generated successfully")
            logger.info(f"Contradiction found: {antithesis.contradiction_found}")
            if antithesis.counter_claim:
                logger.info(f"Counter-claim: {antithesis.counter_claim[:100]}...")
            
            # Prepare contradiction report
            contradiction_report = antithesis.critique if antithesis.contradiction_found else ""
            
            # Return updated state
            return {
                "current_antithesis": antithesis,
                "contradiction_report": contradiction_report,
                "messages": [f"Skeptic: {antithesis.critique[:150]}..."]
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

