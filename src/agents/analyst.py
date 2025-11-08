"""
Analyst agent node for the dialectical synthesis system.

The Analyst generates an initial thesis based on the research query and knowledge graph context.
"""

import logging
from typing import Dict, Any
from pydantic import ValidationError

from src.models import AgentState, Thesis
from src.utils.model_config import get_agent_model
from src.utils.gemini_client import call_gemini_with_retry
from src.tools.kg_tools import query_knowledge_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyst agent node: Generate initial thesis based on query and knowledge graph context.
    
    This node:
    1. Queries the knowledge graph for relevant research context
    2. Constructs a prompt combining the query, KG context, and procedural memory
    3. Calls Gemini with structured output to generate a Thesis
    4. Returns updated state with the thesis
    
    Implements FR-T1-003: Analyst generates structured Thesis with evidence
    
    Args:
        state: Current AgentState with original_query
    
    Returns:
        Dict with updated current_thesis and messages
    
    Raises:
        Exception: If thesis generation fails after retries
    """
    logger.info("---EXECUTING ANALYST NODE---")
    
    # Extract query and procedural memory
    query = state["original_query"]
    procedural_memory = state.get("procedural_memory", "")
    
    # Query knowledge graph for context
    logger.info(f"Querying knowledge graph for: {query}")
    kg_context = query_knowledge_graph(query)
    
    if kg_context:
        logger.info(f"KG context retrieved: {len(kg_context)} characters")
    else:
        logger.info("No KG context found, using LLM knowledge only")
    
    # Get model ID for analyst
    model_id = get_agent_model("analyst")
    
    # Construct prompt
    prompt = f"""You are an expert research analyst tasked with developing a well-supported thesis.

Research Query: {query}

Knowledge Graph Context:
{kg_context if kg_context else "No relevant research papers found in knowledge graph. Use your general knowledge."}

{f"Procedural Heuristics (from past successes):{procedural_memory}" if procedural_memory else ""}

Your task:
1. Analyze the research query and available context
2. Formulate a clear, specific claim or hypothesis
3. Provide detailed reasoning connecting your evidence to the claim
4. Cite at least 2 pieces of evidence with source URLs
   - If citing from the knowledge graph, use the provided URLs
   - If using general knowledge, use authoritative source URLs (e.g., arxiv.org, wikipedia.org)
5. Ensure evidence snippets are meaningful and support your claim

Generate a thesis with:
- claim: A clear, specific claim or hypothesis (minimum 20 characters)
- reasoning: Detailed explanation (minimum 50 characters)
- evidence: List of at least 2 Evidence objects with source_url, snippet, and optional relevance_score

Be thorough and academic in your analysis."""

    # Call Gemini with retry logic
    max_validation_retries = 2
    for attempt in range(max_validation_retries + 1):
        try:
            logger.info(f"Calling Gemini ({model_id}) for thesis generation (attempt {attempt + 1})")
            
            # Get structured output from Gemini
            response_text = call_gemini_with_retry(
                model_id=model_id,
                prompt=prompt,
                response_schema=Thesis.model_json_schema()
            )
            
            # Parse and validate response
            thesis = Thesis.model_validate_json(response_text)
            
            logger.info(f"Thesis generated successfully: {thesis.claim[:100]}...")
            logger.info(f"Evidence count: {len(thesis.evidence)}")
            
            # Return updated state
            return {
                "current_thesis": thesis,
                "messages": [f"Analyst: {thesis.claim}"]
            }
            
        except ValidationError as e:
            logger.warning(f"Pydantic validation error (attempt {attempt + 1}): {e}")
            
            if attempt < max_validation_retries:
                # Add explicit schema guidance to prompt
                prompt += f"\n\nIMPORTANT: Your previous response had validation errors: {e}\nPlease ensure you follow the exact schema requirements."
                continue
            else:
                logger.error(f"Thesis validation failed after {max_validation_retries + 1} attempts")
                raise
        
        except Exception as e:
            logger.error(f"Analyst node error: {e}")
            raise
    
    # Should never reach here
    raise Exception("Analyst node failed to generate valid thesis")

