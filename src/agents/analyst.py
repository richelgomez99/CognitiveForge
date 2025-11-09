"""
Analyst agent node for the dialectical synthesis system.

The Analyst generates an initial thesis based on the research query and knowledge graph context.

Tier 1: Enhanced with multi-keyword discovery per claim (US1).
"""

import logging
import asyncio
from typing import Dict, Any
from pydantic import ValidationError

from src.models import AgentState, Thesis
from src.utils.model_config import get_agent_model
from src.utils.gemini_client import call_gemini_with_retry
from src.tools.kg_tools import query_knowledge_graph, add_papers_to_neo4j
from src.tools.keyword_extraction import extract_keywords
from src.tools.discovery_strategy import determine_discovery_strategy, should_trigger_follow_up, _infer_complexity_level
from src.tools.paper_discovery import discover_papers_for_keywords_parallel
from src.utils.memory import format_memory_for_prompt, get_memory_summary
from src.tools.telemetry import log_discovery_telemetry  # T077

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyst agent node: Generate initial thesis based on query and knowledge graph context.
    
    Tier 1: Enhanced flow with multi-keyword discovery (T030-T035):
    1. Generate preliminary claim/reasoning
    2. Extract 3-5 targeted keywords (T030)
    3. Determine adaptive discovery strategy (T031)
    4. Discover papers for each keyword in parallel (T032)
    5. Add papers to Neo4j with claim_id tagging
    6. Follow-up discovery if needed (T033)
    7. Query KG with claim-specific context (T034)
    8. Generate final thesis with enriched context
    9. Log all steps for transparency (T035)
    
    Args:
        state: Current AgentState with original_query, current_claim_id, iteration_count
    
    Returns:
        Dict with updated current_thesis and messages
    
    Raises:
        Exception: If thesis generation fails after retries
    """
    logger.info("---EXECUTING ANALYST NODE (Tier 1: Multi-Keyword Discovery)---")
    
    # Extract state
    query = state["original_query"]
    procedural_memory = state.get("procedural_memory", "")
    claim_id = state["current_claim_id"]
    iteration_number = state.get("iteration_count", 0)
    debate_memory = state.get("debate_memory", {"rejected_claims": [], "skeptic_objections": [], "weak_evidence_urls": []})
    
    logger.info(f"Query: {query}")
    logger.info(f"Claim ID: {claim_id[:8]}...")
    logger.info(f"Iteration: {iteration_number}")
    
    # T045: Log memory summary
    memory_summary = get_memory_summary(debate_memory)
    logger.info(f"💭 {memory_summary}")
    
    # T030: Generate preliminary claim for keyword extraction
    logger.info("Step 1: Generating preliminary claim for keyword extraction...")
    preliminary_claim = query  # For first pass, use query as claim
    preliminary_reasoning = "Initial analysis based on research query"
    
    # T030: Extract keywords from claim
    logger.info("Step 2: Extracting targeted keywords...")
    keywords_result = extract_keywords(preliminary_claim, preliminary_reasoning)
    logger.info(f"✅ Extracted {len(keywords_result.keywords)} keywords:")  # T035
    for i, kw in enumerate(keywords_result.keywords, 1):
        logger.info(f"   {i}. {kw}")  # T035
    
    # T031: Determine discovery strategy
    logger.info("Step 3: Determining adaptive discovery strategy...")
    strategy = determine_discovery_strategy(preliminary_claim, [])
    logger.info(f"✅ Strategy: {strategy.initial_papers} initial papers, follow-up: {strategy.follow_up_needed}")  # T035
    logger.info(f"   Reasoning: {strategy.reasoning}")  # T035
    
    # T032: Discover papers for each keyword in parallel
    logger.info("Step 4: Discovering papers for each keyword in parallel...")
    discovered_papers, url_to_keywords = asyncio.run(
        discover_papers_for_keywords_parallel(
            keywords=keywords_result.keywords,
            max_results_per_keyword=strategy.initial_papers,
            claim_id=claim_id
        )
    )
    logger.info(f"✅ Discovered {len(discovered_papers)} unique papers across {len(keywords_result.keywords)} keywords")  # T035
    
    # Add papers to Neo4j with claim_id and iteration_number
    if discovered_papers:
        logger.info("Step 5: Adding papers to Neo4j...")
        result = add_papers_to_neo4j(
            papers=discovered_papers,
            discovered_by=f"analyst_keyword_iteration_{iteration_number}",
            claim_id=claim_id,
            iteration_number=iteration_number
        )
        logger.info(f"✅ Neo4j: added={result['added']}, skipped={result['skipped']}, failed={result['failed']}")  # T035
    
    # T033: Follow-up discovery if needed
    if discovered_papers and strategy.follow_up_needed:
        # Calculate average relevance
        relevance_scores = [p.citation_count or 0 for p in discovered_papers]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        # Normalize to 0-1 (assume max citation count of 1000)
        avg_relevance = min(avg_relevance / 1000, 1.0)
        
        if should_trigger_follow_up(strategy, discovered_papers, avg_relevance):
            logger.info(f"Step 6: Triggering follow-up discovery (avg relevance: {avg_relevance:.2f})...")
            # Use first 2 keywords for follow-up (most important)
            follow_up_papers, _ = asyncio.run(
                discover_papers_for_keywords_parallel(
                    keywords=keywords_result.keywords[:2],
                    max_results_per_keyword=strategy.follow_up_papers,
                    claim_id=claim_id
                )
            )
            if follow_up_papers:
                logger.info(f"✅ Follow-up: {len(follow_up_papers)} additional papers")  # T035
                result = add_papers_to_neo4j(
                    papers=follow_up_papers,
                    discovered_by=f"analyst_followup_iteration_{iteration_number}",
                    claim_id=claim_id,
                    iteration_number=iteration_number
                )
                logger.info(f"✅ Follow-up Neo4j: added={result['added']}, skipped={result['skipped']}")  # T035
                
                # T077: Log telemetry for follow-up discovery
                total_papers_fetched = len(discovered_papers) + len(follow_up_papers)
                complexity_level = _infer_complexity_level(strategy.initial_papers)
                log_discovery_telemetry(
                    claim=preliminary_claim,
                    strategy_recommended=strategy.initial_papers + strategy.follow_up_papers,
                    actual_papers_fetched=total_papers_fetched,
                    avg_relevance=avg_relevance,
                    follow_up_triggered=True,
                    complexity_level=complexity_level,
                    claim_id=claim_id
                )
        else:
            # T077: Log telemetry for initial discovery (no follow-up)
            complexity_level = _infer_complexity_level(strategy.initial_papers)
            relevance_scores = [p.citation_count or 0 for p in discovered_papers]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            avg_relevance = min(avg_relevance / 1000, 1.0)
            
            log_discovery_telemetry(
                claim=preliminary_claim,
                strategy_recommended=strategy.initial_papers,
                actual_papers_fetched=len(discovered_papers),
                avg_relevance=avg_relevance,
                follow_up_triggered=False,
                complexity_level=complexity_level,
                claim_id=claim_id
            )
    
    # T034: Query knowledge graph with claim-specific context
    logger.info("Step 7: Querying knowledge graph with claim-specific context...")
    kg_context = query_knowledge_graph(query, claim_id=claim_id)
    
    # Log context availability (T035)
    kg_context_length = len(kg_context) if kg_context else 0
    
    if kg_context_length > 0:
        logger.info(f"✅ KG context retrieved: {kg_context_length} characters (prioritizing papers for claim_id: {claim_id[:8]}...)")
    else:
        logger.info("No KG context found - using LLM general knowledge")
    
    # Get model ID for analyst
    logger.info("Step 8: Generating final thesis with enriched context...")
    model_id = get_agent_model("analyst")
    
    # T043-T044: Add memory-informed context if debate_memory is non-empty
    memory_context = ""
    rejected_claims = debate_memory.get("rejected_claims", [])
    if rejected_claims:
        memory_context = f"\n\n⚠️ MEMORY: Debate History (Avoid Circular Arguments)\n{format_memory_for_prompt(debate_memory)}\n\nIMPORTANT: Do NOT repeat or closely paraphrase the rejected claims above. Generate a genuinely different approach or perspective."
        logger.info(f"📚 Memory: {len(rejected_claims)} rejected claims found, informing thesis generation")  # T045
    
    # Construct prompt with improved quality guidance
    prompt = f"""You are an expert research analyst tasked with developing a well-supported, defensible thesis that can withstand critical scrutiny.

Research Query: {query}

Knowledge Graph Context:
{kg_context if kg_context else "No relevant research papers found in knowledge graph. Use your general knowledge."}

{f"Procedural Heuristics (from past successes):{procedural_memory}" if procedural_memory else ""}{memory_context}

Your task:
1. Analyze the research query and available context deeply
2. Formulate a clear, specific, and DEFENSIBLE claim or hypothesis
   - Avoid overly broad or vague claims
   - Acknowledge scope and limitations within the claim itself
   - Focus on what the evidence actually supports
3. Provide detailed reasoning that:
   - Explicitly connects evidence to the claim
   - Anticipates potential counter-arguments
   - Acknowledges uncertainty where appropriate
           4. Cite at least 2 pieces of STRONG evidence with source URLs
              - If citing from the knowledge graph, use the provided URLs
              - If using general knowledge, use authoritative source URLs (e.g., arxiv.org, scientific journals)
              - Ensure evidence DIRECTLY supports your claim (not tangentially related)
              - T057: When referencing papers in your reasoning, use format: [CITE: paper_url]
           5. Write evidence snippets that are specific and meaningful (not generic descriptions)

Quality checklist before finalizing:
- Would this claim hold up under expert scrutiny?
- Does the evidence directly support the claim (not just related topics)?
- Have I acknowledged important limitations or scope boundaries?
- Is my reasoning logically sound and free of major gaps?

Generate a thesis with:
- claim: A clear, specific, defensible claim (minimum 30 characters)
- reasoning: Detailed explanation with logical flow (minimum 100 characters)
- evidence: List of at least 2 Evidence objects with source_url, specific snippet text, and relevance_score (0.0-1.0)

Be thorough, academic, and intellectually honest in your analysis."""

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

