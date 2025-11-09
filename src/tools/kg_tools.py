"""
Knowledge Graph tools for Neo4j integration.

This module provides functions to query the Neo4j knowledge graph and add insights.
"""

import os
import logging
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import Synthesis


def _get_neo4j_connection():
    """Get Neo4j connection details from environment."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    return uri, user, password


def query_knowledge_graph(query: str, claim_id: str = None) -> str:
    """
    Query the Neo4j knowledge graph using natural language.
    
    Uses GraphCypherQAChain to convert natural language queries to Cypher
    and retrieve relevant context from the knowledge graph.
    
    Tier 1: Enhanced to prioritize papers discovered for the current claim (US1).
    
    Implements FR-T1-006: Query Neo4j for contextual information
    
    Args:
        query: Natural language research query
        claim_id: Optional UUID of current claim for prioritizing claim-specific papers (Tier 1)
    
    Returns:
        str: Context from knowledge graph, or empty string on failure
    
    Example:
        >>> context = query_knowledge_graph("What is LangGraph?", claim_id="abc-123")
        >>> # Returns relevant research paper abstracts, prioritizing papers for this claim
    """
    try:
        uri, user, password = _get_neo4j_connection()
        
        # Initialize Neo4j graph
        graph = Neo4jGraph(
            url=uri,
            username=user,
            password=password
        )
        
        # Get Gemini model for Cypher generation
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set, skipping KG query")
            return ""
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=api_key,
            temperature=0
        )
        
        # T060: Add schema refresh and prioritization hint for recent papers
        graph.refresh_schema()
        
        # Create GraphCypherQAChain with enhanced instructions
        # Note: GraphCypherQAChain uses LLM to generate Cypher dynamically
        # We add context to prioritize claim-specific and recently discovered papers
        
        # Tier 1: T017 - Add claim-specific prioritization
        priority_hints = []
        if claim_id:
            priority_hints.append(f"HIGHEST PRIORITY: Papers where discovered_for_claim_id = '{claim_id}' (discovered specifically for this claim)")
        priority_hints.append("Priority: Papers discovered recently (within last 7 days) using discovered_at timestamp")
        priority_hints.append("Context: Papers with higher citation_count tend to be more authoritative")
        
        priority_context = "\n".join(priority_hints)
        
        enhanced_query = f"""{query}

Priority Context:
{priority_context}"""
        
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True  # Required for executing generated Cypher
        )
        
        # Execute query with enhanced instructions
        result = chain.invoke({"query": enhanced_query})
        
        # Extract and return context
        context = result.get("result", "")
        
        if context:
            logger.info(f"KG query successful. Context length: {len(context)} chars")
            return context
        else:
            logger.info("KG query returned no results")
            return ""
            
    except Exception as e:
        logger.warning(f"KG query failed: {e}. Falling back to LLM-only mode")
        return ""


def add_insight_to_graph(synthesis: Synthesis, thread_id: str = "default") -> bool:
    """
    Add a synthesis insight to the Neo4j knowledge graph.
    
    Creates an Insight node with the synthesis details and creates relationships
    to cited ResearchPaper nodes.
    
    Implements FR-T1-007: Write Insight nodes to Neo4j
    Implements FR-T1-008: Create [:CITES] relationships to evidence sources
    
    Args:
        synthesis: Synthesis object with novel insight and evidence lineage
        thread_id: Thread ID for tracking conversation context
    
    Returns:
        bool: True if successful, False on exception
    
    Example:
        >>> synthesis = Synthesis(novel_insight="...", evidence_lineage=[...], ...)
        >>> success = add_insight_to_graph(synthesis, "session_123")
    """
    try:
        uri, user, password = _get_neo4j_connection()
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Create Insight node
            result = session.run("""
                CREATE (i:Insight {
                    claim: $claim,
                    confidence_score: $confidence,
                    novelty_score: $novelty,
                    reasoning: $reasoning,
                    created_at: datetime(),
                    alert_sent: false,
                    thread_id: $thread_id
                })
                RETURN i.claim AS claim, id(i) AS insight_id
            """,
                claim=synthesis.novel_insight,
                confidence=synthesis.confidence_score,
                novelty=synthesis.novelty_score,
                reasoning=synthesis.reasoning,
                thread_id=thread_id
            )
            
            insight_record = result.single()
            if insight_record:
                insight_id = insight_record["insight_id"]
                claim = insight_record["claim"]
                logger.info(f"Created Insight node (ID: {insight_id}): {claim[:50]}...")
            
            # Create [:CITES] relationships to ResearchPaper nodes
            citations_created = 0
            for source_url in synthesis.evidence_lineage:
                citation_result = session.run("""
                    MATCH (i:Insight)
                    WHERE id(i) = $insight_id
                    MATCH (p:ResearchPaper {url: $url})
                    MERGE (i)-[c:CITES]->(p)
                    RETURN count(c) AS created
                """,
                    insight_id=insight_id,
                    url=source_url
                )
                
                count = citation_result.single()
                if count and count["created"] > 0:
                    citations_created += 1
            
            logger.info(
                f"Created {citations_created}/{len(synthesis.evidence_lineage)} "
                f"[:CITES] relationships"
            )
            
            if citations_created == 0:
                logger.warning(
                    "No citations created. Evidence URLs may not match ResearchPaper nodes."
                )
        
        driver.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to add insight to graph: {e}")
        return False


def get_insight_count() -> int:
    """
    Get the total count of Insight nodes in the knowledge graph.
    
    Returns:
        int: Number of Insight nodes, or -1 on error
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            result = session.run("MATCH (i:Insight) RETURN count(i) AS total")
            total = result.single()["total"]
        
        driver.close()
        return total
        
    except Exception as e:
        logger.error(f"Failed to get insight count: {e}")
        return -1


def add_papers_to_neo4j(papers: list, discovered_by: str, claim_id: str = "unknown", iteration_number: int = 0) -> dict:
    """
    Add papers to Neo4j knowledge graph with deduplication.
    
    Uses MERGE operation to prevent duplicate papers (by URL).
    Supports extended ResearchPaper schema with discovery metadata.
    
    Tier 1: Enhanced to track claim-specific discovery (US1, T027).
    
    Implements:
    - FR-KD-005: Deduplication by URL
    - FR-KD-006: Discovery metadata tracking
    - FR-KD-014: Extended ResearchPaper fields
    - Tier 1: Claim-specific paper tagging
    
    Args:
        papers: List of PaperMetadata objects to add
        discovered_by: Discovery method identifier (e.g., "manual", "query:...", "analyst_keyword", "skeptic_counter")
        claim_id: UUID of the claim this paper was discovered for (Tier 1)
        iteration_number: Which iteration/round of debate (Tier 1)
    
    Returns:
        dict: {
            "added": int,       # Papers successfully added
            "skipped": int,     # Papers skipped (duplicates)
            "failed": int,      # Papers that failed to add
            "details": List[str]  # Detailed messages
        }
    
    Example:
        >>> from src.models import PaperMetadata
        >>> papers = [PaperMetadata(...), PaperMetadata(...)]
        >>> result = add_papers_to_neo4j(papers, discovered_by="analyst_keyword", claim_id="abc-123", iteration_number=1)
        >>> print(f"Added {result['added']}, skipped {result['skipped']}")
    """
    # Import PaperMetadata here to avoid circular import
    from src.models import PaperMetadata
    
    added = 0
    skipped = 0
    failed = 0
    details = []
    
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            for paper in papers:
                try:
                    # Convert PaperMetadata to dict if needed
                    if isinstance(paper, PaperMetadata):
                        paper_dict = paper.model_dump()
                    else:
                        paper_dict = paper
                    
                    # Check if paper already exists (deduplication)
                    check_result = session.run(
                        "MATCH (p:ResearchPaper {url: $url}) RETURN p",
                        url=paper_dict['url']
                    )
                    
                    if check_result.single():
                        skipped += 1
                        details.append(f"Skipped (duplicate): {paper_dict['title'][:60]}...")
                        logger.debug(f"Paper already exists: {paper_dict['url']}")
                        continue
                    
                    # Add paper using MERGE (idempotent, prevents race conditions)
                    # Tier 1: T027 - Add claim_id, iteration_number, discovered_by
                    result = session.run(
                        """
                        MERGE (p:ResearchPaper {url: $url})
                        ON CREATE SET
                            p.title = $title,
                            p.abstract = $abstract,
                            p.authors = $authors,
                            p.published = $published,
                            p.source = $source,
                            p.citation_count = $citation_count,
                            p.fields_of_study = $fields_of_study,
                            p.discovered_at = datetime(),
                            p.discovered_by = $discovered_by,
                            p.discovered_for_claim_id = $claim_id,
                            p.iteration_number = $iteration_number
                        ON MATCH SET
                            p.citation_count = $citation_count,
                            p.fields_of_study = $fields_of_study
                        RETURN p.title AS title
                        """,
                        url=paper_dict['url'],
                        title=paper_dict['title'],
                        abstract=paper_dict['abstract'],
                        authors=paper_dict['authors'],
                        claim_id=claim_id,
                        iteration_number=iteration_number,
                        published=paper_dict['published'],
                        source=paper_dict['source'],
                        citation_count=paper_dict.get('citation_count', 0),
                        fields_of_study=paper_dict.get('fields_of_study', []),
                        discovered_by=discovered_by
                    )
                    
                    if result.single():
                        added += 1
                        details.append(f"Added: {paper_dict['title'][:60]}...")
                        logger.info(f"Added paper: {paper_dict['title'][:60]}... (source: {paper_dict['source']})")
                    
                except Exception as e:
                    failed += 1
                    details.append(f"Failed: {paper_dict.get('title', 'Unknown')[:60]}... - Error: {str(e)}")
                    logger.error(f"Failed to add paper: {e}")
                    continue
        
        driver.close()
        
        logger.info(
            f"Paper ingestion complete: {added} added, {skipped} skipped, {failed} failed"
        )
        
        return {
            "added": added,
            "skipped": skipped,
            "failed": failed,
            "details": details
        }
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return {
            "added": 0,
            "skipped": 0,
            "failed": len(papers),
            "details": [f"Neo4j connection error: {str(e)}"]
        }

