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


def query_knowledge_graph(query: str) -> str:
    """
    Query the Neo4j knowledge graph using natural language.
    
    Uses GraphCypherQAChain to convert natural language queries to Cypher
    and retrieve relevant context from the knowledge graph.
    
    Implements FR-T1-006: Query Neo4j for contextual information
    
    Args:
        query: Natural language research query
    
    Returns:
        str: Context from knowledge graph, or empty string on failure
    
    Example:
        >>> context = query_knowledge_graph("What is LangGraph?")
        >>> # Returns relevant research paper abstracts and relationships
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
        
        # Create GraphCypherQAChain
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True  # Required for executing generated Cypher
        )
        
        # Execute query
        result = chain.invoke({"query": query})
        
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

