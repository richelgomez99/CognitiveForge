#!/usr/bin/env python3
"""
Populate Neo4j knowledge graph with sample research papers.

Usage:
    python scripts/populate_kg.py --sample-data
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sample research papers data
SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "year": 2017,
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "url": "https://arxiv.org/abs/1706.03762",
        "keywords": ["transformer", "attention", "neural networks", "NLP"],
        "citations": 50000,
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Devlin et al."],
        "year": 2018,
        "abstract": "We introduce a new language representation model called BERT...",
        "url": "https://arxiv.org/abs/1810.04805",
        "keywords": ["BERT", "transformers", "pre-training", "NLP"],
        "citations": 40000,
    },
    {
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Brown et al."],
        "year": 2020,
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks through meta-learning...",
        "url": "https://arxiv.org/abs/2005.14165",
        "keywords": ["GPT-3", "few-shot learning", "language models", "scaling"],
        "citations": 25000,
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Wei et al."],
        "year": 2022,
        "abstract": "We explore how generating a chain of thought can improve the ability of language models to perform complex reasoning...",
        "url": "https://arxiv.org/abs/2201.11903",
        "keywords": ["chain-of-thought", "reasoning", "prompting", "LLMs"],
        "citations": 5000,
    },
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": ["Yao et al."],
        "year": 2023,
        "abstract": "We present ReAct, a novel framework that synergizes reasoning and acting with language models...",
        "url": "https://arxiv.org/abs/2210.03629",
        "keywords": ["ReAct", "reasoning", "actions", "agents"],
        "citations": 2000,
    },
    {
        "title": "LangChain: Building Applications with LLMs through Composability",
        "authors": ["Chase"],
        "year": 2022,
        "abstract": "LangChain is a framework for developing applications powered by language models...",
        "url": "https://github.com/langchain-ai/langchain",
        "keywords": ["LangChain", "LLM applications", "composability", "framework"],
        "citations": 3000,
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Lewis et al."],
        "year": 2020,
        "abstract": "Large pre-trained language models have been shown to store factual knowledge in their parameters...",
        "url": "https://arxiv.org/abs/2005.11401",
        "keywords": ["RAG", "retrieval", "knowledge", "generation"],
        "citations": 8000,
    },
]


def populate_knowledge_graph(sample_data: bool = False):
    """Populate Neo4j with research papers."""
    
    # Get connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    
    print(f"Connecting to Neo4j at {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Clear existing data (optional - for fresh start)
            print("Clearing existing ResearchPaper nodes...")
            session.run("MATCH (p:ResearchPaper) DETACH DELETE p")
            
            # Create constraint for unique URLs
            print("Creating constraint on ResearchPaper.url...")
            session.run("""
                CREATE CONSTRAINT research_paper_url IF NOT EXISTS
                FOR (p:ResearchPaper) REQUIRE p.url IS UNIQUE
            """)
            
            # Populate with sample data
            if sample_data:
                print(f"\nPopulating with {len(SAMPLE_PAPERS)} sample research papers...")
                for i, paper in enumerate(SAMPLE_PAPERS, 1):
                    result = session.run("""
                        CREATE (p:ResearchPaper {
                            title: $title,
                            authors: $authors,
                            year: $year,
                            abstract: $abstract,
                            url: $url,
                            keywords: $keywords,
                            citations: $citations,
                            created_at: datetime(),
                            last_accessed: datetime()
                        })
                        RETURN p.title AS title
                    """, **paper)
                    
                    title = result.single()["title"]
                    print(f"  [{i}/{len(SAMPLE_PAPERS)}] Created: {title}")
            
            # Create relationships between related papers (based on shared keywords)
            print("\nCreating relationships between related papers...")
            result = session.run("""
                MATCH (p1:ResearchPaper), (p2:ResearchPaper)
                WHERE p1 <> p2
                  AND size([k IN p1.keywords WHERE k IN p2.keywords]) > 0
                WITH p1, p2, [k IN p1.keywords WHERE k IN p2.keywords] AS shared_keywords
                MERGE (p1)-[r:RELATED_TO]->(p2)
                SET r.shared_keywords = shared_keywords,
                    r.strength = size(shared_keywords) * 1.0 / (size(p1.keywords) + size(p2.keywords) - size(shared_keywords))
                RETURN count(r) AS relationships_created
            """)
            
            relationships = result.single()["relationships_created"]
            print(f"  Created {relationships} RELATED_TO relationships")
            
            # Verify population
            print("\nVerifying data...")
            result = session.run("""
                MATCH (p:ResearchPaper)
                RETURN count(p) AS total_papers
            """)
            total = result.single()["total_papers"]
            print(f"  Total ResearchPaper nodes: {total}")
            
            result = session.run("""
                MATCH ()-[r:RELATED_TO]->()
                RETURN count(r) AS total_relationships
            """)
            total_rel = result.single()["total_relationships"]
            print(f"  Total relationships: {total_rel}")
            
            print("\n✅ Knowledge graph population complete!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Populate Neo4j knowledge graph")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Populate with sample research papers"
    )
    
    args = parser.parse_args()
    
    if not args.sample_data:
        print("Usage: python scripts/populate_kg.py --sample-data")
        print("\nThis script will populate Neo4j with sample research papers.")
        sys.exit(1)
    
    populate_knowledge_graph(sample_data=args.sample_data)


if __name__ == "__main__":
    main()

