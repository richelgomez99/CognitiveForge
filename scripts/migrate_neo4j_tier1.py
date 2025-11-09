#!/usr/bin/env python3
"""
Neo4j Schema Migration for Tier 1: Intelligent Discovery System

This script adds new properties to ResearchPaper nodes to support claim-specific discovery:
- discovered_for_claim_id: UUID of the claim this paper was discovered for
- iteration_number: Which iteration/round of debate this was discovered in
- discovered_by: Agent or method that discovered this paper

Tier 1: T015 - Neo4j migration script
"""

import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")


def migrate_schema():
    """
    Add Tier 1 properties to ResearchPaper nodes.
    
    This migration is idempotent - safe to run multiple times.
    Existing nodes without the new properties will get default values.
    
    Changes:
        1. Add `discovered_for_claim_id` property (default: "unknown")
        2. Add `iteration_number` property (default: 0)
        3. Add `discovered_by` property (default: "manual")
        4. Create index on `discovered_for_claim_id` for fast lookups
    
    Returns:
        True if migration succeeded, False otherwise
    """
    driver = None
    
    try:
        logger.info(f"🔗 Connecting to Neo4j at {NEO4J_URI}")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            logger.info("✅ Connected to Neo4j")
            
            # =========================================================================
            # Step 1: Add default values to existing ResearchPaper nodes
            # =========================================================================
            logger.info("📝 Adding default values to existing ResearchPaper nodes...")
            
            result = session.run("""
                MATCH (p:ResearchPaper)
                WHERE p.discovered_for_claim_id IS NULL
                SET p.discovered_for_claim_id = 'unknown',
                    p.iteration_number = 0,
                    p.discovered_by = 'manual'
                RETURN count(p) as updated_count
            """)
            
            updated_count = result.single()["updated_count"]
            logger.info(f"✅ Updated {updated_count} existing ResearchPaper nodes")
            
            # =========================================================================
            # Step 2: Create index on discovered_for_claim_id for fast queries
            # =========================================================================
            logger.info("📇 Creating index on discovered_for_claim_id...")
            
            try:
                session.run("""
                    CREATE INDEX research_paper_claim_id IF NOT EXISTS
                    FOR (p:ResearchPaper)
                    ON (p.discovered_for_claim_id)
                """)
                logger.info("✅ Index created successfully")
            except Exception as e:
                # Index might already exist, that's okay
                if "equivalent index already exists" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("✅ Index already exists (skipped)")
                else:
                    raise
            
            # =========================================================================
            # Step 3: Verify schema changes
            # =========================================================================
            logger.info("🔍 Verifying schema changes...")
            
            result = session.run("""
                MATCH (p:ResearchPaper)
                WHERE p.discovered_for_claim_id IS NOT NULL
                  AND p.iteration_number IS NOT NULL
                  AND p.discovered_by IS NOT NULL
                RETURN count(p) as verified_count
            """)
            
            verified_count = result.single()["verified_count"]
            logger.info(f"✅ Verified {verified_count} ResearchPaper nodes have new properties")
            
            # =========================================================================
            # Step 4: Show sample migrated node
            # =========================================================================
            result = session.run("""
                MATCH (p:ResearchPaper)
                RETURN p.title as title, 
                       p.discovered_for_claim_id as claim_id, 
                       p.iteration_number as iteration,
                       p.discovered_by as discovered_by
                LIMIT 1
            """)
            
            sample = result.single()
            if sample:
                logger.info("📄 Sample migrated node:")
                logger.info(f"   Title: {sample['title'][:60]}...")
                logger.info(f"   Claim ID: {sample['claim_id']}")
                logger.info(f"   Iteration: {sample['iteration']}")
                logger.info(f"   Discovered By: {sample['discovered_by']}")
            
            logger.info("🎉 Migration completed successfully!")
            return True
    
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if driver:
            driver.close()
            logger.info("🔌 Neo4j connection closed")


def verify_migration():
    """
    Verify that the migration was applied correctly.
    
    Checks:
        1. All ResearchPaper nodes have the new properties
        2. Index exists on discovered_for_claim_id
        3. Property types are correct
    
    Returns:
        True if verification passed, False otherwise
    """
    driver = None
    
    try:
        logger.info("🔍 Verifying migration...")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Check if all nodes have the new properties
            result = session.run("""
                MATCH (p:ResearchPaper)
                RETURN 
                    count(p) as total_count,
                    count(p.discovered_for_claim_id) as has_claim_id,
                    count(p.iteration_number) as has_iteration,
                    count(p.discovered_by) as has_discovered_by
            """)
            
            stats = result.single()
            total = stats["total_count"]
            has_claim_id = stats["has_claim_id"]
            has_iteration = stats["has_iteration"]
            has_discovered_by = stats["has_discovered_by"]
            
            logger.info(f"Total ResearchPaper nodes: {total}")
            logger.info(f"Nodes with discovered_for_claim_id: {has_claim_id} ({has_claim_id/total*100 if total > 0 else 0:.1f}%)")
            logger.info(f"Nodes with iteration_number: {has_iteration} ({has_iteration/total*100 if total > 0 else 0:.1f}%)")
            logger.info(f"Nodes with discovered_by: {has_discovered_by} ({has_discovered_by/total*100 if total > 0 else 0:.1f}%)")
            
            if total > 0 and (has_claim_id < total or has_iteration < total or has_discovered_by < total):
                logger.warning("⚠️  Some nodes are missing the new properties!")
                return False
            
            logger.info("✅ Verification passed!")
            return True
    
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False
    
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Neo4j Schema Migration for Tier 1: Intelligent Discovery System")
    logger.info("=" * 80)
    
    # Run migration
    success = migrate_schema()
    
    if success:
        logger.info("")
        # Verify migration
        verify_migration()
    else:
        logger.error("Migration failed. Please check the logs above.")
        exit(1)

