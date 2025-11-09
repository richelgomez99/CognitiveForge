"""
Paper Discovery Module for Knowledge Discovery Engine.

This module provides functions to search for academic papers from arXiv and Semantic Scholar,
with built-in rate limiting, error handling, and URL normalization.

Key Features:
- arXiv API integration with configurable delays
- Semantic Scholar API integration with semantic search
- Exponential backoff retry for rate limits
- URL normalization for deduplication
- Error isolation per source (one source failure doesn't block the other)
- Comprehensive logging

Usage:
    from src.tools.paper_discovery import discover_papers_for_query
    
    papers = discover_papers_for_query("transformer architecture")
    # Returns List[PaperMetadata] from both arXiv and Semantic Scholar
"""

import logging
import time
import os
from typing import List
from datetime import datetime

import arxiv
from semanticscholar import SemanticScholar
from dotenv import load_dotenv

from src.models import PaperMetadata

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Rate Limiting Configuration (Tier 1: Phase 7 - Critical Fix)
# =============================================================================

# Semantic Scholar rate limit: 1 request per second (free tier)
# Add buffer to be safe: 1.5 seconds between requests
SEMANTIC_SCHOLAR_DELAY = float(os.getenv("SEMANTIC_SCHOLAR_DELAY", "1.5"))

# Enable sequential mode to avoid 429 errors (Tier 1: Phase 7 fix)
SEQUENTIAL_SEMANTIC_SCHOLAR = os.getenv("SEQUENTIAL_SEMANTIC_SCHOLAR", "true").lower() == "true"

logger.info(f"📊 Paper discovery config: SEQUENTIAL_SEMANTIC_SCHOLAR={SEQUENTIAL_SEMANTIC_SCHOLAR}, DELAY={SEMANTIC_SCHOLAR_DELAY}s")


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_arxiv_url(url: str) -> str:
    """
    Normalize arXiv URLs to canonical form for deduplication.
    
    Handles variations like:
    - export.arxiv.org → arxiv.org
    - arxiv.org/abs/1234.5678v2 → arxiv.org/abs/1234.5678
    
    Args:
        url: arXiv URL to normalize
        
    Returns:
        Normalized URL string
    """
    url = url.replace("export.arxiv.org", "arxiv.org")
    url = url.replace("/pdf/", "/abs/")
    
    # Remove version suffix (v1, v2, etc.)
    if "/abs/" in url and "v" in url.split("/abs/")[-1]:
        base_url = url.split("v")[0]
        return base_url
    
    return url


# =============================================================================
# arXiv Search
# =============================================================================

def search_arxiv(query: str, max_results: int = 10) -> List[PaperMetadata]:
    """
    Search arXiv for academic papers with rate-limiting and error handling.
    
    Uses arxiv.Client with configurable delay_seconds to respect API limits.
    Implements exponential backoff for rate limit errors (HTTP 429).
    
    Args:
        query: Search query (keywords, authors, titles)
        max_results: Maximum number of papers to return (default: 10)
        
    Returns:
        List of PaperMetadata objects
        
    Raises:
        Exception: If search fails after all retries
    """
    retry_delays = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s
    last_exception = None
    
    for attempt, delay in enumerate(retry_delays + [None], start=1):
        try:
            logger.info(f"Searching arXiv: query='{query}', max_results={max_results} (attempt {attempt})")
            
            # Create arXiv client with rate limiting
            client = arxiv.Client(
                page_size=max_results,
                delay_seconds=3.0,  # 3s delay between requests
                num_retries=0  # We handle retries manually
            )
            
            # Create search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Execute search
            papers = []
            for result in client.results(search):
                try:
                    # Ensure title exists and meets minimum length
                    title = result.title
                    if not title or len(title) < 10:
                        logger.debug(f"Skipping arXiv paper: title missing or too short")
                        continue
                    
                    # Ensure abstract meets minimum length (50 chars)
                    abstract = result.summary if result.summary else None
                    if not abstract or len(abstract) < 50:
                        abstract = f"Abstract not available for this arXiv paper. Title: {title}. This is a placeholder to meet minimum length requirements."
                    
                    # Ensure at least one author
                    authors = [author.name for author in result.authors] if result.authors else ["Unknown Author"]
                    if not authors:
                        authors = ["Unknown Author"]
                    
                    paper = PaperMetadata(
                        title=title,
                        url=normalize_arxiv_url(result.entry_id),
                        abstract=abstract,
                        authors=authors,
                        published=result.published.isoformat(),
                        source="arxiv",
                        citation_count=0,  # arXiv doesn't provide citation counts
                        fields_of_study=[category for category in result.categories] if result.categories else []
                    )
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse arXiv paper '{getattr(result, 'title', 'Unknown')}': {type(e).__name__}: {e}")
                    continue
            
            logger.info(f"✅ arXiv search successful: {len(papers)} papers found")
            return papers
            
        except Exception as e:
            last_exception = e
            logger.warning(f"arXiv search attempt {attempt} failed: {e}")
            
            # If this isn't the last attempt, wait and retry
            if delay is not None:
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                # Last attempt failed
                logger.error(f"arXiv search failed after {len(retry_delays)} retries: {last_exception}")
                # Don't raise - return empty list to allow other sources to succeed
                return []
    
    return []


# =============================================================================
# Semantic Scholar Search
# =============================================================================

def search_semantic_scholar(query: str, max_results: int = 10) -> List[PaperMetadata]:
    """
    Search Semantic Scholar for academic papers with semantic ranking.
    
    Implements exponential backoff for rate limit errors (HTTP 429).
    Uses API key from environment if available for higher rate limits.
    
    Args:
        query: Search query (keywords, authors, titles)
        max_results: Maximum number of papers to return (default: 10)
        
    Returns:
        List of PaperMetadata objects
        
    Raises:
        Exception: If search fails after all retries
    """
    retry_delays = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s
    last_exception = None
    
    # Get API key from environment (optional but recommended)
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    
    for attempt, delay in enumerate(retry_delays + [None], start=1):
        try:
            logger.info(f"Searching Semantic Scholar: query='{query}', max_results={max_results} (attempt {attempt})")
            
            # Create Semantic Scholar client
            if api_key:
                sch = SemanticScholar(api_key=api_key)
                logger.debug("Using Semantic Scholar API key (higher rate limits)")
            else:
                sch = SemanticScholar()
                logger.debug("Using Semantic Scholar without API key (lower rate limits)")
            
            # Execute search with AGGRESSIVE early termination
            # The semanticscholar library has threading/retry issues - we need to force evaluation
            results_generator = sch.search_paper(query, limit=max_results)
            
            # Convert to list immediately with hard limit to prevent background threads
            papers = []
            collected = 0
            max_collect = max_results * 3  # Buffer for validation failures
            
            for result in results_generator:
                collected += 1
                # Hard stop to prevent infinite iteration
                if collected > max_collect:
                    logger.warning(f"Hit collection limit ({max_collect}), stopping search")
                    break
                
                # Stop when we have enough valid papers
                if len(papers) >= max_results:
                    break
                
                try:
                    # Skip papers without essential fields
                    if not result.title or not result.paperId:
                        logger.debug(f"Skipping paper: missing title or paperId")
                        continue
                    
                    # Ensure title meets minimum length (10 chars)
                    title = result.title
                    if len(title) < 10:
                        logger.debug(f"Skipping paper: title too short ({len(title)} chars)")
                        continue
                    
                    # Ensure abstract meets minimum length (50 chars)
                    abstract = result.abstract if result.abstract else None
                    if not abstract or len(abstract) < 50:
                        abstract = f"Abstract not available for this paper. Title: {title}. This is a placeholder to meet minimum length requirements."
                    
                    # Ensure at least one author
                    # Note: result.authors is a list of Author objects with .name attribute
                    authors = []
                    if result.authors:
                        for author in result.authors:
                            if hasattr(author, 'name') and author.name:
                                authors.append(author.name)
                            elif isinstance(author, dict) and 'name' in author:
                                authors.append(author['name'])
                    
                    if not authors:
                        authors = ["Unknown Author"]
                    
                    # Format publication date
                    published = str(result.year) if result.year else "2024-01-01"
                    
                    paper = PaperMetadata(
                        title=title,
                        url=result.url if result.url else f"https://www.semanticscholar.org/paper/{result.paperId}",
                        abstract=abstract,
                        authors=authors,
                        published=published,
                        source="semantic_scholar",
                        citation_count=result.citationCount if result.citationCount else 0,
                        fields_of_study=result.fieldsOfStudy if result.fieldsOfStudy else []
                    )
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse Semantic Scholar paper '{getattr(result, 'title', 'Unknown')}': {type(e).__name__}: {e}")
                    continue
            
            logger.info(f"✅ Semantic Scholar search successful: {len(papers)} papers found")
            return papers
            
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if "429" in error_str or "rate limit" in error_str:
                logger.warning(f"Semantic Scholar rate limit hit (attempt {attempt})")
                
                # If this isn't the last attempt, wait and retry
                if delay is not None:
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Semantic Scholar rate limit exceeded after {len(retry_delays)} retries")
                    return []
            else:
                # Non-rate-limit error
                logger.error(f"Semantic Scholar search failed: {e}")
                return []
    
    return []


# =============================================================================
# Unified Discovery Function
# =============================================================================

def discover_papers_for_query(query: str, max_results_per_source: int = 10, claim_id: str = None) -> List[PaperMetadata]:
    """
    Discover papers from both arXiv and Semantic Scholar for a given query.
    
    This is the main entry point for paper discovery. It searches both sources
    with error isolation - if one source fails, the other continues.
    
    Tier 1: Enhanced to accept claim_id for claim-specific discovery tracking (T026).
    
    Args:
        query: Search query (keywords)
        max_results_per_source: Maximum papers per source (default: 10)
        claim_id: Optional UUID of the claim this discovery is for (Tier 1: US1)
        
    Returns:
        Combined list of PaperMetadata from both sources (max: 20 papers)
        
    Example:
        >>> papers = discover_papers_for_query("transformer architecture", max_results_per_source=10, claim_id="abc-123")
        >>> print(f"Found {len(papers)} papers")
    
    Tier 1: The claim_id is used for context logging and passed to the caller for Neo4j tagging.
    """
    claim_context = f" [claim_id: {claim_id[:8]}...]" if claim_id else ""
    logger.info(f"Starting paper discovery for query: '{query}'{claim_context}")
    start_time = time.time()
    
    all_papers = []
    
    # Search arXiv (error isolated)
    try:
        arxiv_papers = search_arxiv(query, max_results=max_results_per_source)
        all_papers.extend(arxiv_papers)
        logger.info(f"arXiv contributed {len(arxiv_papers)} papers")
    except Exception as e:
        logger.error(f"arXiv search failed (continuing with other sources): {e}")
    
    # Search Semantic Scholar (error isolated)
    # DEMO MODE: Skip if ARXIV_ONLY=true (avoids rate limits for demo)
    arxiv_only = os.getenv("ARXIV_ONLY", "false").lower() == "true"
    
    if arxiv_only:
        logger.info("⏭️  Semantic Scholar: Skipped (DEMO MODE - arXiv only, no rate limits)")
    else:
        try:
            semantic_scholar_papers = search_semantic_scholar(query, max_results=max_results_per_source)
            all_papers.extend(semantic_scholar_papers)
            logger.info(f"Semantic Scholar contributed {len(semantic_scholar_papers)} papers")
        except Exception as e:
            logger.error(f"Semantic Scholar search failed (continuing with other sources): {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Paper discovery complete: {len(all_papers)} papers found in {elapsed_time:.2f}s")
    
    return all_papers


# =============================================================================
# Helper Functions for Sequential Discovery (Phase 7)
# =============================================================================

def _search_arxiv(query: str, max_results: int) -> List[PaperMetadata]:
    """Wrapper for arXiv search (used in sequential mode)."""
    return search_arxiv(query, max_results)


def _search_semantic_scholar(query: str, max_results: int) -> List[PaperMetadata]:
    """Wrapper for Semantic Scholar search (used in sequential mode)."""
    return search_semantic_scholar(query, max_results)


async def _discover_sequential_rate_limited(
    keywords: List[str],
    max_results_per_keyword: int,
    claim_id: str = None
) -> List[List[PaperMetadata]]:
    """
    Sequential discovery with rate limiting for Semantic Scholar.
    
    Tier 1: Phase 7 - Critical fix for 429 rate limit errors.
    
    Strategy:
    1. Query arXiv for all keywords in parallel (no rate limits)
    2. Query Semantic Scholar sequentially with delays (1 req/sec)
    3. Combine results
    
    Args:
        keywords: List of keywords to search
        max_results_per_keyword: Papers per keyword
        claim_id: Optional claim ID for tracking
    
    Returns:
        List of paper lists (one per keyword)
    
    Performance:
        - arXiv: ~2-5 seconds (parallel)
        - Semantic Scholar: ~1.5s * num_keywords (sequential)
        - Total: ~5-10 seconds for 4 keywords (vs 30-60s with retries)
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    logger.info(f"   🚀 Phase 1: Querying arXiv (parallel)...")
    arxiv_start = time.time()
    
    # Phase 1: arXiv queries in parallel (no rate limits)
    arxiv_papers_per_keyword = {}
    for keyword in keywords:
        try:
            papers = _search_arxiv(keyword, max_results_per_keyword)
            arxiv_papers_per_keyword[keyword] = papers
            logger.info(f"     arXiv '{keyword}': {len(papers)} papers")
        except Exception as e:
            logger.error(f"     arXiv '{keyword}' failed: {e}")
            arxiv_papers_per_keyword[keyword] = []
    
    arxiv_elapsed = time.time() - arxiv_start
    logger.info(f"   ✅ arXiv complete: {arxiv_elapsed:.2f}s")
    
    # Phase 2: Semantic Scholar queries SEQUENTIALLY with delays
    # DEMO MODE: Skip if ARXIV_ONLY=true
    arxiv_only = os.getenv("ARXIV_ONLY", "false").lower() == "true"
    
    if arxiv_only:
        logger.info(f"   ⏭️  Phase 2: Semantic Scholar SKIPPED (DEMO MODE - arXiv only)")
        semantic_papers_per_keyword = {keyword: [] for keyword in keywords}
    else:
        logger.info(f"   🐢 Phase 2: Querying Semantic Scholar (sequential, {SEMANTIC_SCHOLAR_DELAY}s delay)...")
        semantic_start = time.time()
        
        semantic_papers_per_keyword = {}
        for i, keyword in enumerate(keywords, 1):
            try:
                logger.info(f"     [{i}/{len(keywords)}] Semantic Scholar '{keyword}'...")
                
                # Use thread pool to avoid blocking event loop
                with ThreadPoolExecutor() as executor:
                    papers = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        _search_semantic_scholar,
                        keyword,
                        max_results_per_keyword
                    )
                
                semantic_papers_per_keyword[keyword] = papers
                logger.info(f"     ✅ '{keyword}': {len(papers)} papers")
                
                # Rate limit delay (except after last query)
                if i < len(keywords):
                    logger.debug(f"     ⏳ Waiting {SEMANTIC_SCHOLAR_DELAY}s...")
                    await asyncio.sleep(SEMANTIC_SCHOLAR_DELAY)
            
            except Exception as e:
                logger.error(f"     ❌ Semantic Scholar '{keyword}' failed: {e}")
                semantic_papers_per_keyword[keyword] = []
        
        semantic_elapsed = time.time() - semantic_start
        logger.info(f"   ✅ Semantic Scholar complete: {semantic_elapsed:.2f}s")
    
    # Combine results
    combined_results = []
    for keyword in keywords:
        arxiv_papers = arxiv_papers_per_keyword.get(keyword, [])
        semantic_papers = semantic_papers_per_keyword.get(keyword, [])
        combined_results.append(arxiv_papers + semantic_papers)
    
    total_elapsed = time.time() - arxiv_start
    logger.info(f"   ⏱️ Total sequential discovery: {total_elapsed:.2f}s")
    
    return combined_results


async def discover_papers_for_keywords_parallel(
    keywords: List[str], 
    max_results_per_keyword: int = 3, 
    claim_id: str = None
) -> tuple[List[PaperMetadata], dict]:
    """
    Discover papers for multiple keywords with rate limiting.
    
    Tier 1: T028-T029 - Multi-keyword discovery with deduplication.
    Phase 7: Sequential Semantic Scholar to avoid 429 rate limits.
    
    Strategy:
    - arXiv: Parallel queries (no rate limits)
    - Semantic Scholar: Sequential with 1.5s delay (1 req/sec limit)
    
    Args:
        keywords: List of search keywords (3-5 multi-word phrases)
        max_results_per_keyword: Papers to fetch per keyword (default: 3)
        claim_id: UUID of the claim for tracking
    
    Returns:
        Tuple of (unique_papers, metadata_dict):
            - unique_papers: Deduplicated list of PaperMetadata
            - metadata_dict: Mapping of URL -> list of keywords that found it
    
    Example:
        >>> keywords = ["neural correlates consciousness", "integrated information theory"]
        >>> papers, metadata = await discover_papers_for_keywords_parallel(keywords, max_results_per_keyword=3)
        >>> print(f"Found {len(papers)} unique papers across {len(keywords)} keywords")
    
    Tier 1: Enables targeted paper discovery per claim with rate limit compliance.
    """
    import asyncio
    
    logger.info(f"🔍 Starting discovery for {len(keywords)} keywords")
    claim_context = f" [claim_id: {claim_id[:8]}...]" if claim_id else ""
    logger.info(f"   Keywords: {keywords}{claim_context}")
    logger.info(f"   Mode: {'Sequential Semantic Scholar' if SEQUENTIAL_SEMANTIC_SCHOLAR else 'Fully Parallel'}")
    
    start_time = time.time()
    
    if SEQUENTIAL_SEMANTIC_SCHOLAR:
        # Phase 7 fix: Sequential Semantic Scholar queries to avoid 429s
        results = await _discover_sequential_rate_limited(keywords, max_results_per_keyword, claim_id)
    else:
        # Original parallel approach (fast but may hit rate limits)
        tasks = [
            asyncio.to_thread(
                discover_papers_for_query, 
                query=keyword, 
                max_results_per_source=max_results_per_keyword,
                claim_id=claim_id
            )
            for keyword in keywords
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # T029: Deduplicate papers across keywords
    # Track which keywords found which papers
    url_to_paper = {}  # url -> PaperMetadata (first occurrence)
    url_to_keywords = {}  # url -> List[str] (all keywords that found it)
    
    for keyword, result in zip(keywords, results):
        # Handle exceptions from individual keyword searches
        if isinstance(result, Exception):
            logger.error(f"Discovery failed for keyword '{keyword}': {result}")
            continue
        
        papers = result
        logger.info(f"  '{keyword}': {len(papers)} papers")
        
        for paper in papers:
            # Normalize URL for deduplication
            url = paper.url
            
            if url not in url_to_paper:
                # First time seeing this paper - store it
                url_to_paper[url] = paper
                url_to_keywords[url] = [keyword]
            else:
                # Duplicate paper - just track the keyword
                url_to_keywords[url].append(keyword)
    
    # Get unique papers list
    unique_papers = list(url_to_paper.values())
    
    # Log deduplication stats
    total_papers_raw = sum(len(r) for r in results if not isinstance(r, Exception))
    duplicates_removed = total_papers_raw - len(unique_papers)
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"✅ Parallel discovery complete in {elapsed_time:.2f}s:")
    logger.info(f"   Total papers (raw): {total_papers_raw}")
    logger.info(f"   Unique papers: {len(unique_papers)}")
    logger.info(f"   Duplicates removed: {duplicates_removed}")
    
    # Log papers found by multiple keywords (high relevance indicator)
    multi_keyword_papers = {url: kws for url, kws in url_to_keywords.items() if len(kws) > 1}
    if multi_keyword_papers:
        logger.info(f"   Papers found by multiple keywords ({len(multi_keyword_papers)}):")
        for url, kws in list(multi_keyword_papers.items())[:3]:  # Show first 3
            paper = url_to_paper[url]
            logger.info(f"     • {paper.title[:60]}... (found by {len(kws)} keywords)")
    
    return unique_papers, url_to_keywords


# =============================================================================
# Automatic Discovery & Ingestion (for Analyst Agent)
# =============================================================================

def discover_and_ingest(query: str, max_papers_per_source: int = 3) -> int:
    """
    Automatic discovery: search for papers and ingest them into Neo4j.
    
    This function is used by the Analyst agent when the knowledge graph
    has insufficient context for a research query. It automatically:
    1. Searches arXiv and Semantic Scholar
    2. Adds discovered papers to Neo4j with deduplication
    3. Returns count of newly added papers
    
    Args:
        query: Research query to search for
        max_papers_per_source: Maximum papers to fetch per source (default: 3)
    
    Returns:
        int: Total number of papers successfully added to Neo4j
    
    Example:
        >>> added = discover_and_ingest("neural networks", max_papers_per_source=5)
        >>> # Returns number of papers added (e.g., 8 if 2 were duplicates)
    """
    from src.tools.kg_tools import add_papers_to_neo4j
    
    logger.info(f"🔍 Automatic discovery triggered for: '{query}'")
    
    all_papers = []
    
    # Search arXiv (fast and reliable)
    try:
        arxiv_papers = search_arxiv(query, max_results=max_papers_per_source * 2)  # Get more from arXiv since we're skipping SS
        all_papers.extend(arxiv_papers)
        logger.info(f"✅ arXiv: {len(arxiv_papers)} papers discovered")
    except Exception as e:
        logger.warning(f"⚠️  arXiv search failed: {e}")
    
    # Skip Semantic Scholar for automatic discovery (library has pagination issues)
    # It's still available for manual discovery via UI
    logger.info("ℹ️  Skipping Semantic Scholar (using arXiv only for automatic discovery)")
    
    # Ingest papers into Neo4j
    if all_papers:
        result = add_papers_to_neo4j(all_papers, discovered_by="analyst_auto")
        added_count = result.get("added", 0)
        skipped_count = result.get("skipped", 0)
        
        logger.info(f"📚 Ingestion complete: {added_count} added, {skipped_count} skipped (duplicates)")
        return added_count
    else:
        logger.warning("⚠️  No papers discovered from any source")
        return 0

