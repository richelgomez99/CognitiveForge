"""
Epic 5: Task 2 - Paper Curator Agent

The Paper Curator agent filters and ranks discovered papers based on:
- Relevance to the research query
- Citation count and impact
- Publication quality
- Recency and novelty

This reduces noise and ensures downstream agents work with high-quality evidence.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import re

from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority
from src.models import PaperScore, CurationReport

logger = logging.getLogger(__name__)


class PaperCuratorAgent(BaseAgent):
    """
    Paper Curator Agent - Filters and ranks papers for quality and relevance.

    Responsibilities:
    1. Score papers based on multiple quality metrics
    2. Filter out low-quality or irrelevant papers
    3. Rank papers by combined relevance + quality score
    4. Produce a curated list for downstream agents
    """

    def __init__(self, min_quality_threshold: float = 0.4, top_k: int = 10):
        """
        Initialize Paper Curator agent.

        Args:
            min_quality_threshold: Minimum quality score to keep a paper (0-1)
            top_k: Maximum number of papers to keep after ranking
        """
        super().__init__()
        self.min_quality_threshold = min_quality_threshold
        self.top_k = top_k

    def get_role(self) -> AgentRole:
        """Return agent role."""
        return AgentRole.PAPER_CURATOR

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            role=AgentRole.PAPER_CURATOR,
            priority=AgentPriority.HIGH,
            dependencies=[AgentRole.ANALYST],  # Runs after Analyst discovers papers
            can_run_parallel=False,
            timeout_seconds=60,
            description="Filter and rank papers based on relevance and quality"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute paper curation logic.

        Args:
            state: AgentState containing:
                - original_query: Research question
                - discovered_papers: List of paper dicts from discovery
                - current_thesis: Current thesis (optional, for relevance scoring)

        Returns:
            Dict with:
                - curated_papers: List of paper URLs ranked by quality
                - curation_metadata: CurationReport with statistics
        """
        query = state.get("original_query", "")
        discovered_papers = state.get("discovered_papers", [])

        if not discovered_papers:
            self._logger.warning("No papers to curate - discovery returned empty list")
            return {
                "curated_papers": [],
                "curation_metadata": self._create_empty_report()
            }

        self._logger.info(f"Curating {len(discovered_papers)} discovered papers")

        # Score all papers
        scored_papers = self._score_papers(discovered_papers, query)

        # Filter by quality threshold
        filtered_papers = [
            p for p in scored_papers
            if p.quality_score >= self.min_quality_threshold
        ]

        # Rank by combined score
        ranked_papers = self._rank_papers(filtered_papers)

        # Take top K
        top_papers = ranked_papers[:self.top_k]

        # Create curation report
        report = self._create_report(
            curated_papers=top_papers,
            total_discovered=len(discovered_papers),
            filtered_count=len(discovered_papers) - len(top_papers)
        )

        self._logger.info(
            f"✅ Curation complete: {len(top_papers)}/{len(discovered_papers)} papers selected "
            f"(avg quality: {report.avg_quality:.2f})"
        )

        return {
            "curated_papers": [p.url for p in top_papers],
            "curation_metadata": report
        }

    def _score_papers(self, papers: List[Dict[str, Any]], query: str) -> List[PaperScore]:
        """
        Score papers based on relevance and quality metrics.

        Args:
            papers: List of paper dictionaries with title, url, abstract, etc.
            query: Research query for relevance scoring

        Returns:
            List of PaperScore objects
        """
        scored = []

        for paper in papers:
            # Extract paper fields
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            url = paper.get("url", "")
            citation_count = paper.get("citationCount", 0)
            year = paper.get("year")

            # Calculate relevance score (0-1)
            relevance_score = self._calculate_relevance(title, abstract, query)

            # Calculate quality score (0-1)
            quality_score = self._calculate_quality(
                citation_count=citation_count,
                year=year,
                has_abstract=bool(abstract)
            )

            scored.append(PaperScore(
                url=url,
                title=title,
                relevance_score=relevance_score,
                quality_score=quality_score,
                citation_count=citation_count,
                rank=0  # Will be assigned during ranking
            ))

        return scored

    def _calculate_relevance(self, title: str, abstract: str, query: str) -> float:
        """
        Calculate relevance score based on keyword overlap.

        Args:
            title: Paper title
            abstract: Paper abstract
            query: Research query

        Returns:
            Relevance score (0-1)
        """
        # Extract keywords from query
        query_keywords = set(self._extract_keywords(query.lower()))

        # Extract keywords from title and abstract
        title_keywords = set(self._extract_keywords(title.lower()))
        abstract_keywords = set(self._extract_keywords(abstract.lower()))

        # Calculate overlap (Jaccard similarity)
        title_overlap = len(query_keywords & title_keywords) / max(len(query_keywords), 1)
        abstract_overlap = len(query_keywords & abstract_keywords) / max(len(query_keywords), 1)

        # Weight title higher than abstract
        relevance = (0.7 * title_overlap) + (0.3 * abstract_overlap)

        return min(relevance, 1.0)

    def _calculate_quality(
        self,
        citation_count: int,
        year: int = None,
        has_abstract: bool = True
    ) -> float:
        """
        Calculate quality score based on citations, recency, and completeness.

        Args:
            citation_count: Number of citations
            year: Publication year
            has_abstract: Whether paper has an abstract

        Returns:
            Quality score (0-1)
        """
        score = 0.0

        # Citation score (50% weight)
        # Use log scale to avoid extreme outliers
        if citation_count > 0:
            citation_score = min(citation_count / 100.0, 1.0)  # 100+ citations = max score
        else:
            citation_score = 0.1  # Minimum for uncited papers
        score += 0.5 * citation_score

        # Recency score (30% weight)
        if year:
            current_year = datetime.now().year
            age = current_year - year
            recency_score = max(0.0, 1.0 - (age / 10.0))  # Linear decay over 10 years
            score += 0.3 * recency_score
        else:
            score += 0.15  # Neutral if year unknown

        # Completeness score (20% weight)
        if has_abstract:
            score += 0.2

        return min(score, 1.0)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords (alphanumeric tokens, length >= 3)
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)

        # Filter: length >= 3, not common stopwords
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
            "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
            "that", "this", "with", "from", "have", "what", "when", "your", "about"
        }

        keywords = [
            w for w in words
            if len(w) >= 3 and w not in stopwords
        ]

        return keywords

    def _rank_papers(self, papers: List[PaperScore]) -> List[PaperScore]:
        """
        Rank papers by combined relevance + quality score.

        Args:
            papers: List of scored papers

        Returns:
            List of papers sorted by rank (descending)
        """
        # Calculate combined score
        for paper in papers:
            # Weight relevance slightly higher than quality
            paper.combined_score = (0.6 * paper.relevance_score) + (0.4 * paper.quality_score)

        # Sort by combined score (descending)
        ranked = sorted(papers, key=lambda p: p.combined_score, reverse=True)

        # Assign ranks
        for i, paper in enumerate(ranked, start=1):
            paper.rank = i

        return ranked

    def _create_report(
        self,
        curated_papers: List[PaperScore],
        total_discovered: int,
        filtered_count: int
    ) -> CurationReport:
        """
        Create curation report with statistics.

        Args:
            curated_papers: Final curated papers
            total_discovered: Total papers discovered
            filtered_count: Number of papers filtered out

        Returns:
            CurationReport object
        """
        if curated_papers:
            avg_quality = sum(p.quality_score for p in curated_papers) / len(curated_papers)
        else:
            avg_quality = 0.0

        return CurationReport(
            curated_papers=curated_papers,
            total_discovered=total_discovered,
            filtered_count=filtered_count,
            avg_quality=avg_quality
        )

    def _create_empty_report(self) -> CurationReport:
        """Create empty report when no papers available."""
        return CurationReport(
            curated_papers=[],
            total_discovered=0,
            filtered_count=0,
            avg_quality=0.0
        )
