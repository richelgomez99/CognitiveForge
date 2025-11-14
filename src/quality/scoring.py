"""
Quality Scoring Engine for Research Papers - Phase 2 Epic 3.

Implements multi-factor scoring rubric:
- Citation count (30%)
- Venue impact factor (25%)
- Peer review status (25%)
- Recency score (20%)

Target: Overall score 0-100 for ranking papers by quality.
"""

import logging
from typing import Dict, Optional
from datetime import datetime, date
import math

from src.models_v2 import PaperQualityScoreV2
from .venue_rankings import VenueDatabase

logger = logging.getLogger(__name__)


class PaperQualityScorer:
    """
    Multi-factor quality scoring for research papers.

    Scoring Components:
    1. Citations (30%): Normalized citation count
    2. Venue (25%): Impact factor from venue database
    3. Peer Review (25%): Binary indicator (100 if peer-reviewed, 50 if not)
    4. Recency (20%): Exponential decay from publication year

    Formula:
        overall = 0.30 * citation_score + 0.25 * venue_score +
                  0.25 * peer_score + 0.20 * recency_score
    """

    def __init__(self, venue_db: Optional[VenueDatabase] = None):
        """
        Initialize scorer.

        Args:
            venue_db: Optional venue database (default: VenueDatabase())
        """
        self.venue_db = venue_db or VenueDatabase()
        self.stats = {
            'papers_scored': 0,
            'avg_score': 0.0,
            'high_quality_papers': 0,  # score >= 70
        }

    def score_paper(
        self,
        citation_count: int,
        venue: str,
        is_peer_reviewed: bool,
        publication_year: int,
        **kwargs
    ) -> PaperQualityScoreV2:
        """
        Calculate overall quality score for a paper.

        Args:
            citation_count: Number of citations
            venue: Publication venue (conference/journal name)
            is_peer_reviewed: Whether paper is peer-reviewed
            publication_year: Year of publication
            **kwargs: Optional fields (methodology_score, etc.)

        Returns:
            PaperQualityScoreV2 with all scores calculated
        """
        # Component 1: Citation score (0-100)
        citation_score = self._score_citations(citation_count)

        # Component 2: Venue impact factor (0-100)
        venue_score = self.venue_db.get_impact_factor(venue)

        # Component 3: Peer review score (0-100)
        peer_score = 100.0 if is_peer_reviewed else 50.0

        # Component 4: Recency score (0-100)
        recency_score = self._score_recency(publication_year)

        # Calculate overall score
        overall = PaperQualityScoreV2.calculate_overall(
            citation_count=citation_count,
            venue_if=venue_score,
            peer_reviewed=is_peer_reviewed,
            recency=recency_score
        )

        # Create quality score object
        quality = PaperQualityScoreV2(
            citation_count=citation_count,
            venue_impact_factor=venue_score,
            is_peer_reviewed=is_peer_reviewed,
            recency_score=recency_score,
            methodology_score=kwargs.get('methodology_score'),
            overall_score=overall
        )

        # Update statistics
        self.stats['papers_scored'] += 1
        current_avg = self.stats['avg_score']
        n = self.stats['papers_scored']
        self.stats['avg_score'] = (current_avg * (n - 1) + overall) / n

        if overall >= 70.0:
            self.stats['high_quality_papers'] += 1

        logger.debug(
            f"Scored paper: citations={citation_count}, venue={venue[:30]}, "
            f"peer={is_peer_reviewed}, year={publication_year}, overall={overall:.1f}"
        )

        return quality

    def score_paper_from_metadata(self, metadata: Dict) -> PaperQualityScoreV2:
        """
        Score paper from metadata dictionary.

        Args:
            metadata: Dict with keys:
                - citation_count (int)
                - venue (str)
                - is_peer_reviewed (bool)
                - publication_year (int)
                - methodology_score (optional float)

        Returns:
            PaperQualityScoreV2
        """
        return self.score_paper(
            citation_count=metadata.get('citation_count', 0),
            venue=metadata.get('venue', 'Unknown'),
            is_peer_reviewed=metadata.get('is_peer_reviewed', False),
            publication_year=metadata.get('publication_year', datetime.now().year),
            methodology_score=metadata.get('methodology_score')
        )

    def _score_citations(self, citation_count: int) -> float:
        """
        Normalize citation count to 0-100 scale.

        Uses logarithmic scaling for citation counts:
        - 0 citations -> 0
        - 10 citations -> 50
        - 100 citations -> 75
        - 1000+ citations -> 90-100

        Formula: min(100, (log10(citations + 1) / 3) * 100)
        """
        if citation_count <= 0:
            return 0.0

        # Logarithmic scaling
        score = (math.log10(citation_count + 1) / 3.0) * 100
        return min(100.0, score)

    def _score_recency(self, publication_year: int) -> float:
        """
        Score based on publication recency.

        Uses exponential decay:
        - Current year: 100
        - 1 year old: ~95
        - 3 years old: ~80
        - 5 years old: ~67
        - 10 years old: ~37

        Formula: 100 * exp(-years_old / 10)
        """
        current_year = datetime.now().year
        years_old = max(0, current_year - publication_year)

        # Exponential decay with half-life ~7 years
        score = 100.0 * math.exp(-years_old / 10.0)
        return round(score, 2)

    def batch_score(self, papers: list[Dict]) -> list[PaperQualityScoreV2]:
        """
        Score multiple papers in batch.

        Args:
            papers: List of metadata dicts

        Returns:
            List of PaperQualityScoreV2 objects
        """
        scores = []
        for paper in papers:
            try:
                score = self.score_paper_from_metadata(paper)
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to score paper {paper.get('title', 'unknown')}: {e}")
                # Add default low score
                scores.append(PaperQualityScoreV2(
                    citation_count=0,
                    venue_impact_factor=0.0,
                    is_peer_reviewed=False,
                    recency_score=0.0,
                    overall_score=0.0
                ))

        logger.info(f"Batch scored {len(scores)} papers, avg: {self.stats['avg_score']:.1f}")
        return scores

    def get_stats(self) -> Dict:
        """Get scoring statistics."""
        return {
            **self.stats,
            'high_quality_rate': (
                self.stats['high_quality_papers'] / self.stats['papers_scored']
                if self.stats['papers_scored'] > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'papers_scored': 0,
            'avg_score': 0.0,
            'high_quality_papers': 0,
        }


# Singleton instance
_scorer: Optional[PaperQualityScorer] = None


def get_quality_scorer() -> PaperQualityScorer:
    """Get or create PaperQualityScorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = PaperQualityScorer()
    return _scorer
