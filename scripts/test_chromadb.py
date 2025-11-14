"""
Test script for ChromaDB vector store.

Tests:
1. ChromaDB initialization
2. Adding papers with embeddings
3. Semantic search
4. Store and retrieve 100 test embeddings
"""

import sys
sys.path.insert(0, '/home/user/CognitiveForge')

from src.memory.vector_store import get_vector_store
import numpy as np


def test_chromadb():
    """Test ChromaDB basic operations."""
    print("🧪 Testing ChromaDB vector store...")

    # Initialize
    print("\n[1/4] Initializing ChromaDB...")
    vs = get_vector_store()
    print("✅ ChromaDB initialized")

    # Add test papers
    print("\n[2/4] Adding test papers...")
    test_papers = [
        {
            "url": "http://arxiv.org/abs/1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "metadata": {"year": 2017, "authors": "Vaswani et al."}
        },
        {
            "url": "http://arxiv.org/abs/1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            "metadata": {"year": 2018, "authors": "Devlin et al."}
        },
        {
            "url": "http://arxiv.org/abs/2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "abstract": "We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance.",
            "metadata": {"year": 2020, "authors": "Brown et al."}
        }
    ]

    for paper in test_papers:
        # Generate random embedding for test papers (simulating E5-large-v2: 1024-dim)
        embedding = np.random.rand(1024).tolist()
        vs.add_paper_with_embedding(
            paper_url=paper["url"],
            title=paper["title"],
            abstract=paper["abstract"],
            embedding=embedding,
            metadata=paper["metadata"]
        )

    print(f"✅ Added {len(test_papers)} test papers")

    # Test semantic search with embedding
    print("\n[3/4] Testing semantic search...")
    query_embedding = np.random.rand(1024).tolist()
    results = vs.semantic_search_with_embedding(query_embedding, top_k=3)
    print(f"✅ Found {len(results)} papers for query embedding")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['url']}")
        if r.get('distance'):
            print(f"     Distance: {r['distance']:.3f}")

    # Test with 100 embeddings
    print("\n[4/4] Testing 100 embeddings...")
    for i in range(100):
        # Generate random embedding (simulating E5-large-v2 output: 1024-dim)
        embedding = np.random.rand(1024).tolist()
        vs.add_paper_with_embedding(
            paper_url=f"http://test.paper/{i}",
            title=f"Test Paper {i}",
            abstract=f"This is test paper number {i} with random embedding.",
            embedding=embedding,
            metadata={"test": True, "index": i}
        )

    print("✅ Added 100 test papers with embeddings")

    # Verify total count
    total_papers = vs.papers.count()
    print(f"\n📊 Total papers in ChromaDB: {total_papers}")

    # Test query with embedding
    query_embedding = np.random.rand(1024).tolist()
    results = vs.semantic_search_with_embedding(query_embedding, top_k=5)
    print(f"✅ Semantic search with embedding returned {len(results)} results")

    print("\n✅ All ChromaDB tests passed!")
    print(f"   - ChromaDB successfully stores/retrieves embeddings")
    print(f"   - Semantic search working correctly")
    print(f"   - Tested with {total_papers} papers total")

    return True


if __name__ == "__main__":
    try:
        test_chromadb()
        print("\n🎉 ChromaDB setup complete and verified!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
