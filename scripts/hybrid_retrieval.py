"""
Hybrid Retrieval Module for Scripture Search.

Combines sparse (keyword) and dense (semantic) embeddings using
Reciprocal Rank Fusion (RRF) with optional neural reranking.

Approaches:
1. Sparse-only: Uses Pinecone's integrated sparse embeddings (pinecone-sparse-english-v0)
2. Dense-only: Uses OpenAI text-embedding-3-large
3. Hybrid RRF: Combines sparse + dense with Reciprocal Rank Fusion
4. Hybrid + Rerank: RRF fusion followed by neural reranking
"""

import os
import time
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_random_exponential


load_dotenv()

# Index configuration
SPARSE_INDEX = "standard-works-v2"  # Integrated sparse embeddings
DENSE_INDEX = "standard-works"  # OpenAI dense embeddings
SPARSE_NAMESPACE = "scriptures"
EMBEDDING_MODEL = "text-embedding-3-large"

# Reranking models available via Pinecone
RERANK_MODELS = {
    "pinecone": "pinecone-rerank-v0",
    "cohere": "cohere-rerank-3.5",
    "bge": "bge-reranker-v2-m3",
}


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""
    documents: list[dict]
    latency_ms: float
    method: str
    query: str
    sparse_count: int = 0
    dense_count: int = 0
    fusion_count: int = 0


class HybridRetrieval:
    """Hybrid retrieval combining sparse and dense embeddings."""

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.sparse_index = self.pinecone.Index(SPARSE_INDEX)
        self.dense_index = self.pinecone.Index(DENSE_INDEX)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _get_dense_embedding(self, text: str) -> list[float]:
        """Generate dense embedding using OpenAI."""
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def _search_sparse(self, query: str, top_k: int = 50) -> list[dict]:
        """Search using Pinecone's integrated sparse embeddings."""
        results = self.sparse_index.search_records(
            namespace=SPARSE_NAMESPACE,
            query={"inputs": {"text": query}, "top_k": top_k},
        )

        documents = []
        for rank, hit in enumerate(results["result"]["hits"], start=1):
            fields = hit.get("fields", {})
            documents.append({
                "id": hit["_id"],
                "rank": rank,
                "score": hit["_score"],
                "text": fields.get("original_text", ""),
                "reference": fields.get("reference", ""),
                "volume": fields.get("volume", ""),
                "book": fields.get("book", ""),
                "chapter": fields.get("chapter", 0),
                "verse": fields.get("verse", 0),
                "context": fields.get("text", ""),  # Contextualized text
                "source": "sparse",
            })

        return documents

    def _search_dense(self, query: str, top_k: int = 50) -> list[dict]:
        """Search using OpenAI dense embeddings."""
        query_vector = self._get_dense_embedding(query)

        results = self.dense_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        documents = []
        for rank, match in enumerate(results["matches"], start=1):
            metadata = match.get("metadata", {})
            documents.append({
                "id": match["id"],
                "rank": rank,
                "score": match.get("score", 0),
                "text": metadata.get("text", ""),
                "reference": metadata.get("reference", ""),
                "volume": metadata.get("volume", ""),
                "context": metadata.get("context", ""),
                "source": "dense",
            })

        return documents

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict]],
        k: int = 60,
        weights: list[float] | None = None,
    ) -> list[dict]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion (RRF).

        RRF Score = sum(weight / (k + rank)) for each result list

        Args:
            result_lists: List of result lists from different retrievers
            k: Constant to prevent high scores for top results (default 60)
            weights: Optional weights for each result list

        Returns:
            Combined and re-ranked results sorted by RRF score
        """
        if weights is None:
            weights = [1.0] * len(result_lists)

        # Build fusion scores keyed by reference
        fusion_scores: dict[str, dict] = {}

        for weight, results in zip(weights, result_lists):
            for result in results:
                ref = result["reference"]
                rank = result["rank"]
                rrf_score = weight / (k + rank)

                if ref not in fusion_scores:
                    fusion_scores[ref] = {
                        "id": result["id"],
                        "reference": ref,
                        "text": result["text"],
                        "volume": result.get("volume", ""),
                        "context": result.get("context", ""),
                        "rrf_score": 0,
                        "sources": [],
                        "source_ranks": {},
                        "source_scores": {},
                    }

                fusion_scores[ref]["rrf_score"] += rrf_score
                fusion_scores[ref]["sources"].append(result["source"])
                fusion_scores[ref]["source_ranks"][result["source"]] = rank
                fusion_scores[ref]["source_scores"][result["source"]] = result["score"]

        # Sort by RRF score and assign new ranks
        combined = sorted(
            fusion_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        for rank, doc in enumerate(combined, start=1):
            doc["rank"] = rank
            doc["score"] = doc["rrf_score"]
            doc["in_both"] = len(set(doc["sources"])) > 1

        return combined

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _rerank_with_pinecone(
        self,
        query: str,
        documents: list[dict],
        top_n: int,
        model: str = "pinecone",
    ) -> list[dict]:
        """Rerank documents using Pinecone's reranking API."""
        if not documents:
            return []

        model_name = RERANK_MODELS.get(model, model)
        texts = [doc["text"] for doc in documents]

        response = self.pinecone.inference.rerank(
            model=model_name,
            query=query,
            documents=texts,
            top_n=min(top_n, len(documents)),
            return_documents=True,
        )

        reranked = []
        for rank, result in enumerate(response.data, start=1):
            original_doc = documents[result.index].copy()
            original_doc["original_rank"] = original_doc["rank"]
            original_doc["original_score"] = original_doc["score"]
            original_doc["rank"] = rank
            original_doc["score"] = result.score
            reranked.append(original_doc)

        return reranked

    def retrieve_sparse(self, query: str, top_k: int = 20) -> HybridResult:
        """Retrieve using sparse embeddings only."""
        start = time.perf_counter()

        documents = self._search_sparse(query, top_k)

        latency_ms = (time.perf_counter() - start) * 1000

        return HybridResult(
            documents=documents,
            latency_ms=latency_ms,
            method="sparse",
            query=query,
            sparse_count=len(documents),
        )

    def retrieve_dense(self, query: str, top_k: int = 20) -> HybridResult:
        """Retrieve using dense embeddings only."""
        start = time.perf_counter()

        documents = self._search_dense(query, top_k)

        latency_ms = (time.perf_counter() - start) * 1000

        return HybridResult(
            documents=documents,
            latency_ms=latency_ms,
            method="dense",
            query=query,
            dense_count=len(documents),
        )

    def retrieve_hybrid(
        self,
        query: str,
        sparse_k: int = 50,
        dense_k: int = 50,
        final_k: int = 20,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.2,
        rrf_k: int = 60,
    ) -> HybridResult:
        """
        Retrieve using hybrid sparse + dense with RRF fusion.

        Args:
            query: Search query
            sparse_k: Number of sparse results to retrieve
            dense_k: Number of dense results to retrieve
            final_k: Number of final results to return
            sparse_weight: Weight for sparse results in RRF
            dense_weight: Weight for dense results in RRF
            rrf_k: RRF constant (higher = more uniform ranking)

        Returns:
            HybridResult with fused documents
        """
        start = time.perf_counter()

        # Get results from both retrievers
        sparse_results = self._search_sparse(query, sparse_k)
        dense_results = self._search_dense(query, dense_k)

        # Combine with RRF
        fused = self._reciprocal_rank_fusion(
            [sparse_results, dense_results],
            k=rrf_k,
            weights=[sparse_weight, dense_weight],
        )

        documents = fused[:final_k]

        latency_ms = (time.perf_counter() - start) * 1000

        return HybridResult(
            documents=documents,
            latency_ms=latency_ms,
            method="hybrid_rrf",
            query=query,
            sparse_count=len(sparse_results),
            dense_count=len(dense_results),
            fusion_count=len(fused),
        )

    def retrieve_hybrid_rerank(
        self,
        query: str,
        sparse_k: int = 75,
        dense_k: int = 75,
        fusion_k: int = 50,
        top_n: int = 20,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.2,
        rerank_model: str = "pinecone",
    ) -> HybridResult:
        """
        Retrieve using hybrid RRF fusion followed by neural reranking.

        Args:
            query: Search query
            sparse_k: Number of sparse results to retrieve
            dense_k: Number of dense results to retrieve
            fusion_k: Number of fused results to pass to reranker
            top_n: Final number of results after reranking
            sparse_weight: Weight for sparse results in RRF
            dense_weight: Weight for dense results in RRF
            rerank_model: Reranking model ("pinecone", "cohere", or "bge")

        Returns:
            HybridResult with reranked documents
        """
        start = time.perf_counter()

        # Get results from both retrievers
        sparse_results = self._search_sparse(query, sparse_k)
        dense_results = self._search_dense(query, dense_k)

        # Combine with RRF
        fused = self._reciprocal_rank_fusion(
            [sparse_results, dense_results],
            weights=[sparse_weight, dense_weight],
        )

        # Rerank top fusion results
        candidates = fused[:fusion_k]
        reranked = self._rerank_with_pinecone(query, candidates, top_n, rerank_model)

        latency_ms = (time.perf_counter() - start) * 1000

        return HybridResult(
            documents=reranked,
            latency_ms=latency_ms,
            method=f"hybrid_rerank_{rerank_model}",
            query=query,
            sparse_count=len(sparse_results),
            dense_count=len(dense_results),
            fusion_count=len(fused),
        )

    def retrieve_all(
        self,
        query: str,
        top_k: int = 20,
        include_rerank: bool = True,
    ) -> dict[str, HybridResult]:
        """
        Run all hybrid retrieval methods on a query.

        Args:
            query: Search query
            top_k: Number of results for non-rerank methods
            include_rerank: Whether to include reranked results

        Returns:
            Dictionary of method name to HybridResult
        """
        results = {
            "sparse": self.retrieve_sparse(query, top_k),
            "dense": self.retrieve_dense(query, top_k),
            "hybrid_rrf": self.retrieve_hybrid(query, final_k=top_k),
        }

        if include_rerank:
            results["hybrid_rerank"] = self.retrieve_hybrid_rerank(
                query, top_n=top_k
            )

        return results


def main():
    """Test the hybrid retrieval module."""
    retriever = HybridRetrieval()

    test_queries = [
        "faith without works is dead",
        "baptism by immersion",
        "plan of salvation",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        results = retriever.retrieve_all(query, top_k=5)

        for method, result in results.items():
            print(f"\n{method.upper()} ({result.latency_ms:.0f}ms)")

            for doc in result.documents[:3]:
                score_str = f"{doc['score']:.4f}" if isinstance(doc['score'], float) else str(doc['score'])
                in_both = " [BOTH]" if doc.get("in_both") else ""
                print(f"  {doc['rank']}. {doc['reference']} ({score_str}){in_both}")


if __name__ == "__main__":
    main()
