"""
Unified retrieval pipeline for RAG comparison evaluation.

Provides a single interface to query all retrieval approaches:
1. Legacy RAG (raw verse embeddings)
2. Contextual RAG (verse + chapter summary embeddings)
3. Contextual RAG + Cohere Reranking
4. Contextual RAG + Custom Scripture Reranking
5. Hybrid RAG (Sparse + Dense with RRF Fusion)
6. Hybrid RAG + Reranking
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cohere
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_random_exponential

from prompts.retrieval_prompt import system_prompt as RAG_SYSTEM_PROMPT


load_dotenv()

LEGACY_INDEX = "standard-work-nocontext"
CONTEXTUAL_INDEX = "standard-works"
SPARSE_INDEX = "standard-works-v2"
SPARSE_NAMESPACE = "scriptures"
EMBEDDING_MODEL = "text-embedding-3-large"
RESPONSE_MODEL = "gpt-5.2"
CUSTOM_RERANKER_PATH = Path(__file__).parent.parent / "models" / "scripture-reranker" / "final"


@dataclass
class RetrievalResult:
    """Standardized retrieval result."""
    documents: list[dict]
    latency_ms: float
    method: str
    query: str


class RetrievalPipeline:
    """Unified interface for all retrieval methods."""

    def __init__(self, load_custom_reranker: bool = True):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.cohere = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

        self.legacy_index = self.pinecone.Index(LEGACY_INDEX)
        self.contextual_index = self.pinecone.Index(CONTEXTUAL_INDEX)
        self.sparse_index = self.pinecone.Index(SPARSE_INDEX)

        # Load custom scripture reranker if available
        self.custom_reranker = None
        if load_custom_reranker and CUSTOM_RERANKER_PATH.exists():
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                self.custom_reranker = CrossEncoder(str(CUSTOM_RERANKER_PATH))
                print(f"Loaded custom reranker from: {CUSTOM_RERANKER_PATH}")
            except Exception as e:
                print(f"Warning: Could not load custom reranker: {e}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for query text."""
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def _query_index(
        self,
        index,
        query_embedding: list[float],
        top_k: int,
        include_context: bool = True,
    ) -> list[dict]:
        """Query Pinecone and return documents."""
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        documents = []
        for rank, match in enumerate(results["matches"], start=1):
            metadata = match.get("metadata", {})
            doc = {
                "id": match["id"],
                "rank": rank,
                "score": match.get("score", 0),
                "text": metadata.get("text", ""),
                "reference": metadata.get("reference", ""),
                "volume": metadata.get("volume", ""),
            }
            if include_context:
                doc["context"] = metadata.get("context", "")
            documents.append(doc)

        return documents

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _rerank_documents(
        self,
        query: str,
        documents: list[dict],
        top_n: int,
    ) -> list[dict]:
        """Rerank documents using Cohere."""
        if not documents:
            return []

        texts = [doc["text"] for doc in documents]

        response = self.cohere.rerank(
            model="rerank-v3.5",
            query=query,
            documents=texts,
            top_n=top_n,
        )

        reranked = []
        for rank, result in enumerate(response.results, start=1):
            doc = documents[result.index].copy()
            doc["original_rank"] = doc["rank"]
            doc["rank"] = rank
            doc["original_score"] = doc["score"]
            doc["score"] = result.relevance_score
            reranked.append(doc)

        return reranked

    def _custom_rerank_documents(
        self,
        query: str,
        documents: list[dict],
        top_n: int,
    ) -> list[dict]:
        """Rerank documents using custom scripture reranker."""
        if not documents or self.custom_reranker is None:
            return documents[:top_n] if documents else []

        # Create query-document pairs for scoring
        pairs = [(query, doc["text"]) for doc in documents]

        # Get scores from custom reranker
        scores = self.custom_reranker.predict(pairs)

        # Sort by score (descending) and take top_n
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rank, (doc, score) in enumerate(scored_docs[:top_n], start=1):
            doc_copy = doc.copy()
            doc_copy["original_rank"] = doc["rank"]
            doc_copy["rank"] = rank
            doc_copy["original_score"] = doc["score"]
            doc_copy["score"] = float(score)
            reranked.append(doc_copy)

        return reranked

    def retrieve_legacy(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Retrieve using legacy RAG (raw verse embeddings)."""
        start = time.perf_counter()

        embedding = self._get_embedding(query)
        documents = self._query_index(
            self.legacy_index,
            embedding,
            top_k,
            include_context=False,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            documents=documents,
            latency_ms=latency_ms,
            method="legacy",
            query=query,
        )

    def retrieve_contextual(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Retrieve using contextual RAG (verse + chapter summary embeddings)."""
        start = time.perf_counter()

        embedding = self._get_embedding(query)
        documents = self._query_index(
            self.contextual_index,
            embedding,
            top_k,
            include_context=True,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            documents=documents,
            latency_ms=latency_ms,
            method="contextual",
            query=query,
        )

    def retrieve_contextual_rerank(
        self,
        query: str,
        initial_k: int = 150,
        top_n: int = 20,
    ) -> RetrievalResult:
        """Retrieve using contextual RAG with Cohere reranking.

        Args:
            query: Search query
            initial_k: Number of candidates to retrieve before reranking (default: 150)
            top_n: Number of results to keep after reranking (default: 20)
        """
        start = time.perf_counter()

        embedding = self._get_embedding(query)
        documents = self._query_index(
            self.contextual_index,
            embedding,
            initial_k,
            include_context=True,
        )

        reranked = self._rerank_documents(query, documents, top_n)

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            documents=reranked,
            latency_ms=latency_ms,
            method="contextual_rerank",
            query=query,
        )

    def retrieve_contextual_custom_rerank(
        self,
        query: str,
        initial_k: int = 150,
        top_n: int = 20,
    ) -> RetrievalResult:
        """Retrieve using contextual RAG with custom scripture reranker.

        Args:
            query: Search query
            initial_k: Number of candidates to retrieve before reranking (default: 150)
            top_n: Number of results to keep after reranking (default: 20)
        """
        if self.custom_reranker is None:
            raise ValueError("Custom reranker not loaded. Check model path.")

        start = time.perf_counter()

        embedding = self._get_embedding(query)
        documents = self._query_index(
            self.contextual_index,
            embedding,
            initial_k,
            include_context=True,
        )

        reranked = self._custom_rerank_documents(query, documents, top_n)

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            documents=reranked,
            latency_ms=latency_ms,
            method="contextual_custom_rerank",
            query=query,
        )

    def _query_sparse_index(self, query: str, top_k: int) -> list[dict]:
        """Query sparse index using integrated embeddings."""
        results = self.sparse_index.search_records(
            namespace=SPARSE_NAMESPACE,
            query={"inputs": {"text": query}, "top_k": top_k},
        )

        documents = []
        for rank, hit in enumerate(results["result"]["hits"], start=1):
            fields = hit.get("fields", {})
            doc = {
                "id": hit["_id"],
                "rank": rank,
                "score": hit["_score"],
                "text": fields.get("original_text", ""),
                "reference": fields.get("reference", ""),
                "volume": fields.get("volume", ""),
                "context": fields.get("text", ""),
                "source": "sparse",
            }
            documents.append(doc)

        return documents

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict]],
        k: int = 60,
        weights: list[float] | None = None,
    ) -> list[dict]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        if weights is None:
            weights = [1.0] * len(result_lists)

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
                    }

                fusion_scores[ref]["rrf_score"] += rrf_score
                fusion_scores[ref]["sources"].append(result.get("source", "unknown"))
                fusion_scores[ref]["source_ranks"][result.get("source", "unknown")] = rank

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
        model: str = "pinecone-rerank-v0",
    ) -> list[dict]:
        """Rerank documents using Pinecone's reranking API."""
        if not documents:
            return []

        texts = [doc["text"] for doc in documents]

        response = self.pinecone.inference.rerank(
            model=model,
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

    def retrieve_hybrid(
        self,
        query: str,
        sparse_k: int = 50,
        dense_k: int = 50,
        final_k: int = 20,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.2,
    ) -> RetrievalResult:
        """Retrieve using hybrid sparse + dense with RRF fusion.

        Args:
            query: Search query
            sparse_k: Number of sparse results to retrieve
            dense_k: Number of dense results to retrieve
            final_k: Number of final results to return
            sparse_weight: Weight for sparse results in RRF
            dense_weight: Weight for dense results in RRF
        """
        start = time.perf_counter()

        # Get sparse results (keyword-based)
        sparse_results = self._query_sparse_index(query, sparse_k)

        # Get dense results (semantic)
        embedding = self._get_embedding(query)
        dense_results = self._query_index(
            self.contextual_index,
            embedding,
            dense_k,
            include_context=True,
        )
        for doc in dense_results:
            doc["source"] = "dense"

        # Combine with RRF
        fused = self._reciprocal_rank_fusion(
            [sparse_results, dense_results],
            weights=[sparse_weight, dense_weight],
        )

        documents = fused[:final_k]

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            documents=documents,
            latency_ms=latency_ms,
            method="hybrid_rrf",
            query=query,
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
    ) -> RetrievalResult:
        """Retrieve using hybrid RRF fusion followed by Pinecone reranking.

        Args:
            query: Search query
            sparse_k: Number of sparse results to retrieve
            dense_k: Number of dense results to retrieve
            fusion_k: Number of fused results to pass to reranker
            top_n: Final number of results after reranking
            sparse_weight: Weight for sparse results in RRF
            dense_weight: Weight for dense results in RRF
        """
        start = time.perf_counter()

        # Get sparse results
        sparse_results = self._query_sparse_index(query, sparse_k)

        # Get dense results
        embedding = self._get_embedding(query)
        dense_results = self._query_index(
            self.contextual_index,
            embedding,
            dense_k,
            include_context=True,
        )
        for doc in dense_results:
            doc["source"] = "dense"

        # Combine with RRF
        fused = self._reciprocal_rank_fusion(
            [sparse_results, dense_results],
            weights=[sparse_weight, dense_weight],
        )

        # Rerank top fusion results
        candidates = fused[:fusion_k]
        reranked = self._rerank_with_pinecone(query, candidates, top_n)

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            documents=reranked,
            latency_ms=latency_ms,
            method="hybrid_rerank",
            query=query,
        )

    def retrieve_all(
        self,
        query: str,
        top_k: int = 20,
        rerank_initial_k: int = 150,
        rerank_top_n: int = 20,
        include_custom_rerank: bool = True,
        include_hybrid: bool = True,
    ) -> dict[str, RetrievalResult]:
        """Run all retrieval methods on a query.

        Args:
            query: Search query
            top_k: Number of results for legacy/contextual methods
            rerank_initial_k: Number of candidates to retrieve before reranking
            rerank_top_n: Number of results to keep after reranking
            include_custom_rerank: Include custom scripture reranker if available
            include_hybrid: Include hybrid sparse+dense methods
        """
        results = {
            "legacy": self.retrieve_legacy(query, top_k),
            "contextual": self.retrieve_contextual(query, top_k),
            "contextual_rerank": self.retrieve_contextual_rerank(
                query, rerank_initial_k, rerank_top_n
            ),
        }

        # Add custom reranker if available
        if include_custom_rerank and self.custom_reranker is not None:
            results["contextual_custom_rerank"] = self.retrieve_contextual_custom_rerank(
                query, rerank_initial_k, rerank_top_n
            )

        # Add hybrid methods
        if include_hybrid:
            results["hybrid_rrf"] = self.retrieve_hybrid(
                query, final_k=top_k
            )
            results["hybrid_rerank"] = self.retrieve_hybrid_rerank(
                query, top_n=rerank_top_n
            )

        return results

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_response(
        self,
        query: str,
        documents: list[dict],
    ) -> str:
        """Generate RAG response using retrieved documents."""
        docs_text = "\n\n".join(
            f"[{doc['reference']}]: {doc['text']}"
            for doc in documents
        )

        input_text = f"""User Query: {query}

Retrieved Scripture Passages:
{docs_text}

Generate a response following the output contract."""

        response = self.openai.responses.create(
            model=RESPONSE_MODEL,
            reasoning={"effort": "low"},
            instructions=RAG_SYSTEM_PROMPT,
            input=input_text,
        )

        return response.output_text


def main():
    """Test the retrieval pipeline."""
    pipeline = RetrievalPipeline()

    test_query = "Who was Nephi?"

    print("Testing Retrieval Pipeline")
    print("=" * 50)
    print(f"Query: {test_query}\n")

    results = pipeline.retrieve_all(test_query)

    for method, result in results.items():
        print(f"\n{method.upper()}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print(f"  Results: {len(result.documents)}")
        if result.documents:
            top_doc = result.documents[0]
            print(f"  Top result: {top_doc['reference']} (score: {top_doc['score']:.4f})")


if __name__ == "__main__":
    main()
