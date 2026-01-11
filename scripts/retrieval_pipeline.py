"""
Unified retrieval pipeline for RAG comparison evaluation.

Provides a single interface to query all three retrieval approaches:
1. Legacy RAG (raw verse embeddings)
2. Contextual RAG (verse + chapter summary embeddings)
3. Contextual RAG + Cohere Reranking
"""

import os
import time
from dataclasses import dataclass
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
EMBEDDING_MODEL = "text-embedding-3-large"
RESPONSE_MODEL = "gpt-5.2"


@dataclass
class RetrievalResult:
    """Standardized retrieval result."""
    documents: list[dict]
    latency_ms: float
    method: str
    query: str


class RetrievalPipeline:
    """Unified interface for all retrieval methods."""

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.cohere = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

        self.legacy_index = self.pinecone.Index(LEGACY_INDEX)
        self.contextual_index = self.pinecone.Index(CONTEXTUAL_INDEX)

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

    def retrieve_all(
        self,
        query: str,
        top_k: int = 20,
        rerank_initial_k: int = 150,
        rerank_top_n: int = 20,
    ) -> dict[str, RetrievalResult]:
        """Run all three retrieval methods on a query.

        Args:
            query: Search query
            top_k: Number of results for legacy/contextual methods
            rerank_initial_k: Number of candidates to retrieve before reranking
            rerank_top_n: Number of results to keep after reranking
        """
        return {
            "legacy": self.retrieve_legacy(query, top_k),
            "contextual": self.retrieve_contextual(query, top_k),
            "contextual_rerank": self.retrieve_contextual_rerank(
                query, rerank_initial_k, rerank_top_n
            ),
        }

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
