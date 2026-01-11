"""
Generate training data for fine-tuning a domain-specific scripture reranker.

This script:
1. Generates diverse queries about scripture content
2. Retrieves candidate documents from Pinecone
3. Uses GPT-5.2 as judge to label relevance (0-3 scale)
4. Outputs labeled pairs for Sentence Transformers training

Output format compatible with:
- Sentence Transformers CrossEncoder training
- Hugging Face datasets
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "reranker_training"
INDEX_NAME = "standard-works"
EMBEDDING_MODEL = "text-embedding-3-large"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # Claude Sonnet 4.5

MAX_WORKERS = 5  # Lower for Anthropic rate limits
CANDIDATES_PER_QUERY = 50  # Retrieve this many candidates per query

# Query generation prompts
QUERY_GENERATION_PROMPT = """You are generating search queries for a scripture study RAG system covering:
- Book of Mormon
- Doctrine and Covenants
- Pearl of Great Price
- Old Testament
- New Testament

Generate {count} diverse search queries that a student might ask. Include a mix of:

1. FACTUAL queries (who, what, where, when):
   - "Who was Alma the Younger?"
   - "What happened at the waters of Mormon?"
   - "How many sons did Lehi have?"

2. THEMATIC queries (doctrinal topics):
   - "What does the Book of Mormon teach about grace?"
   - "How is the Atonement explained in scripture?"
   - "What are the principles of prayer?"

3. CROSS-REFERENCE queries (spanning volumes):
   - "How do Isaiah's prophecies appear in the Book of Mormon?"
   - "What do different scriptures say about the resurrection?"

4. SPECIFIC VERSE queries:
   - "What does Moroni 10:4-5 say?"
   - "Explain 2 Nephi 2:25"

5. PERSON/PLACE queries:
   - "Tell me about the city of Zarahemla"
   - "What is the story of Abinadi?"

Return as a JSON array of objects with "query" and "type" fields.
Example: [{{"query": "Who was Nephi?", "type": "factual"}}]

Generate exactly {count} queries. Be creative and varied."""

RELEVANCE_JUDGE_PROMPT = """You are an expert evaluator assessing scripture retrieval quality for a RAG system.

Rate the relevance of this scripture passage to the query on a scale of 0-3:

- **0 (Not Relevant)**: The passage has no connection to the query. Different topic, different people, different context.
- **1 (Marginally Relevant)**: The passage is tangentially related. It might mention a related concept or person but does not address the query's core intent.
- **2 (Relevant)**: The passage addresses the query partially. It provides useful context or related information but may not be the most direct answer.
- **3 (Highly Relevant)**: The passage directly answers or addresses the query. It contains the specific information the user is seeking.

Guidelines:
- Focus on semantic relevance, not keyword matching
- Consider the user's likely intent behind the query
- A passage can be highly relevant even if it uses different wording
- Historical context and cross-references count as relevant if they illuminate the query

Respond with JSON only: {{"score": <0-3>, "reasoning": "<brief explanation>"}}"""


@dataclass
class TrainingExample:
    """A single training example for the reranker."""
    query: str
    query_type: str
    document_text: str
    document_reference: str
    document_id: str
    relevance_score: int
    reasoning: str


class RerankerDataGenerator:
    """Generates labeled training data for scripture reranker."""

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone.Index(INDEX_NAME)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_queries(self, count: int) -> list[dict]:
        """Generate diverse queries using Claude Sonnet 4.5."""
        response = self.anthropic.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4096,
            system="You are a helpful assistant that generates search queries.",
            messages=[
                {"role": "user", "content": QUERY_GENERATION_PROMPT.format(count=count)}
            ],
        )

        try:
            # Extract JSON from response
            text = response.content[0].text
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                queries = json.loads(text[start:end])
                return queries
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing queries: {e}")
            return []

        return []

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for query text."""
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def retrieve_candidates(self, query: str, top_k: int = CANDIDATES_PER_QUERY) -> list[dict]:
        """Retrieve candidate documents from Pinecone."""
        embedding = self.get_embedding(query)

        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )

        documents = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            documents.append({
                "id": match["id"],
                "score": match.get("score", 0),
                "text": metadata.get("text", ""),
                "reference": metadata.get("reference", ""),
                "volume": metadata.get("volume", ""),
            })

        return documents

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def judge_relevance(self, query: str, document: dict) -> tuple[int, str]:
        """Use Claude to judge relevance of a document to a query."""
        user_prompt = f"""Query: {query}

Retrieved Passage:
Reference: {document.get('reference', 'Unknown')}
Text: "{document.get('text', '')}"

Rate the relevance (0-3) and provide brief reasoning."""

        response = self.anthropic.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            system=RELEVANCE_JUDGE_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        try:
            text = response.content[0].text
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                score = int(result.get("score", 0))
                reasoning = result.get("reasoning", "")
                return score, reasoning
        except (json.JSONDecodeError, ValueError):
            pass
        return 0, "Failed to parse judgment"

    def judge_batch(self, query: str, documents: list[dict]) -> list[tuple[int, str]]:
        """Judge multiple documents in parallel."""
        results = [None] * len(documents)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.judge_relevance, query, doc): i
                for i, doc in enumerate(documents)
            }

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results

    def generate_examples_for_query(
        self,
        query: str,
        query_type: str,
        num_candidates: int = CANDIDATES_PER_QUERY,
    ) -> list[TrainingExample]:
        """Generate training examples for a single query."""
        # Retrieve candidates
        candidates = self.retrieve_candidates(query, num_candidates)

        if not candidates:
            return []

        # Judge all candidates
        judgments = self.judge_batch(query, candidates)

        # Create training examples
        examples = []
        for doc, (score, reasoning) in zip(candidates, judgments):
            examples.append(TrainingExample(
                query=query,
                query_type=query_type,
                document_text=doc["text"],
                document_reference=doc["reference"],
                document_id=doc["id"],
                relevance_score=score,
                reasoning=reasoning,
            ))

        return examples

    def generate_dataset(
        self,
        num_queries: int = 200,
        candidates_per_query: int = CANDIDATES_PER_QUERY,
        include_existing_queries: bool = True,
    ) -> list[TrainingExample]:
        """Generate complete training dataset."""
        all_examples = []
        queries = []

        # Load existing test queries if requested
        if include_existing_queries:
            test_queries_path = Path(__file__).parent.parent / "evaluations" / "queries" / "test_queries.json"
            if test_queries_path.exists():
                with open(test_queries_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for q in data.get("queries", []):
                        queries.append({
                            "query": q["query"],
                            "type": q["type"],
                        })
                print(f"Loaded {len(queries)} existing test queries")

        # Generate additional queries
        remaining = num_queries - len(queries)
        if remaining > 0:
            print(f"Generating {remaining} additional queries...")
            batch_size = 50  # Generate in batches

            while len(queries) < num_queries:
                batch_count = min(batch_size, num_queries - len(queries))
                new_queries = self.generate_queries(batch_count)
                queries.extend(new_queries)
                print(f"  Generated {len(queries)}/{num_queries} queries")

        # Shuffle queries
        random.shuffle(queries)

        # Process each query
        print(f"\nProcessing {len(queries)} queries with {candidates_per_query} candidates each...")
        print(f"This will generate ~{len(queries) * candidates_per_query:,} training examples")

        for i, q in enumerate(queries):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing query {i + 1}/{len(queries)}: {q['query'][:50]}...")

            examples = self.generate_examples_for_query(
                query=q["query"],
                query_type=q["type"],
                num_candidates=candidates_per_query,
            )
            all_examples.extend(examples)

        return all_examples


def save_dataset(examples: list[TrainingExample], output_dir: Path):
    """Save dataset in multiple formats including Hugging Face Dataset."""
    from datasets import Dataset, DatasetDict

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Full JSON with all fields (for inspection)
    full_path = output_dir / f"training_data_full_{timestamp}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump([asdict(ex) for ex in examples], f, indent=2)
    print(f"Saved full dataset: {full_path}")

    # 2. Hugging Face Dataset format
    hf_data = {
        "query": [ex.query for ex in examples],
        "document": [ex.document_text for ex in examples],
        "reference": [ex.document_reference for ex in examples],
        "label": [ex.relevance_score / 3.0 for ex in examples],  # Normalized 0-1
        "raw_score": [ex.relevance_score for ex in examples],  # Original 0-3
        "query_type": [ex.query_type for ex in examples],
        "reasoning": [ex.reasoning for ex in examples],
    }

    # Create dataset and split into train/test
    full_dataset = Dataset.from_dict(hf_data)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "test": split_dataset["test"],
    })

    # Save to disk
    hf_path = output_dir / f"scripture_reranker_dataset_{timestamp}"
    dataset_dict.save_to_disk(str(hf_path))
    print(f"Saved Hugging Face Dataset: {hf_path}")
    print(f"  Train: {len(dataset_dict['train']):,} examples")
    print(f"  Test: {len(dataset_dict['test']):,} examples")

    # Also save as parquet for easy loading
    parquet_path = output_dir / f"scripture_reranker_{timestamp}.parquet"
    full_dataset.to_parquet(str(parquet_path))
    print(f"Saved Parquet format: {parquet_path}")

    # 3. Sentence Transformers format (labeled pairs) - for compatibility
    st_data = {
        "text1": [ex.query for ex in examples],
        "text2": [ex.document_text for ex in examples],
        "label": [ex.relevance_score / 3.0 for ex in examples],
    }
    st_dataset = Dataset.from_dict(st_data)
    st_path = output_dir / f"st_format_{timestamp}"
    st_dataset.save_to_disk(str(st_path))
    print(f"Saved Sentence Transformers format: {st_path}")

    # 4. CSV format (for easy inspection)
    csv_path = output_dir / f"training_data_{timestamp}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("query\tdocument\treference\tscore\tquery_type\n")
        for ex in examples:
            # Escape tabs and newlines
            query = ex.query.replace("\t", " ").replace("\n", " ")
            doc = ex.document_text.replace("\t", " ").replace("\n", " ")
            ref = ex.document_reference.replace("\t", " ")
            f.write(f"{query}\t{doc}\t{ref}\t{ex.relevance_score}\t{ex.query_type}\n")
    print(f"Saved CSV format: {csv_path}")

    # 5. Statistics
    stats = {
        "total_examples": len(examples),
        "unique_queries": len(set(ex.query for ex in examples)),
        "train_examples": len(dataset_dict["train"]),
        "test_examples": len(dataset_dict["test"]),
        "score_distribution": {
            "0": sum(1 for ex in examples if ex.relevance_score == 0),
            "1": sum(1 for ex in examples if ex.relevance_score == 1),
            "2": sum(1 for ex in examples if ex.relevance_score == 2),
            "3": sum(1 for ex in examples if ex.relevance_score == 3),
        },
        "query_type_distribution": {},
        "timestamp": timestamp,
        "dataset_path": str(hf_path),
    }
    for ex in examples:
        qtype = ex.query_type
        stats["query_type_distribution"][qtype] = stats["query_type_distribution"].get(qtype, 0) + 1

    stats_path = output_dir / f"training_stats_{timestamp}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {stats_path}")

    return stats


def main():
    """Generate reranker training data."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate reranker training data")
    parser.add_argument("--num-queries", type=int, default=200,
                        help="Number of queries to generate (default: 200)")
    parser.add_argument("--candidates", type=int, default=50,
                        help="Candidates per query (default: 50)")
    parser.add_argument("--include-existing", action="store_true", default=True,
                        help="Include existing test queries")
    args = parser.parse_args()

    print("Scripture Reranker Training Data Generator")
    print("=" * 60)
    print(f"Target queries: {args.num_queries}")
    print(f"Candidates per query: {args.candidates}")
    print(f"Expected examples: ~{args.num_queries * args.candidates:,}")
    print()

    generator = RerankerDataGenerator()

    examples = generator.generate_dataset(
        num_queries=args.num_queries,
        candidates_per_query=args.candidates,
        include_existing_queries=args.include_existing,
    )

    print(f"\nGenerated {len(examples):,} training examples")

    stats = save_dataset(examples, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total examples: {stats['total_examples']:,}")
    print(f"Unique queries: {stats['unique_queries']}")
    print(f"\nScore distribution:")
    for score, count in sorted(stats["score_distribution"].items()):
        pct = count / stats["total_examples"] * 100
        print(f"  Score {score}: {count:,} ({pct:.1f}%)")
    print(f"\nQuery type distribution:")
    for qtype, count in sorted(stats["query_type_distribution"].items()):
        print(f"  {qtype}: {count:,}")


if __name__ == "__main__":
    main()
