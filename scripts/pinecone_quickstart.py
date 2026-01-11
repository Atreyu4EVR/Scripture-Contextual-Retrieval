"""
Pinecone Quickstart - Quick Test
This script demonstrates:
1. Connecting to a Pinecone index with integrated embeddings
2. Upserting sample data
3. Performing semantic search with reranking
"""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")

pc = Pinecone(api_key=api_key)

# Sample data from different domains
records = [
    {"_id": "rec1", "content": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history"},
    {"_id": "rec2", "content": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science"},
    {"_id": "rec5", "content": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature"},
    {"_id": "rec7", "content": "The Great Wall of China was built to protect against invasions.", "category": "history"},
    {"_id": "rec15", "content": "Leonardo da Vinci painted the Mona Lisa.", "category": "art"},
    {"_id": "rec17", "content": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history"},
    {"_id": "rec21", "content": "The Statue of Liberty was a gift from France to the United States.", "category": "history"},
    {"_id": "rec26", "content": "Rome was once the center of a vast empire.", "category": "history"},
    {"_id": "rec33", "content": "The violin is a string instrument commonly used in orchestras.", "category": "music"},
    {"_id": "rec38", "content": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history"},
    {"_id": "rec48", "content": "Vincent van Gogh painted Starry Night.", "category": "art"},
    {"_id": "rec50", "content": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy"},
]

# Target the index
index_name = "agentic-quickstart-test"
dense_index = pc.Index(index_name)

# Upsert the records into a namespace
namespace = "example-namespace"
print(f"Upserting {len(records)} records into namespace '{namespace}'...")
dense_index.upsert_records(namespace, records)
print("Upsert complete!")

# Wait for the upserted vectors to be indexed
print("Waiting 10 seconds for vectors to be indexed...")
time.sleep(10)

# View stats for the index
stats = dense_index.describe_index_stats()
print(f"\nIndex Stats:")
print(f"  Total vector count: {stats.total_vector_count}")
print(f"  Namespaces: {list(stats.namespaces.keys())}")

# Define the query
query = "Famous historical structures and monuments"
print(f"\nSearching for: '{query}'")

# Search the dense index and rerank results
reranked_results = dense_index.search(
    namespace=namespace,
    query={
        "top_k": 10,
        "inputs": {
            "text": query
        }
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 10,
        "rank_fields": ["content"]
    }
)

# Print the reranked results
print("\nReranked Search Results:")
print("-" * 80)
for hit in reranked_results["result"]["hits"]:
    print(f"ID: {hit['_id']}")
    print(f"  Score: {round(hit['_score'], 4)}")
    print(f"  Content: {hit['fields']['content']}")
    print(f"  Category: {hit['fields']['category']}")
    print()

print("Quickstart complete!")
