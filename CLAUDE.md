# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository implements **Contextual Retrieval** for the LDS Standard Works, applying Anthropic's methodology to improve scripture search quality while maintaining intellectual property compliance. The project generates AI-powered chapter summaries that enrich verse embeddings, enabling semantic search that bridges modern queries to archaic scriptural language.

If needed, reference the code documentation by Anthropic at `https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide`

The key contribution is a clean-room approach: instead of using copyrighted Church study aids (chapter headings, cross-references, Topical Guide), we generate new retrieval-optimized context using GPT-5.2.

## Repository Structure

```
Scripture-Contextual-Retrieval/
├── scriptures/
│   ├── source/                     # Original scripture JSON files
│   ├── contextualized/             # Verses with AI-generated context
│   │   └── summaries/              # Chapter summaries
│   └── statistics/                 # Token count statistics
├── scripts/
│   ├── generate_chapter_summaries.py   # Step 1: Generate chapter context
│   ├── generate_contextualized_verses.py # Step 2: Combine context + verses
│   ├── generate_embeddings.py          # Step 3: Create embeddings
│   ├── upsert_to_pinecone.py           # Step 4: Upload to vector DB
│   ├── run_evaluation.py               # Step 5: Full evaluation pipeline
│   ├── retrieval_pipeline.py           # Core retrieval logic
│   ├── hybrid_retrieval.py             # Hybrid search implementation
│   ├── llm_judge.py                    # LLM-as-judge relevance scoring
│   ├── metrics_calculator.py           # P@K, NDCG, MRR calculations
│   └── tokenizer.py                    # Token counting utility
├── evaluations/
│   ├── queries/test_queries.json       # 30 curated test queries
│   ├── config/evaluation_config.json   # Evaluation parameters
│   ├── results/                        # Raw retrieval results by run
│   └── reports/                        # Comparison reports (JSON + MD)
├── prompts/
│   └── chapter_summary_prompt.md       # GPT-5.2 context generation prompt
├── models/                             # Custom reranker checkpoints
└── reranker_training/                  # Training data for custom reranker
```

## Scripture Data Structure

Scripture files in `scriptures/` use nested JSON (books → chapters → verses):

```json
{
  "books": [
    {
      "book": "1 Nephi",
      "chapters": [
        {
          "chapter": 1,
          "reference": "1 Nephi 1",
          "verses": [
            {
              "verse": 1,
              "reference": "1 Nephi 1:1",
              "text": "I, Nephi, having been born of goodly parents..."
            }
          ]
        }
      ]
    }
  ]
}
```

**Special case:** Doctrine and Covenants uses "sections" instead of "chapters" but follows the same structure. The code handles this with conditional `unit_type` logic.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Context Generation | OpenAI GPT-5.2 (reasoning: low) |
| Embeddings | OpenAI text-embedding-3-large (3072 dims) |
| Vector Database | Pinecone |
| Reranking | Cohere (general), custom scripture reranker |
| Tokenizer | tiktoken (o200k_base encoding) |
| Evaluation | LLM-as-judge with GPT-5.2 |

## Key Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add API keys

# Full pipeline (recommended)
python scripts/run_pipeline.py                  # Runs all 4 steps
python scripts/run_pipeline.py --dry-run        # Preview without executing
python scripts/run_pipeline.py --start-from 3   # Resume from step N

# Evaluation
python scripts/run_evaluation.py                # Runs all 6 retrieval methods

# Utilities
python scripts/tokenizer.py                     # Count tokens per volume
python scripts/query_pinecone.py                # Test individual queries
```

## Environment Variables

Required in `.env` (see `.env.example`):

```
OPENAI_API_KEY=sk-...        # Required: embeddings, context generation, evaluation
PINECONE_API_KEY=pcsk_...    # Required: vector database operations
COHERE_API_KEY=...           # Optional: reranking in retrieval pipeline
ANTHROPIC_API_KEY=sk-ant-... # Optional: reranker training data generation
```

## Pinecone Index Schema

Index name: `standard-works` (and `standard-works-nocontext` for baseline)

Metadata fields per vector:

- `volume`: e.g., "Book of Mormon"
- `reference`: e.g., "1 Nephi 1:1"
- `context`: AI-generated chapter summary
- `text`: Original verse text (required for LangChain compatibility)

**Important:** LangChain's `PineconeVectorStore` expects `text_key='text'`. Earlier versions used `pageContent` which caused empty retrieval results.

## Evaluation Framework

The evaluation compares 6 retrieval methods:

1. `legacy` - No context (baseline)
2. `contextual` - With AI-generated context
3. `contextual_rerank` - Contextual + Cohere reranking
4. `contextual_custom_rerank` - Contextual + custom scripture reranker
5. `hybrid_rrf` - Sparse + dense with reciprocal rank fusion
6. `hybrid_rerank` - Hybrid + Pinecone reranking

Test set: 30 queries across 3 types (factual, thematic, cross-reference)

Metrics: Precision@K, MRR, NDCG@K, average relevance (0-3 scale)

## Corpus Statistics

| Volume | Verses | Chapters | Tokens |
|--------|-------:|---------:|-------:|
| Old Testament | 23,145 | 929 | 765,236 |
| New Testament | 7,957 | 260 | 223,878 |
| Book of Mormon | 6,604 | 239 | 321,306 |
| Doctrine and Covenants | 3,654 | 138 | 135,366 |
| Pearl of Great Price | 635 | 16 | 32,059 |
| **Total** | **41,995** | **1,582** | **1,477,845** |

## Known Issues and Solutions

**OOM on Pinecone upsert (Exit 137):**

- Removed ThreadPoolExecutor from upsert script
- Added `gc.collect()` after each volume
- Process volumes sequentially

**Empty retrieval results:**

- Cause: LangChain expects `text_key='text'` but metadata had `pageContent`
- Solution: Renamed metadata field to `text` during upsert

## Context Generation Prompt Design

The prompt in `prompts/chapter_summary_prompt.md` is optimized for **retrieval**, not human readability:

- Objective/literal (no theological interpretation)
- Present tense
- Proper nouns over pronouns
- 75-100 token limit
- Prioritizes: setting markers, actors, events, commandments, named topics

This produces context that maximizes embedding similarity for semantic search.
