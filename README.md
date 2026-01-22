# Contextual Retrieval for LDS Standard Works

A complete pipeline for building high-quality, IP-compliant scripture retrieval systems using Anthropic's Contextual Retrieval methodology. This repository contains the code, prompts, evaluation framework, and results for AI-augmented scripture study.

## Key Findings

We evaluated contextual retrieval against traditional RAG approaches across 30 scripture study queries. **AI-generated chapter context improves retrieval quality while maintaining complete intellectual property compliance.**

| Method | P@5 | NDCG@10 | Avg Relevance | Latency |
|--------|-----|---------|---------------|---------|
| Traditional RAG (baseline) | 0.860 | 0.831 | 2.155 | 354ms |
| **Contextual Retrieval** | **0.907** | **0.869** | **2.257** | **327ms** |
| Contextual + Cohere Rerank | 0.820 | 0.796 | 2.112 | 522ms |

Contextual retrieval achieved a **5.4% improvement in Precision@5** and **4.6% improvement in NDCG@10** compared to traditional semantic search—while actually *reducing* latency by 7.6%.

Notably, general-purpose reranking *degraded* performance by 6.4% (p=0.011), suggesting that domain-specific embedding optimization outperforms generic post-hoc reranking for specialized religious texts.

## The Problem

Scripture verses are notoriously difficult to retrieve using standard RAG approaches. A single verse may contain only 10–20 words, providing insufficient semantic signal for effective embedding. Archaic language compounds the problem: terms like "hot drinks" (D&C 89:9) don't match modern queries about the Word of Wisdom. Additionally, verses are meaningful within their chapter and book context, but traditional embeddings treat them as isolated fragments.

The obvious solution would be to incorporate existing study aids (chapter summaries, cross-references, topical guides), but these are **protected intellectual property** of The Church of Jesus Christ of Latter-day Saints.

## Our Solution

We apply Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) methodology to generate **new, retrieval-optimized context** for each verse using AI—creating a clean-room alternative to copyrighted study aids.

### How It Works

1. **Chapter-level context generation**: GPT-5.2 generates a factual, retrieval-optimized summary for each of the 1,582 chapters in the Standard Works.

2. **Verse-level chunking with inherited context**: Each of the 41,995 verses becomes a chunk, with its parent chapter's context prepended before embedding.

3. **Optimized embedding**: The combined context + verse text is embedded using `text-embedding-3-large` (3072 dimensions), capturing both the verse content and its thematic setting.

**Example**: A query about "caffeine" now matches D&C 89:9 because the AI-generated context mentions "hot drinks" and "Word of Wisdom"—terms that bridge the modern query to archaic verse text.

## Corpus Statistics

| Volume | Verses | Chapters | Tokens (o200k_base) |
|--------|-------:|---------:|--------------------:|
| Old Testament | 23,145 | 929 | 765,236 |
| New Testament | 7,957 | 260 | 223,878 |
| Book of Mormon | 6,604 | 239 | 321,306 |
| Doctrine and Covenants | 3,654 | 138 | 135,366 |
| Pearl of Great Price | 635 | 16 | 32,059 |
| **Total** | **41,995** | **1,582** | **1,477,845** |

## Repository Structure

```
standardworks-tokenizer/
├── scriptures/                     # Source scripture data (JSON)
├── prompts/
│   └── chapter_summary_prompt.md   # Context generation prompt
├── scripts/
│   ├── generate_chapter_summaries.py
│   ├── generate_contextualized_verses.py
│   ├── generate_embeddings.py
│   ├── upsert_to_pinecone.py
│   ├── run_evaluation.py           # Full evaluation pipeline
│   ├── llm_judge.py                # LLM-as-judge relevance scoring
│   └── metrics_calculator.py       # P@K, NDCG, MRR calculations
├── evaluations/
│   ├── queries/test_queries.json   # 30 curated test queries
│   ├── config/evaluation_config.json
│   ├── results/                    # Raw retrieval results
│   └── reports/                    # Comparison reports (JSON + Markdown)
└── articles/
    └── enhanced-scripture-rag.md   # Detailed methodology writeup
```

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for embeddings and context generation)
- Pinecone account (for vector storage)

### Installation

```bash
git clone https://github.com/Atreyu4EVR/scripture-contextual-retrieval.git
cd scripture-contextual-retrieval

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env
```

### Run the Pipeline

```bash
# 1. Generate chapter summaries (1,582 API calls to GPT-5.2)
python scripts/generate_chapter_summaries.py

# 2. Create contextualized verses
python scripts/generate_contextualized_verses.py

# 3. Generate embeddings (text-embedding-3-large)
python scripts/generate_embeddings.py

# 4. Upsert to Pinecone
python scripts/upsert_to_pinecone.py

# 5. Run evaluation
python scripts/run_evaluation.py
```

## Context Generation

Our prompt is specifically designed for **retrieval optimization**, not human readability. Key constraints include objective/literal language (no theological interpretation), present tense, proper nouns over pronouns, and a 75–100 token limit. The full prompt is available in [`prompts/chapter_summary_prompt.md`](prompts/chapter_summary_prompt.md).

This produces context optimized for embedding similarity rather than human scanning—a key distinction from traditional chapter summaries.

## Intellectual Property Compliance

This methodology is designed for CES (Church Educational System) institutions that need to respect Church intellectual property.

**Indexed (public domain / freely available):**

- Scripture text

**NOT reproduced (protected IP):**

- Chapter headings and summaries
- Footnotes and cross-references
- Topical Guide entries
- Bible Dictionary content
- Any other study aids

AI-generated context serves as a **clean-room replacement** for copyrighted study aids, providing equivalent retrieval benefit through newly-created content.

## Evaluation Framework

We evaluate using an LLM-as-judge approach with relevance scored on a 0–3 scale. The test set of 30 queries spans three categories: factual queries ("Who was Nephi's father?"), thematic queries ("What does scripture teach about repentance?"), and cross-reference queries ("How do the Bible and Book of Mormon both describe the Messiah?").

Full evaluation results are available in [`evaluations/reports/`](evaluations/reports/).

## Citation

```bibtex
@misc{vallejo2026contextual,
  author       = {Vallejo, Ron},
  title        = {Contextual Retrieval for {LDS} Standard Works: 
                  An {IP}-Compliant Methodology for Scripture Study},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Atreyu4EVR/scripture-contextual-retrieval}}
}
```

## References

- Anthropic. (2024). [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). *Anthropic Engineering Blog*.
- Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). *arXiv:2005.11401*.

## License

MIT License. See [LICENSE](LICENSE) for details.

Scripture text is used in accordance with the Church's policy on scripture availability. AI-generated summaries are original works created for this project.

## Author

**Ron Vallejo**  
AI Engineer, Brigham Young University–Idaho
