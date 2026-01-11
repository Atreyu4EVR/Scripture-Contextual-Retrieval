# Notes

Notes for the project. Purpose: Token counting, contextual embedding generation, and vector storage for all five volumes of the LDS Standard Works using OpenAI's tiktoken, GPT-5.2, and text-embedding-3-large.

---

## Phase 1: Token Counting

### Initial Setup

- Created project structure with `CLAUDE.md`, `tokenizer.py`, `requirements.txt`
- Built token counter using OpenAI's tiktoken library

### Encoding Selection

- Initially used `cl100k_base` encoding, but found this was a legacy encoder for older GPT models
- Researched and switched to `o200k_base` encoding (correct for all GPT-5 models)
- Confirmed via `tiktoken.encoding_for_model("gpt-5")`

### Token Count Results (`o200k_base`)

| Volume | Token Count |
|--------|-------------|
| Book of Mormon | 321,306 |
| Doctrine & Covenants | 135,366 |
| New Testament | 223,878 |
| Old Testament | 765,236 |
| Pearl of Great Price | 32,059 |
| **Grand Total** | **1,477,845** |

### Deliverables

- `tokenizer.py` - Token counting script
- `token-counts.csv` - CSV export of results

---

## Phase 2: Data Normalization

### Problem

- Scripture JSON files had inconsistent structures
- D&C used "sections" instead of "chapters"
- Pearl of Great Price had different book organization

### Solution

- Created `normalize_scriptures.py`
- Standardized all volumes to `books → chapters → verses` structure
- Treated D&C "sections" as semantically equivalent to "chapters"

---

## Phase 3: Chapter Summary Generation

### Approach: Anthropic's Contextual Retrieval Method

- Prepend contextual summaries to each verse chunk before embedding
- Improves retrieval accuracy by providing chapter-level context

### Implementation

- Created `generate_chapter_summaries.py`
- Used GPT-5.2 Responses API with `reasoning={"effort": "low"}` parameter
- Implemented parallel processing with `ThreadPoolExecutor`
- Added Tenacity retry logic with exponential backoff

### Prompt Design

- Objective, factual summaries (50-75 tokens)
- No interpretive or devotional language
- Chapter verse count included
- Key events/themes summarized

### Results

- 1,582 chapter summaries generated successfully
- Saved to `scriptures/contextualized/summaries` directory as JSON files per volume

---

## Phase 4: Contextualized Verse Generation

### Implementation

- Created `generate_contextualized_verses.py`
- Combined chapter summaries with individual verses
- Clear boundary markers for retrieval:

```
**Context**: "The following passage is from **{reference}** from the **{volume}**. {chapter_summary}"
**Verse**: "{verse_text}"
```

### Output

- Contextualized verses saved to `contextualized-verses/` directory

---

## Phase 5: Embedding Generation

### Configuration

- Model: `text-embedding-3-large` (3072 dimensions)
- Batch processing for efficiency
- Tier 5 rate limits: 15,000 RPM, 40M TPM

### Implementation

- Created `generate_embeddings.py`
- Embedded the full contextualized text (context + verse)
- Stored embeddings with metadata (volume, reference, book, chapter, verse, text, context)

### Results

- 41,995 embeddings generated
- ~2.9 GB total data
- Saved to `scriptures/embeddings` directory as JSON files per volume:
  - `book-of-mormon-embeddings.json` (6,604)
  - `doctrine-and-covenants-embeddings.json` (3,654)
  - `new-testament-embeddings.json` (7,957)
  - `old-testament-embeddings.json` (23,145)
  - `pearl-of-great-price-embeddings.json` (635)

---

## Phase 6: Pinecone Vector Database Upsert

### Index Configuration

- Index name: `standard-works`
- Dimensions: 3072 (text-embedding-3-large)

### Challenges & Fixes

1. **Exit Code 137 (OOM)**
   - Removed `ThreadPoolExecutor`
   - Added `gc.collect()` after each volume
   - Process one volume at a time sequentially

2. **Wrong Record Format**
   - Initially used dictionaries: `{"id": ..., "values": ..., "metadata": ...}`
   - Fixed to use tuples: `(id, values, metadata)`

3. **Empty pageContent in Retrieval**
   - Problem: LangChain returned `pageContent: ""` at root, data was in `metadata.pageContent`
   - Root cause: LangChain's `PineconeVectorStore` defaults to `text_key='text'`
   - Solution: Renamed metadata field from `pageContent` to `text`

### Final Metadata Schema

```json
{
    "volume": "Book of Mormon",
    "reference": "1 Nephi 1:1",
    "context": "The following passage is from **1 Nephi 1:1**...",
    "text": "I, Nephi, having been born of goodly parents..."
}
```

### Final Results

- 41,995 vectors successfully upserted
- Proper field mapping verified with LangChain compatibility

---

## Files Created

### Scripts

| File | Purpose |
|------|---------|
| `tokenizer.py` | Token counting with tiktoken |
| `normalize_scriptures.py` | Standardize JSON structure |
| `generate_chapter_summaries.py` | GPT-5.2 chapter summaries |
| `generate_contextualized_verses.py` | Combine context + verses |
| `generate_embeddings.py` | text-embedding-3-large generation |
| `upsert_to_pinecone.py` | Vector database upload |
| `query_pinecone.py` | Search with proper field mapping |

### Documentation

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions for Claude Code |
| `AGENTS.md` | Copy of CLAUDE.md for other agents |
| `README.md` | Project documentation |
| `pinecone-instructions.md` | Pinecone metadata requirements |
| `script-instructions.md` | Script generation guidelines |

### Data Directories

| Directory | Contents |
|-----------|----------|
| `scriptures/` | Original + normalized JSON files |
| `chapter-summaries/` | 1,582 GPT-5.2 generated summaries |
| `contextualized-verses/` | Verses with prepended context |
| `embeddings/` | 41,995 embedding vectors (~2.9 GB) |

### Output

| File | Purpose |
|------|---------|
| `token-counts.csv` | Token count results |
| `results.json` | Query test results |

---

## Tech Stack

- **Language:** Python
- **Tokenizer:** OpenAI tiktoken (`o200k_base` for GPT-5)
- **LLM:** GPT-5.2 Responses API with reasoning
- **Embeddings:** text-embedding-3-large (3072 dimensions)
- **Vector DB:** Pinecone Serverless
- **Retry Logic:** Tenacity with exponential backoff
- **Parallelization:** ThreadPoolExecutor
