# LDS Standard Works Tokenizer

A Python utility that counts tokens in the LDS Standard Works using OpenAI's tiktoken library. Provides token counts for each scripture volume and a grand total across all five books.

## Overview

This tool analyzes the complete text of the LDS Standard Works and calculates token counts using the `o200k_base` encoding. This is useful for understanding context window usage when working with large language models.

## Token Encoding

All GPT-5 models use the `o200k_base` token encoding. You can let tiktoken automatically select the correct encoding for a model:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-5")
```

## Token Counts (GPT-5 / o200k_base)

| Volume | Verses | Tokens |
| :------ | -----: | -----: |
| Book of Mormon | 6,604 | 321,306 |
| Doctrine and Covenants | 3,654 | 135,366 |
| New Testament | 7,957 | 223,878 |
| Old Testament | 23,145 | 765,236 |
| Pearl of Great Price | 635 | 32,059 |
| **Grand Total** | **41,995** | **1,477,845** |

## Requirements

- Python 3.10+
- tiktoken

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python3 scripts/tokenizer.py
```

## Output

The script outputs token counts to the console and a CSV file (`token-counts.csv`) is available with the results.

## Scripture Data Format

Scripture files are stored in the `scriptures/` directory as flat JSON files:

```json
{
  "verses": [
    {
      "reference": "1 Nephi 1:1",
      "text": "verse content..."
    }
  ]
}
```

## Contextual Retrieval

This repository also implements **Contextual Retrieval**, a novel approach to retrieval augmented generation (RAG) that dramatically reduces retrieval failure by up to 35% in certain conditions. Introduced by Anthropic in the paper "Introducing Contextual Retrieval"[^1], this method intuitively embeds background context into each chunk, giving the LLM a significant boost in retrieval performance.

Read more about this our article "*[Enhanced Scripture RAG](articles/enhanced-scripture-rag.md)*"

## Scriptures `scriptures`

The primary dataset for this repository is the LDS Standard Works and is located in the `scriptures` directory. Read more about  `scriptures/README.md`

## Citations

[^1]: Anthropic PBC. (2024, September 19). *Introducing contextual retrieval*. Engineering at Anthropic. <https://www.anthropic.com/engineering/contextual-retrieval>
