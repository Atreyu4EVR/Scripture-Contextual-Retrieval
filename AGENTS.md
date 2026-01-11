# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Purpose

Token counter for the LDS Standard Works using OpenAI's tiktoken library. Calculates token counts for each scripture volume and provides a grand total across all five books.

## Scripture Data Structure

The `scriptures/` directory contains flat JSON files for each volume:
- `book-of-mormon-flat.json`
- `doctrine-and-covenants-flat.json`
- `new-testament-flat.json`
- `old-testament-flat.json`
- `pearl-of-great-price-flat.json`

Each JSON file follows this structure:
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

## Tech Stack

- **Language:** Python
- **Tokenizer:** OpenAI tiktoken library
- **Data Format:** JSON

## Token Encoding

All GPT-5 models use the `o200k_base` token encoding. You can let tiktoken automatically select the correct encoding for a model:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-5")
```

## Build Commands

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the tokenizer
python3 tokenizer.py
```

## Output

- Console output with per-volume and grand total token counts
- `token-counts.csv` - CSV export of results
