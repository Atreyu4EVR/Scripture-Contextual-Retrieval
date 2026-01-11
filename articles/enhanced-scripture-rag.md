# Enhanced Scripture RAG

This repository aims to demonstrates the effectiveness of "Contextual Retrieval" in enhancing RAG performance. We use the LDS "Standard Works" scripture data in `scriptures` as the principle dataset for an experimental Scriptural-focused RAG Agent, that can reliably perform question answering tasks on faith-based topics grounded in doctrine.

## Chunking

Typical RAG pipelines require chunking, however this repository skips that step since the content in the Standard Works is already chunked into "verses", naturally fitting standard chunk sizes.
