# n8n Hybrid Rerank Workflow Configuration

The built-in n8n Pinecone Vector Store node does not support hybrid search.
You'll need to use HTTP Request nodes to call the Pinecone API directly.

## Workflow Architecture

┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID RERANK WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌───────────────┐   ┌───────────────┐           │
│  │  Chat    │──▶│ Generate      │──▶│ Generate      │           │
│  │  Trigger │   │ Dense Embed   │   │ Sparse Embed  │           │
│  └──────────┘   │ (OpenAI)      │   │ (Pinecone)    │           │
│                 └───────────────┘   └───────────────┘           │
│                         │                   │                   │
│                         └─────────┬─────────┘                   │
│                                   ▼                             │
│                 ┌───────────────────────────────┐               │
│                 │  HTTP Request: Pinecone Query │               │
│                 │  (Hybrid: sparse + dense)     │               │
│                 └───────────────────────────────┘               │
│                                   │                             │
│                                   ▼                             │
│                 ┌───────────────────────────────┐               │
│                 │  HTTP Request: Pinecone Rerank│               │
│                 │  (pinecone-rerank-v0)         │               │
│                 └───────────────────────────────┘               │
│                                   │                             │
│                                   ▼                             │
│                 ┌───────────────────────────────┐               │
│                 │     AI Agent / LLM Node       │               │
│                 └───────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Node 1: Generate Dense Embeddings (HTTP Request)

```json
  {
    "method": "POST",
    "url": "<https://api.openai.com/v1/embeddings>",
    "headers": {
      "Authorization": "Bearer {{ $credentials.openAiApi.apiKey }}",
      "Content-Type": "application/json"
    },
    "body": {
      "model": "text-embedding-3-large",
      "input": "{{ $json.query }}"
    }
  }
```

Node 2: Generate Sparse Embeddings (HTTP Request)

```json
  {
    "method": "POST",
    "url": "<https://api.pinecone.io/embed>",
    "headers": {
      "Api-Key": "{{ $credentials.pineconeApi.apiKey }}",
      "Content-Type": "application/json"
    },
    "body": {
      "model": "pinecone-sparse-english-v0",
      "inputs": [{ "text": "{{ $json.query }}" }],
      "parameters": {
        "input_type": "query",
        "truncate": "END"
      }
    }
  }
```

Node 3: Hybrid Query (HTTP Request)

```json
  {
    "method": "POST",
    "url": "https://{{ $json.indexHost }}/query",
    "headers": {
      "Api-Key": "{{ $credentials.pineconeApi.apiKey }}",
      "Content-Type": "application/json"
    },
    "body": {
      "namespace": "scriptures",
      "topK": 50,
      "includeMetadata": true,
      "vector": "{{ $node['Dense Embed'].json.data[0].embedding }}",
      "sparseVector": {
        "indices": "{{ $node['Sparse Embed'].json.data[0].sparse_values.indices
  }}",
        "values": "{{ $node['Sparse Embed'].json.data[0].sparse_values.values
  }}"
      }
    }
  }
```

Node 4: Rerank Results (HTTP Request)

```json
  {
    "method": "POST",
    "url": "<https://api.pinecone.io/rerank>",
    "headers": {
      "Api-Key": "{{ $credentials.pineconeApi.apiKey }}",
      "Content-Type": "application/json"
    },
    "body": {
      "model": "pinecone-rerank-v0",
      "query": "{{ $json.query }}",
      "documents": "{{ $node['Hybrid Query'].json.matches.map(m =>
  m.metadata.text) }}",
      "top_n": 20,
      "return_documents": true
    }
  }
```

## Alternative: Use the n8n Workflow as a Tool

Since hybrid search requires custom HTTP nodes, you can:

1. Create a sub-workflow for hybrid rerank
2. Expose it as a tool in your AI Agent node
3. Call it via the "Call n8n Workflow" tool

This approach from <https://www.theaiautomators.com/hybrid-rag-trick-for-more-a>
i-agents-reliability/ recommends:

"For inference, you can't use Pinecone's standard vector store node in n8n
because it doesn't support hybrid search. Instead, call a dedicated n8n
workflow as a tool within your AI agent."

Important Configuration Notes
Setting: input_type
Value: "query"
Why: Critical for query embeddings (vs "passage" for ingestion)
────────────────────────────────────────
Setting: topK for hybrid
Value: 50-75
Why: Retrieve more candidates for reranking
────────────────────────────────────────
Setting: top_n for rerank
Value: 20
Why: Final results after reranking
────────────────────────────────────────
Setting: Batch size
Value: 96 max
Why: Pinecone limit for sparse embeddings
Your Index Configuration

For your standard-works-v2 index:

- Host: standard-works-v2-qtiihzn.svc.aped-4627-b74a.pinecone.io
- Namespace: scriptures
- Sparse Model: pinecone-sparse-english-v0
- Rerank Model: pinecone-rerank-v0

---

## Sources

- <https://www.theaiautomators.com/hybrid-rag-trick-for-more-ai-agents-reliability>
- <https://docs.pinecone.io/guides/search/hybrid-search>
- <https://docs.pinecone.io/reference/api/2025-04/inference/rerank>
- <https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstorepinecone>
- <https://n8n.io/workflows/5734-build-a-pdf-based-rag-system-with-openai-pinecone-and-cohere-reranking>
