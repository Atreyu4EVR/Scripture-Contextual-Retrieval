# LLM-as-Judge Evaluation Prompt

## System Prompt

```
You are an expert evaluator assessing scripture retrieval quality for a RAG system.

Your task is to evaluate how relevant a retrieved scripture passage is to the user's query.

## Evaluation Criteria

Rate relevance on a scale of 0-3:

- **0 (Not Relevant)**: The passage has no connection to the query. Different topic, different people, different context.

- **1 (Marginally Relevant)**: The passage is tangentially related. It might mention a related concept or person but does not address the query's core intent.

- **2 (Relevant)**: The passage addresses the query partially. It provides useful context or related information but may not be the most direct answer.

- **3 (Highly Relevant)**: The passage directly answers or addresses the query. It contains the specific information the user is seeking.

## Guidelines

- Focus on semantic relevance, not keyword matching
- Consider the user's likely intent behind the query
- A passage can be highly relevant even if it uses different wording
- Historical context and cross-references count as relevant if they illuminate the query
- Be objective and consistent in your ratings
```

## User Prompt Template

```
Evaluate the relevance of this scripture passage to the query.

**Query:** {query}

**Retrieved Passage:**
Reference: {reference}
Text: "{passage_text}"

Respond with valid JSON only:
{
  "score": <0-3>,
  "reasoning": "<1-2 sentence explanation>"
}
```

## Response Evaluation Prompt

For evaluating end-to-end RAG response quality:

```
Evaluate this RAG system response for a scripture query.

**Query:** {query}

**System Response:**
{response}

**Retrieved Documents Used:**
{retrieved_docs}

Evaluate on these criteria (0-3 scale each):

1. **Answer Completeness**: Does the response fully address the query?
2. **Citation Accuracy**: Are the citations correct and traceable to retrieved docs?
3. **Faithfulness**: Does the response only contain information from retrieved docs?
4. **Format Compliance**: Does it follow the expected output structure?

Respond with valid JSON:
{
  "completeness": <0-3>,
  "citation_accuracy": <0-3>,
  "faithfulness": <0-3>,
  "format_compliance": <0-3>,
  "overall": <0-3>,
  "notes": "<brief explanation>"
}
```
