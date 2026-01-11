"""
LLM-as-Judge evaluator for RAG retrieval quality.

Uses GPT-5.2 to evaluate relevance of retrieved documents
and quality of generated responses.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


load_dotenv()

JUDGE_MODEL = "gpt-5.2"
MAX_WORKERS = 20

RELEVANCE_SYSTEM_PROMPT = """You are an expert evaluator assessing scripture retrieval quality for a RAG system.

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
- Be objective and consistent in your ratings"""

RESPONSE_SYSTEM_PROMPT = """You are an expert evaluator assessing RAG system response quality.

Evaluate the response on these criteria (0-3 scale each):

1. **Answer Completeness**: Does the response fully address the query?
   - 0: Does not address the query at all
   - 1: Addresses query minimally or tangentially
   - 2: Addresses most aspects of the query
   - 3: Fully and comprehensively addresses the query

2. **Citation Accuracy**: Are the citations correct and traceable?
   - 0: No citations or completely wrong citations
   - 1: Some citations present but many errors
   - 2: Most citations are correct
   - 3: All citations are accurate and properly formatted

3. **Faithfulness**: Does the response only contain information from retrieved docs?
   - 0: Contains significant fabricated information
   - 1: Contains some information not in retrieved docs
   - 2: Mostly faithful with minor extrapolations
   - 3: Completely faithful to retrieved documents

4. **Format Compliance**: Does it follow the expected output structure?
   - 0: Does not follow format at all
   - 1: Partially follows format
   - 2: Mostly follows format with minor deviations
   - 3: Perfectly follows the expected format"""


@dataclass
class RelevanceJudgment:
    """Result of relevance evaluation."""
    query: str
    document_id: str
    reference: str
    score: int
    reasoning: str


@dataclass
class ResponseJudgment:
    """Result of response evaluation."""
    query: str
    completeness: int
    citation_accuracy: int
    faithfulness: int
    format_compliance: int
    overall: int
    notes: str


class LLMJudge:
    """LLM-based evaluator for retrieval and response quality."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _call_judge(self, system_prompt: str, user_prompt: str) -> str:
        """Make a judge API call with retry logic."""
        response = self.client.responses.create(
            model=JUDGE_MODEL,
            reasoning={"effort": "low"},
            instructions=system_prompt,
            input=user_prompt,
        )
        return response.output_text

    def judge_relevance(
        self,
        query: str,
        document: dict,
    ) -> RelevanceJudgment:
        """Judge the relevance of a single document to a query."""
        user_prompt = f"""Evaluate the relevance of this scripture passage to the query.

**Query:** {query}

**Retrieved Passage:**
Reference: {document.get('reference', 'Unknown')}
Text: "{document.get('text', '')}"

Respond with valid JSON only:
{{"score": <0-3>, "reasoning": "<1-2 sentence explanation>"}}"""

        result = self._call_judge(RELEVANCE_SYSTEM_PROMPT, user_prompt)

        try:
            parsed = json.loads(result)
            score = int(parsed.get("score", 0))
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            score = 0
            reasoning = f"Failed to parse response: {result[:100]}"

        return RelevanceJudgment(
            query=query,
            document_id=document.get("id", ""),
            reference=document.get("reference", ""),
            score=score,
            reasoning=reasoning,
        )

    def judge_relevance_batch(
        self,
        query: str,
        documents: list[dict],
    ) -> list[RelevanceJudgment]:
        """Judge relevance of multiple documents in parallel."""
        judgments = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.judge_relevance, query, doc): doc
                for doc in documents
            }

            for future in as_completed(futures):
                judgment = future.result()
                judgments.append(judgment)

        judgments.sort(key=lambda j: documents.index(
            next(d for d in documents if d.get("id") == j.document_id)
        ))

        return judgments

    def judge_response(
        self,
        query: str,
        response: str,
        retrieved_docs: list[dict],
    ) -> ResponseJudgment:
        """Judge the quality of a RAG response."""
        docs_text = "\n".join(
            f"- [{doc.get('reference', '')}]: {doc.get('text', '')[:200]}..."
            for doc in retrieved_docs[:5]
        )

        user_prompt = f"""Evaluate this RAG system response for a scripture query.

**Query:** {query}

**System Response:**
{response}

**Retrieved Documents Used:**
{docs_text}

Evaluate on the four criteria and respond with valid JSON only:
{{
  "completeness": <0-3>,
  "citation_accuracy": <0-3>,
  "faithfulness": <0-3>,
  "format_compliance": <0-3>,
  "overall": <0-3>,
  "notes": "<brief explanation>"
}}"""

        result = self._call_judge(RESPONSE_SYSTEM_PROMPT, user_prompt)

        try:
            parsed = json.loads(result)
            return ResponseJudgment(
                query=query,
                completeness=int(parsed.get("completeness", 0)),
                citation_accuracy=int(parsed.get("citation_accuracy", 0)),
                faithfulness=int(parsed.get("faithfulness", 0)),
                format_compliance=int(parsed.get("format_compliance", 0)),
                overall=int(parsed.get("overall", 0)),
                notes=parsed.get("notes", ""),
            )
        except (json.JSONDecodeError, ValueError):
            return ResponseJudgment(
                query=query,
                completeness=0,
                citation_accuracy=0,
                faithfulness=0,
                format_compliance=0,
                overall=0,
                notes=f"Failed to parse: {result[:100]}",
            )


def main():
    """Test the LLM judge."""
    judge = LLMJudge()

    test_query = "Who was Nephi?"
    test_doc = {
        "id": "test-1",
        "reference": "1 Nephi 1:1",
        "text": "I, Nephi, having been born of goodly parents, therefore I was taught somewhat in all the learning of my father.",
    }

    print("Testing LLM Judge")
    print("=" * 50)
    print(f"Query: {test_query}")
    print(f"Document: {test_doc['reference']}\n")

    judgment = judge.judge_relevance(test_query, test_doc)

    print(f"Score: {judgment.score}/3")
    print(f"Reasoning: {judgment.reasoning}")


if __name__ == "__main__":
    main()
