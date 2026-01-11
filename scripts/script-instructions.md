# Script Instructions

## Maximizing API Processing

We need to take advantage of the high rate limits available for the service tier we are on at OpenAI, make the script as fast as possible (parallel processing, multiple workers, etc), while **intelligently respecting the rate limits** using the Tenacity library and building in retry logic.

We have the Tier 5 rate limits noted below:

GPT Models:

| TIER | RPM | TPM | BATCH QUEUE LIMIT |
| :---- | ---: | ---: | -----------------: |
| Tier 5 | 15,000 | 40,000,000 | 15,000,000,000 |

Embedding Models:

| TIER | RPM | TPM | BATCH QUEUE LIMIT |
| :---- | ---: | ---: | -----------------: |
| Tier 5 | 10,000 | 10,000,000 | 4,000,000,000 |

## Retrying

One easy way to avoid rate limit errors is to automatically retry requests with a random exponential backoff. Retrying with exponential backoff means performing a short sleep when a rate limit error is hit, then retrying the unsuccessful request. If the request is still unsuccessful, the sleep length is increased and the process is repeated. This continues until the request is successful or until a maximum number of retries is reached. This approach has many benefits:

- Automatic retries means you can recover from rate limit errors without crashes or missing data
- Exponential backoff means that your first retries can be tried quickly, while still benefiting from longer delays if your first few retries fail
- Adding random jitter to the delay helps retries from all hitting at the same time

### Tenacity

Tenacity is an Apache 2.0 licensed general-purpose retrying library, written in Python, to simplify the task of adding retry behavior to just about anything. To add exponential backoff to your requests, you can use the tenacity.retry decorator. The below example uses the tenacity.wait_random_exponential function to add random exponential backoff to a request.

```python
from openai import OpenAI
client = OpenAI()

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.completions.create(**kwargs)

completion_with_backoff(model="gpt-5.2", prompt=prompt)
```

```json

**Context**: "The following passage is from **1 Nephi 1:1** from the **Book of Mormon**. This chapter contains 20 verses and records Nephi’s introduction to his record and language, and sets the scene in the first year of Zedekiah at Jerusalem. Lehi prays, sees a pillar of fire, and receives a vision of heaven, a descending figure with twelve others, and a book. Reading it, he prophesies Jerusalem’s destruction and Babylonian captivity, and testifies of a coming Messiah. The people mock and seek Lehi’s life."
**Verse**: "I, Nephi, having been born of goodly parents, therefore I was taught somewhat in all the learning of my father; and having seen many afflictions in the course of my days, nevertheless, having been highly favored of the Lord in all my days; yea, having had a great knowledge of the goodness and the mysteries of God, therefore I make a record of my proceedings in my days."
