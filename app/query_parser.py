import json
import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a query understanding engine for a movie and TV show recommender.

A user gives you a natural language query describing what they want to watch.
Your job is to extract their intent and return a JSON object with exactly these fields:

{
  "expanded_query": "a rich descriptive string combining mood, themes, genres, and any specific preferences — this will be used for semantic search",
  "explanation": "one sentence describing what the user seems to be in the mood for"
}

Rules:
- expanded_query should be 2-4 sentences, rich with descriptive language
- If the user mentions a specific actor, genre, or title — include it
- If the query is vague (e.g. "something good") — infer a reasonable interpretation
- Return ONLY the JSON object, no markdown, no preamble
"""

def parse_query(user_query: str) -> dict:
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_query}
            ]
        )

        raw = response.content[0].text.strip()
        parsed = json.loads(raw)

        return {
            "expanded_query": parsed.get("expanded_query", user_query),
            "explanation": parsed.get("explanation", "")
        }

    except Exception as e:
        print(f"Query parser error: {e}")
        return {
            "expanded_query": user_query,
            "explanation": ""
        }


if __name__ == "__main__":
    import time

    test_queries = [
        "something light and funny for a Sunday evening with my girlfriend",
        "intense crime thriller like Breaking Bad",
        "a feel good movie for kids"
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = parse_query(q)
        print(f"Expanded: {result['expanded_query']}")
        print(f"Explanation: {result['explanation']}")
        time.sleep(1)