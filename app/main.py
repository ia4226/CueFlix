import os
import asyncio
import anthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.recommender import load_index, recommend
from app.query_parser import parse_query

app = FastAPI()

index = load_index()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class QueryRequest(BaseModel):
    query: str
    content_type: str = "Both"
    min_imdb: float = 0.0
    top_k: int = 5


def generate_explanation(title: str, description: str, user_query: str) -> str:
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"User wants: {user_query}\n"
                    f"Title: {title}\n"
                    f"Description: {description}\n\n"
                    "In one sentence, explain why this title matches what the user wants. "
                    "Be specific and conversational. No preamble."
                )
            }]
        )
        return response.content[0].text.strip()
    except Exception:
        return ""


@app.post("/recommend")
async def get_recommendations(request: QueryRequest):
    # Step 1: Parse query
    parsed = parse_query(request.query)
    expanded_query = parsed["expanded_query"]
    query_explanation = parsed["explanation"]

    # Step 2: Hybrid retrieval with filters
    results = recommend(
        expanded_query,
        index,
        top_k=request.top_k,
        content_type=request.content_type,
        min_imdb=request.min_imdb,
    )

    # Step 3: Generate all match explanations in parallel
    loop = asyncio.get_event_loop()
    explanations = await asyncio.gather(*[
        loop.run_in_executor(None, generate_explanation, r["title"], r["description"], request.query)
        for r in results
    ])

    for result, explanation in zip(results, explanations):
        result["match_reason"] = explanation

    return {
        "query_explanation": query_explanation,
        "results": results
    }


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))