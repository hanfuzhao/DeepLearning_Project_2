"""
NLP Product — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import time

from classifier import analyze_single, analyze_conversation, run_stress_test, _model_info, _model_is_available

app = FastAPI(
    title="NLP Product API",
    description="NLP-powered sarcasm, passive-aggression, and gaslighting detector",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SingleAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Message to analyze")


class ConversationAnalysisRequest(BaseModel):
    messages: List[str] = Field(..., min_items=1, max_items=20)


class StressTestRequest(BaseModel):
    additional_samples: Optional[List[Tuple[str, str]]] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    info = _model_info()
    return {
        "status": "ok",
        "service": "NLP Product API",
        "model": info,
        "fine_tuned_ready": _model_is_available(),
    }


@app.post("/analyze/single")
def analyze_single_endpoint(req: SingleAnalysisRequest):
    """Analyze a single message for toxicity type."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    start = time.time()
    result = analyze_single(req.text.strip())
    elapsed = round(time.time() - start, 3)
    return {
        "input": req.text,
        "label": result.label,
        "confidence": result.confidence,
        "all_scores": result.all_scores,
        "severity": result.severity,
        "sanity_check": result.sanity_check,
        "coping_strategies": result.coping_strategies,
        "suggested_responses": result.suggested_responses,
        "pattern_highlights": result.pattern_highlights,
        "elapsed_seconds": elapsed,
    }


@app.post("/analyze/conversation")
def analyze_conversation_endpoint(req: ConversationAnalysisRequest):
    """Contextual Consistency Audit over a conversation window."""
    messages = [m.strip() for m in req.messages if m.strip()]
    if not messages:
        raise HTTPException(status_code=400, detail="No valid messages provided.")
    start = time.time()
    result = analyze_conversation(messages)
    elapsed = round(time.time() - start, 3)
    result["elapsed_seconds"] = elapsed
    return result


@app.post("/evaluate/stress-test")
def stress_test_endpoint(req: StressTestRequest):
    """Run the Sarcastic Sentiment Flip stress test."""
    return run_stress_test(req.additional_samples)


@app.get("/resources")
def mental_health_resources():
    """Return curated mental health resources."""
    return {
        "crisis": [
            {"name": "Crisis Text Line", "detail": "Text HOME to 741741", "url": "https://www.crisistextline.org"},
            {"name": "National Domestic Violence Hotline", "detail": "1-800-799-7233", "url": "https://www.thehotline.org"},
            {"name": "SAMHSA Helpline", "detail": "1-800-662-4357 (free, 24/7)", "url": "https://www.samhsa.gov/find-help/national-helpline"},
        ],
        "self_help": [
            {"name": "Gaslighting Recovery Workbook", "detail": "Book by Amy Marlow-MaCoy", "url": "https://www.amazon.com/dp/1648481043"},
            {"name": "Loveisrespect", "detail": "Chat, call, or text for relationship help", "url": "https://www.loveisrespect.org"},
            {"name": "7 Cups", "detail": "Free online chat with trained listeners", "url": "https://www.7cups.com"},
        ],
        "education": [
            {"name": "Psychology Today — Gaslighting", "detail": "Articles on recognizing gaslighting", "url": "https://www.psychologytoday.com/us/basics/gaslighting"},
            {"name": "NAMI", "detail": "National Alliance on Mental Illness", "url": "https://www.nami.org"},
        ],
    }
