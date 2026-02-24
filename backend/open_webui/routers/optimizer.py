from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Any, Dict, Optional, List
from open_webui.utils.auth import get_verified_user
import os
import time
import datetime
import json
import httpx
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
LOCAL_MODEL_URL = os.getenv(
    "OPTIMIZER_LOCAL_URL",
    "http://localhost:8080/api/chat/completions",
)
LOCAL_MODEL_NAME = os.getenv("OPTIMIZER_LOCAL_MODEL", "gemma3:latest")

OPENAI_API_URL = os.getenv(
    "OPTIMIZER_OPENAI_URL",
    "https://api.openai.com/v1/chat/completions",
)
OPENAI_MODEL_NAME = os.getenv("OPTIMIZER_OPENAI_MODEL", "gpt-5-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Optional: where to log optimizer stats
LOG_FILE = os.getenv("OPTIMIZER_LOG_FILE", "optimizer_logs.jsonl")

# -------------------------------------------------------------------
# Difficulty heuristic
# -------------------------------------------------------------------
def analyze_difficulty(text: str) -> str:
    """
    Heuristic difficulty analyzer:
    - short/simple text -> "simple"
    - long / math / code / many clauses / keywords -> "complex"
    """
    if not text:
        return "simple"

    text_lower = text.lower()
    complexity_keywords = [
        "explain",
        "compare",
        "summarize",
        "analysis",
        "reason",
        "why",
        "derive",
        "proof",
        "optimize",
        "implement",
        "evaluate",
        "create",
        "plan",
        "multi-step",
        "algorithm",
        "code",
        "function",
        "class",
        "sql",
        "database",
        "mathematics",
        "solve",
    ]

    score = 0
    words = text.split()

    # length heuristic
    if len(words) > 80:
        score += 2
    elif len(words) > 30:
        score += 1

    # punctuation / clauses
    if text.count("\n") >= 3 or text.count(";") + text.count("—") + text.count(",") > 8:
        score += 1

    # keywords
    for kw in complexity_keywords:
        if kw in text_lower:
            score += 1
            break

    # question words
    if any(q in text_lower for q in ["how", "why", "what", "which", "when"]):
        score += 1

    return "complex" if score >= 2 else "simple"


# -------------------------------------------------------------------
# Logging helper (non-fatal)
# -------------------------------------------------------------------
def log_request(entry: Dict[str, Any]) -> None:
    if not LOG_FILE:
        return
    try:
        entry = dict(entry)
        entry.setdefault(
            "timestamp", datetime.datetime.utcnow().isoformat() + "Z"
        )
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # never break main flow because of logging
        pass


# -------------------------------------------------------------------
# Downstream callers
# -------------------------------------------------------------------
async def call_local_model(
    payload: Dict[str, Any],
    auth_header: Optional[str],
) -> Dict[str, Any]:
    """
    Call the local OpenWebUI /api/chat/completions endpoint.

    IMPORTANT: We forward the Authorization header from the original
    request so that get_verified_user() on the backend passes.
    """
    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(LOCAL_MODEL_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


async def call_openai_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the OpenAI API directly using OPENAI_API_KEY.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set for optimizer")
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured for optimizer",
        )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENAI_API_URL, headers=headers, json=payload)
        # If this fails (401, 400, etc.), raise_for_status will bubble up
        resp.raise_for_status()
        return resp.json()


# -------------------------------------------------------------------
# Main optimizer endpoint
# -------------------------------------------------------------------
@router.post("/chat/completions")
async def create_completion(request: Request, user=Depends(get_verified_user)):
    """
    OpenAI-compatible chat completions endpoint with an optimizer.

    Behavior:
    - If model == "Auto" (or KMUTT variants):
        * analyze difficulty of latest user message
        * simple   -> local model (gemma3:latest)
        * complex  -> OpenAI model (gpt-5-nano)
    - If model explicitly == LOCAL_MODEL_NAME  -> local model
    - If model explicitly == OPENAI_MODEL_NAME -> OpenAI model
    """
    body = await request.json()
    model_field_raw = body.get("model") or ""
    model_field = model_field_raw.lower()

    messages: List[Dict[str, Any]] = body.get("messages", [])
    latest_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user = msg.get("content") or ""
            break

    is_auto = model_field in (
        "auto",
        "kmutt",
        "kmutt ai",
        "kmutt ai-as-a-service",
        "kmutt ai-as-a-service (auto)",
    )

    chosen_model = None
    difficulty = None
    call_fn = None

    # Decide model & call function
    if is_auto:
        difficulty = analyze_difficulty(latest_user)
        if difficulty == "simple":
            chosen_model = LOCAL_MODEL_NAME
            call_fn = "local"
        else:
            chosen_model = OPENAI_MODEL_NAME
            call_fn = "openai"
    else:
        # Explicit model selection
        if model_field == LOCAL_MODEL_NAME.lower():
            chosen_model = LOCAL_MODEL_NAME
            call_fn = "local"
        elif model_field == OPENAI_MODEL_NAME.lower():
            chosen_model = OPENAI_MODEL_NAME
            call_fn = "openai"
        else:
            # Fallback: treat unknown as local
            chosen_model = LOCAL_MODEL_NAME
            call_fn = "local"

    # Prepare payload for downstream
    downstream_body = dict(body)
    downstream_body["model"] = chosen_model

    start = time.perf_counter()
    try:
        if call_fn == "local":
            auth_header = request.headers.get("Authorization")
            resp_json = await call_local_model(
                payload=downstream_body, auth_header=auth_header
            )
        else:
            resp_json = await call_openai_model(payload=downstream_body)
    except httpx.HTTPStatusError as e:
        # Bubble up upstream errors with some context
        status_code = e.response.status_code
        logger.error(
            f"Downstream error from {call_fn} model {chosen_model}: "
            f"{status_code} - {e.response.text}"
        )
        raise HTTPException(
            status_code=502,
            detail=f"Downstream {call_fn} model error ({status_code})",
        )
    except Exception as e:
        logger.exception(f"Unexpected error in optimizer: {e}")
        raise HTTPException(
            status_code=500, detail="Optimizer internal error"
        )

    latency = time.perf_counter() - start
    usage = resp_json.get("usage", {})
    total_tokens = usage.get("total_tokens")

    user_label = getattr(user, "id", None) or getattr(user, "email", None) or "unknown"

    log_entry = {
        "user": user_label,
        "difficulty": difficulty or "n/a",
        "chosen_model": chosen_model,
        "request_model_field": model_field_raw,
        "tokens": total_tokens,
        "latency_s": round(latency, 3),
    }
    log_request(log_entry)

    return resp_json


# -------------------------------------------------------------------
# Minimal /models endpoint so OpenWebUI sees "Auto"
# -------------------------------------------------------------------
@router.get("/models")
async def list_models():
    """
    Minimal OpenAI-compatible /models endpoint so OpenWebUI
    can discover the 'Auto' model.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "Auto",
                "object": "model",
                "owned_by": "kmutt-optimizer",
            }
        ],
    }
