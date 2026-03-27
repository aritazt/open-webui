from fastapi import APIRouter, HTTPException, Request
from typing import Any, Dict, List, Optional, Tuple
from open_webui.utils.auth import decode_token
from open_webui.models.users import Users
import os
import time
import datetime
import json
import uuid
import httpx
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Public model exposed to users
# -------------------------------------------------------------------
PUBLIC_MODEL_NAME = "KMUTT-AI-as-a-service"

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
LOCAL_MODEL_URL = os.getenv(
    "OPTIMIZER_LOCAL_URL",
    "http://localhost:11434/v1/chat/completions",
)
LOCAL_MODEL_NAME = os.getenv("OPTIMIZER_LOCAL_MODEL", "gemma3:latest")

OPENAI_API_URL = os.getenv(
    "OPTIMIZER_OPENAI_URL",
    "https://api.openai.com/v1/chat/completions",
)
OPENAI_MODEL_NAME = os.getenv("OPTIMIZER_OPENAI_MODEL", "gpt-5-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

LOG_FILE = os.getenv("OPTIMIZER_LOG_FILE", "")
OPTIMIZER_SERVICE_TOKEN = os.getenv("OPTIMIZER_SERVICE_TOKEN", "KMUTT")

HTTP_TIMEOUT_SECONDS = float(os.getenv("OPTIMIZER_HTTP_TIMEOUT", "60"))

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log_request(entry: Dict[str, Any]) -> None:
    """
    Non-fatal JSONL logging.
    """
    if not LOG_FILE:
        return

    try:
        payload = dict(entry)
        payload.setdefault("timestamp", utc_now_iso())
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_latest_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            return str(content or "")
    return ""


def sanitize_common_payload(payload: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Sanitize payload before sending downstream.
    - Force non-streaming for simpler JSON handling
    - Override model so clients cannot choose their own downstream model
    """
    cleaned = dict(payload)
    cleaned["model"] = model_name
    cleaned["stream"] = False
    return cleaned


def sanitize_openai_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert unsupported parameters for newer OpenAI models.
    """
    cleaned = dict(payload)

    if "max_tokens" in cleaned and "max_completion_tokens" not in cleaned:
        cleaned["max_completion_tokens"] = cleaned.pop("max_tokens")

    return cleaned


# -------------------------------------------------------------------
# Agent-based routing
# -------------------------------------------------------------------
def build_router_prompt(user_text: str) -> List[Dict[str, str]]:
    """
    Prompt for the router model.
    The router must answer with only one label.
    """
    system_prompt = (
        "You are a routing agent for a university AI portal.\n"
        "Your task is to classify whether a user request should be handled by:\n"
        "- SIMPLE: short, factual, lightweight, direct Q&A, casual chat, basic arithmetic\n"
        "- COMPLEX: reasoning-heavy, coding, multi-step analysis, planning, long writing, debugging, technical explanation\n\n"
        "Rules:\n"
        "1. Reply with exactly one word: SIMPLE or COMPLEX\n"
        "2. Do not explain your answer\n"
        "3. If unsure, prefer COMPLEX"
    )

    user_prompt = f"Classify this user request:\n\n{user_text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def call_local_model(
    payload: Dict[str, Any],
    auth_header: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        resp = await client.post(LOCAL_MODEL_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


async def call_openai_model(payload: Dict[str, Any]) -> Dict[str, Any]:
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

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        resp = await client.post(OPENAI_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


async def classify_route(
    latest_user_message: str,
    auth_header: Optional[str],
) -> Tuple[str, str]:
    """
    Agent-based route classification.
    Uses the local model as the router agent.
    Returns:
      - route_label: "simple" or "complex"
      - raw_router_output: raw text returned by the router
    """
    if not latest_user_message.strip():
        return "simple", "EMPTY_USER_MESSAGE"

    router_payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": build_router_prompt(latest_user_message),
        "temperature": 0,
        "stream": False,
    }

    try:
        resp_json = await call_local_model(router_payload, auth_header=auth_header)
        raw_output = (
            resp_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .upper()
        )

        if "SIMPLE" in raw_output and "COMPLEX" not in raw_output:
            return "simple", raw_output
        if "COMPLEX" in raw_output:
            return "complex", raw_output

        return "complex", raw_output or "UNPARSEABLE_ROUTER_OUTPUT"
    except Exception as e:
        logger.warning(f"Router agent failed, defaulting to complex: {e}")
        return "complex", "ROUTER_FAILURE_FALLBACK"


# -------------------------------------------------------------------
# Auth
# -------------------------------------------------------------------
def authenticate_request(auth_header: str):
    """
    Supports:
    1. JWT token
    2. Real API key
    3. Service token
    """
    user = None

    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :]

        token_data = decode_token(token)
        if token_data and "id" in token_data:
            user = Users.get_user_by_id(token_data["id"])

        if user is None:
            user = Users.get_user_by_api_key(token)

        if user is None and token == OPTIMIZER_SERVICE_TOKEN:
            user = Users.get_first_user()

    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return user


# -------------------------------------------------------------------
# Main endpoint
# -------------------------------------------------------------------
@router.post("/chat/completions")
async def create_completion(request: Request):
    """
    Public behavior:
    - Users only use KMUTT-AI-as-a-service
    - Any incoming model field is ignored for routing
    - Optimizer internally decides local vs OpenAI
    """
    request_id = str(uuid.uuid4())
    auth_header = request.headers.get("Authorization", "")
    user = authenticate_request(auth_header)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages: List[Dict[str, Any]] = body.get("messages", [])
    latest_user_message = get_latest_user_message(messages)

    user_label = getattr(user, "email", None) or getattr(user, "id", None) or "unknown"

    logger.info(f"[ROUTER] request_id={request_id} user={user_label} received request")

    # Force public model name regardless of what client sent
    incoming_model = body.get("model", "")
    body["model"] = PUBLIC_MODEL_NAME

    route_label, router_raw_output = await classify_route(
        latest_user_message, auth_header
    )

    if route_label == "simple":
        chosen_backend = "local"
        chosen_model = LOCAL_MODEL_NAME
    else:
        chosen_backend = "openai"
        chosen_model = OPENAI_MODEL_NAME

    logger.info(
        f"[ROUTER] request_id={request_id} incoming_model={incoming_model!r} "
        f"public_model={PUBLIC_MODEL_NAME} route={route_label} "
        f"backend={chosen_backend} chosen_model={chosen_model} "
        f"router_output={router_raw_output!r}"
    )

    downstream_body = sanitize_common_payload(body, chosen_model)

    if chosen_backend == "openai":
        downstream_body = sanitize_openai_payload(downstream_body)

    start = time.perf_counter()

    try:
        if chosen_backend == "local":
            resp_json = await call_local_model(
                payload=downstream_body,
                auth_header=auth_header,
            )
        else:
            resp_json = await call_openai_model(payload=downstream_body)

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        response_text = e.response.text

        logger.error(
            f"[ROUTER] request_id={request_id} downstream_error "
            f"backend={chosen_backend} model={chosen_model} "
            f"status={status_code} body={response_text}"
        )

        log_request(
            {
                "request_id": request_id,
                "user": user_label,
                "incoming_model": incoming_model,
                "public_model": PUBLIC_MODEL_NAME,
                "latest_user_message": latest_user_message,
                "route_label": route_label,
                "router_output": router_raw_output,
                "chosen_backend": chosen_backend,
                "chosen_model": chosen_model,
                "status": "error",
                "downstream_status_code": status_code,
                "downstream_response": response_text,
            }
        )

        raise HTTPException(
            status_code=502,
            detail=f"Downstream {chosen_backend} model error ({status_code})",
        )

    except Exception as e:
        logger.exception(f"[ROUTER] request_id={request_id} unexpected_error: {e}")

        log_request(
            {
                "request_id": request_id,
                "user": user_label,
                "incoming_model": incoming_model,
                "public_model": PUBLIC_MODEL_NAME,
                "latest_user_message": latest_user_message,
                "route_label": route_label,
                "router_output": router_raw_output,
                "chosen_backend": chosen_backend,
                "chosen_model": chosen_model,
                "status": "internal_error",
                "error": str(e),
            }
        )

        raise HTTPException(status_code=500, detail="Optimizer internal error")

    latency = time.perf_counter() - start
    usage = resp_json.get("usage", {})
    total_tokens = usage.get("total_tokens")

    log_request(
        {
            "request_id": request_id,
            "user": user_label,
            "incoming_model": incoming_model,
            "public_model": PUBLIC_MODEL_NAME,
            "latest_user_message": latest_user_message,
            "route_label": route_label,
            "router_output": router_raw_output,
            "chosen_backend": chosen_backend,
            "chosen_model": chosen_model,
            "latency_s": round(latency, 3),
            "tokens": total_tokens,
            "status": "success",
        }
    )

    logger.info(
        f"[ROUTER] request_id={request_id} completed "
        f"backend={chosen_backend} model={chosen_model} latency_s={latency:.3f}"
    )

    answer_tag = f"[Answered by: {chosen_model}]\n\n"

    try:
        if resp_json.get("choices") and resp_json["choices"][0].get("message"):
            original = resp_json["choices"][0]["message"].get("content", "")
            resp_json["choices"][0]["message"]["content"] = answer_tag + original
    except Exception:
        pass

    return resp_json


# -------------------------------------------------------------------
# Models endpoint
# -------------------------------------------------------------------
@router.get("/models")
async def list_models():
    """
    Only expose one model publicly.
    Users should not be able to select downstream models directly.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": PUBLIC_MODEL_NAME,
                "object": "model",
                "owned_by": "kmutt-optimizer",
                "name": PUBLIC_MODEL_NAME,
            }
        ],
    }