import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("replicate-bridge")

REPLICATE_API_BASE = "https://api.replicate.com/v1"
REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]
REPLICATE_VERSION = os.environ["REPLICATE_VERSION"]
MODEL_NAME = os.getenv("MODEL_NAME", "qwen_32b")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
KEEPALIVE_INTERVAL_SECONDS = max(30, int(os.getenv("KEEPALIVE_INTERVAL_SECONDS", "120")))
KEEPALIVE_CHECK_SECONDS = max(5, int(os.getenv("KEEPALIVE_CHECK_SECONDS", "10")))
REPLICATE_MAX_WAIT_SECONDS = max(30, int(os.getenv("REPLICATE_MAX_WAIT_SECONDS", "180")))
REPLICATE_POLL_SECONDS = max(0.5, float(os.getenv("REPLICATE_POLL_SECONDS", "1")))
REPLICATE_MAX_CREATE_RETRIES = max(0, int(os.getenv("REPLICATE_MAX_CREATE_RETRIES", "3")))
REPLICATE_RATE_LIMIT_BUFFER_SECONDS = max(0.0, float(os.getenv("REPLICATE_RATE_LIMIT_BUFFER_SECONDS", "0.25")))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "50"))

client: httpx.AsyncClient | None = None
keepalive_task: asyncio.Task | None = None
state_lock = asyncio.Lock()
ping_lock = asyncio.Lock()
prediction_create_lock = asyncio.Lock()
ping_done_event = asyncio.Event()
ping_done_event.set()
state: dict[str, Any] = {
    "enabled": False,
    "last_activity_monotonic": time.monotonic(),
    "last_real_request": None,
    "active_real_requests": 0,
    "next_create_allowed_monotonic": 0.0,
    "last_rate_limit": None,
    "last_rate_limit_retry_after": None,
    "last_ping": None,
    "last_ping_status": None,
    "last_ping_predict_time": None,
    "last_ping_total_time": None,
    "last_ping_error": None,
    "ping_in_progress": False,
    "ping_started_at": None,
    "current_ping_prediction_id": None,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_bearer(authorization: str | None, expected: str, label: str) -> None:
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=f"Missing {label} bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=401, detail=f"Invalid {label} bearer token")


async def require_bridge_auth(authorization: str | None = Header(default=None)) -> None:
    require_bearer(authorization, BRIDGE_API_KEY, "bridge")


async def require_admin_auth(authorization: str | None = Header(default=None)) -> None:
    require_bearer(authorization, ADMIN_TOKEN, "admin")


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def build_prompt(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system_parts: list[str] = []
    conversation: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = content_to_text(message.get("content"))
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            conversation.append(f"Assistant: {content}")
        elif role == "tool":
            name = message.get("name") or message.get("tool_call_id") or "tool"
            conversation.append(f"Tool ({name}): {content}")
        else:
            conversation.append(f"User: {content}")
    conversation.append("Assistant:")
    return "\n\n".join(system_parts).strip(), "\n\n".join(conversation).strip()


def build_replicate_input(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty array")
    system_prompt, prompt = build_prompt(messages)
    model_input: dict[str, Any] = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_tokens": int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS),
        "temperature": float(payload["temperature"] if payload.get("temperature") is not None else DEFAULT_TEMPERATURE),
        "top_p": float(payload["top_p"] if payload.get("top_p") is not None else DEFAULT_TOP_P),
        "top_k": int(payload.get("top_k") or DEFAULT_TOP_K),
    }
    stop = payload.get("stop")
    if isinstance(stop, list) and stop:
        model_input["stop_sequences"] = ",".join(str(item) for item in stop)
    elif isinstance(stop, str) and stop:
        model_input["stop_sequences"] = stop
    return model_input


def normalize_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(str(item) for item in output)
    return str(output)


def retry_after_seconds(response: httpx.Response) -> float:
    header = (response.headers.get("Retry-After") or "").strip()
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    try:
        body = response.json()
    except Exception:
        body = {}
    if isinstance(body, dict) and isinstance(body.get("retry_after"), (int, float)):
        return max(0.0, float(body["retry_after"]))
    body_text = response.text or ""
    for pattern in [r"resets?\s+in\s+~?\s*([0-9]+(?:\.[0-9]+)?)\s*s", r"retry(?:\s+after)?\s+~?\s*([0-9]+(?:\.[0-9]+)?)\s*s", r"~\s*([0-9]+(?:\.[0-9]+)?)\s*s"]:
        match = re.search(pattern, body_text, flags=re.IGNORECASE)
        if match:
            return max(0.0, float(match.group(1)))
    return 1.0


async def begin_real_activity() -> None:
    async with state_lock:
        state["active_real_requests"] += 1
        state["last_activity_monotonic"] = time.monotonic()
        state["last_real_request"] = utc_now()


async def end_real_activity() -> None:
    async with state_lock:
        state["active_real_requests"] = max(0, int(state["active_real_requests"]) - 1)
        state["last_activity_monotonic"] = time.monotonic()
        state["last_real_request"] = utc_now()


async def wait_for_inflight_ping() -> None:
    async with state_lock:
        ping_in_progress = bool(state["ping_in_progress"])
        prediction_id = state["current_ping_prediction_id"]

    if not ping_in_progress:
        return

    logger.info(
        "real request waiting for keepalive warm-up prediction=%s",
        prediction_id,
    )

    try:
        await asyncio.wait_for(
            ping_done_event.wait(),
            timeout=REPLICATE_MAX_WAIT_SECONDS + 30,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "timed out waiting for keepalive warm-up; "
            "real request will continue"
        )


async def create_prediction(model_input: dict[str, Any], *, request_kind: str, retry_429: bool) -> dict[str, Any] | None:
    assert client is not None
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}", "Content-Type": "application/json"}
    max_attempts = REPLICATE_MAX_CREATE_RETRIES + 1 if retry_429 else 1
    for attempt in range(max_attempts):
        async with prediction_create_lock:
            async with state_lock:
                if request_kind == "keepalive" and state["active_real_requests"] > 0:
                    return None
                cooldown = max(0.0, float(state["next_create_allowed_monotonic"]) - time.monotonic())
            if cooldown > 0:
                if request_kind == "keepalive":
                    logger.info("keepalive skipped; Replicate cooldown %.2fs", cooldown)
                    return None
                logger.info("real request waiting %.2fs for Replicate cooldown", cooldown)
                await asyncio.sleep(cooldown)
            response = await client.post(
                f"{REPLICATE_API_BASE}/predictions",
                headers=headers,
                json={"version": REPLICATE_VERSION, "input": model_input},
                timeout=30,
            )
            if response.status_code == 429:
                delay = retry_after_seconds(response) + REPLICATE_RATE_LIMIT_BUFFER_SECONDS
                async with state_lock:
                    state["next_create_allowed_monotonic"] = max(float(state["next_create_allowed_monotonic"]), time.monotonic() + delay)
                    state["last_rate_limit"] = utc_now()
                    state["last_rate_limit_retry_after"] = delay
                logger.warning("Replicate 429 kind=%s retry_after=%.2fs attempt=%s/%s", request_kind, delay, attempt + 1, max_attempts)
                if request_kind == "keepalive":
                    return None
                if attempt + 1 < max_attempts:
                    continue
                raise HTTPException(status_code=429, detail={"message": "Replicate rate limit exceeded after retries", "retry_after": delay, "body": response.text}, headers={"Retry-After": str(max(1, int(delay)))})
            if response.status_code >= 400:
                raise HTTPException(status_code=502, detail={"message": "Replicate prediction creation failed", "status_code": response.status_code, "body": response.text})
            return response.json()
    raise HTTPException(status_code=502, detail="Replicate prediction creation failed")


async def wait_for_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    assert client is not None
    status = prediction.get("status")
    if status == "succeeded":
        return prediction
    if status in {"failed", "canceled"}:
        raise HTTPException(status_code=502, detail=prediction.get("error") or status)
    get_url = prediction.get("urls", {}).get("get")
    if not get_url:
        raise HTTPException(status_code=502, detail="Replicate response has no polling URL")
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
    deadline = time.monotonic() + REPLICATE_MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(REPLICATE_POLL_SECONDS)
        response = await client.get(get_url, headers=headers, timeout=30)
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Replicate prediction polling failed: {response.text}")
        prediction = response.json()
        status = prediction.get("status")
        if status == "succeeded":
            return prediction
        if status in {"failed", "canceled"}:
            raise HTTPException(status_code=502, detail=prediction.get("error") or status)
    raise HTTPException(status_code=504, detail=f"Replicate prediction exceeded {REPLICATE_MAX_WAIT_SECONDS}s")


async def run_keepalive_ping() -> dict[str, Any]:
    if ping_lock.locked():
        return {
            "skipped": True,
            "reason": "ping already running",
        }

    async with state_lock:
        if state["active_real_requests"] > 0:
            return {
                "skipped": True,
                "reason": "real request active",
            }

    async with ping_lock:
        async with state_lock:
            if state["active_real_requests"] > 0:
                return {
                    "skipped": True,
                    "reason": "real request active",
                }

            ping_done_event.clear()

            state["ping_in_progress"] = True
            state["ping_started_at"] = utc_now()
            state["current_ping_prediction_id"] = None
            state["last_ping_status"] = "starting"
            state["last_ping_error"] = None

        ping_input = {
            "prompt": "OK",
            "system_prompt": "",
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
        }

        started = time.monotonic()

        try:
            prediction = await create_prediction(
                ping_input,
                request_kind="keepalive",
                retry_429=False,
            )

            if prediction is None:
                async with state_lock:
                    state["last_ping"] = utc_now()
                    state["last_ping_status"] = "skipped"
                    state["last_ping_error"] = (
                        "Replicate cooldown/rate limit"
                    )
                    state["last_activity_monotonic"] = (
                        time.monotonic()
                    )

                return {
                    "skipped": True,
                    "reason": "Replicate cooldown/rate limit",
                }

            async with state_lock:
                state["current_ping_prediction_id"] = (
                    prediction.get("id")
                )
                state["last_ping_status"] = (
                    prediction.get("status") or "starting"
                )

            prediction = await wait_for_prediction(prediction)

            metrics = prediction.get("metrics") or {}

            async with state_lock:
                state["last_activity_monotonic"] = (
                    time.monotonic()
                )
                state["last_ping"] = utc_now()
                state["last_ping_status"] = (
                    prediction.get("status")
                )
                state["last_ping_predict_time"] = (
                    metrics.get("predict_time")
                )
                state["last_ping_total_time"] = metrics.get(
                    "total_time",
                    time.monotonic() - started,
                )
                state["last_ping_error"] = None

            logger.info(
                "keepalive succeeded predict_time=%s total_time=%s",
                state["last_ping_predict_time"],
                state["last_ping_total_time"],
            )

            return {
                "status": prediction.get("status"),
                "metrics": metrics,
            }

        except Exception as exc:
            async with state_lock:
                state["last_activity_monotonic"] = (
                    time.monotonic()
                )
                state["last_ping"] = utc_now()
                state["last_ping_status"] = "error"
                state["last_ping_error"] = str(exc)

            logger.exception("keepalive failed")

            return {
                "status": "error",
                "error": str(exc),
            }
        
        finally:
            async with state_lock:
                state["ping_in_progress"] = False
                state["ping_started_at"] = None
                state["current_ping_prediction_id"] = None
        
            ping_done_event.set()


async def keepalive_loop() -> None:
    while True:
        await asyncio.sleep(KEEPALIVE_CHECK_SECONDS)
        async with state_lock:
            enabled = bool(state["enabled"])
            active_real_requests = int(state["active_real_requests"])
            idle_for = time.monotonic() - float(state["last_activity_monotonic"])
        if enabled and active_real_requests == 0 and idle_for >= KEEPALIVE_INTERVAL_SECONDS:
            await run_keepalive_ping()


def openai_chunk(completion_id: str, delta: dict[str, Any], finish_reason: str | None = None) -> str:
    payload = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL_NAME, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def parse_replicate_sse(stream_url: str):
    assert client is not None
    async with client.stream("GET", stream_url, headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}", "Accept": "text/event-stream"}, timeout=None) as response:
        if response.status_code >= 400:
            body = await response.aread()
            raise RuntimeError(body.decode(errors="replace"))
        event_name: str | None = None
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if line.startswith(":"):
                continue
            if line == "":
                if event_name is not None:
                    yield event_name, "\n".join(data_lines)
                event_name = None
                data_lines = []
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())


async def openai_stream(prediction: dict[str, Any], completion_id: str):
    try:
        yield openai_chunk(completion_id, {"role": "assistant"})
        stream_url = prediction.get("urls", {}).get("stream")
        if stream_url:
            try:
                async for event_name, data in parse_replicate_sse(stream_url):
                    if event_name == "output":
                        yield openai_chunk(completion_id, {"content": data})
                    elif event_name == "error":
                        logger.error("Replicate stream error: %s", data)
                    elif event_name == "done":
                        break
                yield openai_chunk(completion_id, {}, "stop")
                yield "data: [DONE]\n\n"
                return
            except Exception:
                logger.exception("Native streaming failed; falling back to polling")
        final_prediction = await wait_for_prediction(prediction)
        content = normalize_output(final_prediction.get("output"))
        if content:
            yield openai_chunk(completion_id, {"content": content})
        yield openai_chunk(completion_id, {}, "stop")
        yield "data: [DONE]\n\n"
    finally:
        await end_real_activity()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, keepalive_task
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))
    keepalive_task = asyncio.create_task(keepalive_loop())
    logger.info("bridge started model=%s keepalive=OFF interval=%ss", MODEL_NAME, KEEPALIVE_INTERVAL_SECONDS)
    try:
        yield
    finally:
        if keepalive_task is not None:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
        if client is not None:
            await client.aclose()


app = FastAPI(title="Replicate OpenAI Bridge", version="1.1.1", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME, "keepalive_enabled": bool(state["enabled"])}


@app.get("/v1/models", dependencies=[Depends(require_bridge_auth)])
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "replicate-bridge"}]}


@app.post("/v1/chat/completions", dependencies=[Depends(require_bridge_auth)])
async def chat_completions(request: Request):
    payload = await request.json()
    requested_model = payload.get("model", MODEL_NAME)

    if requested_model != MODEL_NAME:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{requested_model}'",
        )

    model_input = build_replicate_input(payload)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    stream = bool(payload.get("stream"))

    await begin_real_activity()

    if stream:
        try:
            # Dacă warm-up-ul este deja pornit,
            # așteptăm să termine înainte de requestul real.
            await wait_for_inflight_ping()

            prediction = await create_prediction(
                model_input,
                request_kind="real",
                retry_429=True,
            )

            if prediction is None:
                raise HTTPException(
                    status_code=503,
                    detail="Replicate prediction was not created",
                )

        except Exception:
            await end_real_activity()
            raise

        return StreamingResponse(
            openai_stream(prediction, completion_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        # Dacă warm-up-ul este deja pornit,
        # nu mai creăm încă un prediction.
        await wait_for_inflight_ping()

        prediction = await create_prediction(
            model_input,
            request_kind="real",
            retry_429=True,
        )

        if prediction is None:
            raise HTTPException(
                status_code=503,
                detail="Replicate prediction was not created",
            )

        prediction = await wait_for_prediction(prediction)

        content = normalize_output(
            prediction.get("output")
        )

        metrics = prediction.get("metrics") or {}

        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "replicate": {
                    "prediction_id": prediction.get("id"),
                    "predict_time": metrics.get("predict_time"),
                    "total_time": metrics.get("total_time"),
                },
            }
        )

    finally:
        await end_real_activity()


@app.get("/admin/keepalive/status", dependencies=[Depends(require_admin_auth)])
async def keepalive_status() -> dict[str, Any]:
    async with state_lock:
        idle_for = max(
            0.0,
            time.monotonic()
            - float(state["last_activity_monotonic"]),
        )

        enabled = bool(state["enabled"])
        ping_in_progress = bool(
            state["ping_in_progress"]
        )

        next_ping = (
            max(
                0.0,
                KEEPALIVE_INTERVAL_SECONDS - idle_for,
            )
            if enabled and not ping_in_progress
            else None
        )

        return {
            "enabled": enabled,
            "interval_seconds": KEEPALIVE_INTERVAL_SECONDS,
            "seconds_since_activity": round(idle_for, 2),

            "next_ping_in_seconds": (
                round(next_ping, 2)
                if next_ping is not None
                else None
            ),

            "last_real_request": state["last_real_request"],
            "active_real_requests": state["active_real_requests"],

            "ping_in_progress": ping_in_progress,
            "ping_started_at": state["ping_started_at"],
            "current_ping_prediction_id": state[
                "current_ping_prediction_id"
            ],

            "last_rate_limit": state["last_rate_limit"],
            "last_rate_limit_retry_after": state[
                "last_rate_limit_retry_after"
            ],

            "last_ping": state["last_ping"],
            "last_ping_status": state["last_ping_status"],
            "last_ping_predict_time": state[
                "last_ping_predict_time"
            ],
            "last_ping_total_time": state[
                "last_ping_total_time"
            ],
            "last_ping_error": state["last_ping_error"],

            "model": MODEL_NAME,
        }


@app.post("/admin/keepalive/start", dependencies=[Depends(require_admin_auth)])
async def keepalive_start() -> dict[str, Any]:
    async with state_lock:
        already_enabled = bool(state["enabled"])

        active_real_requests = int(state["active_real_requests"])

        ping_in_progress = bool(state["ping_in_progress"])

        state["enabled"] = True
        state["last_activity_monotonic"] = time.monotonic()

    warmup_started = False

    if (
        not already_enabled
        and active_real_requests == 0
        and not ping_in_progress
    ):
        asyncio.create_task(run_keepalive_ping())
        warmup_started = True

    if warmup_started:
        message = "Keepalive enabled; warm-up ping started"

    elif ping_in_progress:
        message = "Keepalive enabled; warm-up already running"

    elif active_real_requests > 0:
        message = "Keepalive enabled; real request already active"

    else:
        message = "Keepalive already enabled"

    return {
        "enabled": True,
        "warmup_started": warmup_started,
        "message": message,
    }


@app.post("/admin/keepalive/stop", dependencies=[Depends(require_admin_auth)])
async def keepalive_stop() -> dict[str, Any]:
    async with state_lock:
        state["enabled"] = False
    return {"enabled": False, "message": "Keepalive disabled"}



@app.post("/admin/keepalive/ping", dependencies=[Depends(require_admin_auth)])
async def keepalive_ping() -> dict[str, Any]:
    return await run_keepalive_ping()
