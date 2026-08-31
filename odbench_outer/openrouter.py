"""Small dependency-free OpenRouter Chat Completions client."""

from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class ModelAPIError(RuntimeError):
    """A recoverable failure at the remote model boundary."""


class OpenRouterError(ModelAPIError):
    pass


@dataclass(frozen=True)
class Completion:
    message: dict[str, Any]
    model: str
    finish_reason: str | None
    usage: dict[str, Any]
    response_id: str | None
    raw: dict[str, Any]


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = DEFAULT_ENDPOINT,
        request_timeout: float = 300,
        max_retries: int = 3,
        app_title: str = "OD Benchmark",
        http_referer: str | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("OpenRouter API key is required")
        if not isinstance(model, str) or not model:
            raise ValueError("OpenRouter model is required")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("OpenRouter endpoint is required")
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
            or not math.isfinite(float(request_timeout))
            or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be positive")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.app_title = app_title
        self.http_referer = http_referer

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str,
        max_output_tokens: int,
        reasoning_effort: str | None = None,
    ) -> Completion:
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise ValueError("messages and tools must be arrays")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "parallel_tool_calls": False,
            "stream": False,
            "max_completion_tokens": max_output_tokens,
            "session_id": session_id,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.app_title,
            "X-OpenRouter-Metadata": "enabled",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint, data=encoded, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    try:
                        document = json.load(response)
                    except (UnicodeError, json.JSONDecodeError) as error:
                        raise OpenRouterError("OpenRouter returned invalid JSON") from error
                break
            except urllib.error.HTTPError as error:
                detail = error.read(64 * 1024).decode("utf-8", errors="replace")
                retryable = error.code in {408, 409, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.max_retries:
                    raise OpenRouterError(f"OpenRouter HTTP {error.code}: {detail}") from error
                retry_after = (
                    error.headers.get("Retry-After") if error.headers is not None else None
                )
                try:
                    delay = float(retry_after) if retry_after is not None else 2**attempt
                except ValueError:
                    delay = 2**attempt
                if not math.isfinite(delay) or delay < 0:
                    delay = 2**attempt
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt >= self.max_retries:
                    raise OpenRouterError(f"OpenRouter request failed: {error}") from error
                delay = 2**attempt
            time.sleep(min(delay, 60) + random.random() * 0.25)
        else:  # pragma: no cover - loop always breaks or raises
            raise OpenRouterError("OpenRouter retry loop exhausted")

        if not isinstance(document, dict):
            raise OpenRouterError("OpenRouter returned a non-object response")
        if "error" in document:
            raise OpenRouterError(f"OpenRouter error: {document['error']}")
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenRouterError("OpenRouter response has no completion choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise OpenRouterError("OpenRouter response has no assistant message")
        usage = document.get("usage")
        response_id = document.get("id")
        response_model = document.get("model", self.model)
        finish_reason = choice.get("finish_reason")
        if response_id is not None and not isinstance(response_id, str):
            raise OpenRouterError("OpenRouter response id is invalid")
        if not isinstance(response_model, str) or not response_model:
            raise OpenRouterError("OpenRouter response model is invalid")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise OpenRouterError("OpenRouter finish reason is invalid")
        return Completion(
            message=message,
            model=response_model,
            finish_reason=finish_reason,
            usage=usage if isinstance(usage, dict) else {},
            response_id=response_id,
            raw=document,
        )
