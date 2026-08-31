from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from odbench_outer.openrouter import OpenRouterClient, OpenRouterError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class OpenRouterClientTests(unittest.TestCase):
    def test_chat_completion_request_and_response(self) -> None:
        document = {
            "id": "response-1",
            "model": "provider/model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    },
                }
            ],
            "usage": {"total_tokens": 12, "cost": 0.02},
        }
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data)
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            return FakeResponse(json.dumps(document).encode())

        client = OpenRouterClient(api_key="secret", model="provider/model")
        with patch("urllib.request.urlopen", fake_urlopen):
            completion = client.complete(
                messages=[{"role": "user", "content": "task"}],
                tools=[],
                session_id="run-1",
                max_output_tokens=123,
                reasoning_effort="high",
            )

        self.assertEqual(captured["authorization"], "Bearer secret")
        self.assertFalse(captured["payload"]["parallel_tool_calls"])
        self.assertEqual(captured["payload"]["session_id"], "run-1")
        self.assertEqual(captured["payload"]["max_completion_tokens"], 123)
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")
        self.assertEqual(completion.response_id, "response-1")
        self.assertEqual(completion.usage["total_tokens"], 12)

    def test_invalid_json_is_a_recoverable_model_error(self) -> None:
        client = OpenRouterClient(
            api_key="secret", model="provider/model", max_retries=0
        )
        with patch(
            "urllib.request.urlopen", return_value=FakeResponse(b"not json")
        ):
            with self.assertRaisesRegex(OpenRouterError, "invalid JSON"):
                client.complete(
                    messages=[{"role": "user", "content": "task"}],
                    tools=[],
                    session_id="run-1",
                    max_output_tokens=123,
                )


if __name__ == "__main__":
    unittest.main()
