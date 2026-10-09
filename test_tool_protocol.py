import asyncio
import json
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

os.environ.setdefault("REPLICATE_API_TOKEN", "test")
os.environ.setdefault("REPLICATE_VERSION", "test")
import app


class ToolProtocolTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "tools": [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }],
        }

    def test_tool_call_is_structured(self):
        output = "<tool_call><function=web_search><parameter=query>weather today</parameter></function></tool_call>"
        message, reason = app.completion_message(output, self.payload)
        self.assertEqual(reason, "tool_calls")
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "web_search")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"query": "weather today"})

    def test_unknown_tool_stays_as_text(self):
        output = "<tool_call><function=not_allowed></function></tool_call>"
        message, reason = app.completion_message(output, self.payload)
        self.assertEqual(reason, "stop")
        self.assertEqual(message["content"], output)

    def test_malformed_tool_stays_as_text(self):
        output = "<tool_call><function=web_search>not valid XML</function></tool_call>"
        message, reason = app.completion_message(output, self.payload)
        self.assertEqual(reason, "stop")
        self.assertEqual(message["content"], output)

    def test_plain_text_and_titles_unchanged(self):
        for output in ("Great", '{"title":"Example"}'):
            message, reason = app.completion_message(output, self.payload)
            self.assertEqual((message["content"], reason), (output, "stop"))

    def test_no_tools_means_plain_text(self):
        output = "<tool_call><function=web_search></function></tool_call>"
        self.assertEqual(app.completion_message(output, {})[1], "stop")

    def test_tool_definitions_are_supplied_to_model(self):
        with patch.dict(os.environ, {"REPLICATE_PROMPT_FORMAT": "qwen-chatml"}):
            model_input = app.build_replicate_input({
                **self.payload, "messages": [{"role": "user", "content": "search"}],
            })
        self.assertIn("web_search", model_input["prompt"])
        self.assertIn("parameters", model_input["prompt"])

    def test_assistant_tool_history_is_preserved(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "function": {"name": "web_search", "arguments": '{"query":"test"}'},
        }]}
        with patch.dict(os.environ, {"REPLICATE_PROMPT_FORMAT": "qwen-chatml"}):
            _, prompt = app.build_prompt([message])
        self.assertIn("<function=web_search>", prompt)

    def test_json_arguments_and_multiple_calls(self):
        output = (
            "<tool_call><function=web_search><parameter=query>one</parameter></function></tool_call>"
            "<tool_call><function=web_search><parameter=query>two</parameter></function></tool_call>"
        )
        message, reason = app.completion_message(output, self.payload)
        self.assertEqual(reason, "tool_calls")
        self.assertEqual(len(message["tool_calls"]), 2)
        self.assertNotEqual(message["tool_calls"][0]["id"], message["tool_calls"][1]["id"])


    def test_streamed_tool_chunks(self):
        async def consume():
            chunks = []
            async for chunk in app.openai_stream({
                "output": [
                    "<tool_call><function=web_search>",
                    "<parameter=query>today</parameter></function></tool_call>",
                ],
            }, "chatcmpl-test", self.payload):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(consume())
        events = [
            json.loads(row.removeprefix("data: "))
            for row in chunks if not row.startswith("data: [DONE]")
        ]
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "web_search")
        self.assertFalse(any("content" in event["choices"][0]["delta"] for event in events))

    def test_failed_prediction_returns_complete_http_502(self):
        async def create_prediction(*args, **kwargs):
            return {"id": "pred-failed", "status": "starting"}

        async def wait_for_prediction(_prediction):
            raise HTTPException(status_code=502, detail="Prediction failed: upstream 500")

        payload = {
            "model": app.MODEL_NAME,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.object(app, "create_prediction", create_prediction):
            with patch.object(app, "wait_for_prediction", wait_for_prediction):
                with TestClient(app.app) as client:
                    response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Prediction failed: upstream 500")
        self.assertTrue(response.headers["content-type"].startswith("application/json"))

    def test_successful_prediction_returns_complete_sse(self):
        async def create_prediction(*args, **kwargs):
            return {"id": "pred-success", "status": "starting"}

        async def wait_for_prediction(_prediction):
            return {"id": "pred-success", "status": "succeeded", "output": ["Hello!"]}

        payload = {
            "model": app.MODEL_NAME,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.object(app, "create_prediction", create_prediction):
            with patch.object(app, "wait_for_prediction", wait_for_prediction):
                with TestClient(app.app) as client:
                    response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('"content": "Hello!"', response.text)
        self.assertIn('"finish_reason": "stop"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\\n\\n"))


if __name__ == "__main__":
    unittest.main()
