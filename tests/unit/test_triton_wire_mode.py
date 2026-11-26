import json

import httpx
import pytest
import torch

from butler.triton_client import TritonClient


async def _sent_payload(task, language="en"):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"outputs": [{"name": "GENERATED_TEXT", "data": ["ok"]}]})

    client = TritonClient(prompt_protocol="general_json", model_name="voxtral_mini_3b_2507")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await client._generate_text("Can you transcribe this audio?", torch.zeros(1600), 16000, task=task, language=language)
    prompt = next(i for i in seen["body"]["inputs"] if i["name"] == "TEXT_PROMPT")["data"][0][0]
    return json.loads(prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "wire"),
    [("ASR", "transcribe"), ("SQA", "chat"), ("CAPTION", "caption")],
)
async def test_task_types_reach_the_model_as_its_own_modes(task, wire):
    payload = await _sent_payload(task)
    assert payload["mode"] == wire


@pytest.mark.asyncio
async def test_transcription_keeps_its_source_language():
    payload = await _sent_payload("ASR", language="en-US")
    assert payload["language"] == "en"
