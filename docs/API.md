# API examples

The server exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8095/v1` and serves model ID `minicpm-q8`. Keep the server bound to localhost unless you have configured access control for your deployment.

## Vision

Send image content with a text question. Local images can be encoded as a data URL; no external image hosting is required.

```python
import base64
import json
import urllib.request
from pathlib import Path

image = base64.b64encode(Path("screenshot.png").read_bytes()).decode()
body = {
    "model": "minicpm-q8",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this interface and suggest concrete improvements."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
    ]}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    "http://127.0.0.1:8095/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=600) as response:
    print(json.load(response)["choices"][0]["message"])
```

## Tools

Supply standard function schemas in `tools` and use `tool_choice: "auto"`. Execute returned calls in your agent, append the assistant message and corresponding `tool` results to the history, and request the next turn. The server generates tool calls; it does not execute arbitrary functions for you.

```json
{
  "model": "minicpm-q8",
  "messages": [{"role": "user", "content": "What is the weather in Boston?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city.",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "temperature": 0.6,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## Thinking and long work

Set `chat_template_kwargs.enable_thinking` per request. The benchmarked ordinary serving sampler uses temperature 0.6, top-p 0.95, and top-k 20; the graphics task used temperature 1.0. No temperature-zero requirement exists.

The release supports a maximum combined prompt/completion context of 262,144 tokens. The request examples omit a completion cap so the server can use remaining context. Check your client: many agent applications silently impose a shorter output limit. Prefix and image-feature caches benefit repeated histories; keeping the same image bytes avoids unnecessary re-encoding.
