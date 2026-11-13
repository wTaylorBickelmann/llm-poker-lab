"""Small client for Ollama's HTTP chat API.

The evolution loop calls this once per generation. Tests replace it with
a fake, and the CLI points ``--host`` at whatever server is local.

::

    system + user prompt ---> POST /api/chat ---> message content
"""

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """The model server was unreachable or returned a body we cannot use."""


def normalize_host(host: str) -> str:
    """Accept ``localhost:11434`` or a full URL and drop a trailing slash."""
    cleaned = host.strip().rstrip("/")
    if "://" not in cleaned:
        return "http://" + cleaned
    return cleaned


def chat(
    host: str,
    model: str,
    system: str,
    user: str,
    timeout: float = 180,
    temperature: float = 0.7,
    num_predict: int = 1800,
) -> str:
    """Send one non-streaming chat request and return the assistant text."""
    url = normalize_host(host) + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"Could not reach Ollama at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError("Ollama request timed out") from exc
    return _content(raw)


def _content(raw: bytes) -> str:
    """Pull message content out of an Ollama JSON body."""
    try:
        body = json.loads(raw.decode())
        content = body["message"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise OllamaError(f"Unexpected Ollama response: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("Ollama returned an empty response")
    return content
