"""Provider clients, settings precedence, and a fake OpenAI-compatible server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from llm_poker_lab.cli import main
from llm_poker_lab.evolve import evolve
from llm_poker_lab.llm import (
    LLMClient,
    LLMError,
    OllamaClient,
    OpenAICompatClient,
    _redact,
    build_client,
    normalize_base_url,
    read_api_key,
    resolve_settings,
)
from tests.support import VALID_BOT

SECRET = "super-secret-key"


def _mentions_secret(root) -> bool:
    """True when any saved file contains the bearer token bytes."""
    needle = SECRET.encode()
    return any(path.is_file() and needle in path.read_bytes() for path in Path(root).rglob("*"))


class _Reply:
    """Minimal httpx-like response for the mocked transport."""

    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        """Raise the same error httpx would for a failed status."""
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(self.status_code, text=self.text, request=request)
            raise httpx.HTTPStatusError("bad", request=request, response=response)

    def json(self):
        """Return the canned JSON, or fail the way a broken body would."""
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _install(monkeypatch, responder) -> None:
    """Replace httpx.post with a responder of (url, json, headers, timeout)."""

    def post(url, json=None, headers=None, timeout=None):
        """Call the test responder."""
        return responder(url, json, headers, timeout)

    monkeypatch.setattr("llm_poker_lab.llm.httpx.post", post)


def test_ollama_and_openai_clients(monkeypatch):
    """Both clients parse a chat reply and hide transport failures."""
    seen = {}

    def ok(url, payload, headers, timeout):
        """Record a successful Ollama call."""
        seen.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return _Reply(200, {"message": {"content": " STRATEGY:\nfold\n "}})

    _install(monkeypatch, ok)
    client = OllamaClient("localhost:11434/", "qwen", timeout=3, temperature=0.2)
    assert "fold" in client.complete("system", "user")
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["payload"]["model"] == "qwen"
    assert seen["payload"]["messages"][0]["content"] == "system"
    assert seen["timeout"] == 3
    assert client.public() == {"provider": "ollama", "model": "qwen", "base_url": "http://localhost:11434"}
    assert "qwen" in repr(client)
    with pytest.raises(NotImplementedError):
        LLMClient().complete("s", "u")

    def api_ok(url, payload, headers, timeout):
        """Record a successful chat-completions call."""
        seen.update(url=url, payload=payload, headers=dict(headers), timeout=timeout)
        return _Reply(200, {"choices": [{"message": {"content": "hello"}}]})

    _install(monkeypatch, api_ok)
    remote = OpenAICompatClient("https://api.openai.com/v1/", "gpt-4o-mini", SECRET, timeout=9)
    assert remote.complete("sys", "user") == "hello"
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in repr(remote)
    assert SECRET not in json.dumps(remote.public())

    def boom(url, payload, headers, timeout):
        """Return the bearer token in an error body, which must be redacted."""
        return _Reply(500, text=f"denied {SECRET}")

    _install(monkeypatch, boom)
    with pytest.raises(LLMError, match="HTTP 500") as caught:
        remote.complete("sys", "user")
    assert SECRET not in str(caught.value)
    assert "***" in str(caught.value)
    assert caught.value.__cause__ is None

    def refuse(url, payload, headers, timeout):
        """Simulate a connection failure that repeats the token."""
        raise httpx.ConnectError(f"refused {SECRET}")

    _install(monkeypatch, refuse)
    with pytest.raises(LLMError, match="Could not reach") as caught:
        remote.complete("sys", "user")
    assert SECRET not in str(caught.value)

    def stall(url, payload, headers, timeout):
        """Simulate a timeout."""
        raise httpx.TimeoutException("slow")

    _install(monkeypatch, stall)
    with pytest.raises(LLMError, match="timed out"):
        client.complete("sys", "user")

    def broken(url, payload, headers, timeout):
        """Return bytes that are not JSON."""
        return _Reply(200, payload=json.JSONDecodeError("bad", "", 0))

    _install(monkeypatch, broken)
    with pytest.raises(LLMError, match="Unexpected"):
        client.complete("sys", "user")

    def shaped(payload):
        """Return a fixed JSON object."""

        def respond(url, body, headers, timeout):
            return _Reply(200, payload)

        return respond

    _install(monkeypatch, shaped(["nope"]))
    with pytest.raises(LLMError, match="not an object"):
        client.complete("sys", "user")
    _install(monkeypatch, shaped({"message": {}}))
    with pytest.raises(LLMError, match="Unexpected"):
        client.complete("sys", "user")
    _install(monkeypatch, shaped({"message": {"content": "  "}}))
    with pytest.raises(LLMError, match="empty"):
        client.complete("sys", "user")
    _install(monkeypatch, shaped({"message": {"content": 3}}))
    with pytest.raises(LLMError, match="not text"):
        client.complete("sys", "user")
    _install(monkeypatch, shaped({"choices": []}))
    with pytest.raises(LLMError, match="Unexpected"):
        remote.complete("sys", "user")
    assert _redact("plain", None) == "plain"
    assert _redact("plain", SECRET) == "plain"
    assert normalize_base_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"


def test_settings_precedence(tmp_path, monkeypatch):
    """CLI beats the environment, which beats poker-lab.toml, which beats defaults."""
    defaults = resolve_settings(environ={})
    assert defaults == {
        "provider": "ollama",
        "model": "qwen2.5-coder",
        "base_url": "http://127.0.0.1:11434",
        "api_key_env": "POKER_LAB_API_KEY",
    }
    config = tmp_path / "poker-lab.toml"
    config.write_text(
        'provider = "api"\nmodel = "from-file"\nhost = "http://file.test/v1"\n'
        f'api_key = "{SECRET}"\nkey = "{SECRET}"\n',
        encoding="utf-8",
    )
    from_file = resolve_settings(config_path=str(config), environ={})
    assert from_file["provider"] == "api"
    assert from_file["model"] == "from-file"
    assert from_file["base_url"] == "http://file.test/v1"
    assert SECRET not in json.dumps(from_file)
    env = {
        "POKER_LAB_PROVIDER": "ollama",
        "POKER_LAB_MODEL": "from-env",
        "POKER_LAB_BASE_URL": "http://env.test",
    }
    from_env = resolve_settings(config_path=str(config), environ=env)
    assert from_env["provider"] == "ollama"
    assert from_env["model"] == "from-env"
    assert from_env["base_url"] == "http://env.test"
    from_cli = resolve_settings(
        provider="API",
        model="from-cli",
        base_url="https://cli.test/v1",
        host="http://ignored.test",
        api_key_env="OTHER_KEY",
        config_path=str(config),
        environ=env,
    )
    assert from_cli["provider"] == "api"
    assert from_cli["model"] == "from-cli"
    assert from_cli["base_url"] == "https://cli.test/v1"
    assert from_cli["api_key_env"] == "OTHER_KEY"
    host_only = resolve_settings(host="http://host.test", environ={})
    assert host_only["base_url"] == "http://host.test"
    assert resolve_settings(provider="api", environ={})["model"] == "gpt-4o-mini"
    assert resolve_settings(provider="api", environ={})["base_url"] == "https://api.openai.com/v1"
    with pytest.raises(LLMError, match="unknown provider"):
        resolve_settings(environ={"POKER_LAB_PROVIDER": "remote"})
    with pytest.raises(LLMError, match="not found"):
        resolve_settings(config_path=str(tmp_path / "missing.toml"), environ={})
    broken = tmp_path / "broken.toml"
    broken.write_text("provider = [\n", encoding="utf-8")
    with pytest.raises(LLMError, match="could not read"):
        resolve_settings(config_path=str(broken), environ={})
    monkeypatch.chdir(tmp_path)
    implicit = tmp_path / "poker-lab.toml"
    implicit.write_text('provider = "api"\nmodel = ""\nbase_url = ""\n', encoding="utf-8")
    found = resolve_settings(environ={})
    assert found["provider"] == "api"
    assert found["model"] == "gpt-4o-mini"
    monkeypatch.setattr("llm_poker_lab.llm.tomllib.loads", lambda text: ["nope"])
    with pytest.raises(LLMError, match="expected a table"):
        resolve_settings(config_path=str(config), environ={})


def test_api_key_is_env_only():
    """The key comes from the named env var, then OPENAI_API_KEY, and nowhere else."""
    assert read_api_key("POKER_LAB_API_KEY", {"POKER_LAB_API_KEY": f"  {SECRET}  "}) == SECRET
    assert read_api_key("POKER_LAB_API_KEY", {"OPENAI_API_KEY": "fallback-key"}) == "fallback-key"
    assert read_api_key("OPENAI_API_KEY", {"OPENAI_API_KEY": "only-openai"}) == "only-openai"
    assert read_api_key("OTHER", {"OTHER": "custom", "OPENAI_API_KEY": "fallback-key"}) == "custom"
    with pytest.raises(LLMError, match="POKER_LAB_API_KEY or OPENAI_API_KEY"):
        read_api_key("POKER_LAB_API_KEY", {})
    with pytest.raises(LLMError, match="OPENAI_API_KEY") as caught:
        read_api_key("OPENAI_API_KEY", {"OPENAI_API_KEY": "   "})
    assert " or OPENAI_API_KEY" not in str(caught.value)
    client = build_client(
        {"provider": "api", "model": "m", "base_url": "https://api.test/v1", "api_key_env": "OTHER"},
        environ={"OTHER": SECRET},
    )
    assert isinstance(client, OpenAICompatClient)
    assert SECRET not in json.dumps(client.public())
    local = build_client({"provider": "ollama", "model": "qwen", "base_url": "http://127.0.0.1:11434"})
    assert isinstance(local, OllamaClient)
    with pytest.raises(LLMError, match="unknown provider"):
        build_client({"provider": "nope", "model": "m", "base_url": "http://x"})


def test_missing_key_and_key_not_in_results(tmp_path, monkeypatch, capsys):
    """provider=api without a key fails, and a successful run never stores the key."""
    monkeypatch.delenv("POKER_LAB_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(
        ["evolve", "--provider", "api", "--generations", "1", "--hands", "6", "--out", str(tmp_path / "nokey")]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "API key" in err
    assert SECRET not in err

    def respond(url, payload, headers, timeout):
        """Return a reply that is not a bot, so the generation stops immediately."""
        return _Reply(200, {"choices": [{"message": {"content": "not a bot"}}]})

    _install(monkeypatch, respond)
    monkeypatch.setenv("POKER_LAB_API_KEY", SECRET)
    run = tmp_path / "run"
    evolve(
        generations=1,
        hands=6,
        out=run,
        client=build_client(
            {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "POKER_LAB_API_KEY",
            }
        ),
    )
    assert not _mentions_secret(run)
    logged = capsys.readouterr().out
    assert SECRET not in logged
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["provider"] == "api"
    assert config["model"] == "gpt-4o-mini"
    assert "base_url" in config
    assert "api_key" not in config
    results = json.loads((run / "gen-001" / "results.json").read_text(encoding="utf-8"))
    assert results["llm"]["provider"] == "api"

    def local_reply(url, payload, headers, timeout):
        """A default Ollama client should be built when evolve is given none."""
        return _Reply(200, {"message": {"content": "still not a bot"}})

    _install(monkeypatch, local_reply)
    default_run = tmp_path / "default"
    evolve(generations=1, hands=6, out=default_run)
    default_config = json.loads((default_run / "config.json").read_text(encoding="utf-8"))
    assert default_config["provider"] == "ollama"
    assert default_config["model"] == "qwen2.5-coder"


class _API(BaseHTTPRequestHandler):
    """One canned chat-completions response for the end-to-end test."""

    seen: dict = {}

    def do_POST(self):
        """Record the path and authorization header, then return a valid bot."""
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        _API.seen = {"path": self.path, "authorization": self.headers.get("Authorization")}
        content = f"STRATEGY:\ncall the api\nCODE:\n```python\n{VALID_BOT}\n```\n"
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        """Keep the test log quiet."""
        return


def test_evolve_end_to_end_fake_api(tmp_path, monkeypatch, capsys):
    """The CLI talks to a local /chat/completions server and does not save the key."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _API)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setenv("POKER_LAB_API_KEY", SECRET)
    try:
        code = main(
            [
                "evolve",
                "--provider",
                "api",
                "--base-url",
                host,
                "--model",
                "fake-model",
                "--generations",
                "1",
                "--hands",
                "6",
                "--seed",
                "1",
                "--timeout",
                "5",
                "--out",
                str(tmp_path),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
    assert code == 0
    assert _API.seen["path"].endswith("/chat/completions")
    assert _API.seen["authorization"] == f"Bearer {SECRET}"
    assert not _mentions_secret(tmp_path)
    output = capsys.readouterr().out
    config = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert SECRET not in output
    assert "fake-model" in config
    assert '"provider": "api"' in config
