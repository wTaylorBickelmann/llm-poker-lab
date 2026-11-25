"""One chat completion per generation, from Ollama or an OpenAI-compatible API.

The evolution loop asks an ``LLMClient`` for text and does not care which
server answered. Settings come from the command line, then environment
variables, then ``poker-lab.toml``, then defaults. The API key is read
only from the environment and is never written to a result file.

::

    CLI > env > poker-lab.toml > defaults
                  |
                  v
            LLMSettings (no key)
             |              |
             v              v
        OllamaClient   OpenAICompatClient
        POST /api/chat POST /chat/completions
             |              |  Authorization: Bearer (env only)
             +------+-------+
                    v
            complete() once per generation
"""

import json
import os
import tomllib
from pathlib import Path

import httpx

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_API_URL = "https://api.openai.com/v1"
DEFAULT_KEY_ENV = "POKER_LAB_API_KEY"
_SECRET_FIELDS = frozenset(
    {"api_key", "apikey", "api-key", "token", "key", "openai_api_key", "secret"}
)


class LLMError(RuntimeError):
    """The model server was unreachable, refused the call, or was misconfigured."""


class LLMClient:
    """Shared contract. Subclasses implement ``complete``."""

    provider = ""
    model = ""
    base_url = ""

    def complete(self, system: str, user: str) -> str:
        """Return the assistant text for one generation."""
        raise NotImplementedError

    def public(self) -> dict:
        """Provider, model, and base URL for results. Never an API key."""
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url}


class OllamaClient(LLMClient):
    """Local Ollama. Default model is ``qwen2.5-coder``."""

    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 180, temperature: float = 0.7, num_predict: int = 1800) -> None:
        """Remember the server, model, and sampling limits. There is no API key."""
        self.base_url = normalize_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict

    def complete(self, system: str, user: str) -> str:
        """POST ``/api/chat`` and return the assistant message."""
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        body = _post_json(self.base_url + "/api/chat", payload, {}, self.timeout, None)
        return _message_text(body, ("message", "content"))

    def __repr__(self) -> str:
        """Show the model and server, which is all this client stores."""
        return f"OllamaClient(model={self.model!r}, base_url={self.base_url!r})"


class OpenAICompatClient(LLMClient):
    """Any server that speaks ``POST /chat/completions`` with a bearer token."""

    provider = "api"

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 180, temperature: float = 0.7, max_tokens: int = 1800) -> None:
        """Store the endpoint and key. ``repr`` and ``public`` omit the key."""
        self.base_url = normalize_base_url(base_url)
        self.model = model
        self._api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        """POST ``/chat/completions`` and return the assistant message."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = _post_json(self.base_url + "/chat/completions", payload, headers, self.timeout, self._api_key)
        return _message_text(body, ("choices", 0, "message", "content"))

    def __repr__(self) -> str:
        """Show the model and server without the bearer token."""
        return f"OpenAICompatClient(model={self.model!r}, base_url={self.base_url!r})"


def normalize_base_url(url: str) -> str:
    """Accept ``localhost:11434`` or a full URL and drop a trailing slash."""
    cleaned = url.strip().rstrip("/")
    if "://" not in cleaned:
        return "http://" + cleaned
    return cleaned


def read_api_key(env_name: str = DEFAULT_KEY_ENV, environ: dict | None = None) -> str:
    """Read the bearer token from ``env_name``, then ``OPENAI_API_KEY``."""
    env = os.environ if environ is None else environ
    primary = str(env.get(env_name, "")).strip()
    if primary:
        return primary
    if env_name != "OPENAI_API_KEY":
        fallback = str(env.get("OPENAI_API_KEY", "")).strip()
        if fallback:
            return fallback
    extra = "" if env_name == "OPENAI_API_KEY" else " or OPENAI_API_KEY"
    raise LLMError(
        f"provider=api needs an API key in {env_name}{extra}. "
        "Export it in the environment. It is not read from poker-lab.toml."
    )


def resolve_settings(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    host: str | None = None,
    api_key_env: str | None = None,
    config_path: str | None = None,
    environ: dict | None = None,
) -> dict:
    """Pick provider, model, and base URL. CLI beats env beats the toml file."""
    env = os.environ if environ is None else environ
    file_values = _read_config(config_path)
    chosen = _pick(provider, env.get("POKER_LAB_PROVIDER"), file_values.get("provider"), "ollama")
    chosen = str(chosen).strip().lower()
    if chosen not in {"ollama", "api"}:
        raise LLMError(f"unknown provider {chosen!r}; use ollama or api")
    default_model = "qwen2.5-coder" if chosen == "ollama" else "gpt-4o-mini"
    default_url = DEFAULT_OLLAMA_URL if chosen == "ollama" else DEFAULT_API_URL
    cli_url = base_url if _present(base_url) else (host if _present(host) else None)
    file_url = file_values.get("base_url") or file_values.get("host")
    return {
        "provider": chosen,
        "model": str(_pick(model, env.get("POKER_LAB_MODEL"), file_values.get("model"), default_model)),
        "base_url": normalize_base_url(str(_pick(cli_url, env.get("POKER_LAB_BASE_URL"), file_url, default_url))),
        "api_key_env": str(_pick(api_key_env, None, None, DEFAULT_KEY_ENV)),
    }


def build_client(settings: dict, timeout: float = 180, temperature: float = 0.7, environ: dict | None = None) -> LLMClient:
    """Construct the client named by ``settings``. An API client reads the key now."""
    provider = settings["provider"]
    if provider == "ollama":
        return OllamaClient(settings["base_url"], settings["model"], timeout=timeout, temperature=temperature)
    if provider == "api":
        key = read_api_key(settings.get("api_key_env", DEFAULT_KEY_ENV), environ)
        return OpenAICompatClient(settings["base_url"], settings["model"], key, timeout=timeout, temperature=temperature)
    raise LLMError(f"unknown provider {provider!r}; use ollama or api")


def _present(value) -> bool:
    """True when a CLI or env value was actually provided."""
    return value is not None and str(value).strip() != ""


def _pick(cli_value, env_value, file_value, default):
    """Return the first present value in CLI, env, file, default order."""
    if _present(cli_value):
        return cli_value
    if _present(env_value):
        return env_value
    if _present(file_value):
        return file_value
    return default


def _read_config(config_path: str | None) -> dict:
    """Load ``poker-lab.toml`` and drop any field that could hold a secret."""
    path = _config_file(config_path)
    if path is None:
        return {}
    try:
        loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise LLMError(f"could not read {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise LLMError(f"could not read {path}: expected a table")
    return {key: value for key, value in loaded.items() if key not in _SECRET_FIELDS}


def _config_file(config_path: str | None) -> Path | None:
    """Use the explicit path, otherwise ``./poker-lab.toml`` when it exists."""
    if _present(config_path):
        path = Path(str(config_path))
        if not path.is_file():
            raise LLMError(f"config file not found: {path}")
        return path
    candidate = Path("poker-lab.toml")
    if candidate.is_file():
        return candidate
    return None


def _post_json(url: str, payload: dict, headers: dict, timeout: float, secret: str | None) -> dict:
    """POST JSON and return the decoded object. Errors never include ``secret``."""
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except httpx.TimeoutException:
        raise LLMError("LLM request timed out") from None
    except httpx.HTTPStatusError as exc:
        detail = _redact(exc.response.text[:300], secret)
        raise LLMError(f"LLM returned HTTP {exc.response.status_code}: {detail}") from None
    except httpx.HTTPError as exc:
        raise LLMError(f"Could not reach the LLM at {url}: {_redact(str(exc), secret)}") from None
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise LLMError(f"Unexpected LLM response: {exc}") from None
    if not isinstance(body, dict):
        raise LLMError("Unexpected LLM response: body is not an object")
    return body


def _message_text(body: dict, path: tuple) -> str:
    """Walk ``path`` to the assistant string. Indexes and keys both work."""
    value = body
    try:
        for step in path:
            value = value[step]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response: {exc}") from None
    if not isinstance(value, str):
        raise LLMError("Unexpected LLM response: content is not text")
    if not value.strip():
        raise LLMError("LLM returned an empty response")
    return value


def _redact(text: str, secret: str | None) -> str:
    """Replace a bearer token so a logged error cannot repeat it."""
    if secret and secret in text:
        return text.replace(secret, "***")
    return text
