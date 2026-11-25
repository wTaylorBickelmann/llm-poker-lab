"""Run one LLM-written bot in a child process with a short leash.

The parent sends a JSON state per decision and reads one JSON action back.
Imports are limited to a small standard-library set, the child cannot write
files (zero file-size rlimit), and a decision that exceeds the timeout is
a fold. Enough failures disqualify the bot for the rest of the generation.

::

    validate source ---> child process ---> act(state) ---> action or fold
                              |  timeout / crash / bad import
                              +--> fold, and eventually disqualify
"""

import ast
import builtins
import json
import os
import resource
import select
import subprocess
import sys
import time

ALLOWED_MODULES = frozenset(
    {
        "abc", "bisect", "collections", "copy", "dataclasses", "decimal", "enum",
        "fractions", "functools", "heapq", "itertools", "json", "math", "numbers",
        "operator", "random", "re", "statistics", "string", "typing",
    }
)
FORBIDDEN_CALLS = frozenset(
    {
        "__import__", "breakpoint", "compile", "delattr", "eval", "exec", "exit",
        "getattr", "globals", "help", "input", "locals", "open", "quit", "setattr", "vars",
    }
)
MAX_CODE_CHARS = 20_000


class SandboxError(ValueError):
    """The bot source or the child process violated the sandbox contract."""


def validate_source(code: str) -> str:
    """Return the bot class name, or raise if the source is not allowed to run."""
    if len(code) > MAX_CODE_CHARS:
        raise SandboxError(f"code is longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"syntax error: {exc.msg}") from exc
    _Audit().visit(tree)
    name = _bot_class_name(tree)
    if name is None:
        raise SandboxError("code must define a class with act(self, state)")
    return name


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Import hook that allows only ``ALLOWED_MODULES``."""
    if level:
        raise ImportError("relative imports are not allowed")
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"import of {name} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def safe_builtins() -> dict:
    """Builtins handed to exec. No open, eval, or unrestricted import."""
    names = (
        "ArithmeticError", "AssertionError", "AttributeError", "Exception",
        "IndexError", "KeyError", "LookupError", "NotImplementedError",
        "RuntimeError", "StopIteration", "TypeError", "ValueError",
        "SystemExit", "ZeroDivisionError", "abs", "all", "any", "bin", "bool", "chr",
        "classmethod", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "hex", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "object", "oct",
        "ord", "pow", "property", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "zip",
    )
    table = {name: getattr(builtins, name) for name in names}
    table["__build_class__"] = builtins.__build_class__
    table["__import__"] = guarded_import
    table["print"] = silent_print
    return table


def silent_print(*args, **kwargs) -> None:
    """Drop prints so a bot cannot break the JSON line protocol."""
    return None


def load_bot(code: str, seed: int = 0):
    """Validate ``code``, exec it, and return an instance of the bot class."""
    validate_source(code)
    namespace = {"__builtins__": safe_builtins(), "__name__": "evolved_bot"}
    exec(compile(code, "<evolved>", "exec"), namespace, namespace)
    cls = namespace.get("EvolvedBot")
    if not _is_bot_class(cls):
        cls = next((value for value in namespace.values() if _is_bot_class(value)), None)
    if cls is None:
        raise SandboxError("no bot class with act() could be constructed")
    import random

    random.seed(seed)
    return cls()


def parse_reply(line: bytes) -> dict:
    """Decode one child response line into a dict."""
    try:
        data = json.loads(line.decode())
    except json.JSONDecodeError as exc:
        raise SandboxError(f"invalid sandbox output: {exc}") from exc
    if not isinstance(data, dict):
        raise SandboxError("sandbox reply is not an object")
    return data


class SandboxBot:
    """Bot facade. ``act`` talks to the child; failures become folds."""

    name = "evolved"

    def __init__(
        self,
        code: str,
        timeout: float = 0.5,
        max_failures: int = 5,
        max_seconds: float = 120.0,
        load_timeout: float = 5.0,
        seed: int = 0,
    ) -> None:
        """Store limits and reject source that fails the static checks."""
        self.code = code
        self.timeout = timeout
        self.max_failures = max_failures
        self.max_seconds = max_seconds
        self.load_timeout = load_timeout
        self.seed = seed
        self.failures = 0
        self.disqualified = False
        self.last_error: str | None = None
        self.proc: subprocess.Popen | None = None
        self._buf = b""
        self._started_at: float | None = None
        try:
            validate_source(code)
        except SandboxError as exc:
            self._disqualify(str(exc))

    def start(self) -> None:
        """Spawn the child and load the bot. A load failure disqualifies it."""
        if self.disqualified:
            return
        if self.proc is not None and self.proc.poll() is None:
            return
        self._spawn()
        try:
            self._send({"op": "load", "code": self.code, "seed": self.seed})
            reply = self._recv(self.load_timeout)
        except (TimeoutError, SandboxError, OSError) as exc:
            self._disqualify(str(exc))
            return
        if not reply.get("ok"):
            self._disqualify(str(reply.get("error") or "bot failed to load"))

    def act(self, state: dict) -> dict:
        """Return the child's action, or a fold when the child misbehaves."""
        if self.disqualified:
            return {"kind": "fold"}
        if self._started_at is None:
            self._started_at = time.monotonic()
        if time.monotonic() > self._started_at + self.max_seconds:
            self._disqualify("overall time limit exceeded")
            return {"kind": "fold"}
        if self.proc is None or self.proc.poll() is not None:
            self.start()
            if self.disqualified:
                return {"kind": "fold"}
        try:
            self._send({"op": "act", "state": state})
            reply = self._recv(self.timeout)
        except TimeoutError:
            self._fail("decision timed out")
            return {"kind": "fold"}
        except (SandboxError, OSError) as exc:
            self._fail(str(exc))
            return {"kind": "fold"}
        if not reply.get("ok"):
            self._fail(str(reply.get("error") or "invalid sandbox reply"))
            return {"kind": "fold"}
        action = reply.get("action")
        if not isinstance(action, dict) or action.get("kind") not in {"fold", "check_call", "raise"}:
            self._fail("bot returned an invalid action")
            return {"kind": "fold"}
        return action

    def close(self) -> None:
        """Stop the child if it is still running."""
        self._kill()

    def _spawn(self) -> None:
        """Start a fresh worker process with unbuffered stdio."""
        self._kill()
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "llm_poker_lab.sandbox_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _send(self, payload: dict) -> None:
        """Write one JSON line to the child."""
        if self.proc is None or self.proc.stdin is None:
            raise SandboxError("sandbox is not running")
        self.proc.stdin.write(json.dumps(payload).encode() + b"\n")
        self.proc.stdin.flush()

    def _recv(self, timeout: float) -> dict:
        """Read one JSON line, or raise if the child is silent or gone."""
        return parse_reply(self._readline(timeout))

    def _readline(self, timeout: float) -> bytes:
        """Read until a newline, honoring ``timeout`` seconds."""
        if self.proc is None or self.proc.stdout is None:
            raise SandboxError("sandbox is not running")
        buf = self._buf
        deadline = time.monotonic() + timeout
        stream = self.proc.stdout
        while b"\n" not in buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._buf = buf
                raise TimeoutError("decision timed out")
            ready, _, _ = select.select([stream], [], [], remaining)
            if not ready:
                self._buf = buf
                raise TimeoutError("decision timed out")
            chunk = os.read(stream.fileno(), 65536)
            if not chunk:
                self._buf = buf
                raise SandboxError("sandbox process exited")
            buf += chunk
        line, self._buf = buf.split(b"\n", 1)
        return line

    def _fail(self, message: str) -> None:
        """Count a bad decision and disqualify once the budget is spent."""
        self.failures += 1
        self.last_error = message
        self._kill()
        if self.failures >= self.max_failures:
            self.disqualified = True

    def _disqualify(self, message: str) -> None:
        """Stop the bot for the rest of the match and remember why."""
        self.disqualified = True
        self.last_error = message
        self._kill()

    def _kill(self) -> None:
        """Reap the child. Safe to call when no child is running."""
        proc = self.proc
        self.proc = None
        self._buf = b""
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        wait_for(proc)


def wait_for(proc: subprocess.Popen) -> None:
    """Wait for a killed child, ignoring a second timeout."""
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        return


def limit_resource(kind: int, value: int) -> None:
    """Set one rlimit, ignoring platforms that reject the request."""
    try:
        resource.setrlimit(kind, (value, value))
    except (ValueError, OSError):
        return


def apply_limits(memory_bytes: int = 512 * 1024 * 1024, cpu_seconds: int = 30) -> None:
    """Cap address space, CPU, file writes, and extra processes in the child."""
    limit_resource(resource.RLIMIT_AS, memory_bytes)
    limit_resource(resource.RLIMIT_CPU, cpu_seconds)
    limit_resource(resource.RLIMIT_FSIZE, 0)
    nproc = getattr(resource, "RLIMIT_NPROC", None)
    if nproc is not None:
        limit_resource(nproc, 32)


class _Audit(ast.NodeVisitor):
    """Reject imports, calls, and dunder access that escape the sandbox."""

    def visit_Import(self, node: ast.Import) -> None:
        """Block modules outside the allow list."""
        for alias in node.names:
            _check_module(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block relative imports and modules outside the allow list."""
        if node.level:
            raise SandboxError("relative imports are not allowed")
        _check_module(node.module or "")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Block direct calls such as open, eval, and exec."""
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise SandboxError(f"call to {node.func.id} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Block attribute names that start with an underscore."""
        if node.attr.startswith("_"):
            raise SandboxError(f"attribute {node.attr} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Block dunder names other than a normal ``__init__``."""
        if node.id.startswith("_") and node.id != "__init__":
            raise SandboxError(f"name {node.id} is not allowed")
        self.generic_visit(node)


def _check_module(name: str) -> None:
    """Raise if ``name`` is not an allowed standard-library module."""
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise SandboxError(f"import of {name} is not allowed")


def _bot_class_name(tree: ast.AST) -> str | None:
    """Prefer ``EvolvedBot``; otherwise the first class that implements ``act``."""
    fallback = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _has_act(node):
            if node.name == "EvolvedBot":
                return node.name
            if fallback is None:
                fallback = node.name
    return fallback


def _has_act(node: ast.ClassDef) -> bool:
    """True when the class defines ``act`` with room for self and state."""
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "act":
            if len(item.args.args) >= 2 or item.args.vararg is not None:
                return True
    return False


def _is_bot_class(value) -> bool:
    """True for a class object that defines ``act``."""
    return isinstance(value, type) and callable(getattr(value, "act", None))
