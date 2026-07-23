"""
llm_cache.py  --  record / replay wrapper for LP2's OpenAI calls.

WHY THIS EXISTS
    LP2 asks GPT for predictions. GPT is nondeterministic (even at temperature 0),
    and every call costs money. To get reproducible experiments we:
      1. RECORD: run once against the real API, saving every (request -> response) pair.
      2. REPLAY: on later runs, answer from the saved file instead of calling GPT.
         -> free, offline, and identical every time.

HOW IT PLUGS IN (the whole integration is ONE line in LP2)
    In llm_LP2.py, LP2 builds a client like this:

        from openai import OpenAI
        client = OpenAI(api_key=..., organization=...)

    Replace that with:

        from llm_cache import CachingClient
        client = CachingClient()          # reads mode + path from env vars

    Nothing else in LP2 changes. `query_llm` still calls
    `client.chat.completions.create(...)` exactly as before -- it just goes
    through us now. (Same "keep the plug shape" idea as the Yggdrasil swap.)

HOW YOU SWITCH MODES (no code edits -- just environment variables)
    Record a run (spends money, needs a real key):
        export OPENAI_API_KEY=sk-...
        export LLM_CACHE_MODE=record
        export LLM_CACHE_PATH=llm_cache_office.json
        python -m lhmp.main ... project_config_LP2.yaml

    Replay it later (free, offline, deterministic):
        export LLM_CACHE_MODE=replay
        export LLM_CACHE_PATH=llm_cache_office.json
        python -m lhmp.main ... project_config_LP2.yaml

WHAT GETS SAVED PER CALL
    key   (the "question"): a SHA-256 hash of the answer-determining request
          fields -- model, messages (system + prompt + any error_addition),
          response_format, temperature. If any of these change, the key changes.
    value (the "answer"):   the full response, serialized. Replay rebuilds a real
          ChatCompletion from it, so downstream code that reads
          response.choices[0].message.content works identically.

THE CACHE MISS IS A FEATURE
    In replay, if a request's key isn't in the file, it means something upstream
    changed the request -- most likely a spatial-query result differs after the
    Yggdrasil swap. That's exactly what we want to catch, so a miss raises a loud,
    descriptive error (a divergence tripwire) instead of silently calling the API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion

# Transient OpenAI failures worth retrying during recording -- a network blip or a
# rate limit shouldn't silently turn into a permanently missing cache entry.
_TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_MAX_RECORD_ATTEMPTS = 5
_RECORD_BACKOFF_BASE_S = 2.0


class CacheMiss(KeyError):
    """Raised in replay mode when a request was never recorded (a divergence signal)."""


def _canonical_key(request_kwargs: dict) -> str:
    """Build a stable hash from the parts of the request that determine the answer.

    We deliberately key on the *content* of the request (model + messages +
    response_format + temperature), not on object identity, so the same prompt
    always maps to the same slot.
    """
    key_obj = {
        "model": request_kwargs.get("model"),
        "messages": request_kwargs.get("messages"),
        "response_format": request_kwargs.get("response_format"),
        "temperature": request_kwargs.get("temperature"),
    }
    blob = json.dumps(key_obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _Completions:
    """Mimics client.chat.completions -- exposes .create(**kwargs)."""

    def __init__(self, owner: "CachingClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> ChatCompletion:
        return self._owner._handle_create(kwargs)


class _Chat:
    """Mimics client.chat -- holds .completions."""

    def __init__(self, owner: "CachingClient") -> None:
        self.completions = _Completions(owner)


class CachingClient:
    """A drop-in stand-in for openai.OpenAI().

    Exposes exactly the surface LP2 uses: client.chat.completions.create(**kwargs).
    In record mode it delegates to a real OpenAI client and saves the result;
    in replay mode it never touches the network.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        cache_path: Optional[str] = None,
        real_client: Any = None,
    ) -> None:
        self.mode = (mode or os.environ.get("LLM_CACHE_MODE", "record")).lower()
        if self.mode not in ("record", "replay"):
            raise ValueError(
                f"LLM_CACHE_MODE must be 'record' or 'replay', got {self.mode!r}"
            )
        self.cache_path = cache_path or os.environ.get("LLM_CACHE_PATH", "llm_cache.json")

        # LP2 uses client.chat.completions.create(...) -- expose that path.
        self.chat = _Chat(self)

        # Load any existing recordings.
        self._store: dict[str, dict] = self._load_store()

        # A real OpenAI client is only needed for recording. `real_client` is an
        # injection point for testing (so tests never hit the network).
        self._real_client = real_client
        if self.mode == "record" and self._real_client is None:
            from openai import OpenAI  # imported lazily so replay needs no key

            self._real_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_API_ORG"),
            )

    # ---- storage -------------------------------------------------------------

    def _load_store(self) -> dict[str, dict]:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_store(self) -> None:
        # Atomic write so an interrupted run can't corrupt the file.
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    # ---- the intercepted call ------------------------------------------------

    def _handle_create(self, kwargs: dict) -> ChatCompletion:
        key = _canonical_key(kwargs)

        if self.mode == "replay":
            entry = self._store.get(key)
            if entry is None:
                preview = json.dumps(kwargs.get("messages"), ensure_ascii=False)[:600]
                raise CacheMiss(
                    "\n[llm_cache] CACHE MISS during replay.\n"
                    "A request was sent that was never recorded, which usually means an\n"
                    "upstream input changed -- e.g. a spatial-query result differs after\n"
                    "the Yggdrasil swap. This is the divergence tripwire working.\n"
                    f"  cache file : {self.cache_path}\n"
                    f"  missing key: {key}\n"
                    f"  prompt preview: {preview}\n"
                )
            # Rebuild a real ChatCompletion so downstream code is none the wiser.
            return ChatCompletion.model_validate(entry["response"])

        # record mode: call the real API, save, return the genuine response.
        # Transient failures (rate limit, timeout, connection reset, server error) are
        # retried with backoff so a flaky moment doesn't silently orphan this request
        # from the cache -- a real bug we chased down a `CacheMiss` this way once.
        attempt = 0
        while True:
            attempt += 1
            try:
                response: ChatCompletion = self._real_client.chat.completions.create(**kwargs)
                break
            except _TRANSIENT_ERRORS:
                if attempt >= _MAX_RECORD_ATTEMPTS:
                    raise
                time.sleep(_RECORD_BACKOFF_BASE_S * (2 ** (attempt - 1)))
        self._store[key] = {
            # request kept only for human debugging; the key is derived from it.
            "request": {
                "model": kwargs.get("model"),
                "messages": kwargs.get("messages"),
                "response_format": kwargs.get("response_format"),
                "temperature": kwargs.get("temperature"),
            },
            "response": response.model_dump(),
        }
        self._save_store()
        return response

    # ---- small conveniences --------------------------------------------------

    def stats(self) -> dict:
        """Quick summary -- handy for a sanity check after a run."""
        return {"mode": self.mode, "cache_path": self.cache_path, "recorded_calls": len(self._store)}
