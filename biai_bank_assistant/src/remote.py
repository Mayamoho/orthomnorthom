from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .models import AssistantResult


class RemoteError(RuntimeError):
    """The remote model could not be reached or did not answer."""


class RemoteAssistant:
    """A client for a GPU-hosted BankAssistant (Kaggle/Colab), with a local fallback.

    The desktop app keeps its own lexical pipeline so a Bangla question still
    reaches the right FAQ offline, but a laptop CPU writes Bangla badly. Pointing
    `BIAI_API_URL` at the notebook's tunnel makes the app render the reply the
    Qwen model produced on the T4: every gate ran on that host, on the GPU stack,
    before the answer crossed the tunnel.
    """

    def __init__(self, base_url: str, fallback=None, api_token: str | None = None, timeout: float = 25.0):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token if api_token is not None else os.getenv("BIAI_API_TOKEN", "")
        self.timeout = timeout
        self.fallback = fallback
        self.last_error: str | None = None

    def answer(self, question: str) -> AssistantResult:
        try:
            payload = self._post({"question": question})
            return self._to_result(payload, question)
        except Exception as exc:  # a dead tunnel must not fail the desktop app
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.fallback is None:
                raise RemoteError(self.last_error) from exc
            result = self.fallback.answer(question)
            result.metadata = {**result.metadata, "backend": "local", "remote_error": self.last_error}
            return result

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/answer",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_token}"} if self.api_token else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RemoteError(f"endpoint replied {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RemoteError(str(exc)) from exc

    def _to_result(self, payload: dict, question: str) -> AssistantResult:
        if payload.get("status") not in {"answered", "handoff"}:
            raise RemoteError(f"unexpected status {payload.get('status')!r}")
        for required in ("answer", "language", "support_type", "confidence"):
            if required not in payload:
                raise RemoteError(f"endpoint response is missing {required!r}")
        known = {name for name in AssistantResult.__dataclass_fields__} - {"hits", "metadata"}
        fields = {key: payload[key] for key in known if key in payload}
        fields["hits"] = []
        fields["metadata"] = {**payload.get("metadata", {}), "backend": "remote", "question": question}
        return AssistantResult(**fields)

    def health(self) -> dict:
        """Which stack the endpoint is actually running, so the UI can be honest."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {"online": False, "error": str(exc)}


def to_payload(result: AssistantResult) -> dict:
    """The wire format the desktop client expects from `api_server.py`."""
    return {
        "answer": result.answer,
        "language": result.language,
        "status": result.status,
        "support_type": result.support_type,
        "source_id": result.source_id,
        "confidence": round(result.confidence, 4),
        "latency_ms": round(result.latency_ms, 2),
        "metadata": result.metadata,
    }
