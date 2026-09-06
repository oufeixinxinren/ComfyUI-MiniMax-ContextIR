"""Minimal, stdlib-only client for the MiniMax H3 V2 video APIs.

Endpoints implemented:
  POST /v2/h3_context_ir           - H3-Context-IR (returns enhanced prompt)
  POST /v2/video_generation        - create video generation task
  POST /v2/video_regeneration      - 768p -> 2K regeneration
  GET  /v2/query/video_generation/{task_id} - query task status/result

Auth: `Authorization: Bearer <API_KEY>` — the raw API key from
Account Management > API Keys (the platform uses Bearer API_key directly;
no JWT signing is required).
No third-party dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

BASE_URLS = {
    "global": "https://api.minimax.io",
    "cn": "https://api.minimaxi.com",
}

ENDPOINTS = {
    "context_ir": "/v2/h3_context_ir",
    "create": "/v2/video_generation",
    "regenerate": "/v2/video_regeneration",
}


def _request(
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    body: dict | None = None,
    timeout: int = 60,
) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        base_url + path,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        hint = ""
        if exc.code == 401:
            hint = " (check that api_key / MINIMAX_API_KEY is a valid MiniMax API key)"
        raise RuntimeError(f"MiniMax API HTTP {exc.code} on {path}: {detail}{hint}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MiniMax API network error on {path}: {exc.reason}") from exc


def create_task(
    base_url: str,
    api_key: str,
    endpoint: str,
    payload: dict,
    timeout: int = 60,
) -> dict:
    """Create an async task; returns the raw response (contains task_id)."""
    if endpoint not in ENDPOINTS.values():
        raise ValueError(f"Unknown endpoint: {endpoint}")
    return _request(base_url, api_key, "POST", endpoint, body=payload, timeout=timeout)


def query_task(
    base_url: str,
    api_key: str,
    task_id: str,
    timeout: int = 60,
) -> dict:
    return _request(
        base_url,
        api_key,
        "GET",
        f"/v2/query/video_generation/{task_id}",
        timeout=timeout,
    )
