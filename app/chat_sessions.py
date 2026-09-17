"""Read REAL token usage from VS Code `chatSessions` event logs.

Unlike `app/ingest_vscode_sessions` (which proxies cost from transcript text),
this module reads the richer `chatSessions/*.jsonl` files that record actual
prompt/output token counts, the model used, and per-request timing.

Format: each file is one chat session written as an event log.
- `kind == 0`: a base snapshot; `v` is the full session state (with `requests`).
- `kind == 1`: a mutation; `k` is a path (e.g. ["requests", 0, "result"]) and
  `v` is the value to set at that path. Request objects are created implicitly by
  the first mutation that writes into them, so the path is auto-vivified.

Everything here reads local files only and never sends data anywhere.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


def _storage_root() -> Path:
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / "Code" / "User" / "workspaceStorage"


def _workspace_folder_name(ws_dir: Path) -> str:
    ws_json = ws_dir / "workspace.json"
    if ws_json.exists():
        try:
            folder_uri = json.loads(ws_json.read_text(encoding="utf-8")).get("folder", "")
            name = urllib.parse.unquote(folder_uri).rstrip("/").split("/")[-1]
            if name:
                return name
        except (OSError, ValueError):
            pass
    return ws_dir.name


def _find_session_files(storage_root: Path):
    if not storage_root.exists():
        return
    for ws_dir in storage_root.iterdir():
        sessions_dir = ws_dir / "chatSessions"
        if not sessions_dir.is_dir():
            continue
        folder_name = _workspace_folder_name(ws_dir)
        for jsonl_path in sessions_dir.glob("*.jsonl"):
            yield folder_name, jsonl_path


def _ensure(container, key, next_is_int):
    if isinstance(key, int):
        while len(container) <= key:
            container.append(None)
        if container[key] is None:
            container[key] = [] if next_is_int else {}
        return container[key]
    if key not in container or container[key] is None:
        container[key] = [] if next_is_int else {}
    return container[key]


def _apply_mutation(state: dict, path: list, value) -> None:
    cur = state
    for i in range(len(path) - 1):
        cur = _ensure(cur, path[i], isinstance(path[i + 1], int))
        if not isinstance(cur, (dict, list)):
            return
    last = path[-1]
    if isinstance(last, int) and isinstance(cur, list):
        while len(cur) <= last:
            cur.append(None)
        cur[last] = value
    elif isinstance(cur, dict):
        cur[last] = value


def reconstruct_session(jsonl_path: Path) -> dict | None:
    """Replay a chatSessions event log into its final state dict."""
    state: dict | None = None
    with open(jsonl_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == 0:
                snapshot = event.get("v")
                if isinstance(snapshot, dict):
                    state = snapshot
            elif event.get("kind") == 1 and isinstance(state, dict):
                key_path = event.get("k")
                if isinstance(key_path, list) and key_path:
                    try:
                        _apply_mutation(state, key_path, event.get("v"))
                    except (KeyError, IndexError, TypeError):
                        continue
    return state


def _session_default_model(state: dict) -> str | None:
    selected = (
        state.get("inputState", {})
        .get("selectedModel", {})
        .get("identifier")
    )
    return selected or None


def _prompt_text(request: dict) -> str:
    message = request.get("message") or {}
    text = message.get("text") or ""
    if text:
        return text
    parts = message.get("parts") or []
    return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def _kept_edit_count(request: dict) -> int:
    try:
        return json.dumps(request.get("response", [])).count("keepWithId")
    except (TypeError, ValueError):
        return 0


_PATH_KEYS = ("filePath", "fsPath", "path")


def _edited_files(request: dict) -> tuple[str, ...]:
    """Best-effort basenames of code files referenced in a request's response."""
    found: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in _PATH_KEYS and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(request.get("response", []))
    basenames = []
    for raw in found:
        name = raw.replace("\\", "/").rstrip("/").split("/")[-1]
        if "." in name and not name.startswith("."):
            basenames.append(name)
    return tuple(dict.fromkeys(basenames))


def _first_int(request: dict, target_keys: set[str]) -> int:
    """First integer value found under any of target_keys anywhere in the request."""
    stack = [request]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in target_keys and isinstance(value, (int, float)):
                    return int(value)
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return 0


@dataclass
class ChatRequestRecord:
    workspace: str
    session_id: str
    session_title: str
    request_id: str
    timestamp: int | None
    model_id: str | None
    prompt_tokens: int
    output_tokens: int
    prompt_chars: int
    prompt_text: str
    kept_edits: int
    elapsed_ms: int | None
    edited_files: tuple[str, ...] = ()
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    context_length: int = 0


def _extract_records(state: dict, workspace: str) -> list[ChatRequestRecord]:
    session_id = state.get("sessionId", "")
    session_title = (state.get("customTitle") or "").strip()
    default_model = _session_default_model(state)
    records: list[ChatRequestRecord] = []
    for request in state.get("requests", []) or []:
        if not isinstance(request, dict):
            continue
        metadata = (request.get("result") or {}).get("metadata") or {}
        prompt_tokens = int(metadata.get("promptTokens") or 0)
        output_tokens = int(
            metadata.get("outputTokens")
            or request.get("completionTokens")
            or 0
        )
        prompt_text = _prompt_text(request)
        records.append(
            ChatRequestRecord(
                workspace=workspace,
                session_id=session_id,
                session_title=session_title,
                request_id=request.get("requestId", ""),
                timestamp=request.get("timestamp"),
                model_id=request.get("modelId") or default_model,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prompt_chars=len(prompt_text),
                prompt_text=prompt_text,
                kept_edits=_kept_edit_count(request),
                elapsed_ms=request.get("elapsedMs") or request.get("timeSpentWaiting"),
                edited_files=_edited_files(request),
                cached_tokens=_first_int(request, {"cached_tokens"}),
                reasoning_tokens=_first_int(request, {"reasoning_tokens"}),
                context_length=_first_int(request, {"contextLengthBefore"}),
            )
        )
    return records


def load_chat_requests(storage_root: Path | None = None) -> list[ChatRequestRecord]:
    """Scan every workspace's chatSessions and return per-request token records."""
    root = storage_root or _storage_root()
    records: list[ChatRequestRecord] = []
    for workspace, jsonl_path in _find_session_files(root):
        state = reconstruct_session(jsonl_path)
        if isinstance(state, dict):
            records.extend(_extract_records(state, workspace))
    return records


if __name__ == "__main__":
    rows = load_chat_requests()
    total_prompt = sum(r.prompt_tokens for r in rows)
    total_output = sum(r.output_tokens for r in rows)
    print(f"{len(rows)} requests | prompt tokens {total_prompt:,} | output tokens {total_output:,}")
