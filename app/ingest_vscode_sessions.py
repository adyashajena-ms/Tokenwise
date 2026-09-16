"""Ingest real VS Code Copilot chat session transcripts into the yield ledger.

Turns local Copilot activity (across every workspace on this machine) into
Engagement/Case/Attempt rows using the SAME schema as the synthetic data, so
the existing ledger/estimator code shows real rework patterns unmodified.

Everything here reads local files only (never sent anywhere). Cost is a
PROXY estimate (~chars/4 tokens * an assumed $/1K-token rate) because local
transcripts do not record actual billed token counts -- treat costs as
directional, not exact. "Rework" = the same file edited more than once
within one user request; "rejected" attempts = assistant turns that were
superseded by a later turn in the same request.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from collections import Counter
from pathlib import Path

from app.git_outcome import git_outcome_for_case
from app.models import Attempt, Case, Engagement, get_session, init_db

EDIT_TOOL_NAMES = {
    "create_file", "replace_string_in_file", "multi_replace_string_in_file",
    "insert_edit_into_file", "edit_notebook_file",
}
CHARS_PER_TOKEN = 4
PRICE_PER_1K_TOKENS = 0.003  # blended proxy rate -- NOT a real billing figure
TASK_TYPE_PREFIX = "vscode:"


def _vscode_storage_root() -> Path:
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
        except Exception:
            pass
    return ws_dir.name


def _find_transcripts(storage_root: Path):
    if not storage_root.exists():
        return
    for ws_dir in storage_root.iterdir():
        transcripts_dir = ws_dir / "GitHub.copilot-chat" / "transcripts"
        if not transcripts_dir.is_dir():
            continue
        folder_name = _workspace_folder_name(ws_dir)
        for jsonl_path in transcripts_dir.glob("*.jsonl"):
            yield folder_name, jsonl_path


def _text_cost(*texts: str) -> float:
    total_chars = sum(len(t or "") for t in texts)
    tokens = total_chars / CHARS_PER_TOKEN
    return (tokens / 1000) * PRICE_PER_1K_TOKENS


def _safe_json(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def _extract_edit_path(tool_name, args) -> str | None:
    if tool_name not in EDIT_TOOL_NAMES or not isinstance(args, dict):
        return None
    return args.get("filePath") or args.get("path")


def parse_transcript(jsonl_path: Path) -> list[dict]:
    """Split one transcript into cases (one per user message), each with
    per-attempt costs and the file paths it touched."""
    cases: list[dict] = []
    current: dict | None = None
    turn_cost = 0.0
    last_had_content = False

    def close_case():
        nonlocal current, turn_cost, last_had_content
        if current is None:
            return
        incomplete = turn_cost > 0
        if incomplete:
            current["turn_costs"].append(turn_cost)
        current["accepted"] = last_had_content and not incomplete
        if current["turn_costs"]:
            cases.append(current)
        current = None
        turn_cost = 0.0
        last_had_content = False

    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            data = event.get("data") or {}

            if etype == "user.message":
                close_case()
                turn_cost = _text_cost(data.get("content", ""))
                current = {"turn_costs": [], "edited_paths": [], "start_time": event.get("timestamp")}
                continue

            if current is None:
                continue

            if etype == "assistant.message":
                content = data.get("content", "")
                turn_cost += _text_cost(content, data.get("reasoningText", ""))
                last_had_content = bool(content)
                for tool_req in data.get("toolRequests") or []:
                    args = _safe_json(tool_req.get("arguments"))
                    turn_cost += _text_cost(json.dumps(args) if args else str(tool_req.get("arguments", "")))
                    path = _extract_edit_path(tool_req.get("name"), args)
                    if path:
                        current["edited_paths"].append(path)
            elif etype == "assistant.turn_end":
                current["turn_costs"].append(turn_cost)
                turn_cost = 0.0

    close_case()
    return cases


def _clear_previous_ingest(session) -> None:
    stale = session.query(Engagement).filter(Engagement.task_type.like(f"{TASK_TYPE_PREFIX}%")).all()
    for engagement in stale:
        session.delete(engagement)  # cascades to cases/attempts


def ingest_vscode_sessions(use_git_signal: bool = False) -> dict:
    """Scan every local workspace's Copilot transcripts and (re)populate the
    ledger with real usage. Safe to call repeatedly -- previous real-data
    engagements are replaced, not duplicated.

    When `use_git_signal` is True, a request's accepted/abandoned status is
    decided by a read-only git-diff check (were the edited files committed at/
    after the request?) instead of the transcript-shape heuristic, where git
    history is available. See app/git_outcome.py for the (file-level, proxy)
    caveats.
    """
    init_db()
    storage_root = _vscode_storage_root()
    stats = {"workspaces": 0, "cases": 0, "attempts": 0, "skipped_files": 0,
             "git_scored_cases": 0, "git_kept_cases": 0}

    with get_session() as session:
        _clear_previous_ingest(session)

        engagement_ids: dict[str, int] = {}
        for folder_name, jsonl_path in _find_transcripts(storage_root):
            task_type = f"{TASK_TYPE_PREFIX}{folder_name}"
            if task_type not in engagement_ids:
                engagement = Engagement(customer="local-developer", task_type=task_type, complexity_tier="observed")
                session.add(engagement)
                session.flush()
                engagement_ids[task_type] = engagement.id
                stats["workspaces"] += 1

            try:
                cases = parse_transcript(jsonl_path)
            except OSError:
                stats["skipped_files"] += 1
                continue

            for case_data in cases:
                turn_costs = case_data["turn_costs"]
                n_turns = len(turn_costs)
                if n_turns == 0:
                    continue

                path_counts = Counter(case_data["edited_paths"])
                rework_units = sum(c - 1 for c in path_counts.values() if c > 1)
                total_edit_events = sum(path_counts.values())
                total_case_cost = sum(turn_costs)
                rework_fraction = (rework_units / total_edit_events) if total_edit_events else 0.0
                rework_cost = rework_fraction * total_case_cost

                accepted = case_data["accepted"]
                outcome_source = "transcript-heuristic"
                if use_git_signal:
                    git = git_outcome_for_case(case_data["edited_paths"], case_data.get("start_time"))
                    if git is not None:
                        accepted = git["accepted"]
                        outcome_source = "git-diff"
                        stats["git_scored_cases"] += 1
                        if git["accepted"]:
                            stats["git_kept_cases"] += 1
                case = Case(
                    engagement_id=engagement_ids[task_type],
                    status="accepted" if accepted else "abandoned",
                    outcome_source=outcome_source,
                )
                session.add(case)
                session.flush()
                stats["cases"] += 1

                for i, cost in enumerate(turn_costs, start=1):
                    is_last = i == n_turns
                    status = "accepted" if (is_last and accepted) else "rejected"
                    session.add(Attempt(
                        case_id=case.id,
                        attempt_number=i,
                        tokens_used=max(1, int(cost * 1000 / PRICE_PER_1K_TOKENS)),
                        cost=round(cost, 6),
                        status=status,
                        rework_cost=round(rework_cost, 6) if is_last else 0.0,
                    ))
                    stats["attempts"] += 1

    return stats


if __name__ == "__main__":
    print(ingest_vscode_sessions())
