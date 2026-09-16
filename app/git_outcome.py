"""Optional git-based signal for whether a Copilot session's edits were kept.

READ-ONLY: runs only `git rev-parse` and `git log` -- never writes to any repo.
Given the absolute file paths an assistant edited during one request and the
request's start time, it estimates how many of those edits survived (were
committed at/after the request) versus did not.

This is a FILE-LEVEL, directional proxy, NOT ground truth:
- it cannot attribute individual lines/hunks to the assistant,
- uncommitted work reads as "not kept",
- squashed/rebased history can blur the timeline.
Treat "accepted (git)" as evidence a change likely landed, not proof.
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 15


def _run_git(repo: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


@lru_cache(maxsize=512)
def _repo_root(dir_path: str) -> str | None:
    """Top-level git repo containing `dir_path`, or None if not a repo."""
    out = _run_git(dir_path, "rev-parse", "--show-toplevel")
    return out.strip() if out else None


def _file_kept_after(repo_root: str, abs_path: str, since_iso: str | None) -> bool:
    """True if `abs_path` has any commit in `repo_root` at/after `since_iso`."""
    try:
        rel = Path(abs_path).relative_to(Path(repo_root))
    except ValueError:
        return False
    args = ["log", "--pretty=%H"]
    if since_iso:
        args.append(f"--since={since_iso}")
    args += ["--", rel.as_posix()]
    out = _run_git(repo_root, *args)
    return bool(out and out.strip())


def git_outcome_for_case(edited_paths, since_iso: str | None) -> dict | None:
    """Estimate whether a request's edits were kept, using local git history.

    Returns {considered, kept, accepted} where `considered` counts edited files
    that live inside a git repo, or None if none do (caller should then fall
    back to the transcript-shape heuristic).
    """
    considered = 0
    kept = 0
    for path in edited_paths:
        parent = Path(path).parent
        if not parent.exists():
            continue
        root = _repo_root(str(parent))
        if not root:
            continue
        considered += 1
        if _file_kept_after(root, path, since_iso):
            kept += 1

    if considered == 0:
        return None
    return {"considered": considered, "kept": kept, "accepted": kept / considered >= 0.5}
