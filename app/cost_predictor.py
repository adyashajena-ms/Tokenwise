"""Predict a prompt's dollar cost across models BEFORE you send it.

Counts real input tokens for your prompt plus any attached context files (exact
BPE via tiktoken when available), predicts output tokens, and prices the request
against every known model so you can pick the cheapest that fits.

Costs use illustrative per-model rates (app.model_pricing) -- directional, not
billed figures.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from app.model_pricing import MODEL_PRICING, model_details, pricing_for
from app.prompt_optimizer import estimate_tokens

DEFAULT_OUTPUT_TOKENS = 500

# Only treat these as code-context files when auto-collecting from git.
_CODE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".cs", ".java", ".go", ".rb", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".xaml", ".json", ".yaml", ".yml", ".md", ".sql",
}
_MAX_AUTO_FILES = 12


def gather_git_context_files(repo_dir: str = ".") -> list[str]:
    """Auto-collect changed/untracked code files from git as prompt context."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=all"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    files: list[str] = []
    for line in out.stdout.splitlines():
        # porcelain: 'XY <path>' (path starts at column 3)
        path = line[3:].strip().strip('"')
        if " -> " in path:  # renames
            path = path.split(" -> ", 1)[1]
        if not path:
            continue
        if Path(path).suffix.lower() in _CODE_SUFFIXES and Path(path).is_file():
            files.append(path)
        if len(files) >= _MAX_AUTO_FILES:
            break
    return files


def _read_context_files(paths: list[str] | None) -> tuple[int, list[dict]]:
    """Return total context tokens and a per-file token breakdown."""
    breakdown: list[dict] = []
    total = 0
    for raw in paths or []:
        path = Path(raw)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            breakdown.append({"file": raw, "tokens": 0, "error": str(exc)})
            continue
        tokens = estimate_tokens(text)
        total += tokens
        breakdown.append({"file": str(path), "tokens": tokens})
    return total, breakdown


def compute_input_tokens(
    prompt: str,
    context_files: list[str] | None = None,
    extra_context_tokens: int = 0,
) -> dict:
    """Tokenize the prompt and its context, before any model-cost math.

    Lets callers (e.g. the output-token estimator) know the total input size
    without duplicating the prompt.optimizer.estimate_tokens plumbing.
    """
    prompt_tokens = estimate_tokens(prompt)
    context_tokens, file_breakdown = _read_context_files(context_files)
    context_tokens += max(extra_context_tokens, 0)
    return {
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "input_tokens": prompt_tokens + context_tokens,
        "file_breakdown": file_breakdown,
    }


def predict_across_models(
    prompt: str,
    context_files: list[str] | None = None,
    expected_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    extra_context_tokens: int = 0,
) -> dict:
    """Predict input/output cost of a prompt for every known model."""
    if expected_output_tokens < 0:
        raise ValueError("expected_output_tokens must be non-negative")

    tokenized = compute_input_tokens(prompt, context_files, extra_context_tokens)
    prompt_tokens = tokenized["prompt_tokens"]
    context_tokens = tokenized["context_tokens"]
    input_tokens = tokenized["input_tokens"]
    file_breakdown = tokenized["file_breakdown"]

    models = []
    for model_id in MODEL_PRICING:
        price = pricing_for(model_id)
        details = model_details(model_id) or {}
        context_window = details.get("context_window")
        input_cost = (input_tokens / 1_000_000) * price["input"]
        output_cost = (expected_output_tokens / 1_000_000) * price["output"]
        models.append({
            "model": model_id,
            "context_window": context_window,
            "fits": (context_window is None) or (input_tokens <= context_window),
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        })
    models.sort(key=lambda m: m["total_cost"])

    return {
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "input_tokens": input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "file_breakdown": file_breakdown,
        "models": models,
        "cheapest": models[0] if models else None,
        "most_expensive": models[-1] if models else None,
    }


def historical_median_output_tokens() -> int | None:
    """Median real output tokens from local chatSessions history, if available."""
    try:
        from app.prompt_analysis import requests_dataframe

        df = requests_dataframe()
        if df.empty:
            return None
        return int(df["output_tokens"].median())
    except Exception:
        return None


# Minimum requests before a repo's own history is trusted over the global pool.
MIN_REPO_SAMPLE = 5


def _scope_history(df, repo_name: str | None):
    """Split a requests dataframe into (scoped, scope_label) for this repo,
    falling back to the global pool when the repo has too few samples."""
    repo = repo_name or Path.cwd().name
    scoped = df[df["workspace"].astype(str).str.casefold() == repo.casefold()]
    if len(scoped) >= MIN_REPO_SAMPLE:
        return scoped, repo
    return df, "all projects"


def realistic_input_estimate(repo_name: str | None = None) -> dict | None:
    """Median real input tokens scoped to the current repo, with a global fallback.

    Returns {tokens, scope, sample_size}. Uses this repo's own past requests when
    it has at least MIN_REPO_SAMPLE of them; otherwise falls back to all projects.
    """
    try:
        from app.prompt_analysis import requests_dataframe

        df = requests_dataframe()
    except Exception:
        return None
    if df.empty:
        return None
    real = df[df["prompt_tokens"] > 0]
    if real.empty:
        return None

    scoped, scope = _scope_history(real, repo_name)
    return {"tokens": int(scoped["prompt_tokens"].median()),
            "scope": scope, "sample_size": int(len(scoped))}


# Phrases that signal a "big ask" -- broad, open-ended work whose output will
# likely be much larger than a typical request, even if the prompt itself is short.
_BIG_ASK_PATTERNS = [
    r"\brevamp\b", r"\brewrite\b", r"\boverhaul\b", r"\brebuild\b", r"\bredesign\b",
    r"\bmigrate\b", r"\bmigration\b", r"\bfrom scratch\b", r"\bentire\b", r"\bwhole app\b",
    r"\bacross the (app|codebase|repo)\b", r"\bnew architecture\b", r"\bend[- ]to[- ]end\b",
    r"\ball pages\b", r"\bevery page\b", r"\bfull(y)? feature\b", r"\brefactor everything\b",
    r"\brevamping\b",
]
_BIG_ASK_RE = [re.compile(p, re.IGNORECASE) for p in _BIG_ASK_PATTERNS]


def detect_big_ask(prompt: str) -> tuple[bool, list[str]]:
    """Heuristic: does this prompt ask for unusually broad/large work?

    Keyword-based, not semantic -- a fast proxy for "this is an outlier task
    that will produce far more output than a typical request." Not a substitute
    for an LLM's judgment of scope, which an agent invoking this CLI can supply
    directly via --output-tokens.
    """
    matches = [re.sub(r"\\b", "", p.pattern) for p in _BIG_ASK_RE if p.search(prompt)]
    return (len(matches) > 0, matches)


# Signals that a prompt is vague/incomplete and likely to need follow-up
# clarification turns (extra rework cost) rather than being resolved in one shot.
# Map each pattern to a human-readable label for the CLI warning message.
_VAGUE_FILLER_PATTERNS: list[tuple[str, str]] = [
    (r"\bfix (this|it)\b(?!.*[`.]\w)", "fix this/it"),
    (r"\bimprove (this|it)\b(?!.*[`.]\w)", "improve this/it"),
    (r"\bmake (this|it) better\b", "make this/it better"),
    (r"\bdo something\b", "do something"),
    (r"\bhandle (this|it)\b(?!.*[`.]\w)", "handle this/it"),
    (r"\boptimi[sz]e (this|it)\b(?!.*[`.]\w)", "optimize this/it"),
    (r"\bclean(\s|-)?up\b(?!.*[`.]\w)", "clean up"),
    (r"\btake a look\b", "take a look"),
    (r"\bwhatever (you think|works)\b", "whatever you think/works"),
]
_VAGUE_FILLER_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _VAGUE_FILLER_PATTERNS]

# A "concrete reference" -- a named file, function, identifier, or quoted span --
# is evidence the prompt has enough specifics to avoid a guessing round-trip.
_CONCRETE_REFERENCE_RE = re.compile(
    r"`[^`]+`|"                       # `code span`
    r"\b\w+\.[a-zA-Z]{1,4}\b|"        # file.ext
    r"\b[a-z]+_[a-z_]+\b|"            # snake_case identifier
    r"\b[a-z]+[A-Z]\w*\b"             # camelCase identifier
)

# Below this typed-instruction length, an unspecific prompt is treated as risky.
_SHORT_PROMPT_TOKEN_THRESHOLD = 12


def detect_vague_prompt(prompt: str) -> tuple[bool, list[str]]:
    """Heuristic: is this prompt likely too vague/incomplete to resolve in one turn?

    Flags prompts that are short, reference "this"/"it" without naming a
    concrete file/function/identifier, or use generic filler ("do something",
    "clean up", "take a look") -- patterns that historically tend to trigger a
    clarifying follow-up (i.e. rework) rather than a single accepted attempt.
    Keyword-based, not semantic -- a fast proxy, not a judgment of intent.
    """
    reasons: list[str] = []
    has_concrete_reference = bool(_CONCRETE_REFERENCE_RE.search(prompt))

    filler_hits = [label for pattern, label in _VAGUE_FILLER_RE if pattern.search(prompt)]
    if filler_hits:
        reasons.append("generic filler phrasing (" + ", ".join(filler_hits) + ")")

    typed_tokens = estimate_tokens(prompt)
    if typed_tokens <= _SHORT_PROMPT_TOKEN_THRESHOLD and not has_concrete_reference:
        reasons.append(f"short ({typed_tokens} tokens) with no named file/function/identifier")

    return (len(reasons) > 0, reasons)


# Percentile of historical output tokens used when a prompt is flagged as a big ask.
BIG_ASK_PERCENTILE = 90


# Percentile of historical output tokens used when a prompt is flagged as a big ask.
BIG_ASK_PERCENTILE = 90


def _ratio_estimate(scoped, scope: str, denom_col: str, scale_tokens: int, method: str) -> dict:
    ratio = (scoped["output_tokens"] / scoped[denom_col].clip(lower=1)).median()
    tokens = int(scale_tokens * ratio)
    lo, hi = int(scoped["output_tokens"].min()), int(scoped["output_tokens"].max())
    tokens = max(lo, min(tokens, hi)) if hi > 0 else tokens
    return {"tokens": max(tokens, 1), "method": method, "scope": scope,
            "sample_size": int(len(scoped))}


def expected_output_tokens_for(
    typed_tokens: int,
    full_input_tokens: int | None = None,
    repo_name: str | None = None,
    big_ask: bool = False,
) -> dict | None:
    """Estimate output tokens, preferring this repo's own real history.

    Priority (each step only used if the previous one lacks enough samples):
    1. This repo's real typed-instruction -> output ratio (best: ignores
       attached context, which doesn't predict output size).
    2. This repo's real full-input -> output ratio (fallback when the repo
       has no captured typed text, e.g. only first-of-session messages log
       it) -- uses `full_input_tokens` (prompt + context) as the scale factor
       to stay unit-consistent with the historical `prompt_tokens` field.
    3. All-projects typed-instruction ratio.
    4. All-projects full-input ratio.
    Big-ask case: at whichever scope is selected, uses a high percentile of
    historical output tokens instead of ratio-scaling, since broad asks
    produce large output regardless of instruction length.
    """
    try:
        from app.prompt_analysis import requests_dataframe

        df = requests_dataframe()
    except Exception:
        return None
    if df.empty:
        return None

    repo = repo_name or Path.cwd().name
    typed_df = df[(df["prompt_text_tokens"] > 0) & (df["output_tokens"] >= 0)] \
        if "prompt_text_tokens" in df.columns else df.iloc[0:0]
    full_df = df[(df["prompt_tokens"] > 0) & (df["output_tokens"] >= 0)] \
        if "prompt_tokens" in df.columns else df.iloc[0:0]

    candidates = []
    if not typed_df.empty:
        repo_typed = typed_df[typed_df["workspace"].astype(str).str.casefold() == repo.casefold()]
        if len(repo_typed) >= MIN_REPO_SAMPLE:
            candidates.append((repo_typed, repo, "prompt_text_tokens", typed_tokens, "typed-length ratio"))
    if full_input_tokens is not None and not full_df.empty:
        repo_full = full_df[full_df["workspace"].astype(str).str.casefold() == repo.casefold()]
        if len(repo_full) >= MIN_REPO_SAMPLE:
            candidates.append((repo_full, repo, "prompt_tokens", full_input_tokens,
                               "full-input ratio (no typed-text history for this repo)"))
    if not typed_df.empty:
        candidates.append((typed_df, "all projects", "prompt_text_tokens", typed_tokens, "typed-length ratio"))
    if full_input_tokens is not None and not full_df.empty:
        candidates.append((full_df, "all projects", "prompt_tokens", full_input_tokens,
                           "full-input ratio"))

    if not candidates:
        return None

    scoped, scope, denom_col, scale_tokens, method = candidates[0]

    if big_ask:
        tokens = int(scoped["output_tokens"].quantile(BIG_ASK_PERCENTILE / 100))
        return {"tokens": max(tokens, 1), "method": f"p{BIG_ASK_PERCENTILE} (big-ask)",
                "scope": scope, "sample_size": int(len(scoped))}

    return _ratio_estimate(scoped, scope, denom_col, scale_tokens, method)
