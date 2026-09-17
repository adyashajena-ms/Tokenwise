# TokenWise

Prototype for the InSpireD "AI cost/yield" hackathon challenge.

AI spend is easy to measure but hard to justify. Tokens get burned on retries, rework, and
oversized context. **TokenWise** turns local VS Code Copilot activity into **real token and
credit accounting** — showing which project and feature spent what, which prompts were
efficient, and what a task will cost *before* you send it, so you can pick the cheapest model
that fits.

## Three core pillars

1. **Post-Dev Token Analysis** — reads real prompt/output token counts from local VS Code
   `chatSessions` and reports **credits per project folder and per feature**, a good/bad
   prompt breakdown to learn from, and how much a cheaper model would have saved.
   *(`app/chat_sessions.py`, `app/prompt_analysis.py`, dashboard "Post Dev Analysis" page)*
2. **Pre-Dev Cost Predictor (CLI)** — predicts a prompt's **credit cost across every model**
   before you send it, counting real input tokens (prompt + attached files) and sizing output
   from your history. *(`app/predict.py`, `app/cost_predictor.py`, `app/model_pricing.py`)*
3. **Prompt Token Optimizer** — predicts a prompt's token count and cost before you send it,
   then trims it with meaning-preserving rewrites so fewer tokens are spent without
   compromising output. *(`app/prompt_optimizer.py`, dashboard "Prompt Token Optimizer" page)*

> Two earlier pillars — **Yield Ledger** (`app/ledger.py`) and **Presales Estimator**
> (`app/estimator.py`) — are currently **unplugged from the UI** (kept in
> `dashboard/_unplugged/`) and can be re-plugged by moving their page back into
> `dashboard/pages/`.

## Challenge asks -> what's built

| Ask | Status | Where |
|---|---|---|
| Real cost per project/feature, traceable from spend to artifact | **Built (core)** | `app/prompt_analysis.py`, "Post Dev Analysis" page |
| Predict a task's cost before sending, choose a cheaper model | **Built (core)** | `app/predict.py`, `app/cost_predictor.py` |
| Token optimization: predict a prompt's token/cost before sending, and trim it without changing the ask | **Built (core)** | `app/prompt_optimizer.py`, "Prompt Token Optimizer" page |
| Yield ledger / presales estimator (cost-per-accepted-outcome, P10/P50/P90) | **Built, unplugged** | `dashboard/_unplugged/` |
| Evidence standards (described/proposed vs validated) | **Applied throughout** | estimates state their assumptions; rates are the real Copilot credit rates where known, inferred otherwise |

## Data

TokenWise reads **real** local VS Code Copilot data — nothing leaves your machine:

- **chatSessions** (`app/chat_sessions.py`) — reconstructs the event-sourced
  `%APPDATA%\Code\User\workspaceStorage\*\chatSessions\*.jsonl` logs into per-request
  records with **real** prompt/output token counts, the model used, cache/reasoning tokens,
  and the files touched. This powers the Post-Dev analysis and the pre-dev predictor's
  history-based estimates. Cost is reported in **Copilot credits per 1M tokens** (the same
  units as the model picker); per-model rates live in `app/model_pricing.py`.
- **Legacy transcript proxy** (`app/ingest_vscode_sessions.py`) — used by the unplugged
  Yield Ledger; approximates cost from transcript text (~chars/4 tokens). Kept for
  compatibility with the ledger/estimator.

## Run it

```powershell
pip install -r requirements.txt
streamlit run dashboard/Home.py     # dashboard: Real VS Code Activity, Post Dev Analysis, Optimizer
```

If you want `app.predict` to use the LLM-based vague/incomplete-prompt review
instead of the local heuristic, set these environment variables in the shell
you use to run the command:

```powershell
$env:AZURE_OPENAI_API_KEY = "<your-key>"
$env:AZURE_OPENAI_ENDPOINT = "https://tokenwise.services.ai.azure.com/api/projects/tokenwise"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"
```

Optional, if your Foundry resource needs a different API version:

```powershell
$env:AZURE_OPENAI_API_VERSION = "2024-06-01"
```

Predict a prompt's credit cost across models before you send it (CLI, no dashboard):

```powershell
# just the prompt
venv\Scripts\python.exe -m app.predict "add a per-customer breakdown"

# with the files you'll touch + realistic context from your history (most accurate)
venv\Scripts\python.exe -m app.predict "add a per-customer breakdown" --file app/models.py --realistic

# zero-config: pull context from git-changed files, size output from history
venv\Scripts\python.exe -m app.predict "add a per-customer breakdown" --auto --realistic
```

Or run it from VS Code: **Ctrl+Shift+P -> Run Task -> "Predict prompt cost"**.
Copilot also runs it automatically before non-trivial edits (see `.github/copilot-instructions.md`).

## Tests

```powershell
pytest
```

## Model notes

- **Post-dev token analysis**: reconstructs `chatSessions` event logs into per-request rows
  with real prompt/output tokens, model, and files touched; aggregates **credits per project
  and per feature** (one chat session = one feature, labeled by its title with the actual
  files touched shown as grounded evidence). Prompt quality is a heuristic (wasted / bloated /
  efficient), not a validated classifier.
- **Pre-dev cost predictor**: counts real input tokens (exact BPE via `tiktoken`) for the
  prompt plus any attached/`--auto` git-context files, sizes output from your history
  (`--use-history`/`--realistic`), and prices every model in `app/model_pricing.py`. `--realistic`
  adds your real median per-request context (referenced files + history + tools), which is far
  larger than edited files alone. Flags models whose context window can't hold the input.
- **Prompt token optimizer**: predicts a prompt's token count and cost *before* it is sent,
  then trims it using only meaning-preserving rewrites — collapsing redundant whitespace and
  dropping non-instructional filler. It reports each rule applied and the tokens it saved, and
  never rephrases the substantive request. Heuristic — review the optimized prompt before
  relying on it.
- **Pricing**: credits per 1M tokens, matching the VS Code model picker. Known rates are real
  (extracted from local model metadata); a few are inferred from a sibling model and marked as
  such. Regenerate with `python -m tools.refresh_model_catalog`.