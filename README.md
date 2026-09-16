# TokenWise

Prototype for the InSpireD "AI cost/yield" hackathon challenge.

AI spend is easy to measure but hard to justify. Tokens get burned on retries, rework, and
abandoned attempts that never ship. **TokenWise** connects **spend → accepted outcome**, so
every dollar is traceable to the artifact it produced (or flagged as waste when nothing
shipped) — and it cuts waste at the source by predicting and trimming a prompt's token cost
*before* the prompt is ever sent.

## Three core pillars

1. **Yield Ledger** — cost per accepted outcome, traced from spend to artifact, with a waste
   breakdown (retries / rework / abandonment). *(`app/ledger.py`)*
2. **Presales Estimator** — a defensible P10/P50/P90 cost interval for a new engagement, with
   its assumptions stated alongside it. *(`app/estimator.py`)*
3. **Prompt Token Optimizer** — predicts a prompt's token count and cost before you send it,
   then trims it with meaning-preserving rewrites so fewer tokens are spent without
   compromising output. This is the *prevent-waste-before-it-happens* half of the
   spend→outcome story. *(`app/prompt_optimizer.py`)*

## Challenge asks -> what's built

| Ask | Status | Where |
|---|---|---|
| Yield ledger: cost per accepted outcome, traceable from spend to artifact | **Built (core)** | `app/ledger.py`, dashboard "Yield Ledger" page |
| Estimation model: defensible intervals at presales, assumptions stated | **Built (core)** | `app/estimator.py`, dashboard "Presales Estimator" page |
| Token optimization: predict a prompt's token/cost before sending, and trim it without changing the ask | **Built (core)** | `app/prompt_optimizer.py`, dashboard "Prompt Token Optimizer" page |
| Risk-classification rubric | Not built (stretch, out of scope for this pass) | — |
| Chargeback mapping into a FinOps model | Not built (stretch, out of scope for this pass) | — |
| Evidence standards (described/proposed vs validated) | **Applied throughout** | every estimate returns an `assumptions` block; nothing is claimed as validated |

## Data

Two sources feed the same `Engagement`/`Case`/`Attempt` tables:

- **Synthetic** (`app/data_gen.py`) — three task-type archetypes with different cost,
  retry, and abandonment profiles so the ledger/estimator have realistic variance to show.
- **Real** (`app/ingest_vscode_sessions.py`) — scans local VS Code Copilot chat transcripts
  from every workspace on this machine (`%APPDATA%\Code\User\workspaceStorage\*\GitHub.copilot-chat\transcripts\*.jsonl`,
  never sent anywhere) and turns them into the same schema: one `Case` per user request,
  one `Attempt` per assistant turn, "rework" = the same file edited more than once within
  a request. Cost is a **proxy** (~chars/4 tokens x an assumed $/1K-token rate) since local
  transcripts don't record actual billed token counts. Run via the "Real VS Code Activity"
  dashboard page or `python -m app.ingest_vscode_sessions` (safe to re-run — replaces prior
  real-data rows, keeps synthetic rows).

## Run it

```powershell
pip install -r requirements.txt
python -m app.data_gen          # generates hack2026.db
streamlit run dashboard/Home.py # dashboard (also lets you regenerate data)
uvicorn app.api:app --reload    # optional: REST API on http://127.0.0.1:8000
```

## Tests

```powershell
pytest
```

## Model notes

- **Yield ledger**: for each accepted case, cost is the sum of every attempt (including
  rejected retries) plus human rework cost — full spend attributed to the artifact that
  was actually accepted. Abandoned cases contribute their full spend as waste, attributed
  to no outcome.
- **Presales estimator**: empirical bootstrap Monte Carlo over historical accepted-outcome
  costs for the selected task type, optionally scaled by a stated (uncalibrated)
  complexity-tier multiplier. This is a *proposed* method — every result includes the
  sample size, simulation count, and confidence caveat it was built from.
- **Prompt token optimizer**: predicts a prompt's token count and cost *before* it is sent
  (exact BPE count via `tiktoken` when installed, otherwise the same chars/4 proxy the
  ledger uses), then trims it using only meaning-preserving rewrites — collapsing redundant
  whitespace and dropping non-instructional filler/politeness. It reports each rule applied
  and the tokens it saved, and never rephrases the substantive request, so model output is
  not compromised. Heuristic — review the optimized prompt before relying on it.
