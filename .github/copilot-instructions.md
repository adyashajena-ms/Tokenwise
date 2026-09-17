# Copilot instructions for TokenWise

## Check for vague/incomplete prompts first
Before acting on a non-trivial user request, judge whether it is vague or
incomplete (missing a target file/function, ambiguous scope, e.g. "fix it",
"clean this up", "make it better"). If so, ask a brief clarifying question or
state the concrete interpretation you're proceeding with, before implementing --
this is cheaper than a wrong-guess implementation that needs rework. Use your
own judgment here; `app/cost_predictor.py`'s `detect_vague_prompt()` is only a
lightweight keyword heuristic for the standalone CLI, used automatically when
no LLM is configured (see below), not a substitute for this review.

### Optional: LLM-based review for the standalone CLI
`python -m app.predict` can call Azure OpenAI directly to judge vagueness and
suggest a rewrite, instead of the keyword heuristic. It activates automatically
when these environment variables are set (in the user's own shell -- **never**
pass an API key as a CLI argument or paste it into chat):

```powershell
$env:AZURE_OPENAI_API_KEY = "..."
$env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "<deployment name>"
```

If unset, `app/llm_review.py`'s `llm_available()` returns False and the CLI
falls back to the heuristic automatically -- no code change needed either way.

## Predict cost before large edits
Before implementing a non-trivial change the user requests (new feature, refactor
touching multiple files), FIRST estimate its Copilot credit cost so the user can
choose a model:

```powershell
venv\Scripts\python.exe -m app.predict "<short description of the task>" --realistic
```

`--realistic` estimates input from your repo's real median per-request context
(referenced files + history + tools, far larger than just the edited files) and
estimates output from your repo's real input->output relationship, boosting the
estimate automatically if the task looks like a broad "big ask" (rewrite, revamp,
migrate, etc.). Report the cheapest model and the spread, then proceed.

If you already know the exact files involved, pass them explicitly instead --
it's more accurate than the historical estimate:

```powershell
venv\Scripts\python.exe -m app.predict "<task>" --file path\to\a.py --file path\to\b.py
```

Skip this for trivial edits (one-line fixes, typos, comments).

## Cost model
- Costs are in Copilot credits per 1M tokens (see `app/model_pricing.py`), the same
  units shown in the VS Code model picker.
- Real per-model rates and context windows are baked into `app/model_pricing.py`;
  regenerate with `python -m tools.refresh_model_catalog`.
