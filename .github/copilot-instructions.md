# Copilot instructions for TokenWise

## Predict cost before large edits
Before implementing a non-trivial change the user requests (new feature, refactor
touching multiple files), FIRST estimate its Copilot credit cost so the user can
choose a model:

```powershell
venv\Scripts\python.exe -m app.predict "<short description of the task>" --auto --realistic
```

`--auto` pulls context from git-changed files; `--realistic` adds your real median
per-request context (referenced files + history + tools), which is far larger than
edited files alone. Output size comes from your history. Report the cheapest model
and the spread, then proceed with the implementation.

Skip this for trivial edits (one-line fixes, typos, comments).

## Cost model
- Costs are in Copilot credits per 1M tokens (see `app/model_pricing.py`), the same
  units shown in the VS Code model picker.
- Real per-model rates and context windows are baked into `app/model_pricing.py`;
  regenerate with `python -m tools.refresh_model_catalog`.
