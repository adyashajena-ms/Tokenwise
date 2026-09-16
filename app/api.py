"""FastAPI service exposing the yield ledger and presales estimator."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.estimator import estimate_presales_cost
from app.ledger import trace_case, yield_ledger_by_task_type
from app.prompt_optimizer import optimize_prompt, predict_prompt_cost

app = FastAPI(title="TokenWise")


class EstimateRequest(BaseModel):
    task_type: str
    volume: int
    complexity_tier: str | None = None


class PromptRequest(BaseModel):
    prompt: str


@app.get("/ledger")
def get_ledger():
    df = yield_ledger_by_task_type()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data. Run app/data_gen.py first.")
    return df.to_dict(orient="records")


@app.get("/trace/{case_id}")
def get_trace(case_id: int):
    df = trace_case(case_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No case found with id={case_id}")
    return df.to_dict(orient="records")


@app.post("/estimate")
def post_estimate(req: EstimateRequest):
    try:
        return estimate_presales_cost(
            task_type=req.task_type,
            volume=req.volume,
            complexity_tier=req.complexity_tier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict-tokens")
def post_predict_tokens(req: PromptRequest):
    return predict_prompt_cost(req.prompt)


@app.post("/optimize-prompt")
def post_optimize_prompt(req: PromptRequest):
    return optimize_prompt(req.prompt)
