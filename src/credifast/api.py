"""FastAPI adapter for the CrediFast core service."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .model_runtime import LocalModelRuntime, get_local_model_runtime
from .service import evaluate_application


class ScoreRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    application_id: str = Field(min_length=1, max_length=100)
    requested_amount: float = Field(gt=0)
    annual_income: float = Field(gt=0)
    annual_annuity: float = Field(gt=0)
    family_size: int = Field(ge=1)
    employment_years: float | None = Field(default=None, ge=0)
    existing_monthly_obligations: float = Field(ge=0)
    active_accounts: int = Field(ge=0)
    total_outstanding: float = Field(ge=0)
    credit_utilization: float | None = Field(default=None, ge=0, le=5)
    recent_inquiries: int = Field(ge=0)
    delinquent_accounts: int = Field(ge=0)
    bureau_history_months: int | None = Field(default=None, ge=0)
    data_coverage: float = Field(ge=0, le=1)


app = FastAPI(
    title="CrediFast API",
    version="0.1.0",
    description=(
        "Hackathon credit-risk decision-support API. Trained-model endpoints use "
        "frozen local artifacts and remain research-only."
    ),
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    try:
        get_local_model_runtime()
        model_ready = True
    except (FileNotFoundError, RuntimeError, ValueError):
        model_ready = False
    return {
        "status": "ok" if model_ready else "degraded",
        "version": "0.1.0",
        "model_ready": model_ready,
    }


@app.post("/v1/score")
def score(request: ScoreRequest) -> dict:
    try:
        return evaluate_application(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ModelScoreRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    application_id: int = Field(gt=0)
    annual_income: float | None = Field(default=None, gt=0)
    requested_credit: float | None = Field(default=None, gt=0)
    annual_annuity: float | None = Field(default=None, gt=0)
    goods_price: float | None = Field(default=None, gt=0)
    explain: bool = True
    simulate_history_unavailable: bool = False


def model_runtime_dependency() -> LocalModelRuntime:
    return get_local_model_runtime()


ModelRuntime = Annotated[LocalModelRuntime, Depends(model_runtime_dependency)]


@app.get("/v1/model/status")
def model_status(runtime: ModelRuntime) -> dict[str, Any]:
    return runtime.status()


@app.get("/v1/model/applicants")
def model_applicants(
    runtime: ModelRuntime,
) -> dict[str, Any]:
    return {"profiles": runtime.profiles(), "count": len(runtime.profiles())}


@app.get("/v1/model/input-schema")
def model_input_schema(runtime: ModelRuntime) -> dict[str, Any]:
    return runtime.input_schema()


@app.post("/v1/model/score")
def model_score(
    request: ModelScoreRequest,
    runtime: ModelRuntime,
) -> dict[str, Any]:
    overrides = {
        "annual_income": request.annual_income,
        "requested_credit": request.requested_credit,
        "annual_annuity": request.annual_annuity,
        "goods_price": request.goods_price,
    }
    try:
        return runtime.score(
            request.application_id,
            overrides=overrides,
            explain=request.explain,
            simulate_history_unavailable=request.simulate_history_unavailable,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
