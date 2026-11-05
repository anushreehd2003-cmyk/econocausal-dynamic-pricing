"""REST API for EconoCausal: predicted discount effects and budget allocation.

Start it from the project root:
    python -m uvicorn api:app --app-dir backend\\src --port 8000
Interactive docs: http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from model_store import (
    LEVELS,
    load_bundle,
    model_path,
    predict_effects,
    range_warnings,
    version_warnings,
)
from drift import MIN_BATCH, drift_report, untreated_counts
from optimizer import evaluate, greedy_allocation, optimize_allocation, values_from_ite

MAX_CUSTOMERS = 10_000
EXACT_TIME_LIMIT = 20  # seconds for the exact solver


class Customer(BaseModel):
    customer_id: int | str | None = None
    age: float = Field(ge=0, le=120)
    income: float = Field(ge=0)
    loyalty_member: int = Field(ge=0, le=1)
    past_purchases: float = Field(ge=0)
    days_since_last_purchase: float = Field(ge=0)


class CustomerWithBasket(Customer):
    avg_basket_value: float = Field(ge=0)


class DriftCustomer(Customer):
    discount: Literal[0, 10, 20] | None = None
    purchased: Literal[0, 1] | None = None


class DriftRequest(BaseModel):
    customers: list[DriftCustomer] = Field(min_length=MIN_BATCH, max_length=MAX_CUSTOMERS)


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_CUSTOMERS)


class AllocateRequest(BaseModel):
    customers: list[CustomerWithBasket] = Field(min_length=1, max_length=MAX_CUSTOMERS)
    budget: float = Field(gt=0)
    levels: list[int] = Field(default_factory=lambda: list(LEVELS))
    method: Literal["greedy", "exact"] = "greedy"

    @field_validator("levels")
    @classmethod
    def check_levels(cls, value):
        if not value or len(set(value)) != len(value) or not set(value) <= set(LEVELS):
            raise ValueError(
                "levels must be distinct values chosen from " + str(list(LEVELS))
            )
        return sorted(value)


def to_matrix(customers, features):
    return np.array([[getattr(c, f) for f in features] for c in customers], dtype=float)


def create_app(bundle=None):
    """Build the app. Pass a model bundle to skip loading it from disk."""

    @asynccontextmanager
    async def lifespan(app):
        if app.state.bundle is None:
            path = model_path()
            app.state.bundle = load_bundle(path) if path.exists() else None
        yield

    app = FastAPI(title="EconoCausal API", version="1.0.0", lifespan=lifespan)
    app.state.bundle = bundle
    app.state.ref_effects = None  # reference predictions, computed on first /drift call

    origins = os.environ.get(
        "ECONOCAUSAL_CORS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def get_bundle(request: Request):
        if request.app.state.bundle is None:
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run: python backend\\src\\train_final_model.py",
            )
        return request.app.state.bundle

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health(request: Request):
        b = request.app.state.bundle
        if b is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "model_missing",
                    "model_loaded": False,
                    "expected_path": str(model_path()),
                },
            )
        return {
            "status": "ok",
            "model_loaded": True,
            "model_kind": b["kind"],
            "trained_at": b["trained_at"],
            "n_training_rows": b["n_training_rows"],
            "levels": b["levels"],
            "features": b["features"],
            "library_versions": b["library_versions"],
            "drift_reference_available": "reference" in b,
            "warnings": version_warnings(b),
        }

    @app.post("/predict")
    def predict(req: PredictRequest, request: Request):
        b = get_bundle(request)
        X = to_matrix(req.customers, b["features"])
        effects = predict_effects(b, X, intervals=True)
        rows = []
        for i, c in enumerate(req.customers):
            row = {"index": i, "customer_id": c.customer_id}
            for key, values in effects.items():
                row[key] = round(float(values[i]), 5)
            rows.append(row)
        return {
            "model_kind": b["kind"],
            "predictions": rows,
            "warnings": range_warnings(b, X),
        }

    @app.post("/allocate")
    def allocate(req: AllocateRequest, request: Request):
        b = get_bundle(request)
        levels = tuple(req.levels)  # sorted ascending by the validator
        X = to_matrix(req.customers, b["features"])
        basket = np.array([c.avg_basket_value for c in req.customers], dtype=float)

        effects = predict_effects(b, X)
        ite = np.column_stack([effects[f"ite_{lv}"] for lv in levels])
        values = values_from_ite(ite, basket)

        if req.method == "exact":
            alloc = optimize_allocation(
                values, req.budget, levels, time_limit=EXACT_TIME_LIMIT
            )
        else:
            alloc = greedy_allocation(values, req.budget, levels)
        summary = evaluate(alloc, ite, basket, levels)

        # Baseline: the lowest allowed discount to randomly chosen customers
        reach = min(len(basket), int(req.budget // levels[0]))
        blanket = float(values[:, 0].mean() * reach)

        rows = []
        for i, c in enumerate(req.customers):
            level = int(alloc[i])
            effect = float(ite[i, levels.index(level)]) if level else 0.0
            rows.append({
                "index": i,
                "customer_id": c.customer_id,
                "discount": level,
                "predicted_effect": round(effect, 5),
                "predicted_extra_revenue": round(effect * float(basket[i]), 2),
            })
        return {
            "method": req.method,
            "budget": req.budget,
            "spend": summary["spend"],
            "customers_targeted": summary["customers_targeted"],
            "n_10": int((alloc == 10).sum()),
            "n_20": int((alloc == 20).sum()),
            "predicted_extra_purchases": round(summary["extra_purchases"], 3),
            "predicted_extra_revenue": round(summary["extra_revenue"], 2),
            "blanket_expected_revenue": round(blanket, 2),
            "allocation": rows,
            "warnings": range_warnings(b, X),
        }

    @app.post("/drift")
    def drift(req: DriftRequest, request: Request):
        b = get_bundle(request)
        ref = b.get("reference")
        if ref is None:
            raise HTTPException(
                status_code=409,
                detail="This model file has no reference data. "
                       "Rerun: python backend\\src\\train_final_model.py",
            )
        X = to_matrix(req.customers, b["features"])
        if request.app.state.ref_effects is None:
            request.app.state.ref_effects = predict_effects(b, ref["X"])
        cur_effects = predict_effects(b, X)

        def column(name):
            values = [getattr(c, name) for c in req.customers]
            return np.array([np.nan if v is None else v for v in values], dtype=float)

        return drift_report(
            b["features"], ref["X"], X,
            request.app.state.ref_effects, cur_effects,
            (ref["untreated_buys"], ref["untreated_n"]),
            untreated_counts(column("discount"), column("purchased")),
        )

    return app


app = create_app()