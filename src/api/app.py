"""
FastAPI REST Service for Aiqon Shipment Route Optimization Engine.
"""

from __future__ import annotations
import json
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.models.schemas import (
    Shipment,
    OperatorProfile,
    OptimizationWeights,
    RecommendationResult,
)
from src.optimization.scoring import RouteOptimizer
from src.explainer.reasoning import RouteDecisionExplainer
from src.data.generator import RouteGenerator
from src.evaluation.benchmark import BenchmarkSuite

app = FastAPI(
    title="Aiqon Multi-Objective Shipment Route Optimizer API",
    description="Intelligent route recommendation balancing cost, transit time, delay risk, and resilience.",
    version="1.0.0"
)

# In-memory storage / cache
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "shipments.json")
optimizer = RouteOptimizer()
generator = RouteGenerator(seed=42)
benchmark_suite = BenchmarkSuite(optimizer=optimizer)

# Load or generate dataset
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        shipments_db: Dict[str, Shipment] = {s["shipment_id"]: Shipment.model_validate(s) for s in raw_data}
else:
    generated = generator.generate_dataset(40)
    shipments_db = {s.shipment_id: s for s in generated}


class RecommendRequest(BaseModel):
    shipment_id: str
    operator_profile: OperatorProfile = OperatorProfile.BALANCED
    cost_weight: Optional[float] = None
    time_weight: Optional[float] = None
    risk_weight: Optional[float] = None
    max_budget_usd: Optional[float] = None
    max_transit_days: Optional[float] = None
    max_acceptable_risk: Optional[float] = None


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Aiqon Shipment Route Optimizer", "total_shipments": len(shipments_db)}


@app.get("/shipments")
def list_shipments():
    """Returns summary list of all shipments."""
    summaries = []
    for s in shipments_db.values():
        summaries.append({
            "shipment_id": s.shipment_id,
            "cargo_type": s.cargo_type.value,
            "origin": s.origin_city,
            "destination": s.destination_city,
            "cargo_value_usd": s.cargo_value_usd,
            "route_count": len(s.route_options)
        })
    return summaries


@app.get("/shipments/{shipment_id}", response_model=Shipment)
def get_shipment(shipment_id: str):
    """Returns full shipment details with all route options."""
    if shipment_id not in shipments_db:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")
    return shipments_db[shipment_id]


@app.post("/recommend", response_model=RecommendationResult)
def recommend_route(req: RecommendRequest):
    """Recommends optimal route and generates operator explainability breakdown."""
    if req.shipment_id not in shipments_db:
        raise HTTPException(status_code=404, detail=f"Shipment {req.shipment_id} not found")

    shipment = shipments_db[req.shipment_id]

    custom_weights = None
    if req.operator_profile == OperatorProfile.CUSTOM or (req.cost_weight is not None and req.time_weight is not None):
        custom_weights = OptimizationWeights(
            cost_weight=req.cost_weight if req.cost_weight is not None else 0.35,
            time_weight=req.time_weight if req.time_weight is not None else 0.35,
            risk_weight=req.risk_weight if req.risk_weight is not None else 0.30,
            max_budget_usd=req.max_budget_usd,
            max_transit_days=req.max_transit_days,
            max_acceptable_risk=req.max_acceptable_risk
        )

    res = optimizer.recommend(
        shipment=shipment,
        profile=req.operator_profile,
        custom_weights=custom_weights,
        run_monte_carlo=True
    )

    enriched = RouteDecisionExplainer.explain_recommendation(res, shipment)
    return enriched


@app.get("/benchmark")
def run_benchmark():
    """Runs comparative evaluation of proposed optimizer against baseline heuristics."""
    sample = list(shipments_db.values())
    benchmark_res = benchmark_suite.evaluate_shipments(sample)
    # Convert dataframe to dict
    table_records = benchmark_res["summary_table"].to_dict(orient="records")
    return {"benchmark_summary": table_records}
