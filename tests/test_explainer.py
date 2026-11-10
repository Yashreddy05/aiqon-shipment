"""
Unit tests for Explainability Engine.
"""

import pytest
from src.data.generator import RouteGenerator
from src.optimization.scoring import RouteOptimizer
from src.explainer.reasoning import RouteDecisionExplainer
from src.models.schemas import OperatorProfile


def test_explainer_narrative():
    gen = RouteGenerator(seed=42)
    shipment = gen.generate_shipment(2, "Shenzhen", "Los Angeles")
    optimizer = RouteOptimizer()

    rec_res = optimizer.recommend(shipment, profile=OperatorProfile.BALANCED)
    enriched = RouteDecisionExplainer.explain_recommendation(rec_res, shipment)

    # Assertions
    assert enriched.summary_explanation != ""
    assert len(enriched.summary_explanation) > 50
    assert "recommended" in enriched.summary_explanation.lower()
    assert enriched.trade_off_breakdown is not None
    assert "cost_difference_usd" in enriched.trade_off_breakdown
    assert "risk_analysis" in enriched.model_dump()
    assert "composite_risk" in enriched.risk_analysis
