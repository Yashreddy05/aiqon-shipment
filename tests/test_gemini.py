"""
Unit tests for Gemini Route Advisor fallback and integration.
"""

import pytest
from src.explainer.gemini_explainer import GeminiRouteAdvisor
from src.data.generator import RouteGenerator
from src.optimization.scoring import RouteOptimizer
from src.models.schemas import OperatorProfile


def test_gemini_fallback_when_no_key():
    advisor = GeminiRouteAdvisor(api_key=None)
    assert not advisor.is_available()

    gen = RouteGenerator(seed=42)
    shipment = gen.generate_shipment(1, "Shanghai", "Rotterdam")
    optimizer = RouteOptimizer()
    rec = optimizer.recommend(shipment, profile=OperatorProfile.BALANCED)

    # Calling without key returns None cleanly without throwing exceptions
    briefing = advisor.generate_executive_briefing(rec, shipment)
    assert briefing is None

    ans = advisor.answer_operator_question("What is this route?", rec, shipment)
    assert "not configured" in ans
