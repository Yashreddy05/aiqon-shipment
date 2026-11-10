"""
Functional tests for API routing endpoints.
Directly tests endpoint logic to prevent socket blocking in sandboxed environments.
"""

import pytest
from src.api.app import health_check, list_shipments, get_shipment, recommend_route, RecommendRequest
from src.models.schemas import OperatorProfile


def test_health():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["total_shipments"] > 0


def test_list_shipments():
    shipments = list_shipments()
    assert len(shipments) > 0
    assert "shipment_id" in shipments[0]
    assert "origin" in shipments[0]
    assert "destination" in shipments[0]


def test_get_single_shipment():
    shp = get_shipment("SHP-001")
    assert shp.shipment_id == "SHP-001"
    assert len(shp.route_options) >= 3


def test_recommend_endpoint():
    req = RecommendRequest(
        shipment_id="SHP-001",
        operator_profile=OperatorProfile.BALANCED
    )
    result = recommend_route(req)
    assert result.shipment_id == "SHP-001"
    assert result.recommended_route is not None
    assert len(result.summary_explanation) > 30
    assert len(result.rankings) >= 3
    assert result.trade_off_breakdown is not None
