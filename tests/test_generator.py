"""
Unit tests for Synthetic Route Dataset Generator.
"""

import pytest
from src.data.generator import RouteGenerator
from src.models.schemas import Shipment, Route, Leg


def test_generator_shipment_count():
    gen = RouteGenerator(seed=123)
    shipments = gen.generate_dataset(num_shipments=10)
    assert len(shipments) == 10
    for s in shipments:
        assert isinstance(s, Shipment)
        assert len(s.route_options) >= 3
        for r in s.route_options:
            assert isinstance(r, Route)
            assert len(r.legs) >= 1
            assert r.total_cost_usd > 0
            assert r.total_transit_time_days > 0
            assert 0.0 <= r.overall_delay_probability <= 1.0
            assert 0.0 <= r.composite_risk_score <= 1.0


def test_trade_offs_exist():
    """Verify that routes within a shipment exhibit real trade-offs (e.g. cheapest is not fastest)."""
    gen = RouteGenerator(seed=42)
    shp = gen.generate_shipment(1, "Shanghai", "Rotterdam")
    routes = shp.route_options

    costs = [r.total_cost_usd for r in routes]
    times = [r.total_transit_time_days for r in routes]

    # There should be variance in costs and times
    assert min(costs) < max(costs)
    assert min(times) < max(times)

    # Fastest route should not be the cheapest route
    cheapest = min(routes, key=lambda r: r.total_cost_usd)
    fastest = min(routes, key=lambda r: r.total_transit_time_days)
    assert cheapest.route_id != fastest.route_id
