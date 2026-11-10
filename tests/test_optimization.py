"""
Unit tests for Pareto Frontier, TOPSIS, and Route Optimizer.
"""

import pytest
from src.data.generator import RouteGenerator
from src.optimization.pareto import find_pareto_frontier
from src.optimization.topsis import calculate_topsis_scores
from src.optimization.scoring import RouteOptimizer
from src.optimization.simulation import MonteCarloSimulator
from src.models.schemas import OptimizationWeights, OperatorProfile


@pytest.fixture
def sample_shipment():
    gen = RouteGenerator(seed=999)
    return gen.generate_shipment(1, "Shanghai", "Rotterdam")


def test_pareto_frontier(sample_shipment):
    pareto_ids = find_pareto_frontier(sample_shipment.route_options)
    assert len(pareto_ids) > 0
    # Pareto frontier size cannot exceed total routes
    assert len(pareto_ids) <= len(sample_shipment.route_options)


def test_topsis_scoring(sample_shipment):
    weights = OptimizationWeights(cost_weight=0.5, time_weight=0.3, risk_weight=0.2)
    scores = calculate_topsis_scores(sample_shipment.route_options, weights)
    assert len(scores) == len(sample_shipment.route_options)
    for r_id, sc in scores.items():
        assert 0.0 <= sc <= 1.0


def test_monte_carlo_simulation(sample_shipment):
    sim = MonteCarloSimulator(num_simulations=200, seed=42)
    results = sim.simulate_shipment(sample_shipment)
    assert len(results) == len(sample_shipment.route_options)
    for r_id, stats in results.items():
        assert 0.0 <= stats["otif_probability"] <= 1.0
        assert stats["mean_transit_days"] > 0
        assert stats["expected_realized_cost_usd"] >= stats["nominal_cost_usd"]


def test_route_optimizer_profiles(sample_shipment):
    optimizer = RouteOptimizer()

    # Cost minimizer profile should favor lowest cost
    rec_cost = optimizer.recommend(sample_shipment, profile=OperatorProfile.COST_MINIMIZER)
    assert rec_cost.recommended_route is not None

    # Time critical profile should favor lowest transit time
    rec_time = optimizer.recommend(sample_shipment, profile=OperatorProfile.TIME_CRITICAL)
    assert rec_time.recommended_route is not None

    # Recommendations under different objectives should adapt to priorities
    assert rec_time.recommended_route.total_transit_time_days <= rec_cost.recommended_route.total_transit_time_days


def test_hard_constraint_filtering(sample_shipment):
    optimizer = RouteOptimizer()
    min_cost = min(r.total_cost_usd for r in sample_shipment.route_options)

    # Set strict budget constraint that only the cheapest route can satisfy
    strict_weights = OptimizationWeights(
        cost_weight=0.5, time_weight=0.5, risk_weight=0.0,
        max_budget_usd=min_cost + 100.0
    )
    rec = optimizer.recommend(sample_shipment, profile=OperatorProfile.CUSTOM, custom_weights=strict_weights)
    assert rec.recommended_route.total_cost_usd <= min_cost + 100.0
