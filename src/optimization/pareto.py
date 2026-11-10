"""
Pareto Optimality and Non-Dominated Frontier Computation for Logistics Routes.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
from src.models.schemas import Route


def find_pareto_frontier(routes: List[Route]) -> List[str]:
    """
    Computes the Pareto frontier among a set of routes.
    Objectives to minimize:
    1. total_cost_usd
    2. total_transit_time_days
    3. composite_risk_score
    4. overall_delay_probability

    A route A dominates route B if A is no worse than B in all 4 metrics
    and strictly better in at least one metric.

    Returns:
        List of route_ids that belong to the Pareto-optimal frontier.
    """
    if not routes:
        return []

    # Extract vectors: (cost, time, risk, delay_prob)
    candidates = []
    for r in routes:
        candidates.append((
            r.route_id,
            (r.total_cost_usd, r.total_transit_time_days, r.composite_risk_score, r.overall_delay_probability)
        ))

    pareto_route_ids = []

    for id_a, vec_a in candidates:
        is_dominated = False
        for id_b, vec_b in candidates:
            if id_a == id_b:
                continue

            # Check if B dominates A: vec_b[k] <= vec_a[k] for all k, and strictly less for at least one
            b_no_worse = all(b_val <= a_val for a_val, b_val in zip(vec_a, vec_b))
            b_strictly_better = any(b_val < a_val for a_val, b_val in zip(vec_a, vec_b))

            if b_no_worse and b_strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_route_ids.append(id_a)

    return pareto_route_ids


def annotate_routes_with_pareto(routes: List[Route]) -> List[Route]:
    """Marks `is_pareto_optimal` boolean on each route."""
    pareto_ids = set(find_pareto_frontier(routes))
    for r in routes:
        r.is_pareto_optimal = (r.route_id in pareto_ids)
    return routes
