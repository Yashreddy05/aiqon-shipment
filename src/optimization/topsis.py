"""
Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)
Adapted for Multi-Objective Logistics Route Selection.
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple, Any
import numpy as np
from src.models.schemas import Route, OptimizationWeights


def calculate_topsis_scores(
    routes: List[Route],
    weights: OptimizationWeights
) -> Dict[str, float]:
    """
    Computes TOPSIS relative closeness scores for each route.
    Metrics evaluated (all cost criteria where lower is better):
    1. total_cost_usd
    2. total_transit_time_days
    3. composite_risk_score
    4. overall_delay_probability
    5. total_co2_kg (optional, if co2_weight > 0)

    Returns:
        Dict mapping route_id to TOPSIS relative closeness score C_i in [0.0, 1.0].
        Higher score = closer to ideal, farther from anti-ideal.
    """
    if not routes:
        return {}

    n = len(routes)
    if n == 1:
        return {routes[0].route_id: 1.0}

    # Normalize weights
    norm_w = weights.normalized()
    w_vec = [
        norm_w.cost_weight,
        norm_w.time_weight,
        norm_w.risk_weight,
        norm_w.reliability_weight if norm_w.reliability_weight > 0 else 0.15,
    ]
    if norm_w.co2_weight > 0:
        w_vec.append(norm_w.co2_weight)

    w_arr = np.array(w_vec, dtype=float)
    w_arr = w_arr / np.sum(w_arr)  # ensure sum to 1

    # Build decision matrix X: shape (n_routes, m_criteria)
    # All criteria here are cost criteria (minimization)
    raw_matrix = []
    for r in routes:
        row = [
            r.total_cost_usd,
            r.total_transit_time_days,
            r.composite_risk_score,
            r.overall_delay_probability,
        ]
        if norm_w.co2_weight > 0:
            row.append(r.total_co2_kg)
        raw_matrix.append(row)

    X = np.array(raw_matrix, dtype=float)

    # 1. Vector normalization: r_ij = x_ij / sqrt(sum_i(x_ij^2))
    norm_denominators = np.sqrt(np.sum(X ** 2, axis=0))
    # Avoid zero division
    norm_denominators[norm_denominators == 0] = 1.0
    R = X / norm_denominators

    # 2. Weighted normalized decision matrix: V = R * w
    V = R * w_arr

    # 3. Determine Ideal Best (A*) and Ideal Worst (A-)
    # Since all criteria are cost criteria (minimization):
    # Ideal Best A* = min across each column
    # Ideal Worst A- = max across each column
    ideal_best = np.min(V, axis=0)
    ideal_worst = np.max(V, axis=0)

    # 4. Calculate Euclidean separation distances
    d_best = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))

    # 5. Calculate relative closeness to ideal solution: C_i = d_worst / (d_best + d_worst)
    total_dist = d_best + d_worst
    scores = {}
    for i, r in enumerate(routes):
        if total_dist[i] == 0:
            c_score = 0.5
        else:
            c_score = float(d_worst[i] / total_dist[i])
        scores[r.route_id] = round(c_score, 4)

    return scores
