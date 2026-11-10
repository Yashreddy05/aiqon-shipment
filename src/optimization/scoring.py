"""
Scoring and Decision Framework for Shipment Route Optimization.
Combines Pareto filtering, TOPSIS, stochastic risk penalties, and sensitivity testing.
"""

from __future__ import annotations
import copy
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.models.schemas import (
    Route,
    Shipment,
    OptimizationWeights,
    OperatorProfile,
    PROFILE_WEIGHTS,
    RecommendationResult
)
from src.optimization.pareto import find_pareto_frontier, annotate_routes_with_pareto
from src.optimization.topsis import calculate_topsis_scores
from src.optimization.simulation import MonteCarloSimulator


class RouteOptimizer:
    """Core optimization engine for evaluating, scoring, and recommending routes."""

    def __init__(self, simulator: Optional[MonteCarloSimulator] = None):
        self.simulator = simulator or MonteCarloSimulator()

    def filter_feasible_routes(
        self,
        routes: List[Route],
        weights: OptimizationWeights
    ) -> Tuple[List[Route], List[Dict[str, str]]]:
        """
        Filters routes against operator hard constraints (budget, maximum transit time, risk threshold).
        Returns feasible routes and rejection reasons for filtered routes.
        """
        feasible = []
        rejected = []

        for r in routes:
            reasons = []
            if weights.max_budget_usd is not None and r.total_cost_usd > weights.max_budget_usd:
                reasons.append(f"Exceeds max budget (${r.total_cost_usd:.2f} > ${weights.max_budget_usd:.2f})")
            if weights.max_transit_days is not None and r.total_transit_time_days > weights.max_transit_days:
                reasons.append(f"Exceeds max transit time ({r.total_transit_time_days:.1f}d > {weights.max_transit_days:.1f}d)")
            if weights.max_acceptable_risk is not None and r.composite_risk_score > weights.max_acceptable_risk:
                reasons.append(f"Exceeds max acceptable risk ({r.composite_risk_score:.3f} > {weights.max_acceptable_risk:.3f})")

            if reasons:
                rejected.append({"route_id": r.route_id, "route_name": r.name, "reasons": "; ".join(reasons)})
            else:
                feasible.append(r)

        # Fallback: if all routes rejected by overly strict constraints, return all with warning
        if not feasible:
            return routes, [{"route_id": "ALL", "route_name": "All Routes", "reasons": "Warning: Hard constraints too strict; falling back to best-effort ranking."}]

        return feasible, rejected

    def score_routes(
        self,
        routes: List[Route],
        weights: OptimizationWeights
    ) -> Dict[str, float]:
        """
        Computes composite decision score for each candidate route.
        Combines TOPSIS closeness with stochastic simulation penalty.
        Score is in [0, 100], higher is better.
        """
        if not routes:
            return {}

        # 1. TOPSIS Closeness in [0.0, 1.0]
        topsis_scores = calculate_topsis_scores(routes, weights)

        # 2. Add penalty for high tail risk if simulation stats are present
        final_scores = {}
        for r in routes:
            base_topsis = topsis_scores.get(r.route_id, 0.5)

            # Bonus for Pareto-optimal routes (promotes non-dominated choices)
            pareto_bonus = 0.05 if r.is_pareto_optimal else 0.0

            # OTIF reliability adjustment if simulation ran
            sim_boost = 0.0
            if r.simulation_stats:
                otif = r.simulation_stats.get("otif_probability", 0.7)
                sim_boost = 0.10 * (otif - 0.5)  # reward high OTIF

            raw_score = base_topsis + pareto_bonus + sim_boost
            # Normalize to 0-100 scale
            final_scores[r.route_id] = round(float(np.clip(raw_score * 100.0, 0.0, 100.0)), 2)

        return final_scores

    def run_sensitivity_analysis(
        self,
        routes: List[Route],
        base_weights: OptimizationWeights,
        base_recommended_id: str,
        perturbation: float = 0.20
    ) -> Dict[str, Any]:
        """
        Performs sensitivity testing:
        Tests if changing weights (Cost +/-20%, Time +/-20%, Risk +/-20%)
        changes the top-recommended route. Computes recommendation stability.
        """
        scenarios = {
            "Cost Surge (+20% Cost Weight)": OptimizationWeights(
                cost_weight=base_weights.cost_weight * (1 + perturbation),
                time_weight=base_weights.time_weight,
                risk_weight=base_weights.risk_weight
            ),
            "Time Urgent (+20% Time Weight)": OptimizationWeights(
                cost_weight=base_weights.cost_weight,
                time_weight=base_weights.time_weight * (1 + perturbation),
                risk_weight=base_weights.risk_weight
            ),
            "High Volatility (+20% Risk Weight)": OptimizationWeights(
                cost_weight=base_weights.cost_weight,
                time_weight=base_weights.time_weight,
                risk_weight=base_weights.risk_weight * (1 + perturbation)
            ),
            "Fuel / Cost Discount (-20% Cost Weight)": OptimizationWeights(
                cost_weight=base_weights.cost_weight * (1 - perturbation),
                time_weight=base_weights.time_weight,
                risk_weight=base_weights.risk_weight
            ),
        }

        flips = 0
        scenario_results = {}
        for sc_name, sc_weights in scenarios.items():
            sc_scores = calculate_topsis_scores(routes, sc_weights)
            sc_best_id = max(sc_scores.items(), key=lambda x: x[1])[0]
            is_same = (sc_best_id == base_recommended_id)
            if not is_same:
                flips += 1
            scenario_results[sc_name] = {
                "top_route_id": sc_best_id,
                "maintained_rank_1": is_same,
                "scores": sc_scores
            }

        stability_index = round(1.0 - (flips / len(scenarios)), 2)
        return {
            "stability_index": stability_index,
            "flips_detected": flips,
            "scenarios_evaluated": len(scenarios),
            "scenario_details": scenario_results
        }

    def recommend(
        self,
        shipment: Shipment,
        profile: OperatorProfile = OperatorProfile.BALANCED,
        custom_weights: Optional[OptimizationWeights] = None,
        run_monte_carlo: bool = True
    ) -> RecommendationResult:
        """
        End-to-end evaluation and recommendation pipeline for a given shipment.
        """
        # 1. Determine active weights
        if profile == OperatorProfile.CUSTOM and custom_weights is not None:
            active_weights = custom_weights
        else:
            active_weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS[OperatorProfile.BALANCED])

        # 2. Annotate Pareto frontier
        annotate_routes_with_pareto(shipment.route_options)
        pareto_ids = [r.route_id for r in shipment.route_options if r.is_pareto_optimal]

        # 3. Filter feasible routes
        feasible_routes, rejection_log = self.filter_feasible_routes(shipment.route_options, active_weights)

        # 4. Run Monte Carlo stochastic simulation
        sim_summary = {}
        if run_monte_carlo:
            sim_summary = self.simulator.simulate_shipment(shipment)

        # 5. Score routes
        scores = self.score_routes(feasible_routes, active_weights)

        # 6. Rank routes
        ranked_routes = sorted(
            feasible_routes,
            key=lambda r: scores.get(r.route_id, 0.0),
            reverse=True
        )

        recommended = ranked_routes[0]
        runner_up = ranked_routes[1] if len(ranked_routes) > 1 else None

        # 7. Sensitivity analysis
        sensitivity = self.run_sensitivity_analysis(
            feasible_routes,
            active_weights,
            recommended.route_id
        )

        # 8. Rankings table
        rankings = []
        for rank, r in enumerate(ranked_routes, start=1):
            rankings.append({
                "rank": rank,
                "route_id": r.route_id,
                "name": r.name,
                "score": scores.get(r.route_id, 0.0),
                "is_pareto": r.is_pareto_optimal,
                "cost_usd": r.total_cost_usd,
                "transit_time_days": r.total_transit_time_days,
                "delay_probability": r.overall_delay_probability,
                "risk_score": r.composite_risk_score,
                "otif_prob": r.simulation_stats.get("otif_probability") if r.simulation_stats else None,
                "expected_realized_cost": r.simulation_stats.get("expected_realized_cost_usd") if r.simulation_stats else None
            })

        # Return un-summarized result (summary_explanation populated by explainer engine)
        return RecommendationResult(
            shipment_id=shipment.shipment_id,
            origin=shipment.origin_city,
            destination=shipment.destination_city,
            cargo_type=shipment.cargo_type.value,
            operator_profile=profile.value,
            recommended_route_id=recommended.route_id,
            recommended_route_name=recommended.name,
            recommended_route=recommended,
            runner_up_route=runner_up,
            scores=scores,
            rankings=rankings,
            pareto_optimal_route_ids=pareto_ids,
            is_recommended_pareto=recommended.is_pareto_optimal or False,
            summary_explanation="",
            trade_off_breakdown={},
            risk_analysis={},
            sensitivity_analysis=sensitivity,
            simulation_summary=sim_summary
        )
