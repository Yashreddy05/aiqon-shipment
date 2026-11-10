"""
Comprehensive Evaluation and Benchmarking Framework.
Compares the Multi-Objective Route Optimizer against industry baselines:
- Greedy Cost Minimizer
- Greedy Time Minimizer
- Random Route Selector
"""

from __future__ import annotations
import random
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.models.schemas import Shipment, OperatorProfile, Route
from src.optimization.scoring import RouteOptimizer
from src.optimization.pareto import find_pareto_frontier


class BenchmarkSuite:
    """Executes systematic comparative evaluation over shipment datasets."""

    def __init__(self, optimizer: RouteOptimizer = None):
        self.optimizer = optimizer or RouteOptimizer()

    def evaluate_shipments(self, shipments: List[Shipment]) -> Dict[str, Any]:
        """
        Runs full comparative benchmark across all shipments for:
        1. Proposed Balanced Policy
        2. Proposed Risk-Averse Policy
        3. Greedy Cost Minimizer Baseline
        4. Greedy Time Minimizer Baseline
        5. Naive Random Baseline
        """
        strategies = [
            "Proposed (Balanced)",
            "Proposed (Risk-Averse)",
            "Greedy Cost Minimizer",
            "Greedy Time Minimizer",
            "Random Baseline"
        ]

        results_by_strategy = {s: [] for s in strategies}

        for shp in shipments:
            routes = shp.route_options
            if not routes:
                continue

            pareto_ids = set(find_pareto_frontier(routes))

            # 1. Proposed Balanced
            rec_bal = self.optimizer.recommend(shp, profile=OperatorProfile.BALANCED, run_monte_carlo=True)
            results_by_strategy["Proposed (Balanced)"].append(
                self._extract_metrics(rec_bal.recommended_route, pareto_ids, shp)
            )

            # 2. Proposed Risk-Averse
            rec_risk = self.optimizer.recommend(shp, profile=OperatorProfile.RISK_AVERSE, run_monte_carlo=True)
            results_by_strategy["Proposed (Risk-Averse)"].append(
                self._extract_metrics(rec_risk.recommended_route, pareto_ids, shp)
            )

            # 3. Greedy Cost Minimizer
            cheapest = min(routes, key=lambda r: r.total_cost_usd)
            results_by_strategy["Greedy Cost Minimizer"].append(
                self._extract_metrics(cheapest, pareto_ids, shp)
            )

            # 4. Greedy Time Minimizer
            fastest = min(routes, key=lambda r: r.total_transit_time_days)
            results_by_strategy["Greedy Time Minimizer"].append(
                self._extract_metrics(fastest, pareto_ids, shp)
            )

            # 5. Random
            rand_choice = random.choice(routes)
            results_by_strategy["Random Baseline"].append(
                self._extract_metrics(rand_choice, pareto_ids, shp)
            )

        # Aggregate metrics
        summary_rows = []
        for strat, metrics_list in results_by_strategy.items():
            df_m = pd.DataFrame(metrics_list)
            summary_rows.append({
                "Strategy": strat,
                "Pareto Optimality Rate (%)": round(df_m["is_pareto"].mean() * 100, 1),
                "Avg Direct Cost ($)": round(df_m["cost_usd"].mean(), 2),
                "Avg Realized Cost ($)": round(df_m["realized_cost_usd"].mean(), 2),
                "Avg Transit Time (Days)": round(df_m["transit_days"].mean(), 1),
                "Avg Delay Prob (%)": round(df_m["delay_prob"].mean() * 100, 1),
                "Avg Composite Risk": round(df_m["composite_risk"].mean(), 3),
                "Avg OTIF Delivery (%)": round(df_m["otif_prob"].mean() * 100, 1),
                "Multi-Objective Regret": round(df_m["regret_distance"].mean(), 3)
            })

        summary_df = pd.DataFrame(summary_rows)
        return {
            "summary_table": summary_df,
            "detailed_results": results_by_strategy
        }

    def _extract_metrics(self, route: Route, pareto_ids: set, shipment: Shipment) -> Dict[str, Any]:
        """Extracts standard metric vector for a chosen route."""
        # Calculate utopian distance / regret
        # Normalize cost, time, risk relative to available options in this shipment
        all_costs = [r.total_cost_usd for r in shipment.route_options]
        all_times = [r.total_transit_time_days for r in shipment.route_options]
        all_risks = [r.composite_risk_score for r in shipment.route_options]

        c_min, c_max = min(all_costs), max(all_costs)
        t_min, t_max = min(all_times), max(all_times)
        r_min, r_max = min(all_risks), max(all_risks)

        norm_c = (route.total_cost_usd - c_min) / max(c_max - c_min, 1e-5)
        norm_t = (route.total_transit_time_days - t_min) / max(t_max - t_min, 1e-5)
        norm_r = (route.composite_risk_score - r_min) / max(r_max - r_min, 1e-5)

        # Regret is Euclidean distance to utopia point (0, 0, 0)
        regret = float(np.sqrt(norm_c**2 + norm_t**2 + norm_r**2))

        sim = route.simulation_stats or {}
        realized_cost = sim.get("expected_realized_cost_usd", route.total_cost_usd)
        otif = sim.get("otif_probability", 1.0 - route.overall_delay_probability)

        return {
            "route_id": route.route_id,
            "is_pareto": route.route_id in pareto_ids,
            "cost_usd": route.total_cost_usd,
            "realized_cost_usd": realized_cost,
            "transit_days": route.total_transit_time_days,
            "delay_prob": route.overall_delay_probability,
            "composite_risk": route.composite_risk_score,
            "otif_prob": otif,
            "regret_distance": regret
        }
