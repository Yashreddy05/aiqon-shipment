"""
Monte Carlo Stochastic Simulator for Transit Time and Realized Cost Uncertainty.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from src.models.schemas import Route, Shipment


class MonteCarloSimulator:
    """Simulates operational delay realizations and financial penalty exposure under uncertainty."""

    def __init__(self, num_simulations: int = 1000, seed: int = 42):
        self.num_simulations = num_simulations
        self.seed = seed

    def simulate_route(
        self,
        route: Route,
        target_days: float,
        penalty_per_day_usd: float
    ) -> Dict[str, float]:
        """
        Runs Monte Carlo simulation for a single route across all its legs.

        For each leg i:
          - A Bernoulli trial determines whether delay occurs with p = leg.delay_probability
          - If delay occurs, delay duration is drawn from Log-Normal distribution
            calibrated so that mean = leg.expected_delay_days.
        """
        rng = np.random.default_rng(self.seed + abs(hash(route.route_id)) % 10000)

        total_simulated_times = np.zeros(self.num_simulations)
        nominal_time = route.total_transit_time_days
        base_cost = route.total_cost_usd

        # Simulate leg-by-leg
        for leg in route.legs:
            # Did delay occur in this leg?
            delay_occurred = rng.random(self.num_simulations) < leg.delay_probability

            # LogNormal parameters: mu and sigma such that mean = expected_delay_days
            # Let coefficient of variation CV = sigma / mu = 0.6
            mean_delay = max(leg.expected_delay_days, 0.1)
            sigma_ln = 0.55
            mu_ln = np.log(mean_delay) - 0.5 * (sigma_ln ** 2)

            leg_delays = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=self.num_simulations)
            # Only apply delay where delay occurred
            realized_delays = delay_occurred * leg_delays
            total_simulated_times += leg.transit_time_days + realized_delays

        # Calculate late days beyond target
        late_days = np.maximum(0.0, total_simulated_times - target_days)
        realized_penalties = late_days * penalty_per_day_usd

        # Congestion demurrage risk surcharge: if delay exceeds 3 days, add demurrage buffer
        excess_dwell = np.maximum(0.0, total_simulated_times - (nominal_time + 3.0))
        demurrage_surcharge = excess_dwell * 120.0  # $120/day demurrage/storage fee

        total_realized_costs = base_cost + realized_penalties + demurrage_surcharge

        # Metrics
        otif_prob = float(np.mean(total_simulated_times <= target_days))
        mean_time = float(np.mean(total_simulated_times))
        p50_time = float(np.percentile(total_simulated_times, 50))
        p90_time = float(np.percentile(total_simulated_times, 90))
        p99_time = float(np.percentile(total_simulated_times, 99))

        mean_cost = float(np.mean(total_realized_costs))
        p95_cost = float(np.percentile(total_realized_costs, 95))
        # CVaR (Expected Shortfall) at 95%: mean of values >= 95th percentile
        cvar_95_cost = float(np.mean(total_realized_costs[total_realized_costs >= p95_cost]))

        stats = {
            "otif_probability": round(otif_prob, 4),
            "mean_transit_days": round(mean_time, 2),
            "p50_transit_days": round(p50_time, 2),
            "p90_transit_days": round(p90_time, 2),
            "p99_transit_days": round(p99_time, 2),
            "nominal_cost_usd": round(base_cost, 2),
            "expected_realized_cost_usd": round(mean_cost, 2),
            "var_95_cost_usd": round(p95_cost, 2),
            "cvar_95_cost_usd": round(cvar_95_cost, 2),
            "mean_late_days": round(float(np.mean(late_days)), 2),
        }
        return stats

    def simulate_shipment(self, shipment: Shipment) -> Dict[str, Dict[str, float]]:
        """Simulates all route options for a given shipment and updates route simulation stats."""
        results = {}
        for r in shipment.route_options:
            stats = self.simulate_route(
                r,
                target_days=shipment.target_delivery_days,
                penalty_per_day_usd=shipment.delay_penalty_per_day_usd
            )
            r.simulation_stats = stats
            results[r.route_id] = stats
        return results
