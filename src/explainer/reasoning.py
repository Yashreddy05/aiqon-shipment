"""
Explainability Engine for Logistics Operators.
Produces transparent, structured, and natural language explanations of routing decisions,
head-to-head trade-offs, and risk justifications.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from src.models.schemas import Route, Shipment, RecommendationResult


class RouteDecisionExplainer:
    """Translates multi-objective scores, Pareto properties, and stochastic simulations
    into actionable operational reasoning."""

    @staticmethod
    def explain_recommendation(
        result: RecommendationResult,
        shipment: Shipment
    ) -> RecommendationResult:
        """Enriches the RecommendationResult with comprehensive trade-off and risk explanations."""
        rec = result.recommended_route
        runner = result.runner_up_route
        all_routes = shipment.route_options

        # Identify key benchmarks
        cheapest_route = min(all_routes, key=lambda r: r.total_cost_usd)
        fastest_route = min(all_routes, key=lambda r: r.total_transit_time_days)
        safest_route = min(all_routes, key=lambda r: r.composite_risk_score)

        # 1. Generate Head-to-Head Comparison with Runner-Up
        trade_off_breakdown = {}
        if runner:
            cost_diff = rec.total_cost_usd - runner.total_cost_usd
            cost_pct = (cost_diff / runner.total_cost_usd) * 100 if runner.total_cost_usd > 0 else 0
            time_diff = rec.total_transit_time_days - runner.total_transit_time_days
            time_pct = (time_diff / runner.total_transit_time_days) * 100 if runner.total_transit_time_days > 0 else 0
            risk_diff = rec.composite_risk_score - runner.composite_risk_score
            delay_prob_diff = rec.overall_delay_probability - runner.overall_delay_probability

            # Marginal exchange rate
            cost_per_day_saved = abs(cost_diff / time_diff) if abs(time_diff) > 0.01 else 0.0

            trade_off_breakdown = {
                "runner_up_name": runner.name,
                "runner_up_id": runner.route_id,
                "cost_difference_usd": round(cost_diff, 2),
                "cost_difference_pct": round(cost_pct, 1),
                "time_difference_days": round(time_diff, 1),
                "time_difference_pct": round(time_pct, 1),
                "risk_difference": round(risk_diff, 4),
                "delay_prob_difference": round(delay_prob_diff, 4),
                "marginal_cost_per_day_saved": round(cost_per_day_saved, 2),
            }

        # 2. Risk Indicators Breakdown
        risk_analysis = {
            "congestion_risk": round(rec.risk_indicators.congestion_risk, 3),
            "geopolitical_risk": round(rec.risk_indicators.geopolitical_risk, 3),
            "weather_risk": round(rec.risk_indicators.weather_risk, 3),
            "customs_risk": round(rec.risk_indicators.customs_risk, 3),
            "composite_risk": rec.composite_risk_score,
            "vulnerable_legs": []
        }
        for leg in rec.legs:
            if leg.risk_indicators.geopolitical_risk > 0.4 or leg.risk_indicators.congestion_risk > 0.4:
                risk_analysis["vulnerable_legs"].append({
                    "leg_id": leg.leg_id,
                    "origin": leg.origin,
                    "destination": leg.destination,
                    "mode": leg.mode.value,
                    "geopolitical_risk": leg.risk_indicators.geopolitical_risk,
                    "congestion_risk": leg.risk_indicators.congestion_risk
                })

        # 3. Build Natural Language Rationale
        summary = RouteDecisionExplainer._build_narrative(
            rec=rec,
            runner=runner,
            cheapest=cheapest_route,
            fastest=fastest_route,
            safest=safest_route,
            shipment=shipment,
            profile=result.operator_profile,
            trade_off=trade_off_breakdown
        )

        result.summary_explanation = summary
        result.trade_off_breakdown = trade_off_breakdown
        result.risk_analysis = risk_analysis

        return result

    @staticmethod
    def _build_narrative(
        rec: Route,
        runner: Optional[Route],
        cheapest: Route,
        fastest: Route,
        safest: Route,
        shipment: Shipment,
        profile: str,
        trade_off: Dict[str, Any]
    ) -> str:
        """Drafts clear, operational prose for the logistics planner."""
        sentences = []

        # 1. Headline Selection
        sentences.append(
            f"Under the '{profile}' policy, '{rec.name}' ({rec.route_id}) is recommended "
            f"for {shipment.cargo_type.value} moving from {shipment.origin_city} to {shipment.destination_city}."
        )

        # 2. Comparison against Runner-Up
        if runner:
            cost_diff = trade_off["cost_difference_usd"]
            time_diff = trade_off["time_difference_days"]
            risk_diff = trade_off["risk_difference"]

            if rec.route_id == runner.route_id:
                sentences.append("It dominates all other candidate routes across the primary metrics.")
            elif cost_diff > 0 and time_diff < 0:
                sentences.append(
                    f"Compared to the runner-up ('{runner.name}'), this option cuts transit time by {abs(time_diff):.1f} days "
                    f"for an incremental freight investment of ${abs(cost_diff):,.2f} ({abs(trade_off['cost_difference_pct'])}%), "
                    f"yielding a favorable trade-off of ${trade_off['marginal_cost_per_day_saved']:,.2f} per day saved."
                )
            elif cost_diff > 0 and risk_diff < 0:
                sentences.append(
                    f"While costing ${abs(cost_diff):,.2f} more than '{runner.name}', it substantially mitigates operational risk "
                    f"({rec.composite_risk_score:.2f} vs {runner.composite_risk_score:.2f}) and cuts delay probability "
                    f"from {runner.overall_delay_probability*100:.1f}% down to {rec.overall_delay_probability*100:.1f}%."
                )
            elif cost_diff < 0 and time_diff <= 0:
                sentences.append(
                    f"This route strictly dominates the runner-up ('{runner.name}'), saving ${abs(cost_diff):,.2f} "
                    f"while delivering {abs(time_diff):.1f} days faster."
                )
            else:
                sentences.append(
                    f"This route achieves a superior multi-criteria score by balancing a transit time of {rec.total_transit_time_days:.1f} days "
                    f"against a total cost of ${rec.total_cost_usd:,.2f} and risk index of {rec.composite_risk_score:.2f}."
                )

        # 3. Why Extreme Routes Were Rejected
        if rec.route_id != cheapest.route_id:
            extra_cost = rec.total_cost_usd - cheapest.total_cost_usd
            time_saved = cheapest.total_transit_time_days - rec.total_transit_time_days
            risk_avoided = cheapest.composite_risk_score - rec.composite_risk_score
            sentences.append(
                f"The cheapest alternative ('{cheapest.name}' at ${cheapest.total_cost_usd:,.2f}) was rejected because "
                f"it incurs higher disruption risk ({cheapest.composite_risk_score:.2f}) and {cheapest.overall_delay_probability*100:.1f}% delay probability"
                + (f", adding {time_saved:.1f} days in transit." if time_saved > 0 else ".")
            )

        if rec.route_id != fastest.route_id:
            premium = fastest.total_cost_usd - rec.total_cost_usd
            if premium > 0:
                sentences.append(
                    f"The fastest option ('{fastest.name}' at {fastest.total_transit_time_days:.1f} days) was bypassed "
                    f"due to an excessive cost premium of ${premium:,.2f} ({((premium)/rec.total_cost_usd)*100:.0f}% higher) "
                    f"which exceeds the required service level."
                )

        # 4. Pareto & Reliability Assurance
        pareto_txt = "Pareto-optimal (non-dominated)" if rec.is_pareto_optimal else "operationally preferred"
        sentences.append(
            f"The selected route is confirmed {pareto_txt} with an estimated On-Time In-Full (OTIF) reliability of "
            f"{rec.reliability_score*100:.1f}%."
        )

        return " ".join(sentences)
