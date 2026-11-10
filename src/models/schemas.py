"""
Data models and schemas for the Aiqon Multi-Objective Shipment Route Optimizer.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, computed_field


class TransportMode(str, Enum):
    OCEAN = "OCEAN"
    AIR = "AIR"
    RAIL = "RAIL"
    ROAD = "ROAD"
    MULTIMODAL = "MULTIMODAL"


class CargoCategory(str, Enum):
    ELECTRONICS = "High-Tech Electronics"
    PHARMA = "Pharmaceuticals & Cold Chain"
    AUTOMOTIVE = "Automotive Components (JIT)"
    RETAIL = "Retail & Fast Fashion"
    INDUSTRIAL = "Industrial Machinery & Raw Materials"


class OperatorProfile(str, Enum):
    COST_MINIMIZER = "Cost Minimizer"
    TIME_CRITICAL = "Time Critical / Expedited"
    RISK_AVERSE = "Risk Averse / Resilient"
    BALANCED = "Balanced Trade-off"
    CUSTOM = "Custom Weights"


class RiskIndicators(BaseModel):
    congestion_risk: float = Field(..., ge=0.0, le=1.0, description="Port/Hub congestion risk")
    geopolitical_risk: float = Field(..., ge=0.0, le=1.0, description="Choke points, sanctions, conflict risk")
    weather_risk: float = Field(..., ge=0.0, le=1.0, description="Severe weather / seasonal disruption risk")
    customs_risk: float = Field(..., ge=0.0, le=1.0, description="Border clearance & inspection friction")

    @property
    def composite_risk(self) -> float:
        """
        Default composite risk weighted by logistics operational severity:
        Geopolitical disruptions cause highest severity re-routings (0.35),
        Congestion causes major dwell variance (0.30),
        Weather triggers intermittent delays (0.20),
        Customs causes administrative lag (0.15).
        """
        score = (
            0.35 * self.geopolitical_risk +
            0.30 * self.congestion_risk +
            0.20 * self.weather_risk +
            0.15 * self.customs_risk
        )
        return round(score, 4)


class Leg(BaseModel):
    leg_id: str
    origin: str
    destination: str
    mode: TransportMode
    carrier: str
    distance_km: float = Field(..., gt=0)
    cost_usd: float = Field(..., gt=0)
    transit_time_days: float = Field(..., gt=0)
    delay_probability: float = Field(..., ge=0.0, le=1.0)
    expected_delay_days: float = Field(..., ge=0.0)
    risk_indicators: RiskIndicators
    co2_kg: float = Field(..., ge=0.0)


class Route(BaseModel):
    route_id: str
    shipment_id: str
    name: str
    legs: List[Leg]
    corridor_type: str = "Standard"  # e.g., "Direct Suez", "Cape Reroute", "Sea-Air Multimodal", "Express Air"
    is_pareto_optimal: Optional[bool] = None
    simulation_stats: Optional[Dict[str, float]] = None

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        return round(sum(leg.cost_usd for leg in self.legs), 2)

    @computed_field
    @property
    def total_transit_time_days(self) -> float:
        return round(sum(leg.transit_time_days for leg in self.legs), 2)

    @computed_field
    @property
    def total_distance_km(self) -> float:
        return round(sum(leg.distance_km for leg in self.legs), 2)

    @computed_field
    @property
    def total_co2_kg(self) -> float:
        return round(sum(leg.co2_kg for leg in self.legs), 2)

    @computed_field
    @property
    def overall_delay_probability(self) -> float:
        """
        Probability of encountering delay across sequential legs:
        1 - Prod(1 - P_delay_leg_i)
        """
        prob_no_delay = 1.0
        for leg in self.legs:
            prob_no_delay *= (1.0 - leg.delay_probability)
        return round(1.0 - prob_no_delay, 4)

    @computed_field
    @property
    def expected_delay_days(self) -> float:
        return round(sum(leg.expected_delay_days for leg in self.legs), 2)

    @computed_field
    @property
    def risk_indicators(self) -> RiskIndicators:
        """Distance-weighted aggregation of individual risk indicators across route legs."""
        total_dist = sum(leg.distance_km for leg in self.legs)
        if total_dist <= 0:
            total_dist = 1.0
        cg = sum(leg.risk_indicators.congestion_risk * leg.distance_km for leg in self.legs) / total_dist
        geo = sum(leg.risk_indicators.geopolitical_risk * leg.distance_km for leg in self.legs) / total_dist
        wea = sum(leg.risk_indicators.weather_risk * leg.distance_km for leg in self.legs) / total_dist
        cus = sum(leg.risk_indicators.customs_risk * leg.distance_km for leg in self.legs) / total_dist
        return RiskIndicators(
            congestion_risk=round(cg, 4),
            geopolitical_risk=round(geo, 4),
            weather_risk=round(wea, 4),
            customs_risk=round(cus, 4)
        )

    @computed_field
    @property
    def composite_risk_score(self) -> float:
        """Weighted aggregate risk of the route based on leg distances."""
        total_dist = sum(leg.distance_km for leg in self.legs)
        if total_dist <= 0:
            return round(sum(leg.risk_indicators.composite_risk for leg in self.legs) / max(len(self.legs), 1), 4)
        weighted_risk = sum(leg.risk_indicators.composite_risk * leg.distance_km for leg in self.legs) / total_dist
        return round(weighted_risk, 4)

    @computed_field
    @property
    def reliability_score(self) -> float:
        """Reliability index = 1 - overall_delay_probability."""
        return round(1.0 - self.overall_delay_probability, 4)

    @computed_field
    @property
    def intermediate_hubs(self) -> List[str]:
        hubs = []
        for i, leg in enumerate(self.legs[:-1]):
            hubs.append(leg.destination)
        return hubs


class Shipment(BaseModel):
    shipment_id: str
    cargo_type: CargoCategory
    origin_city: str
    origin_country: str
    origin_coords: Tuple[float, float]  # (latitude, longitude)
    destination_city: str
    destination_country: str
    destination_coords: Tuple[float, float]
    cargo_value_usd: float
    cargo_weight_tons: float
    target_delivery_days: float
    delay_penalty_per_day_usd: float
    priority_level: str = "STANDARD"  # STANDARD, HIGH, CRITICAL
    route_options: List[Route] = []


class OptimizationWeights(BaseModel):
    cost_weight: float = Field(0.35, ge=0.0, le=1.0)
    time_weight: float = Field(0.35, ge=0.0, le=1.0)
    risk_weight: float = Field(0.30, ge=0.0, le=1.0)
    reliability_weight: float = Field(0.0, ge=0.0, le=1.0)
    co2_weight: float = Field(0.0, ge=0.0, le=1.0)
    max_budget_usd: Optional[float] = None
    max_transit_days: Optional[float] = None
    max_acceptable_risk: Optional[float] = None

    def normalized(self) -> "OptimizationWeights":
        """Normalize weights to sum to 1.0 across active dimensions."""
        total = self.cost_weight + self.time_weight + self.risk_weight + self.reliability_weight + self.co2_weight
        if total == 0:
            return OptimizationWeights(cost_weight=0.34, time_weight=0.33, risk_weight=0.33)
        return OptimizationWeights(
            cost_weight=round(self.cost_weight / total, 4),
            time_weight=round(self.time_weight / total, 4),
            risk_weight=round(self.risk_weight / total, 4),
            reliability_weight=round(self.reliability_weight / total, 4),
            co2_weight=round(self.co2_weight / total, 4),
            max_budget_usd=self.max_budget_usd,
            max_transit_days=self.max_transit_days,
            max_acceptable_risk=self.max_acceptable_risk
        )


PROFILE_WEIGHTS: Dict[OperatorProfile, OptimizationWeights] = {
    OperatorProfile.COST_MINIMIZER: OptimizationWeights(cost_weight=0.70, time_weight=0.15, risk_weight=0.15),
    OperatorProfile.TIME_CRITICAL: OptimizationWeights(cost_weight=0.15, time_weight=0.70, risk_weight=0.15),
    OperatorProfile.RISK_AVERSE: OptimizationWeights(cost_weight=0.15, time_weight=0.20, risk_weight=0.65),
    OperatorProfile.BALANCED: OptimizationWeights(cost_weight=0.35, time_weight=0.35, risk_weight=0.30),
}


class RecommendationResult(BaseModel):
    shipment_id: str
    origin: str
    destination: str
    cargo_type: str
    operator_profile: str
    recommended_route_id: str
    recommended_route_name: str
    recommended_route: Route
    runner_up_route: Optional[Route] = None
    scores: Dict[str, float]  # route_id -> score (higher is better)
    rankings: List[Dict[str, Any]]
    pareto_optimal_route_ids: List[str]
    is_recommended_pareto: bool
    summary_explanation: str
    trade_off_breakdown: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    simulation_summary: Optional[Dict[str, Any]] = None
