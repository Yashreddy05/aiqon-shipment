"""
Synthetic Dataset Generator for Global Freight Routing.
Simulates realistic multi-leg international shipment corridors with competing
cost, transit time, delay probability, and multi-factor risk profiles.
"""

from __future__ import annotations
import json
import random
from typing import List, Dict, Tuple
import pandas as pd
from src.models.schemas import (
    TransportMode,
    CargoCategory,
    RiskIndicators,
    Leg,
    Route,
    Shipment,
)

# Major Global Logistics Hubs with Coordinates (lat, lon)
GLOBAL_HUBS: Dict[str, Dict[str, Any]] = {
    "Shanghai": {"country": "China", "coords": (31.2304, 121.4737), "type": "Port/Air"},
    "Ningbo": {"country": "China", "coords": (29.8683, 121.5440), "type": "Port"},
    "Shenzhen": {"country": "China", "coords": (22.5431, 114.0579), "type": "Port"},
    "Singapore": {"country": "Singapore", "coords": (1.3521, 103.8198), "type": "Hub"},
    "Rotterdam": {"country": "Netherlands", "coords": (51.9244, 4.4777), "type": "Port"},
    "Hamburg": {"country": "Germany", "coords": (53.5511, 9.9937), "type": "Port"},
    "Antwerp": {"country": "Belgium", "coords": (51.2194, 4.4025), "type": "Port"},
    "Los Angeles": {"country": "USA", "coords": (34.0522, -118.2437), "type": "Port/Air"},
    "Long Beach": {"country": "USA", "coords": (33.7701, -118.1937), "type": "Port"},
    "New York": {"country": "USA", "coords": (40.7128, -74.0060), "type": "Port/Air"},
    "Chicago": {"country": "USA", "coords": (41.8781, -87.6298), "type": "Inland Hub"},
    "Dubai": {"country": "UAE", "coords": (25.2048, 55.2708), "type": "Sea-Air Hub"},
    "Busan": {"country": "South Korea", "coords": (35.1796, 129.0756), "type": "Port"},
    "Tokyo": {"country": "Japan", "coords": (35.6762, 139.6503), "type": "Port/Air"},
    "Mumbai (Nhava Sheva)": {"country": "India", "coords": (18.9499, 72.9511), "type": "Port"},
    "Frankfurt": {"country": "Germany", "coords": (50.1109, 8.6821), "type": "Air/Rail Hub"},
    "Santos": {"country": "Brazil", "coords": (-23.9618, -46.3322), "type": "Port"},
    "Sydney": {"country": "Australia", "coords": (-33.8688, 151.2093), "type": "Port"},
    "Algeciras": {"country": "Spain", "coords": (36.1408, -5.4562), "type": "Hub"},
    "Colombo": {"country": "Sri Lanka", "coords": (6.9271, 79.8612), "type": "Hub"},
    "Klang": {"country": "Malaysia", "coords": (3.0333, 101.4500), "type": "Hub"},
}

CARRIERS_BY_MODE = {
    TransportMode.OCEAN: ["Maersk Line", "MSC Mediterranean", "CMA CGM", "COSCO Shipping", "Hapag-Lloyd", "ONE"],
    TransportMode.AIR: ["Lufthansa Cargo", "Emirates SkyCargo", "Cathay Cargo", "Qatar Cargo", "FedEx Express"],
    TransportMode.RAIL: ["China-Europe Railway Express", "DB Cargo Eurasia", "Trans-Eurasia Logistics"],
    TransportMode.ROAD: ["DHL Freight", "Kuehne+Nagel Road", "DB Schenker", "Local Drayage Services"],
    TransportMode.MULTIMODAL: ["Apex Multimodal Interlink", "BlueSea-Sky Express", "Eurasian Land-Bridge"],
}

TRADE_CORRIDORS = [
    ("Shanghai", "Rotterdam"),
    ("Shenzhen", "Los Angeles"),
    ("Ningbo", "Hamburg"),
    ("Mumbai (Nhava Sheva)", "Rotterdam"),
    ("Busan", "Chicago"),
    ("Tokyo", "Frankfurt"),
    ("Santos", "Antwerp"),
    ("Singapore", "Sydney"),
    ("Shanghai", "New York"),
    ("Mumbai (Nhava Sheva)", "New York"),
]


class RouteGenerator:
    """Generates synthetic shipment instances with multi-leg routes exhibiting realistic trade-offs."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def _generate_legs_for_template(
        self,
        shipment_id: str,
        route_index: int,
        origin: str,
        destination: str,
        strategy: str,
        weight_tons: float
    ) -> Tuple[str, str, List[Leg]]:
        """
        Builds realistic legs based on corridor and routing archetype strategy:
        - "CHEAPEST_OCEAN": Direct ocean via standard shortest choke point (e.g. Suez), lowest cost, high risk/delays.
        - "DEFENSIVE_OCEAN": Re-routing around choke points (e.g. Cape of Good Hope), slightly higher cost & time, very low geopolitical risk.
        - "EXPEDITED_SEA_AIR": Ocean to Middle East hub (Dubai), then Air freight to destination. Fast, medium-high cost.
        - "PREMIUM_EXPRESS_AIR": All-air express, fastest, highest cost, lowest delay risk.
        - "BALANCED_INTERMODAL": Multi-leg combination optimizing trade-offs (e.g., direct ocean with priority berth + express rail).
        """
        route_name = ""
        corridor_type = strategy
        legs: List[Leg] = []

        if strategy == "CHEAPEST_OCEAN":
            route_name = f"Standard Ocean via Primary Chokepoint (Direct)"
            # Leg 1: Port Drayage / Feeder
            leg1 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L1",
                origin=origin,
                destination=f"{origin} Outer Port",
                mode=TransportMode.ROAD,
                carrier=random.choice(CARRIERS_BY_MODE[TransportMode.ROAD]),
                distance_km=60.0,
                cost_usd=round(180.0 + weight_tons * 25, 2),
                transit_time_days=0.8,
                delay_probability=0.08,
                expected_delay_days=0.3,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.25,
                    geopolitical_risk=0.05,
                    weather_risk=0.10,
                    customs_risk=0.15
                ),
                co2_kg=round(45.0 * weight_tons, 2)
            )
            # Leg 2: Ocean Linehaul via Suez/Choke Point
            dist = 18500.0 if "Rotterdam" in destination or "Hamburg" in destination else 12500.0
            base_cost = 2600.0 + weight_tons * 110.0
            time_days = 23.5 if "Rotterdam" in destination else 16.0
            leg2 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L2",
                origin=f"{origin} Outer Port",
                destination=f"{destination} Gateway Port",
                mode=TransportMode.OCEAN,
                carrier=random.choice(CARRIERS_BY_MODE[TransportMode.OCEAN]),
                distance_km=dist,
                cost_usd=round(base_cost, 2),
                transit_time_days=time_days,
                delay_probability=0.42,  # high delay probability due to chokepoints
                expected_delay_days=4.5,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.72,
                    geopolitical_risk=0.84,  # Red Sea / Bab-el-Mandeb / Strait risk
                    weather_risk=0.30,
                    customs_risk=0.20
                ),
                co2_kg=round(dist * 0.015 * weight_tons, 2)
            )
            # Leg 3: Inbound Rail/Truck
            leg3 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L3",
                origin=f"{destination} Gateway Port",
                destination=destination,
                mode=TransportMode.ROAD,
                carrier=random.choice(CARRIERS_BY_MODE[TransportMode.ROAD]),
                distance_km=120.0,
                cost_usd=round(240.0 + weight_tons * 30, 2),
                transit_time_days=1.2,
                delay_probability=0.15,
                expected_delay_days=0.5,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.35,
                    geopolitical_risk=0.05,
                    weather_risk=0.15,
                    customs_risk=0.25
                ),
                co2_kg=round(85.0 * weight_tons, 2)
            )
            legs = [leg1, leg2, leg3]

        elif strategy == "DEFENSIVE_OCEAN":
            route_name = f"Defensive Ocean via Cape of Good Hope (Bypass High-Risk Zones)"
            # Bypasses geopolitical hotspot: extra 3500-5000 km, +7-10 days, slight fuel surcharge, but geopolitical risk near zero!
            dist = 23800.0 if "Rotterdam" in destination or "Hamburg" in destination else 16500.0
            base_cost = 3300.0 + weight_tons * 125.0
            time_days = 32.0 if "Rotterdam" in destination else 21.0

            leg1 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L1",
                origin=origin,
                destination=f"{origin} Outer Port",
                mode=TransportMode.ROAD,
                carrier=random.choice(CARRIERS_BY_MODE[TransportMode.ROAD]),
                distance_km=60.0,
                cost_usd=round(180.0 + weight_tons * 25, 2),
                transit_time_days=0.8,
                delay_probability=0.08,
                expected_delay_days=0.2,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.20,
                    geopolitical_risk=0.02,
                    weather_risk=0.10,
                    customs_risk=0.12
                ),
                co2_kg=round(45.0 * weight_tons, 2)
            )
            leg2 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L2",
                origin=f"{origin} Outer Port",
                destination=f"{destination} Gateway Port",
                mode=TransportMode.OCEAN,
                carrier="Maersk Line (Cape Route Express)",
                distance_km=dist,
                cost_usd=round(base_cost, 2),
                transit_time_days=time_days,
                delay_probability=0.16,  # much lower delay probability!
                expected_delay_days=1.8,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.28,
                    geopolitical_risk=0.08,  # Drastically reduced geopolitical risk!
                    weather_risk=0.45,  # Slightly higher Southern Ocean swells
                    customs_risk=0.15
                ),
                co2_kg=round(dist * 0.016 * weight_tons, 2)
            )
            leg3 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L3",
                origin=f"{destination} Gateway Port",
                destination=destination,
                mode=TransportMode.RAIL,
                carrier="DB Cargo GreenRail",
                distance_km=140.0,
                cost_usd=round(310.0 + weight_tons * 20, 2),
                transit_time_days=1.0,
                delay_probability=0.09,
                expected_delay_days=0.3,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.20,
                    geopolitical_risk=0.02,
                    weather_risk=0.10,
                    customs_risk=0.15
                ),
                co2_kg=round(35.0 * weight_tons, 2)
            )
            legs = [leg1, leg2, leg3]

        elif strategy == "EXPEDITED_SEA_AIR":
            route_name = f"Multimodal Sea-Air via Dubai Global Hub"
            hub = "Dubai"
            dist_sea = 10500.0
            dist_air = 4800.0
            cost_sea = 1900.0 + weight_tons * 90.0
            cost_air = 2800.0 + weight_tons * 380.0

            leg1 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L1",
                origin=origin,
                destination=hub,
                mode=TransportMode.OCEAN,
                carrier="CMA CGM Arabian Express",
                distance_km=dist_sea,
                cost_usd=round(cost_sea, 2),
                transit_time_days=11.5,
                delay_probability=0.18,
                expected_delay_days=1.2,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.30,
                    geopolitical_risk=0.25,
                    weather_risk=0.15,
                    customs_risk=0.20
                ),
                co2_kg=round(dist_sea * 0.015 * weight_tons, 2)
            )
            leg2 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L2",
                origin=hub,
                destination=destination,
                mode=TransportMode.AIR,
                carrier="Emirates SkyCargo",
                distance_km=dist_air,
                cost_usd=round(cost_air, 2),
                transit_time_days=2.5,  # fast air connection including customs transshipment
                delay_probability=0.09,
                expected_delay_days=0.4,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.22,
                    geopolitical_risk=0.18,
                    weather_risk=0.12,
                    customs_risk=0.22
                ),
                co2_kg=round(dist_air * 0.55 * weight_tons, 2)
            )
            legs = [leg1, leg2]

        elif strategy == "PREMIUM_EXPRESS_AIR":
            route_name = f"Priority Direct Air Cargo Carrier"
            dist = 11000.0
            cost = 7500.0 + weight_tons * 1250.0
            time_days = 2.8

            leg1 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L1",
                origin=origin,
                destination=destination,
                mode=TransportMode.AIR,
                carrier=random.choice(CARRIERS_BY_MODE[TransportMode.AIR]),
                distance_km=dist,
                cost_usd=round(cost, 2),
                transit_time_days=time_days,
                delay_probability=0.05,
                expected_delay_days=0.2,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.18,
                    geopolitical_risk=0.08,
                    weather_risk=0.20,
                    customs_risk=0.15
                ),
                co2_kg=round(dist * 0.65 * weight_tons, 2)
            )
            legs = [leg1]

        elif strategy == "BALANCED_INTERMODAL":
            route_name = f"Smart Intermodal with Guaranteed Berth & Priority Rail"
            dist_sea = 17500.0
            cost_sea = 3450.0 + weight_tons * 120.0
            time_days = 20.0

            leg1 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L1",
                origin=origin,
                destination=f"{origin} Priority Pier",
                mode=TransportMode.ROAD,
                carrier="Priority Drayage Services",
                distance_km=50.0,
                cost_usd=round(210.0 + weight_tons * 30, 2),
                transit_time_days=0.6,
                delay_probability=0.06,
                expected_delay_days=0.1,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.15,
                    geopolitical_risk=0.04,
                    weather_risk=0.08,
                    customs_risk=0.10
                ),
                co2_kg=round(40.0 * weight_tons, 2)
            )
            leg2 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L2",
                origin=f"{origin} Priority Pier",
                destination=f"{destination} Dedicated Intermodal Yard",
                mode=TransportMode.OCEAN,
                carrier="Hapag-Lloyd Priority Class",
                distance_km=dist_sea,
                cost_usd=round(cost_sea, 2),
                transit_time_days=time_days,
                delay_probability=0.19,  # moderate
                expected_delay_days=1.9,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.32,
                    geopolitical_risk=0.38,
                    weather_risk=0.25,
                    customs_risk=0.18
                ),
                co2_kg=round(dist_sea * 0.015 * weight_tons, 2)
            )
            leg3 = Leg(
                leg_id=f"{shipment_id}-R{route_index}-L3",
                origin=f"{destination} Dedicated Intermodal Yard",
                destination=destination,
                mode=TransportMode.RAIL,
                carrier="GreenRail Express",
                distance_km=180.0,
                cost_usd=round(380.0 + weight_tons * 25, 2),
                transit_time_days=0.9,
                delay_probability=0.07,
                expected_delay_days=0.2,
                risk_indicators=RiskIndicators(
                    congestion_risk=0.18,
                    geopolitical_risk=0.03,
                    weather_risk=0.12,
                    customs_risk=0.14
                ),
                co2_kg=round(45.0 * weight_tons, 2)
            )
            legs = [leg1, leg2, leg3]

        return route_name, corridor_type, legs

    def generate_shipment(self, index: int, origin: str, dest: str) -> Shipment:
        """Generates a complete shipment with 3-5 distinct route options."""
        shipment_id = f"SHP-{index:03d}"
        cargo_types = list(CargoCategory)
        cargo_type = cargo_types[index % len(cargo_types)]

        # Tailor cargo attributes based on category
        if cargo_type == CargoCategory.PHARMA:
            weight = round(random.uniform(1.5, 6.0), 1)
            cargo_value = round(random.uniform(250000.0, 950000.0), 2)
            target_days = random.choice([5.0, 8.0, 12.0])
            delay_penalty = round(cargo_value * 0.02, 2)  # High penalty for spoiled / cold chain goods
            priority = "CRITICAL"
        elif cargo_type == CargoCategory.ELECTRONICS:
            weight = round(random.uniform(3.0, 15.0), 1)
            cargo_value = round(random.uniform(180000.0, 600000.0), 2)
            target_days = random.choice([7.0, 14.0, 21.0])
            delay_penalty = round(cargo_value * 0.012, 2)
            priority = "HIGH"
        elif cargo_type == CargoCategory.AUTOMOTIVE:
            weight = round(random.uniform(8.0, 24.0), 1)
            cargo_value = round(random.uniform(120000.0, 400000.0), 2)
            target_days = random.choice([16.0, 22.0, 28.0])
            delay_penalty = round(cargo_value * 0.015, 2)  # Assembly line stoppage risk
            priority = "HIGH"
        elif cargo_type == CargoCategory.RETAIL:
            weight = round(random.uniform(10.0, 28.0), 1)
            cargo_value = round(random.uniform(50000.0, 180000.0), 2)
            target_days = random.choice([24.0, 32.0, 38.0])
            delay_penalty = round(cargo_value * 0.005, 2)
            priority = "STANDARD"
        else:  # INDUSTRIAL
            weight = round(random.uniform(15.0, 45.0), 1)
            cargo_value = round(random.uniform(40000.0, 150000.0), 2)
            target_days = random.choice([30.0, 40.0, 50.0])
            delay_penalty = round(cargo_value * 0.003, 2)
            priority = "STANDARD"

        origin_info = GLOBAL_HUBS.get(origin, {"country": "Unknown", "coords": (0.0, 0.0)})
        dest_info = GLOBAL_HUBS.get(dest, {"country": "Unknown", "coords": (0.0, 0.0)})

        # Select 3 to 5 routing strategies for this shipment
        all_strategies = [
            "CHEAPEST_OCEAN",
            "DEFENSIVE_OCEAN",
            "EXPEDITED_SEA_AIR",
            "PREMIUM_EXPRESS_AIR",
            "BALANCED_INTERMODAL"
        ]
        # Ensure at least 4 route options per shipment for rich decision-making
        selected_strategies = random.sample(all_strategies, k=random.choice([4, 5]))

        routes: List[Route] = []
        for r_idx, strat in enumerate(selected_strategies, start=1):
            name, corr_type, legs = self._generate_legs_for_template(
                shipment_id=shipment_id,
                route_index=r_idx,
                origin=origin,
                destination=dest,
                strategy=strat,
                weight_tons=weight
            )
            route = Route(
                route_id=f"{shipment_id}-R{r_idx}",
                shipment_id=shipment_id,
                name=name,
                corridor_type=corr_type,
                legs=legs
            )
            routes.append(route)

        return Shipment(
            shipment_id=shipment_id,
            cargo_type=cargo_type,
            origin_city=origin,
            origin_country=origin_info["country"],
            origin_coords=origin_info["coords"],
            destination_city=dest,
            destination_country=dest_info["country"],
            destination_coords=dest_info["coords"],
            cargo_value_usd=cargo_value,
            cargo_weight_tons=weight,
            target_delivery_days=target_days,
            delay_penalty_per_day_usd=delay_penalty,
            priority_level=priority,
            route_options=routes
        )

    def generate_dataset(self, num_shipments: int = 40) -> List[Shipment]:
        """Generates a comprehensive dataset of realistic shipments."""
        shipments: List[Shipment] = []
        for i in range(1, num_shipments + 1):
            origin, dest = TRADE_CORRIDORS[(i - 1) % len(TRADE_CORRIDORS)]
            shp = self.generate_shipment(i, origin, dest)
            shipments.append(shp)
        return shipments

    def export_dataset(self, shipments: List[Shipment], json_path: str, csv_path: str) -> None:
        """Exports dataset to JSON and flat CSV format for analysis."""
        # JSON export
        data = [shp.model_dump() for shp in shipments]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Flat CSV export of all routes
        flat_rows = []
        for s in shipments:
            for r in s.route_options:
                flat_rows.append({
                    "shipment_id": s.shipment_id,
                    "cargo_type": s.cargo_type.value,
                    "origin": s.origin_city,
                    "destination": s.destination_city,
                    "cargo_value_usd": s.cargo_value_usd,
                    "cargo_weight_tons": s.cargo_weight_tons,
                    "target_delivery_days": s.target_delivery_days,
                    "route_id": r.route_id,
                    "route_name": r.name,
                    "corridor_type": r.corridor_type,
                    "num_legs": len(r.legs),
                    "total_cost_usd": r.total_cost_usd,
                    "total_transit_time_days": r.total_transit_time_days,
                    "overall_delay_probability": r.overall_delay_probability,
                    "expected_delay_days": r.expected_delay_days,
                    "composite_risk": r.composite_risk_score,
                    "reliability_score": r.reliability_score,
                    "total_co2_kg": r.total_co2_kg,
                    "intermediate_hubs": ", ".join(r.intermediate_hubs) if r.intermediate_hubs else "Direct"
                })
        df = pd.DataFrame(flat_rows)
        df.to_csv(csv_path, index=False)
