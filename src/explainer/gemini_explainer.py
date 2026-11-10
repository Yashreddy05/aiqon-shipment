"""
Gemini AI Generative Explainer for Global Logistics Routing.
Integrates Google GenAI SDK (google.genai) to generate executive dispatcher briefings,
contingency recommendations, and interactive logistics reasoning.
"""

from __future__ import annotations
import os
import importlib
from typing import Optional, Dict, Any, List

# Prefer modern google.genai SDK; fallback to google.generativeai if needed
USE_MODERN_SDK = False
try:
    from google import genai
    USE_MODERN_SDK = True
    GENAI_AVAILABLE = True
except ImportError:
    try:
        legacy_genai = importlib.import_module("google.generativeai")
        GENAI_AVAILABLE = True
    except ImportError:
        GENAI_AVAILABLE = False

from src.models.schemas import Shipment, RecommendationResult, Route


class GeminiRouteAdvisor:
    """Generative AI assistant for logistics operators powered by Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash", *args, **kwargs):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = kwargs.get("model_name", model_name)
        self.client = None

        if GENAI_AVAILABLE and self.api_key:
            try:
                if USE_MODERN_SDK:
                    self.client = genai.Client(api_key=self.api_key)
                else:
                    legacy_genai.configure(api_key=self.api_key)
                    self.client = legacy_genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini client: {e}")
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _call_gemini(self, prompt: str) -> str:
        """Invokes Gemini with automatic model fallback if a model is deprecated or unavailable."""
        candidate_models = [self.model_name, "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        # Remove duplicates while preserving order
        unique_models = list(dict.fromkeys(candidate_models))

        last_error = None
        for m in unique_models:
            try:
                if USE_MODERN_SDK:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt
                    )
                    return response.text
                else:
                    gen_model = legacy_genai.GenerativeModel(m)
                    response = gen_model.generate_content(prompt)
                    return response.text
            except Exception as e:
                last_error = e
                # If model is deprecated or not found (404 / 400), try next candidate model
                if "not found" in str(e).lower() or "deprecated" in str(e).lower() or "404" in str(e):
                    continue
                else:
                    raise e

        # If all candidates failed, raise last error
        raise last_error

    def generate_executive_briefing(
        self,
        recommendation: RecommendationResult,
        shipment: Shipment
    ) -> str:
        """Calls Gemini to generate a high-level executive rationale and operational contingency advice."""
        if not self.is_available():
            return None

        rec = recommendation.recommended_route
        runner = recommendation.runner_up_route
        sim = rec.simulation_stats or {}

        runner_name = runner.name if runner else "None"
        runner_cost = f"${runner.total_cost_usd:,.2f}" if runner else "N/A"
        runner_time = f"{runner.total_transit_time_days:.1f} days" if runner else "N/A"
        runner_delay = f"{runner.overall_delay_probability*100:.1f}%" if runner else "N/A"
        runner_risk = f"{runner.composite_risk_score:.3f}" if runner else "N/A"

        prompt = f"""
You are the Chief Freight Operations Officer at a global logistics firm (similar to Flexport / Maersk / Aiqon).
Analyze this multi-objective shipment routing decision and provide a sharp, professional 3-paragraph executive briefing for the logistics dispatch desk.

### SHIPMENT CONTEXT
- Shipment ID: {shipment.shipment_id}
- Cargo Type: {shipment.cargo_type.value}
- Origin: {shipment.origin_city}, {shipment.origin_country}
- Destination: {shipment.destination_city}, {shipment.destination_country}
- Cargo Value: ${shipment.cargo_value_usd:,.2f}
- Weight: {shipment.cargo_weight_tons:.1f} tons
- Target Delivery SLA: {shipment.target_delivery_days:.1f} days
- Delay Penalty: ${shipment.delay_penalty_per_day_usd:,.2f} per day late
- Operating Policy: {recommendation.operator_profile}

### RECOMMENDED ROUTE
- Name: {rec.name} ({rec.route_id})
- Total Freight Cost: ${rec.total_cost_usd:,.2f}
- Planned Transit Time: {rec.total_transit_time_days:.1f} days
- Delay Probability: {rec.overall_delay_probability*100:.1f}%
- Composite Risk Index: {rec.composite_risk_score:.3f} (0 to 1 scale)
- Simulated On-Time In-Full (OTIF) Probability: {sim.get('otif_probability', 0.85)*100:.1f}%
- Expected Realized Cost (with delay penalties): ${sim.get('expected_realized_cost_usd', rec.total_cost_usd):,.2f}
- Pareto Status: {'Pareto Optimal (Non-dominated)' if recommendation.is_recommended_pareto else 'Constrained Selection'}
- Route Legs: {', '.join([f"{leg.mode.value} via {leg.carrier} ({leg.origin}->{leg.destination})" for leg in rec.legs])}

### RUNNER-UP ROUTE
- Name: {runner_name}
- Cost: {runner_cost}
- Transit Time: {runner_time}
- Delay Probability: {runner_delay}
- Composite Risk: {runner_risk}

### YOUR BRIEFING MUST COVER:
1. **Executive Decision Rationale**: Why this route is the superior choice under the {recommendation.operator_profile} policy, specifically addressing the trade-off against cheaper or faster alternatives.
2. **Risk Mitigation & Choke Point Analysis**: Comment on specific route vulnerabilities (such as Red Sea / Bab-el-Mandeb, port congestion at hubs, or customs friction) and why the chosen path defends against them.
3. **Actionable Operational Directives**: 2-3 concrete bullet points for the ground dispatch team (e.g. berth reservations, customs pre-clearance, tracking milestones).

Tone: Professional, direct, analytical, authoritative. Format with clear Markdown headings.
"""

        try:
            return self._call_gemini(prompt)
        except Exception as e:
            return f"*(Gemini API invocation note: {str(e)}. Displaying algorithmic rule-based rationale.)*"

    def answer_operator_question(
        self,
        question: str,
        recommendation: RecommendationResult,
        shipment: Shipment
    ) -> str:
        """Answers an interactive operator query about the recommendation."""
        if not self.is_available():
            return "Gemini API is not configured. Please provide a valid Gemini API key in the sidebar."

        rec = recommendation.recommended_route
        prompt = f"""
You are the Aiqon AI Logistics Copilot. Answer the following operator question concisely and accurately based on the shipment and routing decision:

Shipment: {shipment.shipment_id} ({shipment.cargo_type.value}) from {shipment.origin_city} to {shipment.destination_city}.
Recommended Route: {rec.name} (${rec.total_cost_usd:,.2f}, {rec.total_transit_time_days:.1f} days, risk: {rec.composite_risk_score:.2f}).
Total Route Options Evaluated: {len(shipment.route_options)}.

Operator Question: "{question}"

Answer directly in 2-4 sentences with clear operational facts:
"""
        try:
            return self._call_gemini(prompt)
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"
