"""
Command-Line Interface (CLI) for Aiqon Shipment Route Optimizer.
Provides interactive inspection, recommendations, trade-off breakdowns, and benchmarking.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.models.schemas import (
    Shipment,
    OperatorProfile,
    OptimizationWeights,
    PROFILE_WEIGHTS,
)
from src.data.generator import RouteGenerator
from src.optimization.scoring import RouteOptimizer
from src.explainer.reasoning import RouteDecisionExplainer
from src.evaluation.benchmark import BenchmarkSuite

console = Console()
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHIPMENTS_FILE = os.path.join(DATA_DIR, "shipments.json")
CSV_FILE = os.path.join(DATA_DIR, "routes_summary.csv")


def load_shipments() -> List[Shipment]:
    """Loads shipments from data file or generates if missing."""
    if not os.path.exists(SHIPMENTS_FILE):
        console.print("[yellow]Dataset not found. Generating default 40 shipments...[/yellow]")
        gen = RouteGenerator(seed=42)
        os.makedirs(DATA_DIR, exist_ok=True)
        shipments = gen.generate_dataset(40)
        gen.export_dataset(shipments, SHIPMENTS_FILE, CSV_FILE)
        return shipments

    with open(SHIPMENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Shipment.model_validate(s) for s in data]


def cmd_list(args):
    """List all available shipments."""
    shipments = load_shipments()
    table = Table(title="Available International Shipments", show_header=True, header_style="bold magenta")
    table.add_column("Shipment ID", style="cyan")
    table.add_column("Cargo Type", style="green")
    table.add_column("Origin", style="yellow")
    table.add_column("Destination", style="yellow")
    table.add_column("Weight (t)", justify="right")
    table.add_column("Cargo Value", justify="right")
    table.add_column("Priority", style="red")
    table.add_column("Route Options", justify="center")

    for s in shipments:
        table.add_row(
            s.shipment_id,
            s.cargo_type.value,
            s.origin_city,
            s.destination_city,
            f"{s.cargo_weight_tons:.1f}",
            f"${s.cargo_value_usd:,.0f}",
            s.priority_level,
            str(len(s.route_options))
        )
    console.print(table)


def cmd_inspect(args):
    """Inspect detailed route options and legs for a specific shipment."""
    shipments = load_shipments()
    shipment = next((s for s in shipments if s.shipment_id == args.shipment_id), None)
    if not shipment:
        console.print(f"[bold red]Shipment {args.shipment_id} not found![/bold red]")
        sys.exit(1)

    console.print(Panel(
        f"[bold cyan]Shipment ID:[/bold cyan] {shipment.shipment_id} | "
        f"[bold green]Cargo:[/bold green] {shipment.cargo_type.value} | "
        f"[bold yellow]Route:[/bold yellow] {shipment.origin_city} -> {shipment.destination_city} | "
        f"[bold]Weight:[/bold] {shipment.cargo_weight_tons}t | "
        f"[bold]Value:[/bold] ${shipment.cargo_value_usd:,.2f} | "
        f"[bold]Target Days:[/bold] {shipment.target_delivery_days}d",
        title=f"Shipment Overview: {shipment.shipment_id}",
        border_style="blue"
    ))

    table = Table(title="Available Route Options", show_header=True, header_style="bold blue")
    table.add_column("Route ID", style="cyan")
    table.add_column("Strategy / Name", style="white")
    table.add_column("Legs", justify="center")
    table.add_column("Cost (USD)", justify="right", style="green")
    table.add_column("Transit (d)", justify="right", style="yellow")
    table.add_column("Delay Prob", justify="right", style="red")
    table.add_column("Risk Score", justify="right", style="magenta")
    table.add_column("CO2 (kg)", justify="right")

    for r in shipment.route_options:
        table.add_row(
            r.route_id,
            r.name,
            str(len(r.legs)),
            f"${r.total_cost_usd:,.2f}",
            f"{r.total_transit_time_days:.1f}",
            f"{r.overall_delay_probability*100:.1f}%",
            f"{r.composite_risk_score:.3f}",
            f"{r.total_co2_kg:,.0f}"
        )
    console.print(table)


def cmd_recommend(args):
    """Run recommendation engine with operator explainability for a shipment."""
    shipments = load_shipments()
    shipment = next((s for s in shipments if s.shipment_id == args.shipment_id), None)
    if not shipment:
        console.print(f"[bold red]Shipment {args.shipment_id} not found![/bold red]")
        sys.exit(1)

    profile_map = {
        "cost": OperatorProfile.COST_MINIMIZER,
        "time": OperatorProfile.TIME_CRITICAL,
        "risk": OperatorProfile.RISK_AVERSE,
        "balanced": OperatorProfile.BALANCED,
    }
    selected_profile = profile_map.get(args.profile.lower(), OperatorProfile.BALANCED)

    custom_weights = None
    if args.custom_weights:
        parts = [float(x.strip()) for x in args.custom_weights.split(",")]
        if len(parts) == 3:
            selected_profile = OperatorProfile.CUSTOM
            custom_weights = OptimizationWeights(cost_weight=parts[0], time_weight=parts[1], risk_weight=parts[2])

    optimizer = RouteOptimizer()
    result = optimizer.recommend(
        shipment=shipment,
        profile=selected_profile,
        custom_weights=custom_weights,
        run_monte_carlo=True
    )
    result = RouteDecisionExplainer.explain_recommendation(result, shipment)

    # Output results
    console.print(Panel(
        f"[bold green]RECOMMENDED ROUTE:[/bold green] {result.recommended_route_name} ({result.recommended_route_id})\n"
        f"[bold]Profile Applied:[/bold] {result.operator_profile}\n"
        f"[bold]Pareto Status:[/bold] {'[green]Pareto Optimal[/green]' if result.is_recommended_pareto else '[yellow]Non-Pareto (Constrained/Weighted)[/yellow]'}\n"
        f"[bold]Freight Cost:[/bold] ${result.recommended_route.total_cost_usd:,.2f}\n"
        f"[bold]Planned Transit Time:[/bold] {result.recommended_route.total_transit_time_days:.1f} days\n"
        f"[bold]Delay Probability:[/bold] {result.recommended_route.overall_delay_probability*100:.1f}%\n"
        f"[bold]Composite Risk Index:[/bold] {result.recommended_route.composite_risk_score:.3f}",
        title=f"Optimization Result: {shipment.shipment_id}",
        border_style="green"
    ))

    # Explanations panel
    console.print(Panel(
        f"[bold cyan]Operator Decision Rationale:[/bold cyan]\n\n{result.summary_explanation}",
        title="Explainability & Trade-off Analysis",
        border_style="cyan"
    ))

    # Rankings Table
    table = Table(title="Evaluated Route Rankings", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center")
    table.add_column("Route ID", style="cyan")
    table.add_column("Name")
    table.add_column("Score (0-100)", justify="right", style="green")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Transit (d)", justify="right")
    table.add_column("Delay %", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Pareto?", justify="center")

    for r in result.rankings:
        pareto_mark = "[green]YES[/green]" if r["is_pareto"] else "[red]NO[/red]"
        table.add_row(
            str(r["rank"]),
            r["route_id"],
            r["name"],
            f"{r['score']:.1f}",
            f"${r['cost_usd']:,.2f}",
            f"{r['transit_time_days']:.1f}",
            f"{r['delay_probability']*100:.1f}%",
            f"{r['risk_score']:.3f}",
            pareto_mark
        )
    console.print(table)


def cmd_benchmark(args):
    """Run full comparative benchmark against baseline heuristics."""
    shipments = load_shipments()
    console.print(f"[bold green]Running multi-objective evaluation across {len(shipments)} shipments...[/bold green]")
    suite = BenchmarkSuite()
    res = suite.evaluate_shipments(shipments)
    df = res["summary_table"]

    table = Table(title="Framework Evaluation & Benchmark Comparison", show_header=True, header_style="bold cyan")
    for col in df.columns:
        table.add_column(col, justify="right" if col != "Strategy" else "left")

    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row.values])

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Aiqon Multi-Objective Shipment Route Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    subparsers.add_parser("list", help="List all shipments")

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect routes of a shipment")
    inspect_parser.add_argument("--shipment-id", required=True, help="Shipment ID (e.g. SHP-001)")

    # recommend
    rec_parser = subparsers.add_parser("recommend", help="Recommend route for a shipment")
    rec_parser.add_argument("--shipment-id", required=True, help="Shipment ID (e.g. SHP-001)")
    rec_parser.add_argument("--profile", default="balanced", choices=["balanced", "cost", "time", "risk"], help="Operator profile")
    rec_parser.add_argument("--custom-weights", help="Comma separated cost,time,risk weights (e.g. '0.4,0.3,0.3')")

    # benchmark
    subparsers.add_parser("benchmark", help="Run benchmark across all shipments")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
