from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostBenchmark:
    monthly_recommendations: int
    monthly_explanations: int
    suisensha_monthly_cost_eur: float
    aws_personalize_monthly_cost_eur: float
    gcp_retail_monthly_cost_eur: float
    suisensha_cost_per_million_eur: float
    aws_cost_per_million_eur: float
    gcp_cost_per_million_eur: float
    suisensha_cost_per_thousand_explanations_eur: float
    aws_cost_per_thousand_explanations_eur: float

    def to_markdown(self) -> str:
        savings_vs_aws = (
            (1.0 - self.suisensha_monthly_cost_eur / self.aws_personalize_monthly_cost_eur) * 100
            if self.aws_personalize_monthly_cost_eur > 0
            else 0.0
        )
        savings_vs_gcp = (
            (1.0 - self.suisensha_monthly_cost_eur / self.gcp_retail_monthly_cost_eur) * 100
            if self.gcp_retail_monthly_cost_eur > 0
            else 0.0
        )

        return (
            f"# Cost Benchmark: Self-Hosted SuisenSha vs. Managed Cloud Recommenders\n\n"
            f"**Baseline Assumptions:**\n"
            f"- Monthly Recommendations: {self.monthly_recommendations:,}\n"
            f"- Monthly Explanations: {self.monthly_explanations:,}\n\n"
            f"## Monthly Serving & Infrastructure Cost Comparison\n\n"
            f"| Provider | Architecture | Cost per 1M Recs | Cost per 1k Explanations "
            f"| Total Monthly Cost | Estimated Savings |\n"
            f"|---|---|---|---|---|---|\n"
            f"| **SuisenSha (Self-Hosted)** | Docker / c6i.xlarge + Redis | "
            f"€{self.suisensha_cost_per_million_eur:.2f} | "
            f"€{self.suisensha_cost_per_thousand_explanations_eur:.2f} | "
            f"€{self.suisensha_monthly_cost_eur:.2f} | **Baseline** |\n"
            f"| **AWS Personalize** | Managed SaaS (TPS + API fee) | "
            f"€{self.aws_cost_per_million_eur:.2f} | "
            f"€{self.aws_cost_per_thousand_explanations_eur:.2f} | "
            f"€{self.aws_personalize_monthly_cost_eur:.2f} | -{savings_vs_aws:.1f}% vs AWS |\n"
            f"| **GCP Vertex AI Retail** | Managed Cloud API ($2.50/k req) | "
            f"€{self.gcp_cost_per_million_eur:.2f} | N/A | "
            f"€{self.gcp_retail_monthly_cost_eur:.2f} | -{savings_vs_gcp:.1f}% vs GCP |\n\n"
            f"## Key Takeaways\n"
            f"1. **Cost Efficiency:** Self-hosting SuisenSha yields a {savings_vs_aws:.1f}% "
            f"reduction against AWS Personalize and over {savings_vs_gcp:.1f}% against GCP "
            f"Vertex AI Retail.\n"
            f"2. **Zero Vendor Lock-in:** Models and feature stores remain entirely within your "
            f"own VPC with open telemetry and deterministic latency budgets.\n"
        )


def calculate_serving_costs(
    monthly_recommendations: int = 1_000_000,
    monthly_explanations: int = 1_000,
) -> CostBenchmark:
    """Calculate serving costs comparing self-hosted SuisenSha to managed cloud services."""
    # Conversion rate USD to EUR (~0.92)
    usd_to_eur = 0.92

    # 1. SuisenSha Self-Hosted Cost Model:
    # - 1x AWS c6i.xlarge (4 vCPUs, 8 GB RAM) @ $0.17/hr * 730h = $124.10/mo
    # - 1x Redis instance (cache.t4g.micro) @ $12.50/mo
    # - Egress & EBS volume storage @ $10.00/mo
    # Total infrastructure = ~$146.60 USD = ~€134.87 EUR
    suisensha_infra_eur = 146.60 * usd_to_eur
    # Local explanation generation cost (negligible CPU compute) ~ €0.05 per 1,000 explanations
    suisensha_expl_unit_eur = 0.05
    suisensha_expl_eur = (monthly_explanations / 1_000) * suisensha_expl_unit_eur
    suisensha_total_eur = suisensha_infra_eur + suisensha_expl_eur
    suisensha_per_million = (
        suisensha_total_eur / (monthly_recommendations / 1_000_000)
        if monthly_recommendations >= 1_000_000
        else suisensha_total_eur
    )

    # 2. AWS Personalize Cost Model:
    # - Minimum 1 TPS provisioned @ $0.20/hr * 730h = $146.00/mo
    # - Recommendation generation fee: $0.05 per 1,000 requests = $50.00 per 1M
    # - Retraining compute fee: ~$6.00/mo
    # - LLM explanation via Bedrock Titan/Haiku: ~$2.00 per 1k explanations
    aws_base_usd = 146.00 + 6.00
    aws_recs_usd = (monthly_recommendations / 1_000) * 0.05
    aws_expl_unit_usd = 2.00
    aws_expl_usd = (monthly_explanations / 1_000) * aws_expl_unit_usd
    aws_total_eur = (aws_base_usd + aws_recs_usd + aws_expl_usd) * usd_to_eur
    aws_per_million = (
        (aws_base_usd + aws_recs_usd) * usd_to_eur / (monthly_recommendations / 1_000_000)
    )

    # 3. GCP Vertex AI Retail Cost Model:
    # - Query fee: $2.50 per 1,000 requests = $2,500.00 per 1M requests
    gcp_total_usd = (monthly_recommendations / 1_000) * 2.50
    gcp_total_eur = gcp_total_usd * usd_to_eur
    gcp_per_million = gcp_total_eur / (monthly_recommendations / 1_000_000)

    return CostBenchmark(
        monthly_recommendations=monthly_recommendations,
        monthly_explanations=monthly_explanations,
        suisensha_monthly_cost_eur=round(suisensha_total_eur, 2),
        aws_personalize_monthly_cost_eur=round(aws_total_eur, 2),
        gcp_retail_monthly_cost_eur=round(gcp_total_eur, 2),
        suisensha_cost_per_million_eur=round(suisensha_per_million, 2),
        aws_cost_per_million_eur=round(aws_per_million, 2),
        gcp_cost_per_million_eur=round(gcp_per_million, 2),
        suisensha_cost_per_thousand_explanations_eur=round(suisensha_expl_unit_eur, 2),
        aws_cost_per_thousand_explanations_eur=round(aws_expl_unit_usd * usd_to_eur, 2),
    )
