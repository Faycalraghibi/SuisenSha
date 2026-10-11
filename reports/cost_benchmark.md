# Cost Benchmark: Self-Hosted SuisenSha vs. Managed Cloud Recommenders

**Baseline Assumptions:**
- Monthly Recommendations: 1,000,000
- Monthly Explanations: 1,000

## Monthly Serving & Infrastructure Cost Comparison

| Provider | Architecture | Cost per 1M Recs | Cost per 1k Explanations | Total Monthly Cost | Estimated Savings |
|---|---|---|---|---|---|
| **SuisenSha (Self-Hosted)** | Docker / c6i.xlarge + Redis | €134.92 | €0.05 | €134.92 | **Baseline** |
| **AWS Personalize** | Managed SaaS (TPS + API fee) | €185.84 | €1.84 | €187.68 | -28.1% vs AWS |
| **GCP Vertex AI Retail** | Managed Cloud API ($2.50/k req) | €2300.00 | N/A | €2300.00 | -94.1% vs GCP |

## Key Takeaways
1. **Cost Efficiency:** Self-hosting SuisenSha yields a 28.1% reduction against AWS Personalize and over 94.1% reduction against GCP Vertex AI Retail.
2. **Zero Vendor Lock-in:** Models and feature stores remain entirely within your own VPC with open telemetry and deterministic latency budgets.
3. **Transparent Explanations:** Local template and quantized RAG explanations cost under €0.05 per 1,000 explanations, compared to €1.84+ on managed cloud LLMs.
