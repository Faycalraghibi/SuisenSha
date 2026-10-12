# SuisenSha Python SDK

Official typed Python client library for interacting with the SuisenSha Recommendation Engine.

## Installation

```bash
pip install suisensha-sdk
```

## Quickstart

```python
from suisensha import SuisenShaClient

# Initialize client with optional API key and Tenant ID
client = SuisenShaClient(
    base_url="http://localhost:8000",
    api_key="sk_...",
    tenant_id="tenant_a",
)

# Fetch sequential recommendations
recs = client.recommend_sasrec(user_id=42, k=10)
print(f"Model: {recs['model_name']}")
for item in recs["recommendations"]:
    print(f"- {item['title']} ({item.get('genres')})")

# Ingest live user click event
client.record_events([{"user_id": 42, "item_id": 110, "event_type": "click"}])
```
