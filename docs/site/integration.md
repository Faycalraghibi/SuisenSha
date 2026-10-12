# Client SDKs & Widget Integration

SuisenSha provides ready-to-use client libraries and an embeddable UI carousel widget.

## 1. Embeddable JavaScript Widget

Add real-time recommendation carousels to any web storefront or application with vanilla HTML & JS:

```html
<!-- 1. Include Stylesheet -->
<link rel="stylesheet" href="https://api.suisensha.example.com/static/widget.css" />

<!-- 2. Container element -->
<div id="suisensha-carousel"></div>

<!-- 3. Include and initialize widget -->
<script src="https://api.suisensha.example.com/static/widget.js"></script>
<script>
  const widget = new SuisenShaWidget({
    containerId: "suisensha-carousel",
    apiUrl: "https://api.suisensha.example.com",
    apiKey: "your-client-api-key",
    userId: "user_123",
    model: "hybrid",
    title: "Recommended For You",
    limit: 6,
    onItemClick: (item) => {
      console.log("Clicked item:", item.item_id);
    }
  });
</script>
```

### Widget Features
- Horizontal sliding carousel with navigation arrows.
- **Explainability Badges**: Clicking the badge displays the DSA transparency modal with score breakdown and rationale.
- **Dislike / Exclude button**: Directly emits a `dislike` event, removing the item from future recommendations.

---

## 2. Python Client SDK

Install from `sdks/python/` or private PyPI:

```bash
pip install ./sdks/python
```

Usage:

```python
from suisensha import SuisenShaClient

client = SuisenShaClient(
    base_url="https://api.suisensha.example.com",
    api_key="your-api-key",
    tenant_id="acme_corp",
)

# Get top recommendations
response = client.recommend_hybrid(user_id="user_42", n=5, explain=True)
for item in response.recommendations:
    print(f"Item: {item.item_id}, Score: {item.score:.4f}, Explanation: {item.explanation}")

# Record event
client.send_events([{"user_id": "user_42", "item_id": "item_99", "event_type": "click"}])
```

---

## 3. TypeScript Client SDK

Install from `sdks/typescript/` or npm:

```bash
npm install @suisensha/sdk
```

Usage:

```typescript
import { SuisenShaClient } from "@suisensha/sdk";

const client = new SuisenShaClient({
  baseUrl: "https://api.suisensha.example.com",
  apiKey: "your-api-key",
  tenantId: "acme_corp",
});

const res = await client.recommendHybrid({
  user_id: "user_42",
  n: 5,
  explain: true,
});
console.log(res.recommendations);
```
