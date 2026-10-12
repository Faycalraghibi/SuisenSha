# @suisensha/sdk

Official TypeScript / JavaScript client SDK for **SuisenSha** recommendation platform.

## Features

- Zero runtime external dependencies (uses native `fetch` and `AbortController`)
- Full TypeScript types for all recommendation endpoints, events, sessions, rules, and experiments
- Multi-tenancy support via `tenantId` / `X-Tenant-ID`
- Native support for modern Node.js (18+), Bun, Deno, and modern browsers

## Installation

```bash
npm install @suisensha/sdk
# or
pnpm add @suisensha/sdk
```

## Quick Start

```typescript
import { SuisenShaClient } from "@suisensha/sdk";

const client = new SuisenShaClient({
  baseUrl: "https://api.suisensha.example.com",
  apiKey: "your-api-key",
  tenantId: "tenant-acme",
});

// 1. Get hybrid recommendations
const res = await client.recommendHybrid({
  user_id: "user_42",
  n: 5,
  explain: true,
});
console.log(res.recommendations);

// 2. Ingest user events
await client.sendEvents([
  {
    user_id: "user_42",
    item_id: "item_123",
    event_type: "click",
  },
]);

// 3. Session tracking
const session = await client.recordSessionEvent("sess_abc", "item_456", "view");
console.log("Current session length:", session.length);
```
