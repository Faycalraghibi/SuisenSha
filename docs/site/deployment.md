# Deployment & Kubernetes Helm

SuisenSha provides a production Helm chart located at `deploy/helm/suisensha/`.

## Prerequisites

- Kubernetes cluster 1.25+
- Helm v3.8+
- Ingress controller (e.g. `ingress-nginx`)

## Quick Install

```bash
# Add repo or navigate to helm directory
cd deploy/helm/suisensha

# Inspect default values
helm show values .

# Install release
helm install suisensha . \
  --namespace suisensha-prod \
  --create-namespace \
  --set api.replicaCount=3 \
  --set api.autoscaling.enabled=true \
  --set redis.enabled=true
```

## Autoscaling (HPA)

The API deployment includes Horizontal Pod Autoscaler configurations targeting both CPU and memory:

```yaml
api:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 75
    targetMemoryUtilizationPercentage: 80
```

When load peaks (e.g. during sale events), Kubernetes automatically scales API replicas up to handle incoming throughput, scaling back down when load subsides.

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `api.replicaCount` | `2` | Initial replicas when HPA is disabled |
| `api.resources.requests.cpu` | `500m` | CPU request |
| `api.resources.requests.memory` | `1Gi` | Memory request |
| `api.env.RECOMMENDATION_MAX_N` | `"100"` | Upper bound on items per request |
| `api.env.RATE_LIMIT_PER_MINUTE` | `"60"` | Burst rate limit per key |
| `api.env.DEFAULT_MONTHLY_QUOTA` | `"100000"` | Monthly request limit per tenant |
| `redis.enabled` | `true` | Deploy bundled Redis instance |
| `ingress.enabled` | `false` | Enable Kubernetes Ingress resource |
