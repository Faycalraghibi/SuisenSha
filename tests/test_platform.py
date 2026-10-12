"""Tests for PR-14 platform features: Multi-tenancy, SDKs, Widget, Quotas, and Helm."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdks" / "python"))

import pytest
from fastapi.testclient import TestClient
from suisensha.client import SuisenShaClient

from pipeline.api import dependencies as deps
from pipeline.api.main import app
from pipeline.api.security import SecurityManager, get_security_manager
from pipeline.cache import RedisCache
from pipeline.data.events import EventStore, UserEvent
from pipeline.experiments.manager import ExperimentManager, Variant
from pipeline.mlops.registry import ModelRegistry
from pipeline.ranking.rules import BusinessRule, RuleEngine
from pipeline.sessions.manager import SessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "platform_test.db"


@pytest.fixture
def sec_mgr(tmp_path: Path) -> SecurityManager:
    return SecurityManager(db_path=tmp_path / "sec_platform.db")


@pytest.fixture
def api_client(tmp_path: Path, sec_mgr: SecurityManager):
    ev_store = EventStore(db_path=tmp_path / "events_platform.db")
    sess_mgr = SessionManager(db_path=tmp_path / "sessions_platform.db")

    app.dependency_overrides = {
        get_security_manager: lambda: sec_mgr,
        deps.get_event_store: lambda: ev_store,
        deps.get_session_manager: lambda: sess_mgr,
    }
    yield TestClient(app)
    app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# 1. Multi-Tenant Event & Exclusion Isolation
# ---------------------------------------------------------------------------
def test_event_store_tenant_isolation(tmp_path: Path):
    store = EventStore(db_path=tmp_path / "events.db")

    e_alpha = [
        UserEvent(user_id=1, item_id=10, event_type="click", tenant_id="alpha"),
        UserEvent(user_id=1, item_id=20, event_type="not_interested", tenant_id="alpha"),
    ]
    e_beta = [
        UserEvent(user_id=1, item_id=30, event_type="click", tenant_id="beta"),
    ]
    store.record_batch(e_alpha)
    store.record_batch(e_beta)

    events_alpha = store.get_user_events(1, tenant_id="alpha")
    events_beta = store.get_user_events(1, tenant_id="beta")

    assert len(events_alpha) == 2
    assert {e.item_id for e in events_alpha} == {10, 20}
    assert len(events_beta) == 1
    assert events_beta[0].item_id == 30

    # Exclusion isolation
    excl_alpha = store.get_user_excluded_items(1, tenant_id="alpha")
    excl_beta = store.get_user_excluded_items(1, tenant_id="beta")
    assert 20 in excl_alpha
    assert len(excl_beta) == 0

    # Erasure isolation
    store.erase_user_events(1, tenant_id="alpha")
    assert len(store.get_user_events(1, tenant_id="alpha")) == 0
    assert len(store.get_user_events(1, tenant_id="beta")) == 1


# ---------------------------------------------------------------------------
# 2. Multi-Tenant Session Isolation
# ---------------------------------------------------------------------------
def test_session_manager_tenant_isolation(tmp_path: Path):
    mgr = SessionManager(db_path=tmp_path / "sessions.db")

    s1 = mgr.record_event("sess_100", 10, tenant_id="tenant_x")
    assert len(s1.item_ids) == 1
    assert s1.tenant_id == "tenant_x"

    # Querying from different tenant returns None
    s_other = mgr.get_session("sess_100", tenant_id="tenant_y")
    assert s_other is None

    # Querying from same tenant returns items
    s_same = mgr.get_session("sess_100", tenant_id="tenant_x")
    assert s_same.item_ids == [10]

    assert mgr.count_sessions(tenant_id="tenant_x") == 1
    assert mgr.count_sessions(tenant_id="tenant_y") == 0


# ---------------------------------------------------------------------------
# 3. Multi-Tenant Business Rules Isolation
# ---------------------------------------------------------------------------
def test_business_rules_tenant_isolation():
    engine = RuleEngine()

    rule_alpha = BusinessRule(
        rule_id="r_alpha",
        rule_type="boost",
        item_ids={1},
        value=10.0,
        priority=1,
        tenant_id="tenant_a",
    )
    rule_beta = BusinessRule(
        rule_id="r_beta",
        rule_type="boost",
        item_ids={2},
        value=10.0,
        priority=1,
        tenant_id="tenant_b",
    )
    engine.add_rule(rule_alpha)
    engine.add_rule(rule_beta)

    rules_a = engine.get_active_rules(tenant_id="tenant_a")
    rules_b = engine.get_active_rules(tenant_id="tenant_b")

    assert len(rules_a) == 1
    assert rules_a[0].rule_id == "r_alpha"
    assert len(rules_b) == 1
    assert rules_b[0].rule_id == "r_beta"

    # Applying rules segregated by tenant
    candidates = [{"item_id": 1, "score": 0.5}, {"item_id": 2, "score": 0.5}]
    ranked_a = engine.apply(candidates, tenant_id="tenant_a")
    ranked_b = engine.apply(candidates, tenant_id="tenant_b")

    assert ranked_a[0]["item_id"] == 1
    assert ranked_b[0]["item_id"] == 2

    # Remove rule
    engine.remove_rule(rule_alpha.rule_id, tenant_id="tenant_a")
    assert len(engine.get_active_rules(tenant_id="tenant_a")) == 0
    assert len(engine.get_active_rules(tenant_id="tenant_b")) == 1


# ---------------------------------------------------------------------------
# 4. Multi-Tenant Experiments Isolation
# ---------------------------------------------------------------------------
def test_experiment_manager_tenant_isolation(tmp_path: Path):
    mgr = ExperimentManager(data_dir=tmp_path / "exp")

    v1 = [
        Variant(name="control", weight=1.0, model_type="popular"),
        Variant(name="treatment", weight=1.0, model_type="embedding"),
    ]
    v2 = [
        Variant(name="control", weight=1.0, model_type="popular"),
        Variant(name="treatment", weight=1.0, model_type="sasrec"),
    ]

    exp_1 = mgr.create_experiment(
        exp_id="exp_1",
        name="Exp 1",
        variants=v1,
        tenant_id="client_1",
    )
    exp_2 = mgr.create_experiment(
        exp_id="exp_2",
        name="Exp 2",
        variants=v2,
        tenant_id="client_2",
    )

    list_1 = mgr.list_experiments(tenant_id="client_1")
    list_2 = mgr.list_experiments(tenant_id="client_2")

    assert len(list_1) == 1 and list_1[0].id == exp_1.id
    assert len(list_2) == 1 and list_2[0].id == exp_2.id

    # Lookup cross-tenant
    assert mgr.get_experiment(exp_1.id, tenant_id="client_2") is None
    assert mgr.get_experiment(exp_1.id, tenant_id="client_1") is not None


# ---------------------------------------------------------------------------
# 5. Multi-Tenant Model Registry Isolation
# ---------------------------------------------------------------------------
def test_model_registry_tenant_isolation(tmp_path: Path):
    reg = ModelRegistry(registry_dir=tmp_path / "reg")
    dummy_artifact = tmp_path / "model.pt"
    dummy_artifact.write_text("model weights", encoding="utf-8")

    v_alpha = reg.register_model(
        model_name="hybrid",
        artifact_path=dummy_artifact,
        metrics={"ndcg": 0.42},
        tenant_id="org_alpha",
    )
    reg.promote_to_production(v_alpha.version, model_name="hybrid", tenant_id="org_alpha")

    v_beta = reg.register_model(
        model_name="hybrid",
        artifact_path=dummy_artifact,
        metrics={"ndcg": 0.55},
        tenant_id="org_beta",
    )
    reg.promote_to_production(v_beta.version, model_name="hybrid", tenant_id="org_beta")

    prod_alpha = reg.get_production_version("hybrid", tenant_id="org_alpha")
    prod_beta = reg.get_production_version("hybrid", tenant_id="org_beta")

    assert prod_alpha is not None and prod_alpha.metrics["ndcg"] == 0.42
    assert prod_beta is not None and prod_beta.metrics["ndcg"] == 0.55

    list_alpha = reg.list_versions("hybrid", tenant_id="org_alpha")
    list_beta = reg.list_versions("hybrid", tenant_id="org_beta")
    assert len(list_alpha) == 1 and list_alpha[0].metrics["ndcg"] == 0.42
    assert len(list_beta) == 1 and list_beta[0].metrics["ndcg"] == 0.55


# ---------------------------------------------------------------------------
# 6. Multi-Tenant Cache Key Segregation
# ---------------------------------------------------------------------------
def test_cache_tenant_isolation():
    cache = RedisCache()

    cache.set_top_n(user_id=42, model="hybrid", items=[{"item_id": 10}], tenant_id="tenant_1")
    cache.set_top_n(user_id=42, model="hybrid", items=[{"item_id": 70}], tenant_id="tenant_2")

    items_1 = cache.get_top_n(user_id=42, model="hybrid", tenant_id="tenant_1")
    items_2 = cache.get_top_n(user_id=42, model="hybrid", tenant_id="tenant_2")

    assert items_1 == [{"item_id": 10}]
    assert items_2 == [{"item_id": 70}]

    cache.delete_user(user_id=42, tenant_id="tenant_1")
    assert cache.get_top_n(user_id=42, model="hybrid", tenant_id="tenant_1") is None
    assert cache.get_top_n(user_id=42, model="hybrid", tenant_id="tenant_2") == [{"item_id": 70}]


# ---------------------------------------------------------------------------
# 7. Quota Enforcement & Rate Limits
# ---------------------------------------------------------------------------
def test_tenant_quota_management(sec_mgr: SecurityManager):
    sec_mgr.set_tenant_quota("quota_tenant", limit=3)

    r1 = sec_mgr.check_tenant_quota("quota_tenant")
    assert r1 is True

    r2 = sec_mgr.check_tenant_quota("quota_tenant")
    assert r2 is True

    r3 = sec_mgr.check_tenant_quota("quota_tenant")
    assert r3 is True

    # 4th request exceeds quota
    r4 = sec_mgr.check_tenant_quota("quota_tenant")
    assert r4 is False

    usage = sec_mgr.get_tenant_usage("quota_tenant")
    assert usage["request_count"] == 3
    assert usage["quota_limit"] == 3

    sec_mgr.reset_tenant_usage("quota_tenant")
    r5 = sec_mgr.check_tenant_quota("quota_tenant")
    assert r5 is True


# ---------------------------------------------------------------------------
# 8. Tenant API Security & Header Validation via TestClient
# ---------------------------------------------------------------------------
def test_tenant_header_enforcement(api_client: TestClient, sec_mgr: SecurityManager):
    # Standard key for tenant_a
    key_a, _ = sec_mgr.create_key(
        client_id="client_a", scopes=["read", "write"], tenant_id="tenant_a"
    )
    # Admin key
    key_admin, _ = sec_mgr.create_key(
        client_id="client_admin", scopes=["read", "write", "admin"], tenant_id="default"
    )

    # 1. Matching tenant header -> 200
    res = api_client.get(
        "/health",
        headers={"X-API-Key": key_a, "X-Tenant-ID": "tenant_a"},
    )
    assert res.status_code == 200

    # 2. Mismatched tenant header without admin -> 403 Forbidden
    res_mismatch = api_client.get(
        "/v1/rules",
        headers={"X-API-Key": key_a, "X-Tenant-ID": "tenant_b"},
    )
    assert res_mismatch.status_code == 403
    assert "different tenant" in res_mismatch.json()["detail"].lower()

    # 3. Admin key with tenant override -> 200 OK
    res_admin = api_client.get(
        "/v1/rules",
        headers={"X-API-Key": key_admin, "X-Tenant-ID": "tenant_b"},
    )
    assert res_admin.status_code == 200


# ---------------------------------------------------------------------------
# 9. Static Widget & Demo Endpoints
# ---------------------------------------------------------------------------
def test_static_widget_and_demo(api_client: TestClient):
    js_res = api_client.get("/static/widget.js")
    assert js_res.status_code == 200
    assert "SuisenShaWidget" in js_res.text

    css_res = api_client.get("/static/widget.css")
    assert css_res.status_code == 200
    assert ".suisensha-carousel" in css_res.text

    demo_res = api_client.get("/widget/demo")
    assert demo_res.status_code == 200
    assert "SuisenSha Recommendation Widget Demo" in demo_res.text


# ---------------------------------------------------------------------------
# 10. Python Client SDK Tests
# ---------------------------------------------------------------------------
def test_python_client_sdk_methods():
    client = SuisenShaClient(
        base_url="http://testserver",
        api_key="test-key",
        tenant_id="test-tenant",
    )

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {
            "user_id": 1,
            "recommendations": [{"item_id": 10, "score": 0.95}],
        }
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.recommend_hybrid(user_id=1, k=5)
        assert "recommendations" in res
        assert res["recommendations"][0]["item_id"] == 10

    # Test event recording
    mock_resp.read.return_value = json.dumps({"status": "recorded", "recorded": 1}).encode("utf-8")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        ev_res = client.record_events([{"user_id": 1, "item_id": 10, "event_type": "click"}])
        assert ev_res["status"] == "recorded"

    # Test health
    mock_resp.read.return_value = json.dumps({"status": "healthy"}).encode("utf-8")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        h = client.health()
        assert h["status"] == "healthy"


# ---------------------------------------------------------------------------
# 11. Helm Chart Templates & Config Verification
# ---------------------------------------------------------------------------
def test_helm_chart_structure():
    helm_dir = Path("deploy/helm/suisensha")
    assert helm_dir.exists()

    chart_file = helm_dir / "Chart.yaml"
    values_file = helm_dir / "values.yaml"
    templates_dir = helm_dir / "templates"

    assert chart_file.exists()
    assert values_file.exists()
    assert templates_dir.exists()

    # Read Chart.yaml
    chart_content = chart_file.read_text(encoding="utf-8")
    assert "name: suisensha" in chart_content
    assert "apiVersion: v2" in chart_content

    # Read values.yaml
    values_content = values_file.read_text(encoding="utf-8")
    assert "api:" in values_content
    assert "autoscaling:" in values_content
    assert "redis:" in values_content

    # Check template files
    expected_templates = [
        "_helpers.tpl",
        "api-deployment.yaml",
        "api-service.yaml",
        "api-hpa.yaml",
        "ui-deployment.yaml",
        "ui-service.yaml",
        "redis-deployment.yaml",
        "redis-service.yaml",
        "configmap.yaml",
        "secret.yaml",
        "ingress.yaml",
    ]
    for tmpl in expected_templates:
        tmpl_path = templates_dir / tmpl
        assert tmpl_path.exists(), f"Missing helm template: {tmpl}"
        assert len(tmpl_path.read_text(encoding="utf-8").strip()) > 0
