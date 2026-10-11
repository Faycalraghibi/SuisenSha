"""Client Admin Console for SuisenSha recommendation platform.

Replaces the demo UI with an administrative suite:
- API Keys management
- Business rules & conflict inspection
- A/B testing & uplift tracking with sample size calculator
- Model registry, metrics, and sample explanations
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="SuisenSha — Client Admin Console",
    page_icon="⚙️",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("ADMIN_API_KEY", "")


def get_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = st.session_state.get("admin_api_key", API_KEY)
    if key:
        headers["X-API-Key"] = key
    return headers


def api_request(method: str, path: str, json_data: Any = None, params: Any = None) -> Any:
    try:
        url = f"{API_URL}{path}"
        resp = requests.request(
            method=method,
            url=url,
            headers=get_headers(),
            json=json_data,
            params=params,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            return resp.json()
        st.error(f"API Error ({resp.status_code}): {resp.text}")
        return None
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None


# Sidebar Navigation & Auth
st.sidebar.title("🛠️ Admin Console")
admin_key_input = st.sidebar.text_input("Admin API Key", value=API_KEY, type="password")
st.session_state["admin_api_key"] = admin_key_input

page = st.sidebar.radio(
    "Navigation",
    ["A/B Experiments", "Business Rules", "API Keys & Security", "Models & Explanations"],
)

st.title("⚙️ SuisenSha Recommender Console")

# ---------------------------------------------------------------------------
# Page 1: A/B Experiments
# ---------------------------------------------------------------------------
if page == "A/B Experiments":
    st.header("🧪 A/B Experiments & Uplift Analytics")

    tab1, tab2, tab3 = st.tabs(
        ["Active Experiments", "Create Experiment", "Sample Size Calculator"]
    )

    with tab1:
        st.subheader("Live Experiments")
        experiments = api_request("GET", "/v1/experiments")
        if experiments:
            for exp in experiments:
                with st.expander(
                    f"Experiment: {exp['name']} ({exp['id']}) - Status: {exp['status']}",
                    expanded=True,
                ):
                    st.write(f"**Description:** {exp.get('description', 'N/A')}")
                    st.write(f"**Created:** {exp.get('created_at')}")

                    df_var = pd.DataFrame(exp.get("variants", []))
                    st.dataframe(df_var, use_container_width=True, hide_index=True)

                    # Fetch Uplift Report
                    col_rep, _ = st.columns([2, 1])
                    with col_rep:
                        if st.button(
                            f"Calculate Uplift for {exp['id']}", key=f"uplift_{exp['id']}"
                        ):
                            uplift = api_request("GET", f"/v1/experiments/{exp['id']}/uplift")
                            if uplift:
                                c_m = uplift["control"]
                                t_m = uplift["treatment"]
                                m_col1, m_col2, m_col3 = st.columns(3)
                                m_col1.metric(
                                    "Control CR",
                                    f"{c_m['conversion_rate']:.2%}",
                                    f"{c_m['conversions']}/{c_m['exposures']} users",
                                )
                                m_col2.metric(
                                    "Treatment CR",
                                    f"{t_m['conversion_rate']:.2%}",
                                    f"{t_m['conversions']}/{t_m['exposures']} users",
                                )
                                sig_label = (
                                    "Significant" if uplift["is_significant"] else "Not Significant"
                                )
                                m_col3.metric(
                                    "Relative Uplift",
                                    f"{uplift['relative_uplift_pct']:+.2f}%",
                                    f"p={uplift['p_value']:.4f} ({sig_label})",
                                )

                                ci_msg = (
                                    f"Confidence Interval (95%): "
                                    f"[{uplift['ci_lower']:+.4f}, {uplift['ci_upper']:+.4f}] "
                                    f"| Absolute Uplift: {uplift['absolute_uplift']:+.4f}"
                                )
                                st.info(ci_msg)
        else:
            st.info("No online experiments currently configured.")

    with tab2:
        st.subheader("Launch New Experiment")
        with st.form("create_experiment_form"):
            new_id = st.text_input("Experiment ID (slug)", "exp_sasrec_vs_hybrid")
            new_name = st.text_input("Experiment Name", "SASRec vs. Hybrid MMR Recommender")
            new_desc = st.text_area(
                "Description",
                "Testing conversion uplift of content hybrid embeddings against "
                "sequential transformer.",
            )

            st.write("**Variants**")
            v1_name = st.text_input("Control Variant Name", "control")
            v1_model = st.selectbox("Control Model", ["sasrec", "popularity", "embedding"])
            v1_weight = st.number_input(
                "Control Traffic Weight", min_value=0.1, value=1.0, step=0.1
            )

            v2_name = st.text_input("Treatment Variant Name", "treatment_hybrid")
            v2_model = st.selectbox("Treatment Model", ["hybrid", "sasrec", "embedding"])
            v2_weight = st.number_input(
                "Treatment Traffic Weight", min_value=0.1, value=1.0, step=0.1
            )

            submitted = st.form_submit_button("Create Experiment", type="primary")
            if submitted:
                payload = {
                    "id": new_id,
                    "name": new_name,
                    "description": new_desc,
                    "variants": [
                        {
                            "name": v1_name,
                            "weight": v1_weight,
                            "model_type": v1_model,
                            "params": {},
                        },
                        {
                            "name": v2_name,
                            "weight": v2_weight,
                            "model_type": v2_model,
                            "params": {},
                        },
                    ],
                }
                res = api_request("POST", "/v1/experiments", json_data=payload)
                if res:
                    st.success(f"Experiment '{res['name']}' successfully launched!")
                    st.rerun()

    with tab3:
        st.subheader("Statistical Sample Size Calculator")
        st.markdown(
            "Determine the minimum sample size required per variant to detect "
            "a statistically significant conversion uplift."
        )
        base_cr = st.number_input(
            "Baseline Conversion Rate",
            min_value=0.001,
            max_value=0.99,
            value=0.05,
            step=0.005,
            format="%.3f",
        )
        mde = st.number_input(
            "Minimum Detectable Effect (Relative %)",
            min_value=0.01,
            max_value=2.0,
            value=0.10,
            step=0.01,
            format="%.2f",
        )
        sig_alpha = st.selectbox("Significance Level (Alpha)", [0.05, 0.01, 0.10], index=0)
        power_beta = st.selectbox("Statistical Power (1 - Beta)", [0.80, 0.90, 0.95], index=0)

        if st.button("Calculate Sample Size", type="primary"):
            params = {
                "baseline_conversion": base_cr,
                "mde": mde,
                "alpha": sig_alpha,
                "power": power_beta,
            }
            calc_res = api_request("GET", "/v1/experiments/sample-size", params=params)
            if calc_res:
                c1, c2 = st.columns(2)
                c1.metric(
                    "Required Samples (per variant)",
                    f"{calc_res['required_sample_size_per_variant']:,}",
                )
                c2.metric(
                    "Total Experiment Traffic Required",
                    f"{calc_res['total_required_sample_size']:,}",
                )


# ---------------------------------------------------------------------------
# Page 2: Business Rules
# ---------------------------------------------------------------------------
elif page == "Business Rules":
    st.header("📋 Business Rules & Ranking Controls")

    tab_rules, tab_create = st.tabs(["Active Rules & Conflicts", "Create Rule"])

    with tab_rules:
        rules_data = api_request("GET", "/v1/rules")
        if rules_data:
            st.subheader("Configured Rules")
            if rules_data.get("rules"):
                df_rules = pd.DataFrame(rules_data["rules"])
                st.dataframe(df_rules, use_container_width=True, hide_index=True)
            else:
                st.info("No business rules configured.")

            st.subheader("Rule Conflict Analysis")
            conflicts = rules_data.get("conflicts", [])
            if conflicts:
                for conf in conflicts:
                    st.warning(
                        f"⚠️ Conflict [{conf['conflict_type']}]: "
                        f"{conf['description']} (Rules: {conf['rule_ids']})"
                    )
            else:
                st.success("✅ No conflicting rules detected.")

    with tab_create:
        st.subheader("Add Business Rule")
        with st.form("create_rule_form"):
            rule_id = st.text_input("Rule ID", "boost_action_sci_fi")
            action = st.selectbox("Action", ["exclude", "boost", "pin"])
            genre = st.text_input("Target Genre (Optional)", "Action")
            item_id = st.number_input("Target Item ID (Optional)", min_value=0, value=0)
            boost = st.number_input("Boost Factor (e.g. 1.25)", min_value=0.1, value=1.2, step=0.1)
            pos = st.number_input("Pin Position (1-indexed)", min_value=1, value=1)
            priority = st.number_input("Priority", min_value=1, value=10)

            sub = st.form_submit_button("Save Rule", type="primary")
            if sub:
                payload = {
                    "id": rule_id,
                    "action": action,
                    "genre": genre if genre else None,
                    "item_id": item_id if item_id > 0 else None,
                    "boost": boost if action == "boost" else None,
                    "position": pos if action == "pin" else None,
                    "priority": priority,
                }
                res = api_request("POST", "/v1/rules", json_data=payload)
                if res:
                    st.success(f"Rule '{res['id']}' saved successfully.")
                    st.rerun()


# ---------------------------------------------------------------------------
# Page 3: API Keys & Security
# ---------------------------------------------------------------------------
elif page == "API Keys & Security":
    st.header("🔑 API Keys & Rate Limiting")
    st.markdown("Manage hashed API client keys (SHA-256) and inspect client rate limit tiers.")

    with st.form("create_key_form"):
        st.subheader("Issue New API Key")
        client_name = st.text_input("Client / Application Name", "production-storefront-partner")
        tier = st.selectbox("Rate Limit Tier", ["standard", "elevated", "admin"])
        issue_key = st.form_submit_button("Generate Key", type="primary")

        if issue_key:
            res = api_request(
                "POST", "/v1/auth/keys", json_data={"client_name": client_name, "tier": tier}
            )
            if res:
                st.success("API Key successfully issued! Store this token securely:")
                st.code(res.get("raw_key", "sk_..."))


# ---------------------------------------------------------------------------
# Page 4: Models & Explanations
# ---------------------------------------------------------------------------
elif page == "Models & Explanations":
    st.header("🤖 Model Registry & Explainability")

    st.subheader("Candidate & Production Models")
    st.info("Production models are served by default and hot-swappable with zero downtime.")

    test_user = st.number_input("Inspect User Recommendations", min_value=1, max_value=943, value=1)
    if st.button("Generate Inspection Recommendations", type="primary"):
        recs = api_request("GET", f"/v1/recommend/hybrid/{test_user}")
        if recs:
            st.subheader(f"Hybrid Recommendations for User {test_user}")
            df_recs = pd.DataFrame(recs.get("recommendations", []))
            st.dataframe(df_recs, use_container_width=True, hide_index=True)
