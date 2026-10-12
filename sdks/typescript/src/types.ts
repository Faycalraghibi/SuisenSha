/**
 * Type definitions for SuisenSha Client SDK.
 */

export interface RecommendationItem {
  item_id: string;
  score: number;
  metadata?: Record<string, unknown>;
  explanation?: string;
}

export interface RecommendationResponse {
  recommendations: RecommendationItem[];
  model_version: string;
  cached: boolean;
  user_id?: string | null;
  explanation_type?: string | null;
  feature_importances?: Record<string, number> | null;
}

export interface EventItem {
  user_id: string;
  item_id: string;
  event_type?: string;
  rating?: number | null;
  timestamp?: number | null;
  tenant_id?: string;
}

export interface SessionResponse {
  session_id: string;
  items: string[];
  user_id?: string | null;
  length: number;
  created_at: number;
  updated_at: number;
  tenant_id?: string;
}

export interface BusinessRule {
  rule_id: string;
  rule_type: "pin" | "filter" | "boost" | "diversity";
  name: string;
  enabled: boolean;
  priority: number;
  condition: Record<string, unknown>;
  action: Record<string, unknown>;
  created_at: number;
  tenant_id?: string;
}

export interface BusinessRuleCreate {
  rule_type: "pin" | "filter" | "boost" | "diversity";
  name: string;
  priority?: number;
  condition: Record<string, unknown>;
  action: Record<string, unknown>;
  tenant_id?: string;
}

export interface Experiment {
  id: string;
  name: string;
  control_model: string;
  treatment_model: string;
  traffic_split: number;
  status: "draft" | "running" | "paused" | "concluded";
  tenant_id?: string;
}

export interface ExperimentCreate {
  name: string;
  control_model: string;
  treatment_model: string;
  traffic_split?: number;
  tenant_id?: string;
}

export interface VariantAssignment {
  user_id: string;
  variant: "control" | "treatment";
  model: string;
}

export interface UpliftReport {
  experiment_id: string;
  metric: string;
  control_conversions: number;
  control_impressions: number;
  treatment_conversions: number;
  treatment_impressions: number;
  control_rate: number;
  treatment_rate: number;
  absolute_uplift: number;
  relative_uplift: number;
  p_value: number;
  statistically_significant: boolean;
}

export interface HealthResponse {
  status: string;
  uptime_seconds?: number;
  models_loaded?: string[];
  cache_status?: string;
}

export interface SuisenShaClientOptions {
  baseUrl?: string;
  apiKey?: string;
  tenantId?: string;
  timeoutMs?: number;
}
