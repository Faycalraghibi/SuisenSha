import {
  BusinessRule,
  BusinessRuleCreate,
  EventItem,
  Experiment,
  ExperimentCreate,
  HealthResponse,
  RecommendationResponse,
  SessionResponse,
  SuisenShaClientOptions,
  UpliftReport,
  VariantAssignment,
} from "./types.js";

export class SuisenShaError extends Error {
  public statusCode: number;
  public details?: unknown;

  constructor(message: string, statusCode: number, details?: unknown) {
    super(message);
    this.name = "SuisenShaError";
    this.statusCode = statusCode;
    this.details = details;
  }
}

export class SuisenShaClient {
  public readonly baseUrl: string;
  public readonly apiKey?: string;
  public readonly tenantId?: string;
  public readonly timeoutMs: number;

  constructor(options: SuisenShaClientOptions = {}) {
    this.baseUrl = (options.baseUrl || "http://localhost:8000").replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.tenantId = options.tenantId;
    this.timeoutMs = options.timeoutMs ?? 10000;
  }

  private async request<T>(
    path: string,
    method: "GET" | "POST" | "DELETE" = "GET",
    params?: Record<string, string | number | boolean | undefined>,
    body?: unknown
  ): Promise<T> {
    let url = `${this.baseUrl}${path}`;
    if (params) {
      const searchParams = new URLSearchParams();
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null) {
          searchParams.append(k, String(v));
        }
      }
      const qs = searchParams.toString();
      if (qs) {
        url += `?${qs}`;
      }
    }

    const headers: Record<string, string> = {
      "Accept": "application/json",
    };

    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
      headers["X-API-Key"] = this.apiKey;
    }
    if (this.tenantId) {
      headers["X-Tenant-ID"] = this.tenantId;
    }

    const init: RequestInit = {
      method,
      headers,
    };

    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    init.signal = controller.signal;

    try {
      const res = await fetch(url, init);
      clearTimeout(timer);

      if (!res.ok) {
        let errBody: unknown;
        try {
          errBody = await res.json();
        } catch {
          errBody = await res.text();
        }
        throw new SuisenShaError(
          `Request failed with status ${res.status}: ${JSON.stringify(errBody)}`,
          res.status,
          errBody
        );
      }

      if (res.status === 204) {
        return {} as T;
      }
      return (await res.json()) as T;
    } catch (err: unknown) {
      clearTimeout(timer);
      if (err instanceof SuisenShaError) {
        throw err;
      }
      throw new SuisenShaError(
        (err as Error).message || "Unknown network error",
        0,
        err
      );
    }
  }

  public async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  public async recommendPopular(options: {
    n?: number;
    user_id?: string;
    enable_rules?: boolean;
    explain?: boolean;
  } = {}): Promise<RecommendationResponse> {
    return this.request<RecommendationResponse>("/v1/recommend/popular", "GET", options);
  }

  public async recommendEmbedding(options: {
    user_id: string;
    n?: number;
    enable_rules?: boolean;
    explain?: boolean;
  }): Promise<RecommendationResponse> {
    return this.request<RecommendationResponse>("/v1/recommend/embedding", "GET", options);
  }

  public async recommendSasrec(options: {
    user_id: string;
    n?: number;
    enable_rules?: boolean;
    explain?: boolean;
  }): Promise<RecommendationResponse> {
    return this.request<RecommendationResponse>("/v1/recommend/sasrec", "GET", options);
  }

  public async recommendHybrid(options: {
    user_id: string;
    n?: number;
    alpha?: number;
    enable_rules?: boolean;
    explain?: boolean;
  }): Promise<RecommendationResponse> {
    return this.request<RecommendationResponse>("/v1/recommend/hybrid", "GET", options);
  }

  public async sendEvents(events: EventItem[]): Promise<{ status: string; recorded: number }> {
    return this.request<{ status: string; recorded: number }>("/v1/events", "POST", undefined, events);
  }

  public async getSession(sessionId: string): Promise<SessionResponse> {
    return this.request<SessionResponse>(`/v1/sessions/${encodeURIComponent(sessionId)}`);
  }

  public async recordSessionEvent(
    sessionId: string,
    itemId: string,
    eventType: string = "view"
  ): Promise<SessionResponse> {
    return this.request<SessionResponse>(
      `/v1/sessions/${encodeURIComponent(sessionId)}/events`,
      "POST",
      undefined,
      { item_id: itemId, event_type: eventType }
    );
  }

  public async listRules(): Promise<BusinessRule[]> {
    return this.request<BusinessRule[]>("/v1/rules");
  }

  public async createRule(rule: BusinessRuleCreate): Promise<BusinessRule> {
    return this.request<BusinessRule>("/v1/rules", "POST", undefined, rule);
  }

  public async deleteRule(ruleId: string): Promise<{ status: string }> {
    return this.request<{ status: string }>(`/v1/rules/${encodeURIComponent(ruleId)}`, "DELETE");
  }

  public async listExperiments(): Promise<Experiment[]> {
    return this.request<Experiment[]>("/v1/experiments");
  }

  public async createExperiment(exp: ExperimentCreate): Promise<Experiment> {
    return this.request<Experiment>("/v1/experiments", "POST", undefined, exp);
  }

  public async assignExperimentVariant(expId: string, userId: string): Promise<VariantAssignment> {
    return this.request<VariantAssignment>(
      `/v1/experiments/${encodeURIComponent(expId)}/assignment`,
      "POST",
      { user_id: userId }
    );
  }

  public async getExperimentReport(expId: string): Promise<UpliftReport> {
    return this.request<UpliftReport>(`/v1/experiments/${encodeURIComponent(expId)}/report`);
  }

  public async eraseUserData(userId: string): Promise<{ status: string; user_id: string }> {
    return this.request<{ status: string; user_id: string }>(
      `/v1/users/${encodeURIComponent(userId)}/data`,
      "DELETE"
    );
  }
}
