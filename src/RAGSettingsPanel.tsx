import React from "react";
import "./RAGTheme.css";

const SETTINGS_DEFINITION_ID = "braindrive_rag_service_settings";
const PLUGIN_SLUG = "BrainDriveRAGCommunity";

type ServiceKey = "document_chat" | "document_processing";

type ApiService = {
  get?: (url: string, options?: unknown) => Promise<any>;
  post?: (url: string, body?: unknown, options?: unknown) => Promise<any>;
};

type SettingsService = {
  getSetting?: (definitionId: string, options?: Record<string, unknown>) => Promise<any>;
  setSetting?: (definitionId: string, value: any, options?: Record<string, unknown>) => Promise<any>;
};

type ThemeService = {
  getCurrentTheme: () => string;
  addThemeChangeListener: (callback: (theme: string) => void) => void;
  removeThemeChangeListener: (callback: (theme: string) => void) => void;
};

type Services = {
  api?: ApiService;
  settings?: SettingsService;
  theme?: ThemeService;
};

type PanelProps = {
  services?: Services;
};

type ServiceSettings = {
  enabled: boolean;
  mode: "venv" | "docker";
  protocol: string;
  host: string;
  port: number;
  health_path: string;
  env: Record<string, string>;
};

type SettingsValue = {
  document_chat: ServiceSettings;
  document_processing: ServiceSettings;
  full_install?: boolean;
};

type State = {
  settings: SettingsValue;
  loading: boolean;
  saving: boolean;
  error?: string;
  success?: string;
  health: Record<ServiceKey, string>;
  accordions: Record<string, boolean>;
  currentTheme: string;
};

const defaultServiceSettings = (service: ServiceKey): ServiceSettings => {
  if (service === "document_chat") {
    return {
      enabled: true,
      mode: "venv",
      protocol: "http",
      host: "localhost",
      port: 18000,
      health_path: "/health",
      env: {
        API_HOST: "localhost",
        API_PORT: "18000",
        DOCUMENT_PROCESSOR_API_URL: "http://localhost:18080/documents/",
        LLM_PROVIDER: "ollama",
        EMBEDDING_PROVIDER: "ollama",
        OLLAMA_LLM_BASE_URL: "",
        OLLAMA_LLM_MODEL: "",
        OLLAMA_EMBEDDING_BASE_URL: "",
        OLLAMA_EMBEDDING_MODEL: "",
        ENABLE_CONTEXTUAL_RETRIEVAL: "false",
        OLLAMA_CONTEXTUAL_LLM_BASE_URL: "",
        OLLAMA_CONTEXTUAL_LLM_MODEL: "",
        EVALUATION_PROVIDER: "openai",
        OPENAI_EVALUATION_API_KEY: "",
        OPENAI_EVALUATION_MODEL: "",
      },
    };
  }
  return {
    enabled: true,
    mode: "venv",
    protocol: "http",
    host: "localhost",
    port: 18080,
    health_path: "/health",
    env: {
      API_HOST: "localhost",
      API_PORT: "18080",
      CORS_ALLOW_ANY: "1",
      AUTH_METHOD: "api_key",
      AUTH_API_KEY: "",
      JWT_SECRET: "",
    },
  };
};

const defaultSettings = (): SettingsValue => ({
  document_chat: defaultServiceSettings("document_chat"),
  document_processing: defaultServiceSettings("document_processing"),
  full_install: false,
});

function buildHealthUrl(settings: ServiceSettings): string {
  const healthPath = settings.health_path?.startsWith("/") ? settings.health_path : `/${settings.health_path || "health"}`;
  return `${settings.protocol || "http"}://${settings.host || "localhost"}:${settings.port || 0}${healthPath}`;
}

function mergeServiceSettings(base: ServiceSettings, incoming?: Partial<ServiceSettings>): ServiceSettings {
  if (!incoming) return base;
  const merged: ServiceSettings = {
    ...base,
    ...incoming,
    env: { ...base.env, ...(incoming.env || {}) },
  };
  merged.health_path = merged.health_path?.startsWith("/") ? merged.health_path : `/${merged.health_path || "health"}`;
  merged.port = Number.isFinite(Number(merged.port)) ? Number(merged.port) : base.port;
  return merged;
}

function mergeSettings(loaded?: any): SettingsValue {
  const base = defaultSettings();
  if (!loaded || typeof loaded !== "object") return base;
  const incomingChat = (loaded as any).document_chat;
  const incomingProc = (loaded as any).document_processing;

  return {
    document_chat: mergeServiceSettings(base.document_chat, incomingChat || {}),
    document_processing: mergeServiceSettings(base.document_processing, incomingProc || {}),
    full_install: Boolean((loaded as any).full_install || false),
  };
}

function buildEnvPayload(settings: SettingsValue): Record<string, Record<string, string>> {
  const envPayload: Record<string, Record<string, string>> = {};
  const chat = settings.document_chat;
  const proc = settings.document_processing;

  const procUrl = `${proc.protocol}://${proc.host}:${proc.port}/documents/`;
  envPayload.document_chat = {
    API_HOST: chat.host,
    API_PORT: String(chat.port),
    DOCUMENT_PROCESSOR_API_URL: chat.env.DOCUMENT_PROCESSOR_API_URL || procUrl,
    LLM_PROVIDER: chat.env.LLM_PROVIDER || "ollama",
    EMBEDDING_PROVIDER: chat.env.EMBEDDING_PROVIDER || "ollama",
    OLLAMA_LLM_BASE_URL: chat.env.OLLAMA_LLM_BASE_URL || "",
    OLLAMA_LLM_MODEL: chat.env.OLLAMA_LLM_MODEL || "",
    OLLAMA_EMBEDDING_BASE_URL: chat.env.OLLAMA_EMBEDDING_BASE_URL || "",
    OLLAMA_EMBEDDING_MODEL: chat.env.OLLAMA_EMBEDDING_MODEL || "",
    ENABLE_CONTEXTUAL_RETRIEVAL: chat.env.ENABLE_CONTEXTUAL_RETRIEVAL || "false",
    OLLAMA_CONTEXTUAL_LLM_BASE_URL: chat.env.OLLAMA_CONTEXTUAL_LLM_BASE_URL || "",
    OLLAMA_CONTEXTUAL_LLM_MODEL: chat.env.OLLAMA_CONTEXTUAL_LLM_MODEL || "",
    EVALUATION_PROVIDER: chat.env.EVALUATION_PROVIDER || "openai",
    OPENAI_EVALUATION_API_KEY: chat.env.OPENAI_EVALUATION_API_KEY || "",
    OPENAI_EVALUATION_MODEL: chat.env.OPENAI_EVALUATION_MODEL || "",
  };

  envPayload.document_processing = {
    API_HOST: proc.host,
    API_PORT: String(proc.port),
    CORS_ALLOW_ANY: proc.env.CORS_ALLOW_ANY || "1",
    AUTH_METHOD: proc.env.AUTH_METHOD || "api_key",
    AUTH_API_KEY: proc.env.AUTH_API_KEY || "",
    JWT_SECRET: proc.env.JWT_SECRET || "",
  };

  return envPayload;
}

type AccordionProps = {
  title: string;
  id: string;
  open: boolean;
  onToggle: (id: string) => void;
  children?: React.ReactNode;
};

const Accordion = ({ title, id, open, onToggle, children }: AccordionProps) => {
  return (
    <details
      open={open}
      onToggle={() => onToggle(id)}
      className="rag-accordion"
    >
      <summary>{title}</summary>
      <div className="rag-accordion-content">{children}</div>
    </details>
  );
};

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="rag-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

class BrainDriveRAGSettings extends React.Component<PanelProps, State> {
  constructor(props: PanelProps) {
    super(props);
    this.state = {
      settings: defaultSettings(),
      loading: true,
      saving: false,
      error: undefined,
      success: undefined,
      health: {
        document_chat: "",
        document_processing: ""
      },
      accordions: {
        connectivity_chat: false,
        connectivity_processing: false,
        llm: false,
        embedding: false,
        contextual: false,
        evaluation: false,
        processing_env: false,
      },
  currentTheme: "light",
    };
  }

  componentDidMount(): void {
    this.initializeThemeService();
    void this.loadSettings();
  }

  componentWillUnmount(): void {
    const themeSvc = this.props.services?.theme;
    if (themeSvc?.removeThemeChangeListener) {
      themeSvc.removeThemeChangeListener(this.handleThemeChange);
    }
  }

  initializeThemeService() {
    const ambient = detectAmbientTheme();
    const themeSvc = this.props.services?.theme;
    let resolvedTheme = ambient || "light";

    if (themeSvc?.getCurrentTheme) {
      try {
        resolvedTheme = themeSvc.getCurrentTheme() || resolvedTheme;
      } catch (err) {
        console.warn("BrainDriveRAGSettings: unable to read theme from service", err);
      }
    }

    this.setState({ currentTheme: resolvedTheme });

    if (themeSvc?.addThemeChangeListener) {
      try {
        themeSvc.addThemeChangeListener(this.handleThemeChange);
      } catch (err) {
        console.warn("BrainDriveRAGSettings: unable to attach theme listener", err);
      }
    }
  }

  handleThemeChange = (theme: string) => {
    this.setState({ currentTheme: theme || "light" });
  };

  setAccordion = (id: string) => {
    this.setState((prev) => ({
      ...prev,
      accordions: { ...prev.accordions, [id]: !prev.accordions[id] }
    }));
  };

  setHealthMessage = (key: ServiceKey, message: string) => {
    this.setState((prev) => ({
      ...prev,
      health: { ...prev.health, [key]: message }
    }));
  };

  loadSettings = async () => {
    this.setState((prev) => ({ ...prev, loading: true, error: undefined }));
    const services = this.props.services;
    try {
      let loaded: any = null;
      if (services?.api?.get) {
        const params = new URLSearchParams({
          definition_id: SETTINGS_DEFINITION_ID,
          user_id: "current",
          scope: "user",
        }).toString();
        const url = `/api/v1/settings/instances?${params}`;
        const resp = await services.api.get(url);
        if (Array.isArray(resp) && resp.length > 0) loaded = resp[0]?.value || resp[0];
        else if (resp && typeof resp === "object") loaded = resp.value || resp;
      } else if (services?.settings?.getSetting) {
        loaded = await services.settings.getSetting(SETTINGS_DEFINITION_ID, { userId: "current" });
      }
      const merged = mergeSettings(loaded);
      this.setState((prev) => ({ ...prev, settings: merged, loading: false }));
    } catch (err: any) {
      this.setState((prev) => ({ ...prev, loading: false, error: err?.message || "Failed to load settings" }));
    }
  };

  persistSettings = async (nextSettings: SettingsValue): Promise<boolean> => {
    const services = this.props.services;
    try {
      if (services?.api?.post) {
        const payload = {
          definition_id: SETTINGS_DEFINITION_ID,
          value: nextSettings,
          name: "RAG Services Settings",
          scope: "user",
          user_id: "current",
        };
        await services.api.post("/api/v1/settings/instances", payload);
      } else if (services?.settings?.setSetting) {
        await services.settings.setSetting(SETTINGS_DEFINITION_ID, nextSettings, { userId: "current" });
      } else {
        throw new Error("No API or settings service available");
      }
      return true;
    } catch (err: any) {
      this.setState((prev) => ({ ...prev, error: err?.message || "Failed to save settings" }));
      return false;
    }
  };

  saveSettings = async (restart: boolean, serviceKey?: ServiceKey | null) => {
    this.setState((prev) => ({ ...prev, saving: true, error: undefined, success: undefined }));
    const ok = await this.persistSettings(this.state.settings);
    if (!ok) {
      this.setState((prev) => ({ ...prev, saving: false }));
      return;
    }

    if (restart) {
      await this.controlService("restart", serviceKey ?? null);
    }

    this.setState((prev) => ({
      ...prev,
      saving: false,
      success: restart ? "Saved and restart requested" : "Settings saved"
    }));
    setTimeout(() => this.setState((prev) => ({ ...prev, success: undefined })), 4000);
  };

  controlService = async (action: "start" | "stop" | "restart", serviceKey: ServiceKey | null) => {
    const envPayload = buildEnvPayload(this.state.settings);
    const services = this.props.services;
    try {
      if (services?.api?.post) {
        const payload: Record<string, any> = {
          definition_id: SETTINGS_DEFINITION_ID,
          user_id: "current",
          env: envPayload,
          settings: this.state.settings,
        };
        if (serviceKey) payload.service_name = serviceKey;
        const url = `/api/v1/plugins/${PLUGIN_SLUG}/services/${action}`;
        await services.api.post(url, payload);
      } else {
        this.setState((prev) => ({ ...prev, error: "Control API not available" }));
      }
    } catch (err: any) {
      this.setState((prev) => ({ ...prev, error: err?.message || `Failed to ${action} service` }));
    }
  };

  handleHealthCheck = async (serviceKey: ServiceKey) => {
    const settings = this.state.settings[serviceKey];
    const url = buildHealthUrl(settings);
    this.setHealthMessage(serviceKey, "checking...");
    try {
      const resp = await fetch(url, { method: "GET", credentials: "omit", mode: "cors" });
      if (!resp.ok) {
        this.setHealthMessage(serviceKey, `Unhealthy (${resp.status})`);
        return;
      }
      const text = await resp.text();
      this.setHealthMessage(serviceKey, text || "Healthy");
    } catch (err: any) {
      this.setHealthMessage(serviceKey, `Check failed: ${err?.message || err}`);
    }
  };

  updateServiceField = (serviceKey: ServiceKey, field: keyof ServiceSettings, value: any) => {
    this.setState((prev) => ({
      ...prev,
      settings: {
        ...prev.settings,
        [serviceKey]: {
          ...prev.settings[serviceKey],
          [field]: value
        }
      }
    }));
  };

  updateEnvField = (serviceKey: ServiceKey, key: string, value: string) => {
    this.setState((prev) => ({
      ...prev,
      settings: {
        ...prev.settings,
        [serviceKey]: {
          ...prev.settings[serviceKey],
          env: { ...prev.settings[serviceKey].env, [key]: value }
        }
      }
    }));
  };

  renderConnectivity = (serviceKey: ServiceKey) => {
    const svc = this.state.settings[serviceKey];
    const baseId = serviceKey === "document_chat" ? "connectivity_chat" : "connectivity_processing";
    return (
      <Accordion title="Connectivity" id={baseId} open={this.state.accordions[baseId]} onToggle={this.setAccordion}>
        <div className="rag-grid rag-grid--compact">
          <Field label="Protocol">
            <select className="rag-select" value={svc.protocol} onChange={(e) => this.updateServiceField(serviceKey, "protocol", e.target.value as any)}>
              <option value="http">http</option>
              <option value="https">https</option>
            </select>
          </Field>
          <Field label="Host">
            <input className="rag-input" value={svc.host} onChange={(e) => this.updateServiceField(serviceKey, "host", e.target.value)} />
          </Field>
          <Field label="Port">
            <input className="rag-input" type="number" value={svc.port} onChange={(e) => this.updateServiceField(serviceKey, "port", Number(e.target.value))} />
          </Field>
          <Field label="Health path">
            <input className="rag-input" value={svc.health_path} onChange={(e) => this.updateServiceField(serviceKey, "health_path", e.target.value)} />
          </Field>
        </div>
        <div className="rag-help">Health URL: {buildHealthUrl(svc)}</div>
      </Accordion>
    );
  };

  renderRuntime = () => {
    const chat = this.state.settings.document_chat;
    const contextualEnabled = String(chat.env.ENABLE_CONTEXTUAL_RETRIEVAL).toLowerCase() === "true";
    return (
      <>
        <Accordion title="LLM Provider" id="llm" open={this.state.accordions.llm} onToggle={this.setAccordion}>
          <Field label="Provider">
            <select className="rag-select" value={chat.env.LLM_PROVIDER} onChange={(e) => this.updateEnvField("document_chat", "LLM_PROVIDER", e.target.value)}>
              <option value="ollama">ollama</option>
              <option value="openai">openai</option>
              <option value="groq">groq</option>
              <option value="openrouter">openrouter</option>
            </select>
          </Field>
          <div className="rag-grid rag-grid--wide">
            <Field label="Base URL">
              <input className="rag-input" value={chat.env.OLLAMA_LLM_BASE_URL} onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_LLM_BASE_URL", e.target.value)} />
            </Field>
            <Field label="Model">
              <input className="rag-input" value={chat.env.OLLAMA_LLM_MODEL} onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_LLM_MODEL", e.target.value)} />
            </Field>
          </div>
        </Accordion>

        <Accordion title="Embedding Provider" id="embedding" open={this.state.accordions.embedding} onToggle={this.setAccordion}>
          <Field label="Provider">
            <select className="rag-select" value={chat.env.EMBEDDING_PROVIDER} onChange={(e) => this.updateEnvField("document_chat", "EMBEDDING_PROVIDER", e.target.value)}>
              <option value="ollama">ollama</option>
              <option value="openai">openai</option>
            </select>
          </Field>
          <div className="rag-grid rag-grid--wide">
            <Field label="Base URL">
              <input className="rag-input" value={chat.env.OLLAMA_EMBEDDING_BASE_URL} onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_EMBEDDING_BASE_URL", e.target.value)} />
            </Field>
            <Field label="Model">
              <input className="rag-input" value={chat.env.OLLAMA_EMBEDDING_MODEL} onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_EMBEDDING_MODEL", e.target.value)} />
            </Field>
          </div>
        </Accordion>

        <Accordion title="Contextual Retrieval" id="contextual" open={this.state.accordions.contextual} onToggle={this.setAccordion}>
          <label className="rag-checkbox-row">
            <input type="checkbox" checked={contextualEnabled} onChange={(e) => this.updateEnvField("document_chat", "ENABLE_CONTEXTUAL_RETRIEVAL", e.target.checked ? "true" : "false")} />
            Enable contextual retrieval
          </label>
          <div className="rag-grid rag-grid--wide">
            <Field label="Base URL">
              <input className="rag-input" value={chat.env.OLLAMA_CONTEXTUAL_LLM_BASE_URL} onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_CONTEXTUAL_LLM_BASE_URL", e.target.value)} />
            </Field>
            <Field label="Model">
              <input className="rag-input" value={chat.env.OLLAMA_CONTEXTUAL_LLM_MODEL} onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_CONTEXTUAL_LLM_MODEL", e.target.value)} />
            </Field>
          </div>
        </Accordion>

        <Accordion title="Evaluation Settings" id="evaluation" open={this.state.accordions.evaluation} onToggle={this.setAccordion}>
          <Field label="Provider">
            <select className="rag-select" value={chat.env.EVALUATION_PROVIDER} onChange={(e) => this.updateEnvField("document_chat", "EVALUATION_PROVIDER", e.target.value)}>
              <option value="openai">openai</option>
              <option value="ollama">ollama</option>
              <option value="groq">groq</option>
            </select>
          </Field>
          <div className="rag-grid rag-grid--wide">
            <Field label="API Key">
              <input className="rag-input" type="password" value={chat.env.OPENAI_EVALUATION_API_KEY} onChange={(e) => this.updateEnvField("document_chat", "OPENAI_EVALUATION_API_KEY", e.target.value)} />
            </Field>
            <Field label="Model">
              <input className="rag-input" value={chat.env.OPENAI_EVALUATION_MODEL} onChange={(e) => this.updateEnvField("document_chat", "OPENAI_EVALUATION_MODEL", e.target.value)} />
            </Field>
          </div>
        </Accordion>
      </>
    );
  };

  renderProcessingEnv = () => {
    const proc = this.state.settings.document_processing;
    return (
      <Accordion title="Processing Auth & CORS" id="processing_env" open={this.state.accordions.processing_env} onToggle={this.setAccordion}>
        <div className="rag-grid rag-grid--wide">
          <Field label="CORS allow any (1/0)">
            <input className="rag-input" value={proc.env.CORS_ALLOW_ANY} onChange={(e) => this.updateEnvField("document_processing", "CORS_ALLOW_ANY", e.target.value)} />
          </Field>
          <Field label="Auth method">
            <input className="rag-input" value={proc.env.AUTH_METHOD} onChange={(e) => this.updateEnvField("document_processing", "AUTH_METHOD", e.target.value)} />
          </Field>
          <Field label="Auth API Key">
            <input className="rag-input" value={proc.env.AUTH_API_KEY} onChange={(e) => this.updateEnvField("document_processing", "AUTH_API_KEY", e.target.value)} />
          </Field>
          <Field label="JWT Secret">
            <input className="rag-input" value={proc.env.JWT_SECRET} onChange={(e) => this.updateEnvField("document_processing", "JWT_SECRET", e.target.value)} />
          </Field>
        </div>
      </Accordion>
    );
  };

  renderServiceCard = (serviceKey: ServiceKey, title: string, description: string) => {
    const svc = this.state.settings[serviceKey];
    const healthMessage = this.state.health[serviceKey];
    return (
      <div className="rag-card">
        <div className="rag-card-header">
          <div>
            <div className="rag-card-title">{title}</div>
            <div className="rag-card-description">{description}</div>
          </div>
          <div className="rag-meta">
            <label className="rag-checkbox-row">
              <input type="checkbox" checked={svc.enabled} onChange={(e) => this.updateServiceField(serviceKey, "enabled", e.target.checked)} />
              Enabled
            </label>
            <div>Mode: {svc.mode || "venv"}</div>
          </div>
        </div>

        <div className="rag-actions">
          <button onClick={() => this.saveSettings(false)} disabled={this.state.saving || this.state.loading} className="rag-button rag-button--primary">{this.state.saving ? "Saving..." : "Save"}</button>
          <button onClick={() => this.saveSettings(true, serviceKey)} disabled={this.state.saving || this.state.loading} className="rag-button rag-button--primary">Save & Restart</button>
          <button onClick={() => this.controlService("start", serviceKey)} className="rag-button">Start</button>
          <button onClick={() => this.controlService("stop", serviceKey)} className="rag-button">Stop</button>
          <button onClick={() => this.controlService("restart", serviceKey)} className="rag-button">Restart</button>
          <button onClick={() => this.handleHealthCheck(serviceKey)} className="rag-button">Health</button>
          <span className="rag-health">{healthMessage}</span>
        </div>

        <div className="rag-section-stack">
          {this.renderConnectivity(serviceKey)}
          {serviceKey === "document_chat" ? this.renderRuntime() : this.renderProcessingEnv()}
        </div>
      </div>
    );
  };

  render(): React.ReactNode {
    const rootClass = this.state.currentTheme.toLowerCase().includes("dark") ? "dark-theme" : "";
    return (
      <div className={`rag-settings ${rootClass}`}>
        {this.state.error && <div className="rag-banner rag-banner--error">{this.state.error}</div>}
        {this.state.success && <div className="rag-banner rag-banner--success">{this.state.success}</div>}
        {this.state.loading ? (
          <div>Loading settings...</div>
        ) : (
          <>
            {this.renderServiceCard("document_chat", "Document Chat Service", "User-facing chat API serving retrieved answers.")}
            {this.renderServiceCard("document_processing", "Document Processing Service", "Ingests and chunks documents for retrieval.")}
          </>
        )}
      </div>
    );
  }
}

export default BrainDriveRAGSettings;
export { BrainDriveRAGSettings };

function detectAmbientTheme(): string | null {
  try {
    if (typeof document === "undefined") return null;
    const root = document.documentElement;
    const body = document.body;
    const hasDarkClass =
      root.classList.contains("dark") ||
      root.classList.contains("dark-theme") ||
      body.classList.contains("dark") ||
      body.classList.contains("dark-theme");
    return hasDarkClass ? "dark" : null;
  } catch {
    return null;
  }
}
