import React from "react";
import "./RAGTheme.css";
import { fetchOllamaServers, fetchProviderCatalog, fetchProviderModels } from "./shared/provider-selection/api";
import { filterModelsByPurpose } from "./shared/provider-selection/modelPurpose";
import type { OllamaServer, ProviderCatalogEntry, ProviderModel, ModelPurpose } from "./shared/provider-selection/types";

const SETTINGS_DEFINITION_ID = "braindrive_rag_service_settings";
const PLUGIN_SLUG = "BrainDriveRAGCommunity";

type ServiceKey = "document_chat" | "document_processing";

type HealthBadgeStatus = "idle" | "checking" | "ok" | "bad";

type HealthBadge = {
  status: HealthBadgeStatus;
  label: string;
  details?: string;
};

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

type ProviderSectionKey = "llm" | "embedding" | "contextual" | "evaluation";

type State = {
  settings: SettingsValue;
  loading: boolean;
  saving: boolean;
  error?: string;
  success?: string;
  health: Record<ServiceKey, HealthBadge>;
  accordions: Record<string, boolean>;
  currentTheme: string;
  providerCatalogLoading: boolean;
  providerCatalogError?: string;
  providerCatalog: ProviderCatalogEntry[];
  ollamaServers: OllamaServer[];
  modelsBySection: Record<ProviderSectionKey, ProviderModel[]>;
  modelLoadingBySection: Record<ProviderSectionKey, boolean>;
  modelErrorBySection: Record<ProviderSectionKey, string | undefined>;
  modelFilterBySection: Record<ProviderSectionKey, string>;
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
        OLLAMA_LLM_SERVER_ID: "",
        OLLAMA_LLM_BASE_URL: "",
        OLLAMA_LLM_MODEL: "",
        OLLAMA_EMBEDDING_SERVER_ID: "",
        OLLAMA_EMBEDDING_BASE_URL: "",
        OLLAMA_EMBEDDING_MODEL: "",
        ENABLE_CONTEXTUAL_RETRIEVAL: "false",
        OLLAMA_CONTEXTUAL_LLM_SERVER_ID: "",
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

const PROVIDER_SECTION_KEYS: ProviderSectionKey[] = ["llm", "embedding", "contextual", "evaluation"];

type ProviderSectionConfig = {
  purpose: ModelPurpose;
  allowedProviders: string[];
  providerEnvKey?: string;
  serverIdEnvKey?: string;
  baseUrlEnvKey?: string;
  modelEnvKey: string;
  supportsServers: (provider: string) => boolean;
};

const SECTION_CONFIG: Record<ProviderSectionKey, ProviderSectionConfig> = {
  llm: {
    purpose: "chat",
    allowedProviders: ["ollama", "openai", "groq", "openrouter"],
    providerEnvKey: "LLM_PROVIDER",
    serverIdEnvKey: "OLLAMA_LLM_SERVER_ID",
    baseUrlEnvKey: "OLLAMA_LLM_BASE_URL",
    modelEnvKey: "OLLAMA_LLM_MODEL",
    supportsServers: (provider) => provider === "ollama",
  },
  embedding: {
    purpose: "embedding",
    allowedProviders: ["ollama", "openai"],
    providerEnvKey: "EMBEDDING_PROVIDER",
    serverIdEnvKey: "OLLAMA_EMBEDDING_SERVER_ID",
    baseUrlEnvKey: "OLLAMA_EMBEDDING_BASE_URL",
    modelEnvKey: "OLLAMA_EMBEDDING_MODEL",
    supportsServers: (provider) => provider === "ollama",
  },
  contextual: {
    purpose: "chat",
    allowedProviders: ["ollama"],
    serverIdEnvKey: "OLLAMA_CONTEXTUAL_LLM_SERVER_ID",
    baseUrlEnvKey: "OLLAMA_CONTEXTUAL_LLM_BASE_URL",
    modelEnvKey: "OLLAMA_CONTEXTUAL_LLM_MODEL",
    supportsServers: (provider) => provider === "ollama",
  },
  evaluation: {
    purpose: "evaluation",
    allowedProviders: ["openai", "ollama", "groq"],
    providerEnvKey: "EVALUATION_PROVIDER",
    modelEnvKey: "OPENAI_EVALUATION_MODEL",
    supportsServers: () => false,
  },
};

function resolveOllamaServerSelection(
  servers: OllamaServer[],
  storedServerId: string | undefined,
  storedBaseUrl: string | undefined,
): { serverId: string; baseUrl: string; isCustom: boolean } {
  const baseUrl = storedBaseUrl || "";
  if (!Array.isArray(servers) || servers.length === 0) {
    return { serverId: "", baseUrl, isCustom: true };
  }

  const byId = storedServerId ? servers.find((s) => s.id === storedServerId) : undefined;
  if (byId) {
    return {
      serverId: byId.id,
      baseUrl: byId.serverAddress || baseUrl,
      isCustom: false,
    };
  }

  const byUrl = baseUrl ? servers.find((s) => s.serverAddress === baseUrl) : undefined;
  if (byUrl) {
    return { serverId: byUrl.id, baseUrl: byUrl.serverAddress, isCustom: false };
  }

  if (!baseUrl) {
    const first = servers[0];
    return { serverId: first.id, baseUrl: first.serverAddress || "", isCustom: false };
  }

  return { serverId: "", baseUrl, isCustom: true };
}

function applyOllamaDefaultsToSettings(settings: SettingsValue, servers: OllamaServer[]): SettingsValue {
  const next = { ...settings };
  const chat = next.document_chat;
  const chatEnv = { ...chat.env };

  const llmSel = resolveOllamaServerSelection(servers, chatEnv.OLLAMA_LLM_SERVER_ID, chatEnv.OLLAMA_LLM_BASE_URL);
  chatEnv.OLLAMA_LLM_SERVER_ID = llmSel.serverId;
  chatEnv.OLLAMA_LLM_BASE_URL = llmSel.baseUrl;

  const embSel = resolveOllamaServerSelection(servers, chatEnv.OLLAMA_EMBEDDING_SERVER_ID, chatEnv.OLLAMA_EMBEDDING_BASE_URL);
  chatEnv.OLLAMA_EMBEDDING_SERVER_ID = embSel.serverId;
  chatEnv.OLLAMA_EMBEDDING_BASE_URL = embSel.baseUrl;

  const ctxSel = resolveOllamaServerSelection(servers, chatEnv.OLLAMA_CONTEXTUAL_LLM_SERVER_ID, chatEnv.OLLAMA_CONTEXTUAL_LLM_BASE_URL);
  chatEnv.OLLAMA_CONTEXTUAL_LLM_SERVER_ID = ctxSel.serverId;
  chatEnv.OLLAMA_CONTEXTUAL_LLM_BASE_URL = ctxSel.baseUrl;

  next.document_chat = { ...chat, env: chatEnv };
  return next;
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
  private healthClearTimers: Partial<Record<ServiceKey, number>> = {};

  constructor(props: PanelProps) {
    super(props);
    this.state = {
      settings: defaultSettings(),
      loading: true,
      saving: false,
      error: undefined,
      success: undefined,
      health: {
        document_chat: { status: "idle", label: "" },
        document_processing: { status: "idle", label: "" },
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
      providerCatalogLoading: false,
      providerCatalogError: undefined,
      providerCatalog: [],
      ollamaServers: [],
      modelsBySection: {
        llm: [],
        embedding: [],
        contextual: [],
        evaluation: [],
      },
      modelLoadingBySection: {
        llm: false,
        embedding: false,
        contextual: false,
        evaluation: false,
      },
      modelErrorBySection: {
        llm: undefined,
        embedding: undefined,
        contextual: undefined,
        evaluation: undefined,
      },
      modelFilterBySection: {
        llm: "",
        embedding: "",
        contextual: "",
        evaluation: "",
      },
    };
  }

  componentDidMount(): void {
    this.initializeThemeService();
    void (async () => {
      await this.loadSettings();
      await this.loadProviderSelectionData();
    })();
  }

  componentWillUnmount(): void {
    const themeSvc = this.props.services?.theme;
    if (themeSvc?.removeThemeChangeListener) {
      themeSvc.removeThemeChangeListener(this.handleThemeChange);
    }
    Object.values(this.healthClearTimers).forEach((timerId) => window.clearTimeout(timerId));
    this.healthClearTimers = {};
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

  setHealthBadge = (key: ServiceKey, badge: HealthBadge) => {
    this.setState((prev) => ({
      ...prev,
      health: { ...prev.health, [key]: badge }
    }));
  };

  clearHealthTimer = (key: ServiceKey) => {
    const timerId = this.healthClearTimers[key];
    if (timerId) {
      window.clearTimeout(timerId);
      delete this.healthClearTimers[key];
    }
  };

  scheduleHealthClear = (key: ServiceKey, delayMs: number = 5000) => {
    this.clearHealthTimer(key);
    this.healthClearTimers[key] = window.setTimeout(() => {
      this.setHealthBadge(key, { status: "idle", label: "" });
      delete this.healthClearTimers[key];
    }, delayMs);
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

  updateEnvFields = (serviceKey: ServiceKey, updates: Record<string, string>, callback?: () => void) => {
    this.setState((prev) => ({
      ...prev,
      settings: {
        ...prev.settings,
        [serviceKey]: {
          ...prev.settings[serviceKey],
          env: { ...prev.settings[serviceKey].env, ...updates }
        }
      }
    }), callback);
  };

  setModelFilter = (section: ProviderSectionKey, value: string) => {
    this.setState((prev) => ({
      ...prev,
      modelFilterBySection: { ...prev.modelFilterBySection, [section]: value }
    }));
  };

  loadProviderSelectionData = async (): Promise<void> => {
    const api = this.props.services?.api;
    if (!api?.get) {
      this.setState({
        providerCatalogLoading: false,
        providerCatalogError: "API service not available",
        providerCatalog: [],
        ollamaServers: [],
      });
      return;
    }

    this.setState({ providerCatalogLoading: true, providerCatalogError: undefined });
    try {
      const [catalog, servers] = await Promise.all([
        fetchProviderCatalog(api, { userId: "current", includeUnconfigured: true }),
        fetchOllamaServers(api, { userId: "current" }),
      ]);

      this.setState((prev) => ({
        ...prev,
        providerCatalogLoading: false,
        providerCatalogError: undefined,
        providerCatalog: catalog,
        ollamaServers: servers,
        settings: applyOllamaDefaultsToSettings(prev.settings, servers),
      }), () => {
        void this.refreshModelsForSection("llm");
        void this.refreshModelsForSection("embedding");
        const contextualEnabled = String(this.state.settings.document_chat.env.ENABLE_CONTEXTUAL_RETRIEVAL).toLowerCase() === "true";
        if (contextualEnabled) void this.refreshModelsForSection("contextual");
        void this.refreshModelsForSection("evaluation");
      });
    } catch (err: any) {
      this.setState({
        providerCatalogLoading: false,
        providerCatalogError: err?.message || "Failed to load provider catalog",
        providerCatalog: [],
        ollamaServers: [],
      });
    }
  };

  private resolveProviderMeta(providerId: string): { settingsId: string | null; defaultServerId: string | null; configured: boolean } {
    const entry = this.state.providerCatalog.find((p) => p.id === providerId);
    return {
      settingsId: entry?.settingsId ?? null,
      defaultServerId: entry?.defaultServerId ?? null,
      configured: Boolean(entry?.configured),
    };
  }

  refreshModelsForSection = async (section: ProviderSectionKey): Promise<void> => {
    const api = this.props.services?.api;
    if (!api?.get) return;

    const cfg = SECTION_CONFIG[section];
    const chatEnv = this.state.settings.document_chat.env;

    const providerId = section === "contextual"
      ? "ollama"
      : (cfg.providerEnvKey ? (chatEnv[cfg.providerEnvKey] || "") : "");

    if (section === "evaluation" && providerId && providerId !== "openai") {
      this.setState((prev) => ({
        ...prev,
        modelsBySection: { ...prev.modelsBySection, [section]: [] },
        modelErrorBySection: { ...prev.modelErrorBySection, [section]: undefined },
      }));
      return;
    }

    if (!providerId) {
      this.setState((prev) => ({
        ...prev,
        modelsBySection: { ...prev.modelsBySection, [section]: [] },
        modelErrorBySection: { ...prev.modelErrorBySection, [section]: "Select a provider first" },
      }));
      return;
    }

    this.setState((prev) => ({
      ...prev,
      modelLoadingBySection: { ...prev.modelLoadingBySection, [section]: true },
      modelErrorBySection: { ...prev.modelErrorBySection, [section]: undefined },
    }));

    try {
      let settingsId: string | null = null;
      let serverId: string | null = null;

      if (cfg.supportsServers(providerId)) {
        const serverKey = cfg.serverIdEnvKey || "";
        const baseUrlKey = cfg.baseUrlEnvKey || "";
        const sel = resolveOllamaServerSelection(
          this.state.ollamaServers,
          chatEnv[serverKey],
          chatEnv[baseUrlKey],
        );
        if (!sel.serverId) {
          // Custom URL not backed by ollama_servers_settings -> no backend model list available
          this.setState((prev) => ({
            ...prev,
            modelsBySection: { ...prev.modelsBySection, [section]: [] },
            modelLoadingBySection: { ...prev.modelLoadingBySection, [section]: false },
            modelErrorBySection: { ...prev.modelErrorBySection, [section]: "Custom server URL: model list unavailable" },
          }));
          return;
        }
        settingsId = "ollama_servers_settings";
        serverId = sel.serverId;
      } else {
        const meta = this.resolveProviderMeta(providerId);
        if (!meta.configured) {
          this.setState((prev) => ({
            ...prev,
            modelsBySection: { ...prev.modelsBySection, [section]: [] },
            modelLoadingBySection: { ...prev.modelLoadingBySection, [section]: false },
            modelErrorBySection: { ...prev.modelErrorBySection, [section]: "Provider not configured" },
          }));
          return;
        }
        settingsId = meta.settingsId;
        serverId = meta.defaultServerId;
      }

      if (!settingsId || !serverId) {
        this.setState((prev) => ({
          ...prev,
          modelsBySection: { ...prev.modelsBySection, [section]: [] },
          modelLoadingBySection: { ...prev.modelLoadingBySection, [section]: false },
          modelErrorBySection: { ...prev.modelErrorBySection, [section]: "Provider metadata unavailable" },
        }));
        return;
      }

      const models = await fetchProviderModels(api, {
        provider: providerId,
        settingsId,
        serverId,
        userId: "current",
      });

      const filtered = filterModelsByPurpose(models, cfg.purpose);
      this.setState((prev) => ({
        ...prev,
        modelsBySection: { ...prev.modelsBySection, [section]: filtered },
        modelLoadingBySection: { ...prev.modelLoadingBySection, [section]: false },
        modelErrorBySection: { ...prev.modelErrorBySection, [section]: undefined },
      }));
    } catch (err: any) {
      this.setState((prev) => ({
        ...prev,
        modelsBySection: { ...prev.modelsBySection, [section]: [] },
        modelLoadingBySection: { ...prev.modelLoadingBySection, [section]: false },
        modelErrorBySection: { ...prev.modelErrorBySection, [section]: err?.message || "Failed to load models" },
      }));
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
    this.clearHealthTimer(serviceKey);
    this.setHealthBadge(serviceKey, { status: "checking", label: "Checking…" });
    try {
      const resp = await fetch(url, { method: "GET", credentials: "omit", mode: "cors" });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        const details = body ? `HTTP ${resp.status}: ${body}` : `HTTP ${resp.status}`;
        this.setHealthBadge(serviceKey, { status: "bad", label: "Not running", details });
        this.scheduleHealthClear(serviceKey);
        return;
      }
      const text = await resp.text().catch(() => "");
      this.setHealthBadge(serviceKey, {
        status: "ok",
        label: "Running",
        details: text ? text : `HTTP ${resp.status}`,
      });
      this.scheduleHealthClear(serviceKey);
    } catch (err: any) {
      this.setHealthBadge(serviceKey, { status: "bad", label: "Not running", details: err?.message || String(err) });
      this.scheduleHealthClear(serviceKey);
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
    const chatEnv = chat.env;
    const contextualEnabled = String(chatEnv.ENABLE_CONTEXTUAL_RETRIEVAL).toLowerCase() === "true";
    const catalog = this.state.providerCatalog;
    const servers = this.state.ollamaServers;

    const providerLabel = (id: string) => catalog.find((p) => p.id === id)?.label || id;

    const buildProviderOptions = (section: ProviderSectionKey, selectedProvider: string) => {
      const allowed = SECTION_CONFIG[section].allowedProviders;
      const fromCatalog = catalog.filter((p) => allowed.includes(p.id));
      const available = fromCatalog.filter((p) => p.configured && p.enabled);
      const baseIds = available.length > 0 ? available.map((p) => p.id) : allowed;
      const ids = baseIds.includes(selectedProvider) ? baseIds : [...baseIds, selectedProvider];
      return ids.map((id) => ({ id, label: providerLabel(id) }));
    };

    const filterModels = (section: ProviderSectionKey) => {
      const query = (this.state.modelFilterBySection[section] || "").trim().toLowerCase();
      const models = this.state.modelsBySection[section] || [];
      if (!query) return models;
      return models.filter((m) => (m.name || "").toLowerCase().includes(query));
    };

    const llmProvider = chatEnv.LLM_PROVIDER || "ollama";
    const embeddingProvider = chatEnv.EMBEDDING_PROVIDER || "ollama";
    const evaluationProvider = chatEnv.EVALUATION_PROVIDER || "openai";

    const llmServerSel = resolveOllamaServerSelection(servers, chatEnv.OLLAMA_LLM_SERVER_ID, chatEnv.OLLAMA_LLM_BASE_URL);
    const embeddingServerSel = resolveOllamaServerSelection(servers, chatEnv.OLLAMA_EMBEDDING_SERVER_ID, chatEnv.OLLAMA_EMBEDDING_BASE_URL);
    const contextualServerSel = resolveOllamaServerSelection(servers, chatEnv.OLLAMA_CONTEXTUAL_LLM_SERVER_ID, chatEnv.OLLAMA_CONTEXTUAL_LLM_BASE_URL);

    return (
      <>
        <Accordion title="LLM Provider" id="llm" open={this.state.accordions.llm} onToggle={this.setAccordion}>
          {this.state.providerCatalogError && (
            <div className="rag-banner rag-banner--error">
              Provider catalog unavailable: {this.state.providerCatalogError}
            </div>
          )}
          <Field label="Provider">
            <select
              className="rag-select"
              value={llmProvider}
              onChange={(e) => {
                const next = e.target.value;
                this.updateEnvField("document_chat", "LLM_PROVIDER", next);
                this.setState((prev) => ({
                  ...prev,
                  modelsBySection: { ...prev.modelsBySection, llm: [] },
                  modelErrorBySection: { ...prev.modelErrorBySection, llm: undefined },
                }), () => void this.refreshModelsForSection("llm"));
              }}
            >
              {buildProviderOptions("llm", llmProvider).map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </Field>
          <div className="rag-grid rag-grid--server-model">
            {llmProvider === "ollama" ? (
              <Field label="Server">
                {servers.length > 0 ? (
                  <>
                    <select
                      className="rag-select"
                      value={llmServerSel.isCustom ? "__custom__" : llmServerSel.serverId}
                      onChange={(e) => {
                        const next = e.target.value;
                        if (next === "__custom__") {
                          this.updateEnvFields("document_chat", {
                            OLLAMA_LLM_SERVER_ID: "",
                          }, () => void this.refreshModelsForSection("llm"));
                          return;
                        }
                        const server = servers.find((s) => s.id === next);
                        this.updateEnvFields("document_chat", {
                          OLLAMA_LLM_SERVER_ID: next,
                          OLLAMA_LLM_BASE_URL: server?.serverAddress || chatEnv.OLLAMA_LLM_BASE_URL || "",
                        }, () => void this.refreshModelsForSection("llm"));
                      }}
                    >
                      {servers.map((s) => (
                        <option key={s.id} value={s.id}>{s.serverName} ({s.serverAddress})</option>
                      ))}
                      <option value="__custom__">Custom URL…</option>
                    </select>
                    {llmServerSel.isCustom && (
                      <input
                        className="rag-input"
                        value={chatEnv.OLLAMA_LLM_BASE_URL || ""}
                        onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_LLM_BASE_URL", e.target.value)}
                        placeholder="http://localhost:11434"
                      />
                    )}
                  </>
                ) : (
                  <input
                    className="rag-input"
                    value={chatEnv.OLLAMA_LLM_BASE_URL || ""}
                    onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_LLM_BASE_URL", e.target.value)}
                    placeholder="http://localhost:11434"
                  />
                )}
              </Field>
            ) : (
              <Field label="Server">
                <input className="rag-input" value="(no server selection)" disabled />
              </Field>
            )}

            <div className="rag-field rag-field--action">
              <span aria-hidden="true">Action</span>
              <button className="rag-button" type="button" onClick={() => void this.refreshModelsForSection("llm")} disabled={this.state.modelLoadingBySection.llm}>
                {this.state.modelLoadingBySection.llm ? "Loading…" : "Refresh models"}
              </button>
            </div>

            <Field label="Model">
              {this.state.modelsBySection.llm.length > 20 && (
                <input
                  className="rag-input"
                  value={this.state.modelFilterBySection.llm}
                  onChange={(e) => this.setModelFilter("llm", e.target.value)}
                  placeholder="Filter models…"
                />
              )}
              {this.state.modelsBySection.llm.length > 0 ? (
                <select
                  className="rag-select"
                  value={chatEnv.OLLAMA_LLM_MODEL || ""}
                  onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_LLM_MODEL", e.target.value)}
                >
                  <option value="">Select model…</option>
                  {filterModels("llm").map((m) => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </select>
              ) : (
                <input
                  className="rag-input"
                  value={chatEnv.OLLAMA_LLM_MODEL || ""}
                  onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_LLM_MODEL", e.target.value)}
                  placeholder="Model name…"
                />
              )}
              {this.state.modelErrorBySection.llm && (
                <div className="rag-help">{this.state.modelErrorBySection.llm}</div>
              )}
            </Field>
          </div>
        </Accordion>

        <Accordion title="Embedding Provider" id="embedding" open={this.state.accordions.embedding} onToggle={this.setAccordion}>
          <Field label="Provider">
            <select
              className="rag-select"
              value={embeddingProvider}
              onChange={(e) => {
                const next = e.target.value;
                this.updateEnvField("document_chat", "EMBEDDING_PROVIDER", next);
                this.setState((prev) => ({
                  ...prev,
                  modelsBySection: { ...prev.modelsBySection, embedding: [] },
                  modelErrorBySection: { ...prev.modelErrorBySection, embedding: undefined },
                }), () => void this.refreshModelsForSection("embedding"));
              }}
            >
              {buildProviderOptions("embedding", embeddingProvider).map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </Field>
          <div className="rag-grid rag-grid--server-model">
            {embeddingProvider === "ollama" ? (
              <Field label="Server">
                {servers.length > 0 ? (
                  <>
                    <select
                      className="rag-select"
                      value={embeddingServerSel.isCustom ? "__custom__" : embeddingServerSel.serverId}
                      onChange={(e) => {
                        const next = e.target.value;
                        if (next === "__custom__") {
                          this.updateEnvFields("document_chat", {
                            OLLAMA_EMBEDDING_SERVER_ID: "",
                          }, () => void this.refreshModelsForSection("embedding"));
                          return;
                        }
                        const server = servers.find((s) => s.id === next);
                        this.updateEnvFields("document_chat", {
                          OLLAMA_EMBEDDING_SERVER_ID: next,
                          OLLAMA_EMBEDDING_BASE_URL: server?.serverAddress || chatEnv.OLLAMA_EMBEDDING_BASE_URL || "",
                        }, () => void this.refreshModelsForSection("embedding"));
                      }}
                    >
                      {servers.map((s) => (
                        <option key={s.id} value={s.id}>{s.serverName} ({s.serverAddress})</option>
                      ))}
                      <option value="__custom__">Custom URL…</option>
                    </select>
                    {embeddingServerSel.isCustom && (
                      <input
                        className="rag-input"
                        value={chatEnv.OLLAMA_EMBEDDING_BASE_URL || ""}
                        onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_EMBEDDING_BASE_URL", e.target.value)}
                        placeholder="http://localhost:11434"
                      />
                    )}
                  </>
                ) : (
                  <input
                    className="rag-input"
                    value={chatEnv.OLLAMA_EMBEDDING_BASE_URL || ""}
                    onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_EMBEDDING_BASE_URL", e.target.value)}
                    placeholder="http://localhost:11434"
                  />
                )}
              </Field>
            ) : (
              <Field label="Server">
                <input className="rag-input" value="(no server selection)" disabled />
              </Field>
            )}

            <div className="rag-field rag-field--action">
              <span aria-hidden="true">Action</span>
              <button className="rag-button" type="button" onClick={() => void this.refreshModelsForSection("embedding")} disabled={this.state.modelLoadingBySection.embedding}>
                {this.state.modelLoadingBySection.embedding ? "Loading…" : "Refresh models"}
              </button>
            </div>

            <Field label="Model">
              {this.state.modelsBySection.embedding.length > 20 && (
                <input
                  className="rag-input"
                  value={this.state.modelFilterBySection.embedding}
                  onChange={(e) => this.setModelFilter("embedding", e.target.value)}
                  placeholder="Filter models…"
                />
              )}
              {this.state.modelsBySection.embedding.length > 0 ? (
                <select
                  className="rag-select"
                  value={chatEnv.OLLAMA_EMBEDDING_MODEL || ""}
                  onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_EMBEDDING_MODEL", e.target.value)}
                >
                  <option value="">Select model…</option>
                  {filterModels("embedding").map((m) => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </select>
              ) : (
                <input
                  className="rag-input"
                  value={chatEnv.OLLAMA_EMBEDDING_MODEL || ""}
                  onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_EMBEDDING_MODEL", e.target.value)}
                  placeholder="Model name…"
                />
              )}
              {this.state.modelErrorBySection.embedding && (
                <div className="rag-help">{this.state.modelErrorBySection.embedding}</div>
              )}
            </Field>
          </div>
        </Accordion>

        <Accordion title="Contextual Retrieval" id="contextual" open={this.state.accordions.contextual} onToggle={this.setAccordion}>
          <label className="rag-checkbox-row">
            <input
              type="checkbox"
              checked={contextualEnabled}
              onChange={(e) => {
                const next = e.target.checked ? "true" : "false";
                this.updateEnvField("document_chat", "ENABLE_CONTEXTUAL_RETRIEVAL", next);
                if (e.target.checked) {
                  this.setState((prev) => ({
                    ...prev,
                    modelsBySection: { ...prev.modelsBySection, contextual: [] },
                    modelErrorBySection: { ...prev.modelErrorBySection, contextual: undefined },
                  }), () => void this.refreshModelsForSection("contextual"));
                }
              }}
            />
            Enable contextual retrieval
          </label>
          {contextualEnabled && (
            <div className="rag-grid rag-grid--server-model">
              <Field label="Server">
                {servers.length > 0 ? (
                  <>
                    <select
                      className="rag-select"
                      value={contextualServerSel.isCustom ? "__custom__" : contextualServerSel.serverId}
                      onChange={(e) => {
                        const next = e.target.value;
                        if (next === "__custom__") {
                          this.updateEnvFields("document_chat", {
                            OLLAMA_CONTEXTUAL_LLM_SERVER_ID: "",
                          }, () => void this.refreshModelsForSection("contextual"));
                          return;
                        }
                        const server = servers.find((s) => s.id === next);
                        this.updateEnvFields("document_chat", {
                          OLLAMA_CONTEXTUAL_LLM_SERVER_ID: next,
                          OLLAMA_CONTEXTUAL_LLM_BASE_URL: server?.serverAddress || chatEnv.OLLAMA_CONTEXTUAL_LLM_BASE_URL || "",
                        }, () => void this.refreshModelsForSection("contextual"));
                      }}
                    >
                      {servers.map((s) => (
                        <option key={s.id} value={s.id}>{s.serverName} ({s.serverAddress})</option>
                      ))}
                      <option value="__custom__">Custom URL…</option>
                    </select>
                    {contextualServerSel.isCustom && (
                      <input
                        className="rag-input"
                        value={chatEnv.OLLAMA_CONTEXTUAL_LLM_BASE_URL || ""}
                        onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_CONTEXTUAL_LLM_BASE_URL", e.target.value)}
                        placeholder="http://localhost:11434"
                      />
                    )}
                  </>
                ) : (
                  <input
                    className="rag-input"
                    value={chatEnv.OLLAMA_CONTEXTUAL_LLM_BASE_URL || ""}
                    onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_CONTEXTUAL_LLM_BASE_URL", e.target.value)}
                    placeholder="http://localhost:11434"
                  />
                )}
              </Field>

              <div className="rag-field rag-field--action">
                <span aria-hidden="true">Action</span>
                <button className="rag-button" type="button" onClick={() => void this.refreshModelsForSection("contextual")} disabled={this.state.modelLoadingBySection.contextual}>
                  {this.state.modelLoadingBySection.contextual ? "Loading…" : "Refresh models"}
                </button>
              </div>

              <Field label="Model">
                {this.state.modelsBySection.contextual.length > 20 && (
                  <input
                    className="rag-input"
                    value={this.state.modelFilterBySection.contextual}
                    onChange={(e) => this.setModelFilter("contextual", e.target.value)}
                    placeholder="Filter models…"
                  />
                )}
                {this.state.modelsBySection.contextual.length > 0 ? (
                  <select
                    className="rag-select"
                    value={chatEnv.OLLAMA_CONTEXTUAL_LLM_MODEL || ""}
                    onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_CONTEXTUAL_LLM_MODEL", e.target.value)}
                  >
                    <option value="">Select model…</option>
                    {filterModels("contextual").map((m) => (
                      <option key={m.name} value={m.name}>{m.name}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    className="rag-input"
                    value={chatEnv.OLLAMA_CONTEXTUAL_LLM_MODEL || ""}
                    onChange={(e) => this.updateEnvField("document_chat", "OLLAMA_CONTEXTUAL_LLM_MODEL", e.target.value)}
                    placeholder="Model name…"
                  />
                )}
                {this.state.modelErrorBySection.contextual && (
                  <div className="rag-help">{this.state.modelErrorBySection.contextual}</div>
                )}
              </Field>
            </div>
          )}
        </Accordion>

        <Accordion title="Evaluation Settings" id="evaluation" open={this.state.accordions.evaluation} onToggle={this.setAccordion}>
          <Field label="Provider">
            <select
              className="rag-select"
              value={evaluationProvider}
              onChange={(e) => {
                const next = e.target.value;
                this.updateEnvField("document_chat", "EVALUATION_PROVIDER", next);
                this.setState((prev) => ({
                  ...prev,
                  modelsBySection: { ...prev.modelsBySection, evaluation: [] },
                  modelErrorBySection: { ...prev.modelErrorBySection, evaluation: undefined },
                }), () => void this.refreshModelsForSection("evaluation"));
              }}
            >
              {buildProviderOptions("evaluation", evaluationProvider).map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </Field>
          <div className="rag-grid rag-grid--wide">
            <Field label="API Key">
              <input className="rag-input" type="password" value={chatEnv.OPENAI_EVALUATION_API_KEY} onChange={(e) => this.updateEnvField("document_chat", "OPENAI_EVALUATION_API_KEY", e.target.value)} />
            </Field>
            <Field label="Model">
              {evaluationProvider === "openai" && this.state.modelsBySection.evaluation.length > 0 ? (
                <select
                  className="rag-select"
                  value={chatEnv.OPENAI_EVALUATION_MODEL || ""}
                  onChange={(e) => this.updateEnvField("document_chat", "OPENAI_EVALUATION_MODEL", e.target.value)}
                >
                  <option value="">Select model…</option>
                  {filterModels("evaluation").map((m) => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </select>
              ) : (
                <input className="rag-input" value={chatEnv.OPENAI_EVALUATION_MODEL} onChange={(e) => this.updateEnvField("document_chat", "OPENAI_EVALUATION_MODEL", e.target.value)} />
              )}
              <div className="rag-actions">
                <button className="rag-button" type="button" onClick={() => void this.refreshModelsForSection("evaluation")} disabled={this.state.modelLoadingBySection.evaluation}>
                  {this.state.modelLoadingBySection.evaluation ? "Loading…" : "Refresh models"}
                </button>
              </div>
              {this.state.modelErrorBySection.evaluation && (
                <div className="rag-help">{this.state.modelErrorBySection.evaluation}</div>
              )}
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
    const health = this.state.health[serviceKey];
    const healthClass =
      health.status === "ok"
        ? "rag-health rag-health--ok"
        : health.status === "bad"
          ? "rag-health rag-health--bad"
          : health.status === "checking"
            ? "rag-health rag-health--checking"
            : "rag-health";
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
          {health.label && (
            <span className={healthClass} title={health.details || ""} aria-live="polite">
              {health.label}
            </span>
          )}
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
