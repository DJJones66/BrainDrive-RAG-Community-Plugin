export type ApiService = {
  get?: (url: string, options?: unknown) => Promise<any>;
  post?: (url: string, body?: unknown, options?: unknown) => Promise<any>;
};

export type ProviderServerStrategy = "settings_servers" | "single" | "unknown";

export type ProviderCatalogEntry = {
  id: string;
  label: string;
  settingsId: string | null;
  serverStrategy: ProviderServerStrategy;
  defaultServerId: string | null;
  configured: boolean;
  configuredVia: "settings" | "env" | null;
  enabled: boolean;
  serverCount: number;
};

export type OllamaServer = {
  id: string;
  serverName: string;
  serverAddress: string;
  apiKey?: string;
  connectionStatus?: string;
};

export type ProviderModel = {
  id: string;
  name: string;
  provider: string;
  metadata?: any;
};

export type ModelPurpose = "chat" | "embedding" | "evaluation";

