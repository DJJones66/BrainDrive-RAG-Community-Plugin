import type { ProviderCatalogEntry } from "./types";

// Fallback mapping when /api/v1/ai/providers/catalog is unavailable.
export const FALLBACK_PROVIDER_CATALOG: ProviderCatalogEntry[] = [
  {
    id: "ollama",
    label: "Ollama",
    settingsId: "ollama_servers_settings",
    serverStrategy: "settings_servers",
    defaultServerId: null,
    configured: true, // assume at least localhost exists; UI should still verify servers/models
    configuredVia: "settings",
    enabled: true,
    serverCount: 0,
  },
  {
    id: "openai",
    label: "Openai",
    settingsId: "openai_api_keys_settings",
    serverStrategy: "single",
    defaultServerId: "openai_default_server",
    configured: false,
    configuredVia: null,
    enabled: true,
    serverCount: 0,
  },
  {
    id: "openrouter",
    label: "Openrouter",
    settingsId: "openrouter_api_keys_settings",
    serverStrategy: "single",
    defaultServerId: "openrouter_default_server",
    configured: false,
    configuredVia: null,
    enabled: true,
    serverCount: 0,
  },
  {
    id: "claude",
    label: "Claude",
    settingsId: "claude_api_keys_settings",
    serverStrategy: "single",
    defaultServerId: "claude_default_server",
    configured: false,
    configuredVia: null,
    enabled: true,
    serverCount: 0,
  },
  {
    id: "groq",
    label: "Groq",
    settingsId: "groq_api_keys_settings",
    serverStrategy: "single",
    defaultServerId: "groq_default_server",
    configured: false,
    configuredVia: null,
    enabled: true,
    serverCount: 0,
  },
];

