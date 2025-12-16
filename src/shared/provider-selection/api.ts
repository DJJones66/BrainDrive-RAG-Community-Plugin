import type { ApiService, OllamaServer, ProviderCatalogEntry, ProviderModel } from "./types";
import { FALLBACK_PROVIDER_CATALOG } from "./providerMeta";

function coerceArray<T = any>(value: any): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function unwrapApiResponse(resp: any): any {
  if (!resp) return resp;
  if (resp.data) return resp.data;
  return resp;
}

export async function fetchProviderCatalog(
  api: ApiService | undefined,
  options?: { userId?: string; includeUnconfigured?: boolean },
): Promise<ProviderCatalogEntry[]> {
  const userId = options?.userId ?? "current";
  const includeUnconfigured = options?.includeUnconfigured ?? false;

  if (!api?.get) {
    return includeUnconfigured
      ? FALLBACK_PROVIDER_CATALOG
      : FALLBACK_PROVIDER_CATALOG.filter((p) => p.configured && p.enabled);
  }

  try {
    const qs = new URLSearchParams({ user_id: userId }).toString();
    const resp = unwrapApiResponse(await api.get(`/api/v1/ai/providers/catalog?${qs}`));
    const rawProviders = coerceArray(resp?.providers);

    const mapped: ProviderCatalogEntry[] = rawProviders.map((p: any) => ({
      id: String(p.id ?? ""),
      label: String(p.label ?? p.id ?? ""),
      settingsId: p.settings_id ?? p.settingsId ?? null,
      serverStrategy: (p.server_strategy ?? p.serverStrategy ?? "unknown") as any,
      defaultServerId: p.default_server_id ?? p.defaultServerId ?? null,
      configured: Boolean(p.configured),
      configuredVia: (p.configured_via ?? p.configuredVia ?? null) as any,
      enabled: Boolean(p.enabled ?? true),
      serverCount: Number.isFinite(Number(p.server_count ?? p.serverCount))
        ? Number(p.server_count ?? p.serverCount)
        : 0,
    }));

    const filtered = includeUnconfigured ? mapped : mapped.filter((p) => p.configured && p.enabled);
    return filtered.filter((p) => p.id);
  } catch {
    return includeUnconfigured
      ? FALLBACK_PROVIDER_CATALOG
      : FALLBACK_PROVIDER_CATALOG.filter((p) => p.configured && p.enabled);
  }
}

export async function fetchOllamaServers(
  api: ApiService | undefined,
  options?: { userId?: string },
): Promise<OllamaServer[]> {
  const userId = options?.userId ?? "current";
  if (!api?.get) return [];

  try {
    const qs = new URLSearchParams({
      definition_id: "ollama_servers_settings",
      scope: "user",
      user_id: userId,
    }).toString();
    const resp = unwrapApiResponse(await api.get(`/api/v1/settings/instances?${qs}`));

    const first = Array.isArray(resp) ? resp[0] : resp;
    const value = first?.value ?? first;
    const parsed = typeof value === "string" ? JSON.parse(value) : value;
    const servers = coerceArray<OllamaServer>(parsed?.servers);
    return servers.filter((s) => Boolean(s?.id));
  } catch {
    return [];
  }
}

export async function fetchProviderModels(
  api: ApiService | undefined,
  params: {
    provider: string;
    settingsId: string;
    serverId: string;
    userId?: string;
  },
): Promise<ProviderModel[]> {
  if (!api?.get) return [];

  const qs = new URLSearchParams({
    provider: params.provider,
    settings_id: params.settingsId,
    server_id: params.serverId,
    user_id: params.userId ?? "current",
  }).toString();

  try {
    const resp = unwrapApiResponse(await api.get(`/api/v1/ai/providers/models?${qs}`));
    const rawModels = coerceArray(resp?.models ?? resp);
    return rawModels
      .map((m: any) => ({
        id: String(m.id ?? m.name ?? ""),
        name: String(m.name ?? m.id ?? ""),
        provider: String(m.provider ?? params.provider),
        metadata: m.metadata,
      }))
      .filter((m) => Boolean(m.name));
  } catch {
    return [];
  }
}

