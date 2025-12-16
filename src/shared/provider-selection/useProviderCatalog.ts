import { useCallback, useEffect, useMemo, useState } from "react";
import type { ApiService, OllamaServer, ProviderCatalogEntry } from "./types";
import { fetchOllamaServers, fetchProviderCatalog } from "./api";

export function useProviderCatalog(
  api: ApiService | undefined,
  options?: { userId?: string; includeUnconfigured?: boolean },
) {
  const userId = options?.userId ?? "current";
  const includeUnconfigured = options?.includeUnconfigured ?? false;

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderCatalogEntry[]>([]);
  const [ollamaServers, setOllamaServers] = useState<OllamaServer[]>([]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [catalog, servers] = await Promise.all([
        fetchProviderCatalog(api, { userId, includeUnconfigured }),
        fetchOllamaServers(api, { userId }),
      ]);
      setProviders(catalog);
      setOllamaServers(servers);
      setLoading(false);
    } catch (err: any) {
      setProviders([]);
      setOllamaServers([]);
      setError(err?.message || "Failed to load provider catalog");
      setLoading(false);
    }
  }, [api, userId, includeUnconfigured]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const providersById = useMemo(() => {
    const map = new Map<string, ProviderCatalogEntry>();
    for (const p of providers) map.set(p.id, p);
    return map;
  }, [providers]);

  return { loading, error, providers, providersById, ollamaServers, refresh };
}

