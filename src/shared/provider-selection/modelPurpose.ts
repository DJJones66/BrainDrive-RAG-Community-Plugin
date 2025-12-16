import type { ModelPurpose, ProviderModel } from "./types";

const EMBEDDING_HINTS = [
  "embed",
  "embedding",
  "text-embedding",
  "nomic-embed",
  "mxbai-embed",
  "bge-",
  "e5-",
];

function looksLikeEmbedding(modelName: string): boolean {
  const name = (modelName || "").toLowerCase();
  return EMBEDDING_HINTS.some((hint) => name.includes(hint));
}

export function filterModelsByPurpose(
  models: ProviderModel[],
  purpose: ModelPurpose,
): ProviderModel[] {
  if (!Array.isArray(models) || models.length === 0) return [];

  if (purpose === "embedding") {
    const filtered = models.filter((m) => looksLikeEmbedding(m.name || m.id));
    return filtered.length > 0 ? filtered : models;
  }

  // chat/evaluation: prefer non-embedding models
  const filtered = models.filter((m) => !looksLikeEmbedding(m.name || m.id));
  return filtered.length > 0 ? filtered : models;
}

