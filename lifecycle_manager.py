#!/usr/bin/env python3
"""
BrainDrive RAG Community Plugin Lifecycle Manager

Orchestrates plugin install/update/delete and coordinates docker + venv
flows for Document-Chat-Service and Document-Processing-Service.
"""

import asyncio
import datetime
import importlib.util
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from urllib.parse import unquote

import httpx
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

try:
  from app.core.encryption import encryption_service  # type: ignore
  from app.core.job_manager_provider import get_job_manager  # type: ignore
except Exception:  # pragma: no cover - fallback for remote install
  encryption_service = None
  get_job_manager = None

try:
  from app.utils.ollama import normalize_server_base, make_dedupe_key  # type: ignore
except Exception:  # pragma: no cover - fallback for remote install
  def normalize_server_base(url: str) -> str:
    url = unquote(url or "").strip()
    url = url.rstrip("/")
    if url.endswith("/api/pull"):
      url = url[: -len("/api/pull")]
    if url.endswith("/api"):
      url = url[: -len("/api")]
    return url

  def make_dedupe_key(server_base: str, name: str) -> str:
    return f"{server_base}|{name}"

CURRENT_DIR = Path(__file__).resolve().parent

HELPER_PATH = CURRENT_DIR / "community_lifecycle_manager.py"
spec = importlib.util.spec_from_file_location("rag.community_lifecycle_manager", HELPER_PATH)
helper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_module)
CommunityPluginLifecycleBase = helper_module.CommunityPluginLifecycleBase

SERVICE_OPS_PATH = CURRENT_DIR / "service_ops.py"
service_ops_spec = importlib.util.spec_from_file_location("rag.service_ops", SERVICE_OPS_PATH)
service_ops_module = importlib.util.module_from_spec(service_ops_spec)
service_ops_spec.loader.exec_module(service_ops_module)
prepare_service = service_ops_module.prepare_service
shutdown_service = service_ops_module.shutdown_service
restart_service = service_ops_module.restart_service

start_service = service_ops_module.start_service

health_check = service_ops_module.health_check
get_service_metadata = service_ops_module.get_service_metadata
get_required_env_vars_map = service_ops_module.get_required_env_vars_map
materialize_env_file = service_ops_module.materialize_env_file
update_env_and_restart = service_ops_module.update_env_and_restart
update_health_urls = getattr(service_ops_module, "update_health_urls", None)
_parse_env_template = getattr(service_ops_module, "_parse_env_template", None)
_template_candidates = getattr(service_ops_module, "_template_candidates", None)
SERVICE_CONFIG = getattr(service_ops_module, "SERVICE_CONFIG", {})

logger = structlog.get_logger()

# Curated env keys that map to runtime settings (avoid exposing every env)
DOC_CHAT_ENV_KEYS: List[str] = [
  "API_HOST",
  "API_PORT",
  "DOCUMENT_PROCESSOR_API_URL",
  "LLM_PROVIDER",
  "EMBEDDING_PROVIDER",
  "OLLAMA_LLM_BASE_URL",
  "OLLAMA_LLM_MODEL",
  "OLLAMA_EMBEDDING_BASE_URL",
  "OLLAMA_EMBEDDING_MODEL",
  "ENABLE_CONTEXTUAL_RETRIEVAL",
  "OLLAMA_CONTEXTUAL_LLM_BASE_URL",
  "OLLAMA_CONTEXTUAL_LLM_MODEL",
  "EVALUATION_PROVIDER",
  "OPENAI_EVALUATION_API_KEY",
  "OPENAI_EVALUATION_MODEL",
]

DOC_PROCESSING_ENV_KEYS: List[str] = [
  "API_HOST",
  "API_PORT",
  "CORS_ALLOW_ANY",
  "AUTH_METHOD",
  "AUTH_API_KEY",
  "JWT_SECRET",
]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
HARD_CODED_OLLAMA_MODELS: Tuple[Tuple[str, str, str], ...] = (
  ("llama3.1:8b", DEFAULT_OLLAMA_BASE_URL, "OLLAMA_LLM_MODEL"),
  ("nomic-embed-text:latest", DEFAULT_OLLAMA_BASE_URL, "OLLAMA_EMBEDDING_MODEL"),
  ("llama3.2:3b", DEFAULT_OLLAMA_BASE_URL, "OLLAMA_CONTEXTUAL_LLM_MODEL"),
)


def _coerce_int(value: Any, default: int) -> int:
  try:
    return int(value)
  except Exception:
    return default


def _normalize_service_settings(raw: Dict[str, Any], fallback_port: int) -> Dict[str, Any]:
  """Ensure required fields exist and derive base/health URLs."""
  data = deepcopy(raw) if raw else {}
  protocol = (data.get("protocol") or "http").lower()
  host = data.get("host") or "localhost"
  port = _coerce_int(data.get("port"), fallback_port)
  health_path = data.get("health_path") or "/health"
  if not str(health_path).startswith("/"):
    health_path = f"/{health_path}"
  data.update({
    "protocol": protocol,
    "host": host,
    "port": port,
    "health_path": health_path,
    "base_url": f"{protocol}://{host}:{port}",
    "health_url": f"{protocol}://{host}:{port}{health_path}",
    "env": data.get("env") or {},
  })
  return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively merge dictionaries (override wins)."""
  result: Dict[str, Any] = deepcopy(base)
  for key, value in (override or {}).items():
    if isinstance(value, dict) and isinstance(result.get(key), dict):
      result[key] = _deep_merge(result[key], value)
    else:
      result[key] = value
  return result


def _expand_model_tokens(model: Dict[str, Any]) -> Set[str]:
  tokens: Set[str] = set()
  for key in ("name", "model", "digest"):
    value = model.get(key)
    if value:
      tokens.add(str(value))
  aliases = model.get("aliases") or []
  if isinstance(aliases, list):
    for alias in aliases:
      if isinstance(alias, str):
        tokens.add(alias)
      elif isinstance(alias, dict):
        for value in alias.values():
          if value:
            tokens.add(str(value))
  expanded: Set[str] = set()
  for token in tokens:
    cleaned = str(token).strip().lower()
    if not cleaned:
      continue
    expanded.add(cleaned)
    if ":" in cleaned:
      expanded.add(cleaned.split(":", 1)[0])
  return expanded


def _build_model_token_index(payload: Dict[str, Any]) -> Set[str]:
  tokens: Set[str] = set()
  models = payload.get("models") or []
  if not isinstance(models, list):
    return tokens
  for model in models:
    if isinstance(model, dict):
      tokens.update(_expand_model_tokens(model))
  return tokens


def _build_service_defaults(service_key: str) -> Dict[str, Any]:
  """Seed defaults from env templates (when available) plus sensible fallbacks."""
  defaults: Dict[str, Any] = {}
  if service_key == "document_chat":
    defaults = {
      "enabled": True,
      "mode": SERVICE_CONFIG.get(service_key).default_mode if SERVICE_CONFIG.get(service_key) else "venv",
      "protocol": "http",
      "host": "localhost",
      "port": 18000,
      "health_path": "/health",
      "env": {}
    }
  else:
    defaults = {
      "enabled": True,
      "mode": SERVICE_CONFIG.get(service_key).default_mode if SERVICE_CONFIG.get(service_key) else "venv",
      "protocol": "http",
      "host": "localhost",
      "port": 18080,
      "health_path": "/health",
      "env": {}
    }

  if _parse_env_template and _template_candidates and service_key in SERVICE_CONFIG:
    template_keys: List[str] = []
    parsed_defaults: Dict[str, str] = {}
    for candidate in _template_candidates(SERVICE_CONFIG[service_key]):
      keys, values = _parse_env_template(candidate)
      if keys:
        template_keys.extend(keys)
      if values:
        parsed_defaults.update(values)

    # Connectivity from template if present
    defaults["host"] = parsed_defaults.get("API_HOST", defaults["host"])
    defaults["port"] = _coerce_int(parsed_defaults.get("API_PORT"), defaults["port"])

    if service_key == "document_chat":
      defaults["env"] = {
        "API_HOST": defaults["host"],
        "API_PORT": str(defaults["port"]),
        "DOCUMENT_PROCESSOR_API_URL": parsed_defaults.get("DOCUMENT_PROCESSOR_API_URL", f"http://localhost:18080/documents/"),
        "LLM_PROVIDER": parsed_defaults.get("LLM_PROVIDER", "ollama"),
        "EMBEDDING_PROVIDER": parsed_defaults.get("EMBEDDING_PROVIDER", "ollama"),
        "OLLAMA_LLM_BASE_URL": parsed_defaults.get("OLLAMA_LLM_BASE_URL", ""),
        "OLLAMA_LLM_MODEL": parsed_defaults.get("OLLAMA_LLM_MODEL", ""),
        "OLLAMA_EMBEDDING_BASE_URL": parsed_defaults.get("OLLAMA_EMBEDDING_BASE_URL", ""),
        "OLLAMA_EMBEDDING_MODEL": parsed_defaults.get("OLLAMA_EMBEDDING_MODEL", ""),
        "ENABLE_CONTEXTUAL_RETRIEVAL": parsed_defaults.get("ENABLE_CONTEXTUAL_RETRIEVAL", "false"),
        "OLLAMA_CONTEXTUAL_LLM_BASE_URL": parsed_defaults.get("OLLAMA_CONTEXTUAL_LLM_BASE_URL", ""),
        "OLLAMA_CONTEXTUAL_LLM_MODEL": parsed_defaults.get("OLLAMA_CONTEXTUAL_LLM_MODEL", ""),
        "EVALUATION_PROVIDER": parsed_defaults.get("EVALUATION_PROVIDER", "openai"),
        "OPENAI_EVALUATION_API_KEY": parsed_defaults.get("OPENAI_EVALUATION_API_KEY", ""),
        "OPENAI_EVALUATION_MODEL": parsed_defaults.get("OPENAI_EVALUATION_MODEL", ""),
      }
    else:
      defaults["env"] = {
        "API_HOST": defaults["host"],
        "API_PORT": str(defaults["port"]),
        "CORS_ALLOW_ANY": parsed_defaults.get("CORS_ALLOW_ANY", "1"),
        "AUTH_METHOD": parsed_defaults.get("AUTH_METHOD", "api_key"),
        "AUTH_API_KEY": parsed_defaults.get("AUTH_API_KEY", ""),
        "JWT_SECRET": parsed_defaults.get("JWT_SECRET", ""),
      }

  return _normalize_service_settings(defaults, defaults.get("port", 0))

PLUGIN_DATA: Dict[str, Any] = {
  "name": "BrainDrive-RAG-Community-Plugin",
  "description": "Settings-driven installer for Document Chat and Document Processing services with docker + venv flows.",
  "version": "1.0.0",
  "type": "frontend",
  "icon": "Settings2",
  "category": "ai",
  "official": False,
  "author": "BrainDrive Community",
  "compatibility": "1.0.0",
  "scope": "BrainDriveRAGCommunity",
  "bundle_method": "webpack",
  "bundle_location": "dist/remoteEntry.js",
  "is_local": False,
  "long_description": "Manage and install document services from the plugin UI with toggles for venv or docker execution and live health checks.",
  "plugin_slug": "BrainDriveRAGCommunity",
  "source_type": "github",
  "source_url": "https://github.com/DJJones66/BrainDrive-RAG-Community-Plugin",
  "permissions": ["storage.read", "storage.write", "api.access", "settings.manage"]
}

SETTINGS_DEFINITION_ID = "braindrive_rag_service_settings"

MODULE_DATA: List[Dict[str, Any]] = [
  {
    "name": "BrainDriveRAGSettings",
    "display_name": "RAG Service Controls",
    "description": "Toggle Document Chat and Document Processing services, choose docker or venv, and check health.",
    "icon": "Settings",
    "category": "Settings",
    "priority": 1,
    "props": {
      "title": "RAG Control Panel",
      "subtitle": "Enable services and manage runtime mode."
    },
    "config_fields": {
      "document_chat_enabled": {
        "type": "boolean",
        "description": "Enable Document Chat Service",
        "default": False
      },
      "document_processing_enabled": {
        "type": "boolean",
        "description": "Enable Document Processing Service",
        "default": False
      }
    },
    "required_services": {
      "api": {"methods": ["get", "post"], "version": "1.0.0"},
      "settings": {"methods": ["getSetting", "setSetting", "getSettingDefinitions"], "version": "1.0.0"},
      "theme": {"methods": ["getCurrentTheme", "addThemeChangeListener", "removeThemeChangeListener"], "version": "1.0.0"}
    },
    "layout": {
      "minWidth": 8,
      "minHeight": 4,
      "defaultWidth": 12,
      "defaultHeight": 6
    },
    # Tags drive Settings page discovery: first non-"Settings" tag is used as setting name.
    "tags": ["Settings", SETTINGS_DEFINITION_ID, "rag", "installer", "services"]
  }
]


class BrainDriveRAGCommunityLifecycleManager(CommunityPluginLifecycleBase):
  """Lifecycle manager for BrainDrive RAG Community Plugin."""

  def __init__(self, plugins_base_dir: Optional[str] = None):
    self.plugin_data = PLUGIN_DATA
    self.module_data = MODULE_DATA
    self.settings_definition_id = SETTINGS_DEFINITION_ID
    self.plugins_base_dir = plugins_base_dir
    self.default_settings = {
      "document_chat": _build_service_defaults("document_chat"),
      "document_processing": _build_service_defaults("document_processing"),
    }
    self.set_plugin_root(Path(__file__).resolve().parent)
    chat_defaults = self.default_settings["document_chat"]
    proc_defaults = self.default_settings["document_processing"]
    self.required_services_runtime = [
      {
        "name": "document_chat",
        "source_url": "https://github.com/DJJones66/Document-Chat-Service",
        "type": "venv_process",
        "install_command": "python service_scripts/install_with_venv.py --full",
        "start_command": "python service_scripts/start_with_venv.py",
        "stop_command": "python service_scripts/shutdown_with_venv.py",
        "restart_command": "python service_scripts/restart_with_venv.py",
        "healthcheck_url": chat_defaults.get("health_url", "http://localhost:18000/health"),
        "definition_id": SETTINGS_DEFINITION_ID,
        "runtime_dir_key": "Document-Chat-Service",
        "env_inherit": "minimal",
        "env_overrides": {"PROCESS_PORT": str(chat_defaults.get("port", 18000))},
        "required_env_vars": []
      },
      {
        "name": "document_processing",
        "source_url": "https://github.com/DJJones66/Document-Processing-Service",
        "type": "venv_process",
        "install_command": "python service_scripts/install_with_venv.py --full",
        "start_command": "python service_scripts/start_with_venv.py",
        "stop_command": "python service_scripts/shutdown_with_venv.py",
        "restart_command": "python service_scripts/restart_with_venv.py",
        "healthcheck_url": proc_defaults.get("health_url", "http://localhost:18080/health"),
        "definition_id": SETTINGS_DEFINITION_ID,
        "runtime_dir_key": "Document-Processing-Service",
        "env_inherit": "minimal",
        "env_overrides": {"PROCESS_PORT": str(proc_defaults.get("port", 18080))},
        "required_env_vars": []
      }
    ]

    if plugins_base_dir:
      shared_path = Path(plugins_base_dir) / "shared" / PLUGIN_DATA["plugin_slug"] / f"v{PLUGIN_DATA['version']}"
    else:
      shared_path = Path(__file__).resolve().parent.parent.parent / "backend" / "plugins" / "shared" / PLUGIN_DATA["plugin_slug"] / f"v{PLUGIN_DATA['version']}"

    # Refresh required_env_vars from current templates/fallbacks for compatibility.
    env_map = get_required_env_vars_map()
    self._apply_required_env_vars(env_map)

    super().__init__(
      plugin_slug=PLUGIN_DATA["plugin_slug"],
      version=PLUGIN_DATA["version"],
      shared_storage_path=shared_path
    )

  def _apply_required_env_vars(self, env_map: Dict[str, List[str]]) -> None:
    """
    Update in-memory required_services_runtime to include required_env_vars
    derived from templates and fallbacks (keeps legacy compatibility).
    """
    for service in self.required_services_runtime:
      name = service.get("name")
      service["required_env_vars"] = env_map.get(name, service.get("required_env_vars", []))

  def _refresh_required_services_runtime_from_settings(self, settings_value: Dict[str, Any]) -> None:
    """
    Update runtime healthcheck URLs from settings (per-user/per-instance).
    """
    if not settings_value:
      return
    overrides: Dict[str, str] = {}
    for service in self.required_services_runtime:
      name = service.get("name")
      svc_settings = settings_value.get(name) or settings_value.get(name.replace("document_", "document_"))
      if svc_settings and isinstance(svc_settings, dict):
        normalized = _normalize_service_settings(
          svc_settings,
          svc_settings.get("port", 0) or (18000 if name == "document_chat" else 18080)
        )
        service["healthcheck_url"] = normalized.get("health_url", service.get("healthcheck_url"))
        service["env_overrides"] = {"PROCESS_PORT": str(normalized.get("port", 0) or (18000 if name == "document_chat" else 18080))}
        overrides[name] = service["healthcheck_url"]
    if update_health_urls and overrides:
      try:
        update_health_urls(overrides)
      except Exception:
        # Safe fallback: do not let a health URL override crash lifecycle init
        pass

  def _stringify_env_value(self, value: Any) -> str:
    if isinstance(value, bool):
      return "true" if value else "false"
    return "" if value is None else str(value)

  def _build_env_payload_from_settings(self, settings_value: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Build curated env payloads for services based on saved settings.
    """
    env_payload: Dict[str, Dict[str, str]] = {}
    chat_settings = settings_value.get("document_chat", {})
    proc_settings = settings_value.get("document_processing", {})

    if chat_settings:
      chat_norm = _normalize_service_settings(chat_settings, _coerce_int(chat_settings.get("port"), 18000))
      proc_norm = _normalize_service_settings(proc_settings, _coerce_int(proc_settings.get("port"), 18080)) if proc_settings else {"host": "localhost", "port": 18080, "protocol": "http"}
      chat_env = deepcopy(chat_norm.get("env") or {})
      chat_env.update({
        "API_HOST": chat_norm.get("host"),
        "API_PORT": str(chat_norm.get("port")),
        "DOCUMENT_PROCESSOR_API_URL": chat_env.get("DOCUMENT_PROCESSOR_API_URL") or f"{proc_norm.get('protocol','http')}://{proc_norm.get('host','localhost')}:{proc_norm.get('port',18080)}/documents/",
      })
      env_payload["document_chat"] = {k: self._stringify_env_value(v) for k, v in chat_env.items() if k in DOC_CHAT_ENV_KEYS}

    if proc_settings:
      proc_norm = _normalize_service_settings(proc_settings, _coerce_int(proc_settings.get("port"), 18080))
      proc_env = deepcopy(proc_norm.get("env") or {})
      proc_env.update({
        "API_HOST": proc_norm.get("host"),
        "API_PORT": str(proc_norm.get("port")),
      })
      env_payload["document_processing"] = {k: self._stringify_env_value(v) for k, v in proc_env.items() if k in DOC_PROCESSING_ENV_KEYS}

    return env_payload

  def _hardcoded_ollama_models(self) -> List[Dict[str, Any]]:
    specs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for model_name, server_url, source in HARD_CODED_OLLAMA_MODELS:
      model = str(model_name).strip()
      base = normalize_server_base(str(server_url).strip())
      if not model or not base.startswith(("http://", "https://")):
        continue
      key = (base, model)
      record = specs.get(key)
      if not record:
        record = {"model_name": model, "server_url": base, "sources": []}
        specs[key] = record
      record["sources"].append(source)
    return list(specs.values())

  async def _fetch_ollama_tags(self, client: httpx.AsyncClient, server_base: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    try:
      response = await client.get(f"{server_base}/api/tags", headers=headers)
      response.raise_for_status()
      return response.json()
    except Exception as exc:
      logger.warning("Failed to fetch Ollama tags", server_url=server_base, error=str(exc))
      return None

  async def _enqueue_missing_ollama_models(self, user_id: str) -> Dict[str, Any]:
    jobs_env = os.environ.get("RAG_USE_JOB_MANAGER")
    use_jobs = (jobs_env or "1").lower() in {"1", "true", "yes", "on"}
    if not use_jobs:
      return {"skipped": True, "reason": "job_manager_disabled"}
    if not get_job_manager:
      return {"skipped": True, "reason": "job_manager_unavailable"}

    try:
      model_specs = self._hardcoded_ollama_models()
      if not model_specs:
        return {"skipped": True, "reason": "no_ollama_models"}

      server_groups: Dict[str, List[Dict[str, Any]]] = {}
      for spec in model_specs:
        server_groups.setdefault(spec["server_url"], []).append(spec)

      results: List[Dict[str, Any]] = []
      try:
        job_manager = await get_job_manager()
      except Exception as exc:
        logger.warning("Job manager unavailable for model installs", error=str(exc))
        return {"skipped": True, "reason": "job_manager_unavailable", "error": str(exc)}

      async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
        for server_base, specs in server_groups.items():
          headers = {"Content-Type": "application/json"}
          tags_payload = await self._fetch_ollama_tags(client, server_base, headers)
          if not tags_payload:
            results.append({
              "server_url": server_base,
              "status": "skipped",
              "reason": "ollama_unreachable",
            })
            continue
          tokens = _build_model_token_index(tags_payload)

          for spec in specs:
            model_name = spec["model_name"]
            model_key = model_name.strip().lower()
            if model_key in tokens:
              results.append({
                "model_name": model_name,
                "server_url": server_base,
                "status": "skipped",
                "reason": "already_present",
                "sources": spec.get("sources", []),
              })
              continue
            try:
              payload = {
                "model_name": model_name,
                "server_url": server_base,
              }
              job, created = await job_manager.enqueue_job(
                job_type="ollama.install",
                payload=payload,
                user_id=user_id,
                idempotency_key=make_dedupe_key(server_base, model_name),
                max_retries=1,
              )
              results.append({
                "model_name": model_name,
                "server_url": server_base,
                "status": "queued",
                "job_id": job.id,
                "deduped": not created,
                "sources": spec.get("sources", []),
              })
            except Exception as exc:
              logger.warning("Failed to enqueue Ollama install", model=model_name, server_url=server_base, error=str(exc))
              results.append({
                "model_name": model_name,
                "server_url": server_base,
                "status": "error",
                "error": str(exc),
                "sources": spec.get("sources", []),
              })

      return {"skipped": False, "results": results}
    except Exception as exc:
      logger.warning("Skipping Ollama model installs due to unexpected error", error=str(exc))
      return {"skipped": True, "reason": "error", "error": str(exc)}

  # ---------------------------------------------------------------------------
  # Install / uninstall / update
  # ---------------------------------------------------------------------------

  async def _perform_user_installation(self, user_id: str, db: AsyncSession, shared_plugin_path: Path) -> Dict[str, Any]:
    try:
      # Refresh required_env_vars before persisting runtime rows
      env_map = get_required_env_vars_map()
      self._apply_required_env_vars(env_map)

      records = await self._create_database_records(user_id, db)
      if not records.get("success"):
        return records

      settings_result = await self._ensure_settings(user_id, db)
      settings_value = settings_result.get("value") or await self._load_settings_value(user_id, db)
      if settings_value:
        self._refresh_required_services_runtime_from_settings(settings_value)
      service_installs = await self._prepare_services(user_id)
      model_installs = await self._enqueue_missing_ollama_models(user_id)
      service_starts = None
      auto_start = os.environ.get("RAG_AUTO_START", "1").lower() in {"1", "true", "yes", "on"}
      if auto_start and service_installs.get("mode") == "sync":
        service_starts = await self._start_services(settings_value)
      health = None
      if service_installs.get("mode") == "sync":
        health = await self._collect_service_health(settings_value)

      return {
        "success": True,
        "plugin_id": records["plugin_id"],
        "modules_created": records["modules_created"],
        "settings": settings_result,
        "settings_value": settings_value,
        "service_installs": service_installs,
        "model_installs": model_installs,
        "service_starts": service_starts,
        "service_health": health,
        "service_metadata": get_service_metadata(),
        "required_env_vars": env_map,
      }
    except Exception as error:  # pragma: no cover
      logger.error("RAG plugin installation failed", error=str(error))
      return {"success": False, "error": str(error)}

  async def _perform_user_uninstallation(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
    try:
      shutdown_results = await self._shutdown_services()
      plugin_id = f"{user_id}_{self.plugin_data['plugin_slug']}"
      page_cleanup = await self._delete_plugin_page(user_id, db)
      if not page_cleanup.get("success"):
        return page_cleanup
      delete_result = await self._delete_database_records(user_id, plugin_id, db)
      settings_cleanup = await self._delete_settings_instance(user_id, db)
      return {
        **delete_result,
        "settings_cleanup": settings_cleanup,
        "service_shutdown": shutdown_results
      }
    except Exception as error:  # pragma: no cover
      logger.error("RAG plugin uninstallation failed", error=str(error))
      return {"success": False, "error": str(error)}

  async def _perform_user_update(self, user_id: str, db: AsyncSession, shared_plugin_path: Path) -> Dict[str, Any]:
    logger.info("RAG plugin update invoked; running service prep")
    env_map = get_required_env_vars_map()
    self._apply_required_env_vars(env_map)
    settings_value = await self._load_settings_value(user_id, db)
    if settings_value:
      self._refresh_required_services_runtime_from_settings(settings_value)
    await self._refresh_required_env_vars_in_db(user_id, db, env_map)
    installs = await self._prepare_services(user_id)
    return {"success": True, "service_installs": installs, "required_env_vars": env_map, "settings_value": settings_value}

  # ---------------------------------------------------------------------------
  # Database helpers
  # ---------------------------------------------------------------------------

  async def _create_database_records(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
    try:
      existing = await self._check_existing_plugin(user_id, db)
      if existing.get("exists"):
        return {"success": False, "error": "Plugin already installed for user", "plugin_id": existing.get("plugin_id")}

      now = datetime.datetime.utcnow()
      plugin_id = f"{user_id}_{self.plugin_data['plugin_slug']}"

      plugin_stmt = text("""
        INSERT INTO plugin (
          id, name, description, icon, category, type, version,
          bundle_method, bundle_location, created_at, updated_at, user_id,
          plugin_slug, scope, permissions
        ) VALUES (
          :id, :name, :description, :icon, :category, :type, :version,
          :bundle_method, :bundle_location, :created_at, :updated_at, :user_id,
          :plugin_slug, :scope, :permissions
        )
      """)

      await db.execute(plugin_stmt, {
        "id": plugin_id,
        "name": self.plugin_data["name"],
        "description": self.plugin_data["description"],
        "icon": self.plugin_data["icon"],
        "category": self.plugin_data["category"],
        "type": self.plugin_data["type"],
        "version": self.plugin_data["version"],
        "bundle_method": self.plugin_data["bundle_method"],
        "bundle_location": self.plugin_data["bundle_location"],
        "created_at": now,
        "updated_at": now,
        "user_id": user_id,
        "plugin_slug": self.plugin_data["plugin_slug"],
        "scope": self.plugin_data["scope"],
        "permissions": json.dumps(self.plugin_data.get("permissions", []))
      })

      modules_created: List[str] = []
      for module in self.module_data:
        module_id = f"{user_id}_{self.plugin_data['plugin_slug']}_{module['name']}"
        module_stmt = text("""
          INSERT INTO module (
            id, plugin_id, name, display_name, description, icon, category,
            enabled, priority, props, config_fields, required_services,
            layout, tags, created_at, updated_at, user_id
          ) VALUES (
            :id, :plugin_id, :name, :display_name, :description, :icon, :category,
            :enabled, :priority, :props, :config_fields, :required_services,
            :layout, :tags, :created_at, :updated_at, :user_id
          )
        """)

        await db.execute(module_stmt, {
          "id": module_id,
          "plugin_id": plugin_id,
          "name": module["name"],
          "display_name": module["display_name"],
          "description": module["description"],
          "icon": module["icon"],
          "category": module["category"],
          "enabled": True,
          "priority": module["priority"],
          "props": json.dumps(module["props"]),
          "config_fields": json.dumps(module["config_fields"]),
          "required_services": json.dumps(module["required_services"]),
          "layout": json.dumps(module["layout"]),
          "tags": json.dumps(module["tags"]),
          "created_at": now,
          "updated_at": now,
          "user_id": user_id
        })
        modules_created.append(module_id)

      services_created: List[str] = []
      for service_data in self.required_services_runtime:
        service_id = f"{user_id}_{self.plugin_data['plugin_slug']}_{service_data['name']}"
        service_stmt = text("""
          INSERT INTO plugin_service_runtime
          (id, plugin_id, plugin_slug, name, source_url, type, install_command, start_command,
          stop_command, restart_command, runtime_dir_key, env_inherit, env_overrides,
          healthcheck_url, definition_id, required_env_vars, status, created_at, updated_at, user_id)
          VALUES
          (:id, :plugin_id, :plugin_slug, :name, :source_url, :type, :install_command, :start_command,
          :stop_command, :restart_command, :runtime_dir_key, :env_inherit, :env_overrides,
          :healthcheck_url, :definition_id, :required_env_vars, :status, :created_at, :updated_at, :user_id)
        """)
        await db.execute(service_stmt, {
          "id": service_id,
          "plugin_id": plugin_id,
          "plugin_slug": self.plugin_data["plugin_slug"],
          "name": service_data["name"],
          "source_url": service_data.get("source_url"),
          "type": service_data.get("type"),
          "install_command": service_data.get("install_command"),
          "start_command": service_data.get("start_command"),
          "stop_command": service_data.get("stop_command"),
          "restart_command": service_data.get("restart_command"),
          "runtime_dir_key": service_data.get("runtime_dir_key"),
          "env_inherit": service_data.get("env_inherit"),
          "env_overrides": json.dumps(service_data.get("env_overrides")) if service_data.get("env_overrides") is not None else None,
          "healthcheck_url": service_data.get("healthcheck_url"),
          "definition_id": service_data.get("definition_id"),
          "required_env_vars": json.dumps(service_data.get("required_env_vars", [])),
          "status": "pending",
          "created_at": now,
          "updated_at": now,
          "user_id": user_id
        })
        services_created.append(service_id)

      await db.commit()
      return {"success": True, "plugin_id": plugin_id, "modules_created": modules_created, "services_created": services_created}
    except Exception as error:  # pragma: no cover
      await db.rollback()
      logger.error("Failed to create RAG plugin records", error=str(error))
      return {"success": False, "error": str(error)}

  async def _refresh_required_env_vars_in_db(self, user_id: str, db: AsyncSession, env_map: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Keep plugin_service_runtime.required_env_vars in sync with current templates/fallbacks.
    """
    try:
      plugin_id = f"{user_id}_{self.plugin_data['plugin_slug']}"
      now = datetime.datetime.utcnow()
      for service in self.required_services_runtime:
        name = service.get("name")
        required_env_vars = env_map.get(name, service.get("required_env_vars", []))
        await db.execute(
          text("""
            UPDATE plugin_service_runtime
            SET required_env_vars = :required_env_vars, updated_at = :updated_at
            WHERE plugin_id = :plugin_id AND name = :name AND user_id = :user_id
          """),
          {
            "required_env_vars": json.dumps(required_env_vars),
            "updated_at": now,
            "plugin_id": plugin_id,
            "name": name,
            "user_id": user_id,
          },
        )
      await db.commit()
      return {"success": True}
    except Exception as error:  # pragma: no cover
      await db.rollback()
      logger.error("Failed to refresh required_env_vars", error=str(error))
      return {"success": False, "error": str(error)}

  async def _delete_database_records(self, user_id: str, plugin_id: str, db: AsyncSession) -> Dict[str, Any]:
    try:
      module_delete = text("""
        DELETE FROM module
        WHERE plugin_id = :plugin_id AND user_id = :user_id
      """)
      await db.execute(module_delete, {"plugin_id": plugin_id, "user_id": user_id})

      plugin_delete = text("""
        DELETE FROM plugin
        WHERE id = :plugin_id AND user_id = :user_id
      """)
      await db.execute(plugin_delete, {"plugin_id": plugin_id, "user_id": user_id})

      service_delete = text("""
        DELETE FROM plugin_service_runtime
        WHERE plugin_id = :plugin_id AND user_id = :user_id
      """)
      await db.execute(service_delete, {"plugin_id": plugin_id, "user_id": user_id})

      await db.commit()
      return {"success": True}
    except Exception as error:  # pragma: no cover
      await db.rollback()
      logger.error("Failed to delete RAG plugin records", error=str(error))
      return {"success": False, "error": str(error)}

  # ---------------------------------------------------------------------------
  # Settings
  # ---------------------------------------------------------------------------

  async def _load_settings_value(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
    """
    Fetch settings instance value for the user, returning dict or {}.
    """
    try:
      result = await db.execute(
        text("SELECT value FROM settings_instances WHERE definition_id = :definition_id AND user_id = :user_id"),
        {"definition_id": self.settings_definition_id, "user_id": user_id}
      )
      raw_val = result.scalar_one_or_none()
      if not raw_val:
        return {}
      raw_val = self._maybe_decrypt("settings_instances", "value", raw_val)
      if isinstance(raw_val, str):
        return json.loads(raw_val)
      return raw_val if isinstance(raw_val, dict) else {}
    except Exception:
      return {}

  def _serialize_plain(self, value: Any) -> Optional[str]:
    if value is None:
      return None
    if isinstance(value, (dict, list)):
      return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)

  def _maybe_encrypt(self, table: str, field: str, value: Any) -> Optional[str]:
    if not encryption_service:
      return self._serialize_plain(value)
    try:
      if encryption_service.should_encrypt_field(table, field):
        return encryption_service.encrypt_field(table, field, value)
    except Exception:
      # Fallback to plaintext if encryption is unavailable
      pass
    return self._serialize_plain(value)

  def _maybe_decrypt(self, table: str, field: str, value: Any) -> Any:
    if not encryption_service or value is None:
      return value
    try:
      if encryption_service.should_encrypt_field(table, field) and isinstance(value, str) and encryption_service.is_encrypted_value(value):
        return encryption_service.decrypt_field(table, field, value)
    except Exception:
      return value
    return value

  async def _ensure_settings(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
    try:
      definition_id = self.settings_definition_id
      default_settings_value = {
        "document_chat": _build_service_defaults("document_chat"),
        "document_processing": _build_service_defaults("document_processing"),
        "full_install": False,
      }

      definition = await db.execute(
        text("SELECT id FROM settings_definitions WHERE id = :definition_id"),
        {"definition_id": definition_id}
      )
      definition = definition.scalar_one_or_none()

      if not definition:
        definition_data = {
          "id": definition_id,
          "name": "BrainDrive RAG Services",
          "description": "Toggle Document Chat/Processing and choose docker or venv execution.",
          "category": "RAG",
          "type": "object",
          "default_value": json.dumps(default_settings_value),
          "allowed_scopes": json.dumps(["user"]),
          "validation": json.dumps({}),
          "is_multiple": False,
          "tags": json.dumps(["rag", "services", "venv", "docker"]),
          "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        definition_stmt = text("""
          INSERT INTO settings_definitions
          (id, name, description, category, type, default_value, allowed_scopes, validation, is_multiple, tags, created_at, updated_at)
          VALUES
          (:id, :name, :description, :category, :type, :default_value, :allowed_scopes, :validation, :is_multiple, :tags, :created_at, :updated_at)
        """)
        await db.execute(definition_stmt, definition_data)
      else:
        # Ensure existing definition has valid JSON default_value
        await db.execute(
          text("UPDATE settings_definitions SET default_value = :val, updated_at = :updated WHERE id = :definition_id"),
          {
            "val": json.dumps(default_settings_value),
            "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "definition_id": definition_id
          }
        )

      existing_instance_row = await db.execute(
        text("SELECT id, value FROM settings_instances WHERE definition_id = :definition_id AND user_id = :user_id"),
        {"definition_id": definition_id, "user_id": user_id}
      )
      row = existing_instance_row.fetchone()
      existing_instance = row[0] if row else None
      existing_value_raw = row[1] if row and len(row) > 1 else None
      existing_value_raw = self._maybe_decrypt("settings_instances", "value", existing_value_raw)
      try:
        if isinstance(existing_value_raw, str):
          existing_value = json.loads(existing_value_raw)
        elif isinstance(existing_value_raw, dict):
          existing_value = existing_value_raw
        else:
          existing_value = existing_value_raw or {}
      except Exception:
        existing_value = {}

      # Handle legacy flat fields -> new structure
      legacy_chat = {
        "enabled": existing_value.get("document_chat_enabled"),
        "mode": existing_value.get("document_chat_mode"),
        "port": existing_value.get("document_chat_port"),
      }
      legacy_proc = {
        "enabled": existing_value.get("document_processing_enabled"),
        "mode": existing_value.get("document_processing_mode"),
        "port": existing_value.get("document_processing_port"),
      }

      merged_chat = _deep_merge(default_settings_value["document_chat"], existing_value.get("document_chat", {}))
      merged_proc = _deep_merge(default_settings_value["document_processing"], existing_value.get("document_processing", {}))
      if legacy_chat["enabled"] is not None:
        merged_chat["enabled"] = bool(legacy_chat["enabled"])
      if legacy_chat["mode"]:
        merged_chat["mode"] = legacy_chat["mode"]
      if legacy_chat["port"]:
        merged_chat["port"] = legacy_chat["port"]

      if legacy_proc["enabled"] is not None:
        merged_proc["enabled"] = bool(legacy_proc["enabled"])
      if legacy_proc["mode"]:
        merged_proc["mode"] = legacy_proc["mode"]
      if legacy_proc["port"]:
        merged_proc["port"] = legacy_proc["port"]

      merged_value = {
        "document_chat": _normalize_service_settings(merged_chat, _coerce_int(merged_chat.get("port"), 18000)),
        "document_processing": _normalize_service_settings(merged_proc, _coerce_int(merged_proc.get("port"), 18080)),
        "full_install": bool(existing_value.get("full_install", False)),
      }

      self._refresh_required_services_runtime_from_settings(merged_value)
      instance_id = f"rag_services_settings_{user_id}"
      if not existing_instance:
        instance_data = {
          "id": instance_id,
          "name": "RAG Services Settings",
          "definition_id": definition_id,
          "scope": "user",
          "user_id": user_id,
          "value": self._maybe_encrypt("settings_instances", "value", merged_value),
          "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        instance_stmt = text("""
          INSERT INTO settings_instances
          (id, name, definition_id, scope, user_id, value, created_at, updated_at)
          VALUES
          (:id, :name, :definition_id, :scope, :user_id, :value, :created_at, :updated_at)
        """)
        await db.execute(instance_stmt, instance_data)
      else:
        # Normalize existing instance value to valid JSON
        await db.execute(
          text("UPDATE settings_instances SET value = :val, updated_at = :updated WHERE id = :id"),
          {
            "val": self._maybe_encrypt("settings_instances", "value", merged_value),
            "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": existing_instance
          }
        )

      await db.commit()
      return {"success": True, "definition_id": definition_id, "instance_id": instance_id, "value": merged_value}
    except Exception as error:  # pragma: no cover
      await db.rollback()
      logger.error("Failed to create RAG settings", error=str(error))
      return {"success": False, "error": str(error)}

  async def _delete_settings_instance(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
    """Remove the user-specific settings instance so uninstall succeeds cleanly."""
    try:
      instance_id = f"rag_services_settings_{user_id}"
      await db.execute(
        text("DELETE FROM settings_instances WHERE id = :id AND user_id = :user_id"),
        {"id": instance_id, "user_id": user_id},
      )
      await db.commit()
      return {"success": True, "deleted_instance": instance_id}
    except Exception as error:  # pragma: no cover
      await db.rollback()
      logger.error("Failed to delete RAG settings instance", error=str(error))
      return {"success": False, "error": str(error)}

  # ---------------------------------------------------------------------------
  # Pages (unused for this settings-only plugin)
  # ---------------------------------------------------------------------------

  async def _create_plugin_page(self, user_id: str, db: AsyncSession, modules_created: List[str]) -> Dict[str, Any]:
    return {"success": True, "page_id": None, "created": False}

  async def _delete_plugin_page(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
    return {"success": True, "deleted_rows": 0}

  # ---------------------------------------------------------------------------
  # Service orchestration helpers
  # ---------------------------------------------------------------------------

  async def _prepare_services(self, user_id: str) -> Dict[str, Any]:
    """Create venv + install dependencies for both services using service scripts."""
    skip = os.environ.get("RAG_SKIP_SERVICE_INSTALL", "").lower() in {"1", "true", "yes", "on"}
    full_install = os.environ.get("RAG_FULL_INSTALL", "").lower() in {"1", "true", "yes", "on"}
    force_recreate = os.environ.get("RAG_FORCE_VENV", "").lower() in {"1", "true", "yes", "on"}
    auto_start = os.environ.get("RAG_AUTO_START", "1").lower() in {"1", "true", "yes", "on"}
    async_env = os.environ.get("RAG_ASYNC_INSTALL")
    jobs_env = os.environ.get("RAG_USE_JOB_MANAGER")
    async_mode = (async_env or "1").lower() in {"1", "true", "yes", "on"}
    use_jobs = (jobs_env or "1").lower() in {"1", "true", "yes", "on"}
    if skip:
      return {"skipped": True, "reason": "RAG_SKIP_SERVICE_INSTALL env set"}

    installs: List[Dict[str, Any]] = []
    service_keys = ["document_chat", "document_processing"]

    async def _install_then_start(key: str) -> None:
      result = await prepare_service(key, full_install=full_install, force_recreate=force_recreate)
      if auto_start and result.get("success"):
        await start_service(key)

    if use_jobs and get_job_manager:
      try:
        jm = await get_job_manager()
        job, _ = await jm.enqueue_job(
          job_type="service.install",
          payload={
            "service_ops_path": str(self.shared_path / "service_ops.py"),
            "service_keys": service_keys,
            "full_install": full_install,
            "force_recreate": force_recreate,
            "auto_start": auto_start,
          },
          user_id=user_id,
          workspace_id=None,
          # Use a unique idempotency key per request to avoid reusing old completed jobs
          idempotency_key=f"rag_install_{user_id}_{self.plugin_data['version']}_{uuid.uuid4().hex}",
          max_retries=1,
        )
        return {"skipped": False, "installs": installs, "mode": "job", "job_id": job.id}
      except Exception as exc:
        logger.warning("Job manager not available, falling back to async tasks", error=str(exc))

    if async_mode:
      loop = asyncio.get_running_loop()
      for key in service_keys:
        loop.create_task(_install_then_start(key))
        installs.append({"service": key, "scheduled": True})
      return {"skipped": False, "installs": installs, "mode": "async", "auto_start": auto_start}
    else:
      for key in service_keys:
        install_result = await prepare_service(key, full_install=full_install, force_recreate=force_recreate)
        installs.append({"service": key, **install_result})
        if auto_start and install_result.get("success"):
          await start_service(key)
      return {"skipped": False, "installs": installs, "mode": "sync", "auto_start": auto_start}

  async def _shutdown_services(self) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for key in ("document_chat", "document_processing"):
      shut_result = await shutdown_service(key)
      results.append({"service": key, **shut_result})
    return {"results": results}

  async def _restart_services(self, settings_value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if settings_value:
      await self.apply_settings_env(settings_value, restart=False)
    results: List[Dict[str, Any]] = []
    for key in ("document_chat", "document_processing"):
      restart_result = await restart_service(key)
      results.append({"service": key, **restart_result})
    return {"results": results}

  async def _start_services(self, settings_value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if settings_value:
      await self.apply_settings_env(settings_value, restart=False)
    results: List[Dict[str, Any]] = []
    for key in ("document_chat", "document_processing"):
      enabled = True
      if settings_value and isinstance(settings_value.get(key), dict):
        enabled = bool(settings_value.get(key, {}).get("enabled", True))
      if not enabled:
        results.append({"service": key, "skipped": True, "reason": "disabled"})
        continue
      start_result = await start_service(key)
      results.append({"service": key, **start_result})
    return {"results": results}

  async def _collect_service_health(self, settings_value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if settings_value:
      self._refresh_required_services_runtime_from_settings(settings_value)
    health: List[Dict[str, Any]] = []
    for key in ("document_chat", "document_processing"):
      svc_settings = (settings_value or {}).get(key, {})
      url_override = svc_settings.get("health_url") if isinstance(svc_settings, dict) else None
      status = await health_check(key, url_override)
      health.append({"service": key, **status})
    return {"results": health}

  # ---------------------------------------------------------------------------
  # Env materialization + restart (venv)
  # ---------------------------------------------------------------------------

  async def apply_env_updates(self, env_payload: Dict[str, Dict[str, str]], restart: bool = True) -> Dict[str, Any]:
    """
    Update .env files for services using provided values (keyed by service) and optionally restart.
    Intended to be invoked by settings/UI flows.
    """
    env_map = get_required_env_vars_map()
    self._apply_required_env_vars(env_map)

    results: List[Dict[str, Any]] = []
    for service_key, values in env_payload.items():
      allowed = env_map.get(service_key, [])
      update_result = await update_env_and_restart(service_key, values, allowed_keys=allowed, restart=restart)
      results.append({"service": service_key, **update_result})

    success = all(item.get("success", False) for item in results) if results else True
    return {"success": success, "results": results}

  async def materialize_env_without_restart(self, env_payload: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Write .env files without restarting services (e.g., batch updates).
    """
    env_map = get_required_env_vars_map()
    self._apply_required_env_vars(env_map)

    results: List[Dict[str, Any]] = []
    for service_key, values in env_payload.items():
      allowed = env_map.get(service_key, [])
      write_result = await materialize_env_file(service_key, values, allowed_keys=allowed, backup=True)
      results.append({"service": service_key, **write_result})

    success = all(item.get("changed", True) is not False for item in results) if results else True
    return {"success": success, "results": results}

  async def apply_settings_env(self, settings_value: Dict[str, Any], restart: bool = True) -> Dict[str, Any]:
    """
    Convenience: build env payload from settings_value and apply to services.
    """
    env_payload = self._build_env_payload_from_settings(settings_value)
    return await self.apply_env_updates(env_payload, restart=restart)

  async def control_service_with_settings(self, action: str, service_key: str, settings_value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Control a single service (start/stop/restart) optionally ensuring env is applied first.
    """
    if settings_value:
      await self.apply_settings_env(settings_value, restart=False)
    if action == "start":
      return await start_service(service_key)
    if action == "stop":
      return await shutdown_service(service_key)
    if action == "restart":
      return await restart_service(service_key)
    return {"success": False, "error": f"Unsupported action: {action}"}


__all__ = ["BrainDriveRAGCommunityLifecycleManager"]
