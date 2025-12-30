import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib import error, request

def _is_windows_store_stub(path: str) -> bool:
  if os.name != "nt":
    return False
  return "windowsapps" in path.lower()


def _resolve_python_bin() -> str:
  candidates = [
    os.environ.get("RAG_PYTHON_BIN", ""),
    sys.executable,
    "python3.11",
    "python3",
    "python",
  ]
  seen: set[str] = set()
  for candidate in candidates:
    if not candidate:
      continue
    candidate_path = Path(candidate)
    resolved = str(candidate_path) if candidate_path.is_file() else shutil.which(str(candidate))
    if not resolved or resolved in seen:
      continue
    seen.add(resolved)
    if _is_windows_store_stub(resolved):
      continue
    return resolved
  return sys.executable or "python"


DEFAULT_PYTHON_BIN = _resolve_python_bin()
SERVICES_RUNTIME_ENV_VAR = "BRAINDRIVE_SERVICES_RUNTIME_DIR"


def resolve_services_runtime_dir(
  env: Optional[Mapping[str, str]] = None,
  current_file: Optional[Path] = None,
) -> Path:
  """
  Resolve the BrainDrive services runtime directory in a portable way.

  Priority:
  1) BRAINDRIVE_SERVICES_RUNTIME_DIR env override (path to backend/services_runtime)
  2) If running from backend/plugins/... find backend ancestor and use backend/services_runtime
  3) If running from source tree, find a parent containing backend/ and use backend/services_runtime
  4) Fallback to <cwd>/backend/services_runtime if it already exists
  """
  env_map = env or os.environ
  override = str(env_map.get(SERVICES_RUNTIME_ENV_VAR, "")).strip()
  if override:
    return Path(override).expanduser().resolve()

  current_file = current_file or Path(__file__).resolve()

  # Installed/shared mode: .../backend/plugins/.../service_ops.py
  for parent in [current_file.parent, *current_file.parents]:
    if parent.name == "backend":
      return (parent / "services_runtime").resolve()

  # Dev/source mode: <repo_root>/PluginBuild/.../service_ops.py
  for parent in [current_file.parent, *current_file.parents]:
    backend_dir = parent / "backend"
    if backend_dir.is_dir():
      return (backend_dir / "services_runtime").resolve()

  # Running inside BrainDrive (common during plugin validation): use CWD or its parents.
  cwd = Path.cwd().resolve()
  for parent in [cwd, *cwd.parents]:
    if parent.name == "backend":
      return (parent / "services_runtime").resolve()
    backend_dir = parent / "backend"
    if backend_dir.is_dir():
      return (backend_dir / "services_runtime").resolve()

  # Additional fallback: resolve via sys.path entries (covers cases where CWD is not near the repo).
  for entry in sys.path:
    if not entry:
      continue
    candidate = Path(entry).expanduser()
    if candidate.name == "backend":
      return (candidate / "services_runtime").resolve()
    backend_dir = candidate / "backend"
    if backend_dir.is_dir():
      return (backend_dir / "services_runtime").resolve()

  raise RuntimeError(
    "Unable to resolve BrainDrive services_runtime directory. "
    f"Set {SERVICES_RUNTIME_ENV_VAR} to <BrainDriveRoot>/backend/services_runtime."
  )


BASE_RUNTIME_DIR = resolve_services_runtime_dir()

# Fallback env lists ensure backward compatibility with the legacy Chat-With-Docs plugin.
DOC_PROCESSING_FALLBACK_ENV_VARS: List[str] = [
  # Auth + app
  "DISABLE_AUTH",
  "AUTH_METHOD",
  "AUTH_API_KEY",
  "JWT_SECRET",
  "JWT_ALGORITHM",
  "JWT_EXPIRE_MINUTES",
  "DEBUG",
  "LOG_LEVEL",
  "API_HOST",
  "API_PORT",
  # Processing
  "SPACY_MODEL",
  "DEFAULT_CHUNKING_STRATEGY",
  "DEFAULT_CHUNK_SIZE",
  "DEFAULT_CHUNK_OVERLAP",
  "MIN_CHUNK_SIZE",
  "MAX_CHUNK_SIZE",
  "UPLOADS_DIR",
  "TEMP_DIR",
  "DATABASE_URL",
  # Logging
  "LOG_FORMAT",
  "LOG_FILE",
]

DOC_CHAT_FALLBACK_ENV_VARS: List[str] = [
  # Providers
  "LLM_PROVIDER",
  "EMBEDDING_PROVIDER",
  # Ollama
  "OLLAMA_LLM_BASE_URL",
  "OLLAMA_LLM_MODEL",
  "OLLAMA_EMBEDDING_BASE_URL",
  "OLLAMA_EMBEDDING_MODEL",
  "EMBEDDING_BATCH_SIZE",
  "EMBEDDING_CONCURRENCY",
  "EMBEDDING_TIMEOUT",
  "EMBEDDING_MAX_RETRIES",
  "EMBEDDING_RETRY_DELAY",
  # Contextual retrieval
  "ENABLE_CONTEXTUAL_RETRIEVAL",
  "OLLAMA_CONTEXTUAL_LLM_BASE_URL",
  "OLLAMA_CONTEXTUAL_LLM_MODEL",
  "CONTEXTUAL_BATCH_SIZE",
  "CONTEXTUAL_CHUNK_TIMEOUT",
  "CONTEXTUAL_DOC_MAX_LENGTH",
  # Evaluation / OpenAI
  "OPENAI_EVALUATION_API_KEY",
  "OPENAI_EVALUATION_MODEL",
  # Service URLs (common integrations)
  "DOCUMENT_PROCESSOR_API_URL",
  "DOCUMENT_PROCESSOR_TIMEOUT",
  "DOCUMENT_PROCESSOR_MAX_RETRIES",
]


@dataclass
class ServiceConfig:
  key: str
  label: str
  repo_path: Path
  repo_url: str
  venv_path: Path
  health_url: str
  scripts_dir: Path
  docker_compose: Optional[Path]
  default_mode: str = "venv"
  use_full_install: bool = False


SERVICE_CONFIG: Dict[str, ServiceConfig] = {
  "document_chat": ServiceConfig(
    key="document_chat",
    label="Document Chat Service",
    repo_path=BASE_RUNTIME_DIR / "Document-Chat-Service",
    repo_url="https://github.com/DJJones66/Document-Chat-Service",
    venv_path=BASE_RUNTIME_DIR / "Document-Chat-Service/.venv",
    health_url="http://localhost:18000/health",
    scripts_dir=BASE_RUNTIME_DIR / "Document-Chat-Service/service_scripts",
    docker_compose=BASE_RUNTIME_DIR / "Document-Chat-Service/docker-compose.yml",
    default_mode="venv",
    use_full_install=True
  ),
  "document_processing": ServiceConfig(
    key="document_processing",
    label="Document Processing Service",
    repo_path=BASE_RUNTIME_DIR / "Document-Processing-Service",
    repo_url="https://github.com/DJJones66/Document-Processing-Service",
    venv_path=BASE_RUNTIME_DIR / "Document-Processing-Service/.venv",
    health_url="http://localhost:18080/health",
    scripts_dir=BASE_RUNTIME_DIR / "Document-Processing-Service/service_scripts",
    docker_compose=BASE_RUNTIME_DIR / "Document-Processing-Service/docker-compose.yml",
    default_mode="venv"
  )
}


def _parse_env_lines(env_text: str) -> Dict[str, str]:
  """Parse simple KEY=VALUE lines, ignoring comments/blank lines."""
  values: Dict[str, str] = {}
  for raw_line in env_text.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, val = line.split("=", 1)
    key = key.strip()
    if not key:
      continue
    values[key] = val.strip()
  return values


def _parse_env_template(template_path: Path) -> Tuple[List[str], Dict[str, str]]:
  """Return (keys, defaults) from a template .env* file."""
  if not template_path.exists():
    return [], {}
  text = template_path.read_text()
  values = _parse_env_lines(text)
  return list(values.keys()), values


def _template_candidates(service: ServiceConfig) -> List[Path]:
  """Candidate template files used to derive required env vars."""
  if service.key == "document_chat":
    return [
      service.repo_path / ".env.local.example",
      service.repo_path / ".env.example",
      service.repo_path / ".env.local",
    ]
  return [
    service.repo_path / ".env.local.example",
    service.repo_path / ".env.local",
  ]


def get_required_env_vars(service_key: str) -> List[str]:
  """Derive required_env_vars from templates with backward-compatible fallbacks."""
  service = SERVICE_CONFIG[service_key]
  keys: List[str] = []
  for candidate in _template_candidates(service):
    parsed_keys, _ = _parse_env_template(candidate)
    keys.extend(parsed_keys)
  if service_key == "document_chat":
    keys.extend(DOC_CHAT_FALLBACK_ENV_VARS)
  elif service_key == "document_processing":
    keys.extend(DOC_PROCESSING_FALLBACK_ENV_VARS)
  # Deduplicate while preserving order
  seen = set()
  deduped: List[str] = []
  for key in keys:
    if key in seen or not key:
      continue
    seen.add(key)
    deduped.append(key)
  return deduped


def get_required_env_vars_map() -> Dict[str, List[str]]:
  """Convenience helper to return required_env_vars for all services."""
  return {key: get_required_env_vars(key) for key in SERVICE_CONFIG}


def update_health_urls(overrides: Dict[str, str]) -> None:
  """Override service health URLs in-memory (used by lifecycle settings)."""
  for key, url in (overrides or {}).items():
    service = SERVICE_CONFIG.get(key)
    if service and url:
      service.health_url = url


def _resolve_log_file(cwd: Optional[Path], script: Optional[Path]) -> Path:
  if cwd and cwd.exists():
    return cwd / "service_runtime.log"
  if script and script.parent.exists():
    return script.parent / "service_runtime.log"
  return BASE_RUNTIME_DIR / "service_runtime.log"


def _append_log(log_file: Path, message: str) -> None:
  try:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
      fh.write(message)
  except Exception:
    pass


async def _run_python(script: Path, env: Optional[Dict[str, str]] = None, args: Optional[List[str]] = None, cwd: Optional[Path] = None) -> Dict[str, Any]:
  python_bin = DEFAULT_PYTHON_BIN or _resolve_python_bin()
  cmd = [python_bin, str(script)]
  if args:
    cmd.extend(args)
  log_file = _resolve_log_file(cwd, script)
  workdir = cwd or script.parent
  log_handle = None

  try:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("ab")
  except Exception:
    log_handle = None

  def _write_log(data: bytes) -> None:
    if not log_handle:
      return
    try:
      log_handle.write(data)
      log_handle.flush()
    except Exception:
      pass

  _write_log(f"\n[{cmd}] starting\n".encode("utf-8", errors="replace"))

  try:
    proc = await asyncio.create_subprocess_exec(
      *cmd,
      cwd=str(workdir),
      env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"},
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
  except Exception as exc:
    _write_log(f"\n[{cmd}] spawn failed: {exc}\n".encode("utf-8", errors="replace"))
    if log_handle:
      log_handle.close()
    return {
      "success": False,
      "code": None,
      "stdout": "",
      "stderr": str(exc),
      "cmd": " ".join(cmd),
      "log_file": str(log_file),
    }

  async def _drain(stream: Optional[asyncio.StreamReader], label: Optional[str]) -> bytes:
    if not stream:
      return b""
    chunks: List[bytes] = []
    wrote_label = False
    while True:
      data = await stream.read(4096)
      if not data:
        break
      if label and not wrote_label:
        _write_log(b"\n-- stderr --\n")
        wrote_label = True
      _write_log(data)
      chunks.append(data)
    return b"".join(chunks)

  stdout_task = asyncio.create_task(_drain(proc.stdout, None))
  stderr_task = asyncio.create_task(_drain(proc.stderr, "stderr"))
  rc = await proc.wait()
  stdout = await stdout_task
  stderr = await stderr_task

  _write_log(f"\n[{cmd}] rc={rc}\n".encode("utf-8", errors="replace"))
  if log_handle:
    try:
      log_handle.close()
    except Exception:
      pass

  return {
    "success": rc == 0,
    "code": rc,
    "stdout": stdout.decode(errors="replace"),
    "stderr": stderr.decode(errors="replace"),
    "cmd": " ".join(cmd),
    "log_file": str(log_file),
  }


def _sanitize_int_env(value: Optional[str]) -> Optional[str]:
  if not value:
    return None
  cleaned = value.split("#", 1)[0].strip()
  if not cleaned:
    return None
  try:
    return str(int(cleaned))
  except Exception:
    return None


def _needs_windows_home_env() -> bool:
  if os.name != "nt":
    return False
  home = os.environ.get("HOME", "").strip()
  userprofile = os.environ.get("USERPROFILE", "").strip()
  homedrive = os.environ.get("HOMEDRIVE", "").strip()
  homepath = os.environ.get("HOMEPATH", "").strip()
  return not (home or userprofile or (homedrive and homepath))


def _service_env(service: ServiceConfig) -> Dict[str, str]:
  """Base environment overrides for service scripts."""
  env = {
    # Force async driver even if host environment sets DATABASE_URL differently
    "DATABASE_URL": "sqlite+aiosqlite:///./data/app.db",
    "PYTHON_BIN": DEFAULT_PYTHON_BIN,
  }
  max_file_size = _sanitize_int_env(os.environ.get("MAX_FILE_SIZE"))
  if max_file_size:
    env["MAX_FILE_SIZE"] = max_file_size
  if _needs_windows_home_env():
    home_dir = service.repo_path / "data" / "home"
    try:
      home_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
      pass
    env["HOME"] = str(home_dir)
    env["USERPROFILE"] = str(home_dir)
  return env


async def _start_background(script: Path, env: Optional[Dict[str, str]] = None, args: Optional[List[str]] = None, cwd: Optional[Path] = None) -> Dict[str, Any]:
  """Start a long-running service script without waiting for completion."""
  python_bin = DEFAULT_PYTHON_BIN or _resolve_python_bin()
  cmd = [python_bin, str(script)]
  if args:
    cmd.extend(args)
  env = {**os.environ, **(env or {})}
  env.setdefault("PYTHONUNBUFFERED", "1")
  env.setdefault("PYTHONIOENCODING", "utf-8")
  workdir = cwd or script.parent
  log_file = _resolve_log_file(workdir, script)
  log_file.parent.mkdir(parents=True, exist_ok=True)

  popen_kwargs: Dict[str, Any] = {}
  if os.name != "nt":
    popen_kwargs["start_new_session"] = True
  elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

  log_handle = log_file.open("ab")
  try:
    proc = subprocess.Popen(
      cmd,
      cwd=str(workdir),
      env=env,
      stdin=subprocess.DEVNULL,
      stdout=log_handle,
      stderr=log_handle,
      **popen_kwargs,
    )
  except Exception as exc:
    log_handle.close()
    return {"success": False, "error": str(exc), "cmd": " ".join(cmd), "log_file": str(log_file)}
  finally:
    try:
      log_handle.close()
    except Exception:
      pass

  await asyncio.sleep(1)
  rc = proc.poll()
  return {
    "success": rc is None or rc == 0,
    "code": rc if rc is not None else 0,
    "pid": proc.pid,
    "cmd": " ".join(cmd),
    "log_file": str(log_file),
  }


async def ensure_venv(service_key: str, force_recreate: bool = False) -> Dict[str, Any]:
  service = SERVICE_CONFIG[service_key]
  await _ensure_repo(service)
  _ensure_env_file(service)
  env = _service_env(service)
  if force_recreate:
    env["VENV_FORCE_RECREATE"] = "1"
  script = service.scripts_dir / "create_venv.py"
  return await _run_python(script, env=env, cwd=service.repo_path)


async def install_venv(service_key: str, full: bool = False, upgrade_pip: bool = True) -> Dict[str, Any]:
  service = SERVICE_CONFIG[service_key]
  await _ensure_repo(service)
  _ensure_env_file(service)
  env = _service_env(service)
  if full or service.use_full_install:
    env["FULL_INSTALL"] = "1"
  if not upgrade_pip:
    env["UPGRADE_PIP"] = "0"

  script = service.scripts_dir / "install_with_venv.py"
  return await _run_python(script, env=env, cwd=service.repo_path)


async def restart_service(service_key: str) -> Dict[str, Any]:
  service = SERVICE_CONFIG[service_key]
  await _ensure_repo(service)
  # Do a graceful shutdown then start again without blocking on the long-running process
  await shutdown_service(service_key)
  return await start_service(service_key)


async def start_service(service_key: str) -> Dict[str, Any]:
  service = SERVICE_CONFIG[service_key]
  # Avoid port conflicts: if it's already healthy, treat as started.
  try:
    healthy = await health_check(service_key, timeout=2)
    if healthy.get("success"):
      return {"success": True, "skipped": True, "reason": "already healthy", "health": healthy}
  except Exception:
    pass

  await _ensure_repo(service)
  # If venv is missing, create it (install step is still separate).
  if not service.venv_path.exists():
    venv_result = await ensure_venv(service_key, force_recreate=False)
    if not venv_result.get("success"):
      return {"success": False, "step": "create_venv", **venv_result}
  script = service.scripts_dir / "start_with_venv.py"
  start_result = await _start_background(script, env=_service_env(service), cwd=service.repo_path)
  pid = start_result.get("pid")
  if pid:
    pidfile = service.repo_path / "data" / "service.pid"
    try:
      pidfile.parent.mkdir(parents=True, exist_ok=True)
      pidfile.write_text(str(pid))
      start_result["pidfile"] = str(pidfile)
    except Exception:
      pass
  return start_result


async def shutdown_service(service_key: str) -> Dict[str, Any]:
  service = SERVICE_CONFIG[service_key]
  await _ensure_repo(service)
  script = service.scripts_dir / "shutdown_with_venv.py"
  return await _run_python(script, env=_service_env(service), cwd=service.repo_path)


async def health_check(service_key: str, override_url: Optional[str] = None, timeout: int = 6) -> Dict[str, Any]:
  service = SERVICE_CONFIG[service_key]
  url = override_url or service.health_url

  def _check():
    try:
      with request.urlopen(url, timeout=timeout) as resp:
        body_bytes = resp.read()
        body = body_bytes.decode() if body_bytes else ""
        return {"success": resp.status == 200, "status": resp.status, "body": body}
    except error.HTTPError as http_err:
      return {"success": False, "status": http_err.code, "error": str(http_err)}
    except Exception as exc:  # pragma: no cover - runtime safety
      return {"success": False, "error": str(exc)}

  return await asyncio.to_thread(_check)


async def prepare_service(service_key: str, full_install: bool = False, force_recreate: bool = False) -> Dict[str, Any]:
  steps: List[Dict[str, Any]] = []

  create_result = await ensure_venv(service_key, force_recreate=force_recreate)
  steps.append({"step": "create_venv", **create_result})
  if not create_result.get("success"):
    return {"success": False, "steps": steps}

  install_result = await install_venv(service_key, full=full_install)
  steps.append({"step": "install", **install_result})
  if not install_result.get("success"):
    return {"success": False, "steps": steps}

  return {"success": True, "steps": steps}


def get_service_metadata() -> List[Dict[str, Any]]:
  items: List[Dict[str, Any]] = []
  for service in SERVICE_CONFIG.values():
    items.append({
      "key": service.key,
      "label": service.label,
      "health_url": service.health_url,
      "default_mode": service.default_mode,
      "default_venv_path": str(service.venv_path),
      "docker_compose": str(service.docker_compose) if service.docker_compose else None,
      "repo_path": str(service.repo_path),
      "repo_url": service.repo_url
    })
  return items


async def _ensure_repo(service: ServiceConfig) -> None:
  """Clone the service repo into the backend runtime dir if missing."""
  if service.repo_path.exists():
    return
  service.repo_path.parent.mkdir(parents=True, exist_ok=True)
  if not service.repo_url:
    raise RuntimeError(f"No repo_url defined for service {service.key}")
  cmd = ["git", "clone", service.repo_url, str(service.repo_path)]
  log_file = _resolve_log_file(service.repo_path.parent, None)
  try:
    proc = await asyncio.create_subprocess_exec(
      *cmd,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
  except Exception as exc:
    _append_log(log_file, f"\n[{cmd}] clone spawn failed: {exc}\n")
    raise RuntimeError(f"Failed to clone {service.repo_url}: {exc}") from exc
  if proc.returncode != 0:
    _append_log(log_file, f"\n[{cmd}] rc={proc.returncode}\n{stderr.decode(errors='replace')}\n")
    raise RuntimeError(f"Failed to clone {service.repo_url}: {stderr.decode(errors='replace')}")


def _ensure_env_file(service: ServiceConfig) -> None:
  """Ensure a default .env exists by copying the template/local file."""
  env_path = service.repo_path / ".env"
  if env_path.exists():
    return
  if service.key == "document_chat":
    src = service.repo_path / ".env.local.example"
    if not src.exists():
      src = service.repo_path / ".env.example"
  else:
    # Prefer the new example file; fallback to legacy name if needed
    src = service.repo_path / ".env.local.example"
    if not src.exists():
      src = service.repo_path / ".env.local"
  if src.exists():
    env_text = src.read_text()
    lines = env_text.splitlines()
    # Ensure DATABASE_URL is set to async sqlite
    db_url = 'DATABASE_URL="sqlite+aiosqlite:///./data/app.db"'
    if not any(line.strip().startswith("DATABASE_URL") and "sqlite+aiosqlite" in line for line in lines):
      lines.append(db_url)
    env_path.write_text("\n".join(lines))


def _treat_empty_as_unset(key: str) -> bool:
  """
  Certain env vars should never be written as empty strings because that can
  override useful template defaults and break service startup.
  """
  return key.endswith("_MODEL") or key in {"JWT_EXPIRE_MINUTES", "JWT_SECRET", "AUTH_API_KEY"}


def _render_env_file(existing_text: str, updates: Dict[str, str], allowed_keys: List[str], defaults: Dict[str, str]) -> str:
  """
  Render env content, updating only allowed_keys while preserving other lines/comments.
  """
  existing_values = _parse_env_lines(existing_text)
  seen: set[str] = set()

  def resolve_value(key: str) -> str:
    if key in updates:
      candidate = str(updates.get(key, ""))
      if not (_treat_empty_as_unset(key) and candidate.strip() == ""):
        return candidate

    existing = existing_values.get(key)
    if existing is not None:
      existing_str = str(existing)
      if not (_treat_empty_as_unset(key) and existing_str.strip() == ""):
        return existing_str

    return str(defaults.get(key, ""))

  new_lines: List[str] = []
  for raw_line in existing_text.splitlines():
    line = raw_line.rstrip("\n")
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      new_lines.append(line)
      continue
    key = stripped.split("=", 1)[0].strip()
    if key in allowed_keys:
      value = resolve_value(key)
      new_lines.append(f"{key}={value}")
      seen.add(key)
    else:
      new_lines.append(line)
  # Append any missing allowed keys
  for key in allowed_keys:
    if key in seen:
      continue
    value = resolve_value(key)
    new_lines.append(f"{key}={value}")
  return "\n".join(new_lines).rstrip() + "\n"


async def materialize_env_file(
  service_key: str,
  values: Dict[str, str],
  allowed_keys: Optional[List[str]] = None,
  backup: bool = True,
) -> Dict[str, Any]:
  """
  Write a .env file for the service using provided values for allowed_keys.
  Preserves non-managed lines/comments and creates a backup if content changes.
  """
  service = SERVICE_CONFIG[service_key]
  await _ensure_repo(service)
  _ensure_env_file(service)

  env_path = service.repo_path / ".env"
  allowed = allowed_keys or get_required_env_vars(service_key)

  defaults: Dict[str, str] = {}
  for candidate in _template_candidates(service):
    _, parsed_defaults = _parse_env_template(candidate)
    defaults.update(parsed_defaults)
  fallback_defaults: Dict[str, Dict[str, str]] = {
    "document_processing": {
      "JWT_EXPIRE_MINUTES": "60",
      "JWT_ALGORITHM": "HS256",
      "AUTH_METHOD": "api_key",
    }
  }
  for k, v in fallback_defaults.get(service_key, {}).items():
    defaults.setdefault(k, v)

  existing_text = env_path.read_text() if env_path.exists() else ""
  rendered = _render_env_file(existing_text, values, allowed, defaults)

  if existing_text == rendered:
    return {"changed": False, "env_path": str(env_path)}

  backup_path = None
  if backup and env_path.exists():
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_path = env_path.with_suffix(env_path.suffix + f".bak.{timestamp}")
    backup_path.write_text(existing_text)

  env_path.write_text(rendered)
  return {"changed": True, "env_path": str(env_path), "backup": str(backup_path) if backup_path else None}


async def update_env_and_restart(
  service_key: str,
  values: Dict[str, str],
  allowed_keys: Optional[List[str]] = None,
  restart: bool = True,
) -> Dict[str, Any]:
  """
  Materialize env file for the service, then restart (or skip if unchanged).
  """
  write_result = await materialize_env_file(service_key, values, allowed_keys=allowed_keys, backup=True)
  if not restart and not write_result.get("changed"):
    return {"success": True, "changed": False, **write_result}

  if not restart:
    return {"success": True, "changed": write_result.get("changed", False), **write_result}

  restart_result = await restart_service(service_key)
  return {
    **write_result,
    "restart": restart_result,
    "success": restart_result.get("success", False),
  }


__all__ = [
  "ensure_venv",
  "install_venv",
  "prepare_service",
  "start_service",
  "shutdown_service",
  "restart_service",
  "health_check",
  "get_service_metadata",
  "get_required_env_vars",
  "get_required_env_vars_map",
  "update_health_urls",
  "materialize_env_file",
  "update_env_and_restart",
]
