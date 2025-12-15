import React from "react";

type ServiceKey = "documentChat" | "documentProcessing";

type ServiceConfig = {
  key: ServiceKey;
  label: string;
  description: string;
  defaultPort: number;
  healthPath: string;
  defaultMode: "venv" | "docker";
};

type ToggleState = {
  status: "unknown" | "running" | "stopped" | "error";
  lastMessage?: string;
};

type ApiService = {
  get?: (url: string, options?: unknown) => Promise<{ status?: number; data?: unknown }>;
  post?: (url: string, body?: unknown, options?: unknown) => Promise<{ status?: number; data?: unknown }>;
};

type ThemeService = {
  getCurrentTheme: () => string;
  addThemeChangeListener: (callback: (theme: string) => void) => void;
  removeThemeChangeListener: (callback: (theme: string) => void) => void;
};

type PanelProps = {
  title?: string;
  subtitle?: string;
  initialState?: Partial<Record<ServiceKey, ToggleState>>;
  services?: {
    api?: ApiService;
    theme?: ThemeService;
  };
};

type PanelState = {
  toggles: Record<ServiceKey, ToggleState>;
  healthMessages: Record<ServiceKey, string>;
  currentTheme: string;
};

type Palette = {
  surface: string;
  surfaceBorder: string;
  card: string;
  cardBorder: string;
  text: string;
  muted: string;
  pillBg: string;
  pillBorder: string;
  pillText: string;
  activeBorder: string;
  activeBg: string;
  activeText: string;
  enabledBg: string;
  enabledBorder: string;
  enabledText: string;
  disabledBg: string;
  disabledBorder: string;
  disabledText: string;
  statusOk: string;
  statusWarn: string;
  statusErr: string;
  shadow: string;
};

const SERVICE_CONFIG: ServiceConfig[] = [
  {
    key: "documentChat",
    label: "Document Chat Service",
    description: "User-facing chat API serving retrieved answers.",
    defaultPort: 18000,
    healthPath: "/health",
    defaultMode: "venv"
  },
  {
    key: "documentProcessing",
    label: "Document Processing Service",
    description: "Ingests and chunks documents for retrieval.",
    defaultPort: 18080,
    healthPath: "/health",
    defaultMode: "venv"
  }
];

function getPalette(theme: string): Palette {
  const dark = (theme || "").toLowerCase().includes("dark");
  if (dark) {
    return {
      surface: "#0f172a",
      surfaceBorder: "#16213a",
      card: "#1E293B",          // align with Settings card background
      cardBorder: "#25374f",    // subtle border similar to settings card
      text: "#e5e7eb",
      muted: "#9ba3b4",
      pillBg: "#152037",
      pillBorder: "#22314a",
      pillText: "#e5e7eb",
      activeBorder: "#3b82f6",
      activeBg: "#12274a",
      activeText: "#e0e7ff",
      enabledBg: "#0f3a2e",
      enabledBorder: "#22c55e",
      enabledText: "#bbf7d0",
      disabledBg: "#402c0f",
      disabledBorder: "#f59e0b",
      disabledText: "#fcd34d",
      statusOk: "#22c55e",
      statusWarn: "#f59e0b",
      statusErr: "#ef4444",
      shadow: "0 8px 28px rgba(0,0,0,0.32)"
    };
  }
  return {
    surface: "linear-gradient(135deg, #f8fafc 0%, #ffffff 60%)",
    surfaceBorder: "#e5e7eb",
    card: "#ffffff",
    cardBorder: "#e2e8f0",
    text: "#111827",
    muted: "#4b5563",
    pillBg: "#eef2ff",
    pillBorder: "#d0d7ff",
    pillText: "#283048",
    activeBorder: "#2563eb",
    activeBg: "#eff6ff",
    activeText: "#1d4ed8",
    enabledBg: "#e6fffa",
    enabledBorder: "#14b8a6",
    enabledText: "#0f766e",
    disabledBg: "#fff7ed",
    disabledBorder: "#f59e0b",
    disabledText: "#c2410c",
    statusOk: "#16a34a",
    statusWarn: "#d97706",
    statusErr: "#dc2626",
    shadow: "0 6px 24px rgba(0,0,0,0.08)"
  };
}

function pillStyle(palette: Palette): React.CSSProperties {
  return {
    display: "inline-flex",
    alignItems: "center",
    padding: "4px 8px",
    borderRadius: "12px",
    fontSize: "12px",
    fontWeight: 600,
    background: palette.pillBg,
    color: palette.pillText,
    border: `1px solid ${palette.pillBorder}`,
    textTransform: "uppercase",
    letterSpacing: "0.4px"
  };
}

function cardStyle(palette: Palette): React.CSSProperties {
  return {
    border: `1px solid ${palette.cardBorder}`,
    borderRadius: "12px",
    padding: "14px 16px",
    background: palette.card,
    boxShadow: palette.shadow,
    display: "grid",
    gap: "12px"
  };
}

function ServiceCard({
  config,
  state,
  onToggle,
  onStart,
  onStop,
  onRestart,
  onHealthCheck,
  palette
}: {
  config: ServiceConfig;
  state: ToggleState;
  onToggle: (next: boolean) => void;
  onStart: () => void;
  onStop: () => void;
  onRestart: () => void;
  onHealthCheck: () => void;
  palette: Palette;
}) {
  const statusColor =
    state.status === "running"
      ? palette.statusOk
      : state.status === "stopped"
        ? palette.statusWarn
        : state.status === "error"
          ? palette.statusErr
          : palette.muted;

  return (
    <div style={cardStyle(palette)}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: palette.text }}>{config.label}</span>
            <span style={{
              ...pillStyle(palette),
              background: state.status === "running" ? palette.enabledBg : state.status === "error" ? palette.disabledBg : palette.disabledBg,
              borderColor: statusColor,
              color: statusColor
            }}>
              {state.status === "running" ? "Running" : state.status === "stopped" ? "Stopped" : state.status === "error" ? "Error" : "Unknown"}
            </span>
          </div>
          <p style={{ margin: "6px 0 0 0", color: palette.muted }}>
            {state.lastMessage || config.description}
          </p>
        </div>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
        <button type="button" onClick={onStart} style={buttonStyle(false, palette)}>
          Start
        </button>
        <button type="button" onClick={onStop} style={buttonStyle(false, palette)}>
          Stop
        </button>
        <button type="button" onClick={onRestart} style={buttonStyle(false, palette)}>
          Restart
        </button>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ color: palette.muted, fontSize: 12 }}>Default port: {config.defaultPort}</span>
          <button type="button" onClick={onHealthCheck} style={buttonStyle(false, palette)}>
            Check status
          </button>
        </div>
      </div>
    </div>
  );
}

function buttonStyle(active: boolean, palette: Palette): React.CSSProperties {
  return {
    padding: "8px 12px",
    borderRadius: "10px",
    border: active ? `1px solid ${palette.activeBorder}` : `1px solid ${palette.cardBorder}`,
    background: active ? palette.activeBg : palette.card,
    color: active ? palette.activeText : palette.text,
    cursor: "pointer",
    fontWeight: 600
  };
}

class BrainDriveRAGSettings extends React.Component<PanelProps, PanelState> {
  private themeListener?: (theme: string) => void;

  constructor(props: PanelProps) {
    super(props);
    this.state = {
      toggles: this.buildDefaults(props.initialState),
      healthMessages: {
        documentChat: "",
        documentProcessing: ""
      },
      currentTheme: "light"
    };
  }

  componentDidMount() {
    const themeSvc = this.props.services?.theme;
    if (themeSvc?.getCurrentTheme) {
      const current = themeSvc.getCurrentTheme();
      this.setState({ currentTheme: current });
    }
    if (themeSvc?.addThemeChangeListener) {
      this.themeListener = (theme) => this.setState({ currentTheme: theme || "light" });
      themeSvc.addThemeChangeListener(this.themeListener);
    }
  }

  componentWillUnmount() {
    const themeSvc = this.props.services?.theme;
    if (themeSvc?.removeThemeChangeListener && this.themeListener) {
      themeSvc.removeThemeChangeListener(this.themeListener);
    }
  }

  componentDidUpdate(prevProps: PanelProps) {
    if (prevProps.initialState !== this.props.initialState) {
      this.setState({ toggles: this.buildDefaults(this.props.initialState) });
    }
  }

  private buildDefaults(initialState?: Partial<Record<ServiceKey, ToggleState>>): Record<ServiceKey, ToggleState> {
    const initChatEnabled = (initialState as any)?.documentChat?.enabled ?? false;
    const initProcEnabled = (initialState as any)?.documentProcessing?.enabled ?? false;
    return {
      documentChat: initialState?.documentChat
        ? { status: initChatEnabled ? "running" : "stopped" }
        : { status: initChatEnabled ? "running" : "unknown" },
      documentProcessing: initialState?.documentProcessing
        ? { status: initProcEnabled ? "running" : "stopped" }
        : { status: initProcEnabled ? "running" : "unknown" }
    };
  }

  private setHealth = (key: ServiceKey, msg: string) => {
    this.setState((prev) => ({
      ...prev,
      healthMessages: { ...prev.healthMessages, [key]: msg },
      toggles: {
        ...prev.toggles,
        [key]: {
          ...prev.toggles[key],
          status: msg.toLowerCase().includes("healthy") || msg.toLowerCase().includes("ok") ? "running" : prev.toggles[key].status === "running" ? "error" : "stopped",
          lastMessage: msg
        }
      }
    }));
  };

  private handleToggle = (key: ServiceKey, next: boolean) => {
    this.setState((prev) => ({
      ...prev,
      toggles: { ...prev.toggles, [key]: { ...prev.toggles[key], status: next ? "running" : "stopped" } }
    }));
  };

  private handleHealthCheck = async (cfg: ServiceConfig) => {
    const url = `http://localhost:${cfg.defaultPort}${cfg.healthPath}`;
    try {
      this.setHealth(cfg.key, "checking...");

      const response = await fetch(url, { method: "GET", credentials: "omit", mode: "cors" });
      if (!response.ok) {
        this.setHealth(cfg.key, `Unhealthy (${response.status})`);
        return;
      }
      const text = await response.text();
      const formatted = formatHealthMessage(text);
      this.setHealth(cfg.key, formatted || "Healthy");
    } catch (error: any) {
      this.setHealth(cfg.key, `Check failed: ${error?.message || error}`);
    }
  };

  private callControl = async (cfg: ServiceConfig, action: "start" | "stop" | "restart") => {
    // Control endpoints are not available; surface a message without calling the network.
    this.setHealth(cfg.key, `${action} not supported in this build`);
  };

  render() {
    const { title, subtitle } = this.props;
    const { toggles, healthMessages, currentTheme } = this.state;
    const palette = getPalette(currentTheme);

    return (
      <div style={{ display: "grid", gap: 12 }}>
        {SERVICE_CONFIG.map((cfg) => (
          <div key={cfg.key} style={{ display: "grid", gap: 6 }}>
            <ServiceCard
              config={cfg}
              state={toggles[cfg.key]}
              onToggle={(next) => this.handleToggle(cfg.key, next)}
              onStart={() => this.callControl(cfg, "start")}
              onStop={() => this.callControl(cfg, "stop")}
              onRestart={() => this.callControl(cfg, "restart")}
              onHealthCheck={() => this.handleHealthCheck(cfg)}
              palette={palette}
            />
          </div>
        ))}
      </div>
    );
  }
}

export default BrainDriveRAGSettings;
export { BrainDriveRAGSettings };

function formatHealthMessage(raw: string): string {
  const trimmed = (raw || "").trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object") {
      if (typeof parsed.status === "string") {
        return parsed.version ? `${parsed.status} (${parsed.version})` : parsed.status;
      }
      return Object.entries(parsed)
        .map(([k, v]) => `${k}: ${String(v)}`)
        .join(", ");
    }
  } catch {
    // not JSON
  }
  return trimmed;
}
