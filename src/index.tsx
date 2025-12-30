import "./bootstrap";
import React from "react";
import { BrainDriveRAGSettings } from "./RAGSettingsPanel";

export { BrainDriveRAGSettings };
export default BrainDriveRAGSettings;

export const version = "0.1.0";

export const metadata = {
  name: "BrainDriveRAGCommunity",
  description: "Manage Document Chat and Document Processing services (docker or venv).",
  version
};

if (process.env.NODE_ENV === "development" && typeof window !== "undefined") {
  const rootEl = document.getElementById("root");
  if (rootEl) {
    import("react-dom/client").then(({ createRoot }) => {
      const root = createRoot(rootEl);
      root.render(
        <React.StrictMode>
          <div style={{ maxWidth: 900, margin: "40px auto", fontFamily: "Inter, system-ui, sans-serif" }}>
            <BrainDriveRAGSettings />
          </div>
        </React.StrictMode>
      );
    }).catch((err) => console.error("Failed to render dev shell", err));
  }
}
