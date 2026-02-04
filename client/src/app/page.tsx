"use client";

import {
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from "@pipecat-ai/voice-ui-kit";
import { useEffect, useMemo, useState } from "react";

import ConfigPanel from "../components/ConfigPanel";
import {
  ConfigStore,
  ConfigValues,
  DEFAULT_CONFIG_VALUES,
  createDefaultStore,
} from "@/lib/configDefaults";

const isConfigEqual = (left: ConfigValues, right: ConfigValues) =>
  left.connectionUrl === right.connectionUrl &&
  left.noUserVideo === right.noUserVideo &&
  left.whisperModel === right.whisperModel &&
  left.whisperLanguage === right.whisperLanguage &&
  left.ttsModel === right.ttsModel &&
  left.ttsLanguage === right.ttsLanguage &&
  left.ttsVoice === right.ttsVoice &&
  left.llmProvider === right.llmProvider &&
  left.llmBaseUrl === right.llmBaseUrl &&
  left.llmModel === right.llmModel &&
  left.systemPrompt === right.systemPrompt;

export default function Home() {
  const [activeTab, setActiveTab] = useState<"console" | "config">("console");
  const [configStore, setConfigStore] = useState<ConfigStore>(
    createDefaultStore()
  );
  const [config, setConfig] = useState<ConfigValues>(DEFAULT_CONFIG_VALUES);
  const [newPresetName, setNewPresetName] = useState("");
  const [consoleKey, setConsoleKey] = useState(0);

  useEffect(() => {
    let mounted = true;
    fetch("/api/config", { cache: "no-store" })
      .then((response) => response.json())
      .then((store: ConfigStore) => {
        if (!mounted) return;
        setConfigStore(store);
        const activePreset =
          store.presets.find((preset) => preset.id === store.activePresetId) ??
          store.presets[0];
        setConfig(activePreset?.values ?? DEFAULT_CONFIG_VALUES);
      })
      .catch(() => {
        if (!mounted) return;
        setConfigStore(createDefaultStore());
        setConfig(DEFAULT_CONFIG_VALUES);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const saveStore = async (store: ConfigStore) => {
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(store),
    });
    const saved = (await response.json()) as ConfigStore;
    setConfigStore(saved);
    const activePreset =
      saved.presets.find((preset) => preset.id === saved.activePresetId) ??
      saved.presets[0];
    setConfig(activePreset?.values ?? DEFAULT_CONFIG_VALUES);
    return saved;
  };

  const handlePresetChange = (presetId: string) => {
    const selected = configStore.presets.find((preset) => preset.id === presetId);
    if (!selected) return;
    const updatedStore: ConfigStore = {
      ...configStore,
      activePresetId: presetId,
    };
    setConfigStore(updatedStore);
    setConfig(selected.values);
    void saveStore(updatedStore);
  };

  const handleSavePreset = () => {
    const updatedPresets = configStore.presets.map((preset) =>
      preset.id === configStore.activePresetId
        ? {
            ...preset,
            values: config,
            updatedAt: new Date().toISOString(),
          }
        : preset
    );
    void saveStore({
      ...configStore,
      presets: updatedPresets,
    }).then(() => {
      setConsoleKey((prev) => prev + 1);
    });
  };

  const handleSaveAsNewPreset = () => {
    const name = newPresetName.trim();
    if (!name) return;
    const id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `preset-${Date.now()}`;
    const newPreset = {
      id,
      name,
      values: config,
      updatedAt: new Date().toISOString(),
    };
    const updatedStore: ConfigStore = {
      ...configStore,
      activePresetId: id,
      presets: [...configStore.presets, newPreset],
    };
    setNewPresetName("");
    void saveStore(updatedStore).then(() => {
      setConsoleKey((prev) => prev + 1);
    });
  };

  const handleResetDefaults = () => {
    const defaults = DEFAULT_CONFIG_VALUES;
    setConfig(defaults);
  };

  const consoleContent = useMemo(
    () => (
      <ConsoleTemplate
        key={consoleKey}
        transportType="smallwebrtc"
        connectParams={{
          connectionUrl: config.connectionUrl,
        }}
        noUserVideo={config.noUserVideo}
      />
    ),
    [config, consoleKey]
  );

  const activePreset =
    configStore.presets.find((preset) => preset.id === configStore.activePresetId) ??
    configStore.presets[0];

  const isDirty = Boolean(activePreset && !isConfigEqual(activePreset.values, config));

  return (
    <ThemeProvider>
      <FullScreenContainer>
        <div
          style={{
            height: "100%",
            width: "100%",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: "8px",
              padding: "8px 12px",
              borderBottom: "1px solid rgba(255,255,255,0.08)",
              background: "rgba(0,0,0,0.25)",
            }}
          >
            <button
              type="button"
              onClick={() => setActiveTab("console")}
              style={{
                padding: "6px 10px",
                borderRadius: "6px",
                border:
                  activeTab === "console"
                    ? "1px solid rgba(255,255,255,0.35)"
                    : "1px solid rgba(255,255,255,0.15)",
                background:
                  activeTab === "console"
                    ? "rgba(255,255,255,0.12)"
                    : "transparent",
                color: "inherit",
                cursor: "pointer",
              }}
            >
              Console
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("config")}
              style={{
                padding: "6px 10px",
                borderRadius: "6px",
                border:
                  activeTab === "config"
                    ? "1px solid rgba(255,255,255,0.35)"
                    : "1px solid rgba(255,255,255,0.15)",
                background:
                  activeTab === "config"
                    ? "rgba(255,255,255,0.12)"
                    : "transparent",
                color: "inherit",
                cursor: "pointer",
              }}
            >
              Config
            </button>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            {activeTab === "console" ? (
              consoleContent
            ) : (
              <ConfigPanel
                presets={configStore.presets}
                activePresetId={configStore.activePresetId}
                activePresetName={activePreset?.name ?? "Preset"}
                config={config}
                newPresetName={newPresetName}
                isDirty={isDirty}
                onPresetChange={handlePresetChange}
                onConfigChange={setConfig}
                onNewPresetNameChange={setNewPresetName}
                onSavePreset={handleSavePreset}
                onSaveAsNewPreset={handleSaveAsNewPreset}
                onResetDefaults={handleResetDefaults}
              />
            )}
          </div>
        </div>
      </FullScreenContainer>
    </ThemeProvider>
  );
}