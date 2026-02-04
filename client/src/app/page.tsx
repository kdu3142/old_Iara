"use client";

import {
  ConsoleTemplate,
  FullScreenContainer,
  PipecatLogo,
  ThemeProvider,
} from "@pipecat-ai/voice-ui-kit";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import ConfigPanel from "../components/ConfigPanel";
import LatencyMetricsPortal from "../components/LatencyMetricsPanel";
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
  left.ttsSentenceStreaming === right.ttsSentenceStreaming &&
  JSON.stringify(left.qwenTts) === JSON.stringify(right.qwenTts) &&
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
  const [warmupStatus, setWarmupStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle"
  );
  const [warmupError, setWarmupError] = useState<string | null>(null);
  const warmupPromiseRef = useRef<Promise<boolean> | null>(null);
  const allowConnectRef = useRef(false);

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

  const runWarmup = useCallback(async () => {
    if (warmupPromiseRef.current) {
      return warmupPromiseRef.current;
    }
    setWarmupStatus("loading");
    setWarmupError(null);
    const promise = (async () => {
      const start = Date.now();
      const maxWaitMs = 10 * 60 * 1000;
      const pollIntervalMs = 2000;
      while (Date.now() - start < maxWaitMs) {
        try {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 10000);
          const response = await fetch("/api/warmup", {
            method: "POST",
            cache: "no-store",
            signal: controller.signal,
          });
          clearTimeout(timeout);
          const payload = (await response.json()) as {
            status?: "idle" | "loading" | "ready" | "error";
            error?: string | null;
          };
          if (payload.status === "ready") {
            setWarmupStatus("ready");
            setWarmupError(null);
            return true;
          }
          if (payload.status === "error") {
            setWarmupStatus("error");
            setWarmupError(payload.error ?? "Warmup failed.");
            return false;
          }
          setWarmupStatus("loading");
        } catch (error) {
          setWarmupStatus("loading");
          setWarmupError(
            error instanceof Error ? `${error.message}. Retrying...` : "Warmup request failed. Retrying..."
          );
        }
        await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
      }
      setWarmupStatus("error");
      setWarmupError("Warmup timed out. Try again.");
      return false;
    })().finally(() => {
      warmupPromiseRef.current = null;
    });
    warmupPromiseRef.current = promise;
    return promise;
  }, []);

  useEffect(() => {
    const updateConnectLabel = () => {
      const buttons = Array.from(document.querySelectorAll("button"));
      for (const button of buttons) {
        if (button.textContent?.trim() === "Connect") {
          button.textContent = "Start Chat";
        }
      }
    };
    updateConnectLabel();
    const observer = new MutationObserver(updateConnectLabel);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setWarmupStatus("idle");
    setWarmupError(null);
    allowConnectRef.current = false;
  }, [
    config.whisperModel,
    config.whisperLanguage,
    config.ttsModel,
    config.ttsLanguage,
    config.ttsVoice,
    config.ttsSentenceStreaming,
    config.qwenTts.mode,
    config.qwenTts.model,
    config.qwenTts.language,
    config.qwenTts.speaker,
    config.qwenTts.instruct,
    config.qwenTts.refAudioPath,
    config.qwenTts.refText,
    config.qwenTts.seed,
  ]);

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      const button = target?.closest("button");
      if (!button) return;
      const label = button.textContent?.trim();
      if (label !== "Connect" && label !== "Start Chat") return;
      if (allowConnectRef.current) return;
      event.preventDefault();
      event.stopPropagation();
      void runWarmup().then((ready) => {
        if (ready) {
          allowConnectRef.current = true;
          setTimeout(() => button.click(), 0);
        }
      });
    };
    document.addEventListener("click", handleClick, true);
    return () => {
      document.removeEventListener("click", handleClick, true);
    };
  }, [runWarmup]);

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
        logoComponent={
          <>
            <PipecatLogo className="h-6 w-auto text-foreground" />
            <LatencyMetricsPortal />
          </>
        }
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
          <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
            {activeTab === "console" ? (
              <>
                {consoleContent}
                {warmupStatus === "loading" && (
                  <div
                    style={{
                      position: "absolute",
                      right: "16px",
                      bottom: "16px",
                      padding: "8px 12px",
                      borderRadius: "8px",
                      border: "1px solid rgba(255,255,255,0.2)",
                      background: "rgba(0,0,0,0.35)",
                      fontSize: "12px",
                    }}
                  >
                    Preparing speech models...
                  </div>
                )}
                {warmupStatus === "error" && (
                  <div
                    style={{
                      position: "absolute",
                      right: "16px",
                      bottom: "16px",
                      padding: "8px 12px",
                      borderRadius: "8px",
                      border: "1px solid rgba(255,255,255,0.2)",
                      background: "rgba(0,0,0,0.35)",
                      color: "#ffb4b4",
                      fontSize: "12px",
                    }}
                  >
                    {warmupError ?? "Warmup failed. Try again."}
                  </div>
                )}
              </>
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
