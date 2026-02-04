import type { ConfigValues, Preset } from "@/lib/configDefaults";
import { useEffect, useMemo, useState } from "react";
import {
  KOKORO_VOICES,
  MARVIS_DEFAULT_VOICE,
  LLM_PROVIDERS,
  TTS_LANGUAGES,
  TTS_MODELS,
  WHISPER_LANGUAGES,
  WHISPER_MODELS,
} from "@/lib/configOptions";

type ConfigPanelProps = {
  presets: Preset[];
  activePresetId: string;
  activePresetName: string;
  config: ConfigValues;
  newPresetName: string;
  isDirty: boolean;
  onPresetChange: (presetId: string) => void;
  onConfigChange: (config: ConfigValues) => void;
  onNewPresetNameChange: (value: string) => void;
  onSavePreset: () => void;
  onSaveAsNewPreset: () => void;
  onResetDefaults: () => void;
};

export default function ConfigPanel({
  presets,
  activePresetId,
  activePresetName,
  config,
  newPresetName,
  isDirty,
  onPresetChange,
  onConfigChange,
  onNewPresetNameChange,
  onSavePreset,
  onSaveAsNewPreset,
  onResetDefaults,
}: ConfigPanelProps) {
  const activePresetLabel = activePresetName || "Preset";
  const isMarvisModel = config.ttsModel.startsWith("Marvis-AI");
  const voicesForLanguage = KOKORO_VOICES.filter(
    (voice) => voice.language === config.ttsLanguage
  );
  const kokoroVoices =
    voicesForLanguage.length > 0 ? voicesForLanguage : KOKORO_VOICES;
  const hasCustomVoice = Boolean(
    config.ttsVoice &&
      !KOKORO_VOICES.some((voice) => voice.id === config.ttsVoice)
  );
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaStatus, setOllamaStatus] = useState<
    "idle" | "loading" | "error"
  >("idle");
  const ollamaEnabled = config.llmProvider === "ollama";
  const modelOptions = useMemo(() => {
    if (!ollamaEnabled || ollamaModels.length === 0) return [];
    return ollamaModels;
  }, [ollamaEnabled, ollamaModels]);
  const ollamaModelMissing =
    ollamaEnabled &&
    modelOptions.length > 0 &&
    !modelOptions.includes(config.llmModel);

  useEffect(() => {
    if (!ollamaEnabled) return;
    const controller = new AbortController();
    setOllamaStatus("loading");
    fetch("/api/ollama/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ baseUrl: config.llmBaseUrl }),
      signal: controller.signal,
    })
      .then((response) => response.json())
      .then((data: { models?: string[] }) => {
        const models = data.models ?? [];
        setOllamaModels(models);
        if (models.length > 0 && !models.includes(config.llmModel)) {
          onConfigChange({ ...config, llmModel: models[0] });
        }
        setOllamaStatus("idle");
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        setOllamaStatus("error");
      });
    return () => controller.abort();
  }, [ollamaEnabled, config.llmBaseUrl, config, onConfigChange]);

  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        overflow: "auto",
        padding: "24px",
        boxSizing: "border-box",
      }}
    >
      <button
        type="button"
        onClick={onSavePreset}
        disabled={!isDirty}
        style={{
          position: "fixed",
          right: "24px",
          bottom: "24px",
          padding: "12px 16px",
          borderRadius: "999px",
          border: "1px solid rgba(255,255,255,0.25)",
          background: isDirty ? "rgba(90,135,255,0.25)" : "rgba(255,255,255,0.08)",
          color: "inherit",
          cursor: isDirty ? "pointer" : "default",
          boxShadow: "0 8px 18px rgba(0,0,0,0.35)",
          backdropFilter: "blur(6px)",
          zIndex: 20,
        }}
      >
        {isDirty ? `Save ${activePresetLabel}` : "Saved"}
      </button>

      <div style={{ maxWidth: "760px", margin: "0 auto" }}>
        <h1 style={{ fontSize: "24px", marginBottom: "8px" }}>
          Configuration
        </h1>
        <p style={{ marginBottom: "24px", opacity: 0.8 }}>
          Settings are stored in a local config file and applied immediately.
        </p>

        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>Presets</h2>
          <p style={{ marginBottom: "12px", opacity: 0.7 }}>
            Save changes to the active preset or create a new one.
          </p>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Active preset</div>
            <select
              value={activePresetId}
              onChange={(event) => onPresetChange(event.target.value)}
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
              }}
            >
              {presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
            </select>
          </label>
          <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={onSavePreset}
              style={{
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.08)",
                color: "inherit",
                cursor: "pointer",
              }}
            >
              Save {activePresetLabel}
            </button>
            <label style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <input
                type="text"
                placeholder="New preset name"
                value={newPresetName}
                onChange={(event) => onNewPresetNameChange(event.target.value)}
                style={{
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: "rgba(255,255,255,0.04)",
                  color: "inherit",
                }}
              />
              <button
                type="button"
                onClick={onSaveAsNewPreset}
                style={{
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: "rgba(255,255,255,0.08)",
                  color: "inherit",
                  cursor: "pointer",
                }}
              >
                Save as new
              </button>
            </label>
          </div>
        </section>

        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>
            Connection
          </h2>
          <label style={{ display: "block", marginBottom: "8px" }}>
            <div style={{ marginBottom: "6px" }}>WebRTC offer URL</div>
            <input
              type="text"
              value={config.connectionUrl}
              onChange={(event) =>
                onConfigChange({ ...config, connectionUrl: event.target.value })
              }
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
              }}
            />
          </label>
          <label style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <input
              type="checkbox"
              checked={config.noUserVideo}
              onChange={(event) =>
                onConfigChange({ ...config, noUserVideo: event.target.checked })
              }
            />
            Disable local camera video
          </label>
        </section>

        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>
            Speech-to-text (Whisper MLX)
          </h2>
          <p style={{ marginBottom: "12px", opacity: 0.7 }}>
            These affect transcription quality and speed.
          </p>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Model</div>
            <select
              value={config.whisperModel}
              onChange={(event) =>
                onConfigChange({ ...config, whisperModel: event.target.value })
              }
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
              }}
            >
              {WHISPER_MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
          <label style={{ display: "block" }}>
            <div style={{ marginBottom: "6px" }}>Language</div>
            <select
              value={config.whisperLanguage}
              onChange={(event) =>
                onConfigChange({
                  ...config,
                  whisperLanguage: event.target.value,
                })
              }
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
              }}
            >
              {WHISPER_LANGUAGES.map((language) => (
                <option key={language.id} value={language.id}>
                  {language.label}
                </option>
              ))}
            </select>
          </label>
        </section>

        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>
            Text-to-speech (MLX)
          </h2>
          <p style={{ marginBottom: "12px", opacity: 0.7 }}>
            Model and voice choices apply after reconnecting.
          </p>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Model</div>
            <select
              value={config.ttsModel}
              onChange={(event) => {
                const nextModel = event.target.value;
                const nextVoice = nextModel.startsWith("Marvis-AI")
                  ? MARVIS_DEFAULT_VOICE
                  : config.ttsVoice;
                onConfigChange({
                  ...config,
                  ttsModel: nextModel,
                  ttsVoice: nextVoice,
                });
              }}
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
              }}
            >
              {TTS_MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>

          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Language</div>
            <select
              value={config.ttsLanguage}
              onChange={(event) => {
                const nextLanguage = event.target.value;
                const voiceForLanguage = KOKORO_VOICES.find(
                  (voice) => voice.language === nextLanguage
                );
                const nextVoice = voiceForLanguage
                  ? voiceForLanguage.id
                  : config.ttsVoice;
                onConfigChange({
                  ...config,
                  ttsLanguage: nextLanguage,
                  ttsVoice: isMarvisModel ? config.ttsVoice : nextVoice,
                });
              }}
              disabled={isMarvisModel}
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
                opacity: isMarvisModel ? 0.6 : 1,
              }}
            >
              {TTS_LANGUAGES.map((language) => (
                <option key={language.id} value={language.id}>
                  {language.label}
                </option>
              ))}
            </select>
          </label>

          {isMarvisModel ? (
            <label style={{ display: "block" }}>
              <div style={{ marginBottom: "6px" }}>Voice (optional)</div>
              <input
                type="text"
                value={config.ttsVoice}
                onChange={(event) =>
                  onConfigChange({ ...config, ttsVoice: event.target.value })
                }
                placeholder={MARVIS_DEFAULT_VOICE}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: "rgba(255,255,255,0.04)",
                  color: "inherit",
                }}
              />
            </label>
          ) : (
            <label style={{ display: "block" }}>
              <div style={{ marginBottom: "6px" }}>Voice</div>
              <select
                value={config.ttsVoice}
                onChange={(event) =>
                  onConfigChange({ ...config, ttsVoice: event.target.value })
                }
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: "rgba(255,255,255,0.04)",
                  color: "inherit",
                }}
              >
                {hasCustomVoice && (
                  <option value={config.ttsVoice}>
                    {config.ttsVoice} (custom)
                  </option>
                )}
                {kokoroVoices.map((voice) => (
                  <option key={voice.id} value={voice.id}>
                    {voice.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label
            style={{
              display: "flex",
              gap: "8px",
              alignItems: "center",
              marginTop: "12px",
            }}
          >
            <input
              type="checkbox"
              checked={config.ttsSentenceStreaming}
              onChange={(event) =>
                onConfigChange({
                  ...config,
                  ttsSentenceStreaming: event.target.checked,
                })
              }
            />
            Stream TTS per sentence (faster, more segmented)
          </label>
          <p style={{ marginTop: "8px", opacity: 0.7 }}>
            When enabled, long replies are split and synthesized sentence by
            sentence while preserving playback order.
          </p>
          <p style={{ marginTop: "8px", opacity: 0.7 }}>
            Model changes apply to new sessions; reconnect to use updated
            settings.
          </p>
        </section>

        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>LLM</h2>
          <p style={{ marginBottom: "12px", opacity: 0.7 }}>
            Provider and model changes apply after reconnecting.
          </p>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Provider</div>
            <select
              value={config.llmProvider}
              onChange={(event) => {
                const provider = event.target.value as ConfigValues["llmProvider"];
                const nextBaseUrl =
                  provider === "ollama"
                    ? "http://127.0.0.1:11434/v1"
                    : "http://127.0.0.1:1234/v1";
                onConfigChange({
                  ...config,
                  llmProvider: provider,
                  llmBaseUrl: nextBaseUrl,
                });
              }}
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
              }}
            >
              {LLM_PROVIDERS.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.label}
                </option>
              ))}
            </select>
          </label>

          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>API base URL</div>
            <input
              type="text"
              value={config.llmBaseUrl}
              onChange={(event) =>
                onConfigChange({ ...config, llmBaseUrl: event.target.value })
              }
              style={{
                width: "100%",
                padding: "8px 10px",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.04)",
                color: "inherit",
              }}
            />
          </label>

          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>
              Model
              {ollamaEnabled && (
                <button
                  type="button"
                  onClick={() => {
                    setOllamaStatus("loading");
                    fetch("/api/ollama/models", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ baseUrl: config.llmBaseUrl }),
                    })
                      .then((response) => response.json())
                      .then((data: { models?: string[] }) => {
                        const models = data.models ?? [];
                        setOllamaModels(models);
                        if (models.length > 0 && !models.includes(config.llmModel)) {
                          onConfigChange({ ...config, llmModel: models[0] });
                        }
                        setOllamaStatus("idle");
                      })
                      .catch(() => setOllamaStatus("error"));
                  }}
                  style={{
                    marginLeft: "12px",
                    padding: "4px 8px",
                    borderRadius: "6px",
                    border: "1px solid rgba(255,255,255,0.2)",
                    background: "rgba(255,255,255,0.08)",
                    color: "inherit",
                    cursor: "pointer",
                    fontSize: "12px",
                  }}
                >
                  Refresh
                </button>
              )}
            </div>
            {modelOptions.length > 0 ? (
              <select
                value={config.llmModel}
                onChange={(event) =>
                  onConfigChange({ ...config, llmModel: event.target.value })
                }
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: "rgba(255,255,255,0.04)",
                  color: "inherit",
                }}
              >
                {modelOptions.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={config.llmModel}
                onChange={(event) =>
                  onConfigChange({ ...config, llmModel: event.target.value })
                }
                placeholder="Enter model name"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: "rgba(255,255,255,0.04)",
                  color: "inherit",
                }}
              />
            )}
            {ollamaEnabled && ollamaStatus === "error" && (
              <div style={{ marginTop: "6px", color: "#ffb4b4" }}>
                Unable to fetch Ollama models. Check the base URL and that
                Ollama is running.
              </div>
            )}
            {ollamaModelMissing && (
              <div style={{ marginTop: "6px", color: "#ffdf8a" }}>
                Selected model not found in Ollama list. Updated to the first
                available model.
              </div>
            )}
          </label>
        </section>

        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>
            System prompt
          </h2>
          <p style={{ marginBottom: "12px", opacity: 0.7 }}>
            This prompt is sent as the first message in every new session.
          </p>
          <textarea
            value={config.systemPrompt}
            onChange={(event) =>
              onConfigChange({ ...config, systemPrompt: event.target.value })
            }
            rows={8}
            style={{
              width: "100%",
              padding: "10px",
              borderRadius: "8px",
              border: "1px solid rgba(255,255,255,0.2)",
              background: "rgba(255,255,255,0.04)",
              color: "inherit",
              resize: "vertical",
              lineHeight: 1.4,
            }}
          />
        </section>

        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>Server</h2>
          <p style={{ marginBottom: "12px", opacity: 0.8 }}>
            API keys live in `server/.env`. If you update them, restart
            `./run.sh` so the server reloads your changes.
          </p>
          <button
            type="button"
            onClick={onResetDefaults}
            style={{
              padding: "8px 12px",
              borderRadius: "6px",
              border: "1px solid rgba(255,255,255,0.2)",
              background: "rgba(255,255,255,0.08)",
              color: "inherit",
              cursor: "pointer",
            }}
          >
            Reset to defaults
          </button>
        </section>
      </div>
    </div>
  );
}
