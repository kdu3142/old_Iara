import type { ConfigValues, Preset } from "@/lib/configDefaults";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  KOKORO_VOICES,
  MARVIS_DEFAULT_VOICE,
  LLM_PROVIDERS,
  QWEN_TTS_LANGUAGES,
  QWEN_TTS_MODELS,
  QWEN_TTS_SPEAKERS,
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

const RECORDING_MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/ogg",
];

const pickRecordingMimeType = () => {
  if (typeof MediaRecorder === "undefined") return "";
  for (const candidate of RECORDING_MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(candidate)) return candidate;
  }
  return "";
};

const writeString = (view: DataView, offset: number, value: string) => {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
};

const encodeWav = (audioBuffer: AudioBuffer) => {
  const numChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const sampleRate = audioBuffer.sampleRate;
  const mono = new Float32Array(length);
  if (numChannels === 1) {
    mono.set(audioBuffer.getChannelData(0));
  } else {
    for (let channel = 0; channel < numChannels; channel += 1) {
      const data = audioBuffer.getChannelData(channel);
      for (let i = 0; i < length; i += 1) {
        mono[i] += data[i];
      }
    }
    for (let i = 0; i < length; i += 1) {
      mono[i] /= numChannels;
    }
  }
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (let i = 0; i < length; i += 1) {
    const sample = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return buffer;
};

const blobToWav = async (blob: Blob) => {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new (window.AudioContext ||
    (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
  await audioContext.close();
  const wavBuffer = encodeWav(audioBuffer);
  return new Blob([wavBuffer], { type: "audio/wav" });
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
  const isQwenModel = config.ttsModel.startsWith("mlx-community/Qwen3-TTS");
  const isKokoroModel = !isMarvisModel && !isQwenModel;
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
  const qwenMode = config.qwenTts.mode;
  const qwenNeedsInstruct = qwenMode === "customVoice" || qwenMode === "voiceDesign";
  const qwenNeedsCloneInputs = qwenMode === "voiceCloning";
  const refAudioPath = config.qwenTts.refAudioPath?.trim() ?? "";
  const refAudioUrl = refAudioPath
    ? `/api/qwen/ref-audio?path=${encodeURIComponent(refAudioPath)}`
    : "";
  const turnTaking = config.turnTaking;
  const vadSettings = turnTaking.vad;
  const smartTurnSettings = turnTaking.smartTurn;
  const smartTurnEnabled = smartTurnSettings.enabled;
  const configRef = useRef(config);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const [recordingStatus, setRecordingStatus] = useState<
    "idle" | "recording" | "processing" | "saving" | "error"
  >("idle");
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [recordingSavedPath, setRecordingSavedPath] = useState<string | null>(null);
  const pickQwenModelForMode = (
    mode: ConfigValues["qwenTts"]["mode"],
    currentModel: string
  ) => {
    const size = currentModel.includes("1.7B") ? "1.7B" : "0.6B";
    if (mode === "voiceDesign") {
      return "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16";
    }
    if (mode === "customVoice") {
      return size === "1.7B"
        ? "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
        : "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16";
    }
    return size === "1.7B"
      ? "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
      : "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16";
  };
  const modeFromQwenModel = (model: string): ConfigValues["qwenTts"]["mode"] => {
    if (model.includes("VoiceDesign")) return "voiceDesign";
    if (model.includes("CustomVoice")) return "customVoice";
    return "base";
  };
  const modelOptions = useMemo(() => {
    if (!ollamaEnabled || ollamaModels.length === 0) return [];
    return ollamaModels;
  }, [ollamaEnabled, ollamaModels]);
  const qwenModelOptions = useMemo(() => {
    if (qwenMode === "customVoice") {
      return QWEN_TTS_MODELS.filter((model) => model.includes("CustomVoice"));
    }
    if (qwenMode === "voiceDesign") {
      return QWEN_TTS_MODELS.filter((model) => model.includes("VoiceDesign"));
    }
    return QWEN_TTS_MODELS.filter((model) => model.includes("Base"));
  }, [qwenMode]);
  const ollamaModelMissing =
    ollamaEnabled &&
    modelOptions.length > 0 &&
    !modelOptions.includes(config.llmModel);

  useEffect(() => {
    configRef.current = config;
  }, [config]);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
      if (recordingUrl) {
        URL.revokeObjectURL(recordingUrl);
      }
    };
  }, [recordingUrl]);

  useEffect(() => {
    if (!qwenNeedsCloneInputs) {
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
      setRecordingStatus("idle");
      setRecordingError(null);
      setRecordingSavedPath(null);
      setNewRecordingUrl(null);
    }
  }, [qwenNeedsCloneInputs]);

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

  const canRecord =
    typeof window !== "undefined" &&
    typeof MediaRecorder !== "undefined" &&
    Boolean(navigator.mediaDevices?.getUserMedia);

  const setNewRecordingUrl = (url: string | null) => {
    setRecordingUrl((prev) => {
      if (prev) {
        URL.revokeObjectURL(prev);
      }
      return url;
    });
  };

  const handleStartRecording = async () => {
    if (
      !canRecord ||
      recordingStatus === "recording" ||
      recordingStatus === "processing" ||
      recordingStatus === "saving"
    )
      return;
    setRecordingError(null);
    setRecordingSavedPath(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const mimeType = pickRecordingMimeType();
      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined
      );
      recordedChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        mediaRecorderRef.current = null;
        setRecordingStatus("processing");
        try {
          const rawBlob = new Blob(recordedChunksRef.current, {
            type: recorder.mimeType || "audio/webm",
          });
          recordedChunksRef.current = [];
          const wavBlob =
            rawBlob.type === "audio/wav" ? rawBlob : await blobToWav(rawBlob);
          const url = URL.createObjectURL(wavBlob);
          setNewRecordingUrl(url);
          setRecordingStatus("saving");
          const formData = new FormData();
          formData.append("audio", wavBlob, "qwen-reference.wav");
          const response = await fetch("/api/qwen/ref-audio", {
            method: "POST",
            body: formData,
          });
          if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || "Upload failed.");
          }
          const payload = (await response.json()) as { path?: string };
          if (!payload.path) {
            throw new Error("Upload did not return a file path.");
          }
          setRecordingSavedPath(payload.path);
          const current = configRef.current;
          onConfigChange({
            ...current,
            qwenTts: {
              ...current.qwenTts,
              refAudioPath: payload.path,
            },
          });
          setRecordingStatus("idle");
        } catch (error) {
          setRecordingStatus("error");
          setRecordingError(
            error instanceof Error
              ? error.message
              : "Recording upload failed."
          );
        }
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecordingStatus("recording");
    } catch (error) {
      setRecordingStatus("error");
      setRecordingError(
        error instanceof Error
          ? error.message
          : "Microphone permission denied."
      );
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

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
            Turn Taking
          </h2>
          <p style={{ marginBottom: "12px", opacity: 0.7 }}>
            Controls how the system decides the user is done speaking.
          </p>
          <p style={{ marginTop: "-6px", marginBottom: "12px", opacity: 0.7 }}>
            VAD detects speech vs silence, while smart-turn uses a model to decide
            when a thought is complete for more natural pauses.
          </p>
          <label
            style={{
              display: "flex",
              gap: "8px",
              alignItems: "center",
              marginBottom: "12px",
            }}
          >
            <input
              type="checkbox"
              checked={smartTurnEnabled}
              onChange={(event) =>
                onConfigChange({
                  ...config,
                  turnTaking: {
                    ...config.turnTaking,
                    smartTurn: {
                      ...config.turnTaking.smartTurn,
                      enabled: event.target.checked,
                    },
                  },
                })
              }
            />
            Enable smart-turn model
          </label>
          {!smartTurnEnabled && (
            <p style={{ marginTop: "-6px", marginBottom: "12px", opacity: 0.7 }}>
              Turns end on VAD silence only.
            </p>
          )}

          <div style={{ marginBottom: "8px", fontSize: "13px", opacity: 0.8 }}>
            VAD
          </div>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Confidence</div>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={vadSettings.confidence}
              onChange={(event) => {
                const raw = event.target.value.trim();
                const next = raw === "" ? 0 : Number.parseFloat(raw);
                const value = Number.isFinite(next)
                  ? Math.min(1, Math.max(0, next))
                  : vadSettings.confidence;
                onConfigChange({
                  ...config,
                  turnTaking: {
                    ...config.turnTaking,
                    vad: { ...config.turnTaking.vad, confidence: value },
                  },
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
            />
            <div style={{ marginTop: "6px", opacity: 0.7 }}>
              Minimum VAD confidence to treat audio as speech. Higher is stricter.
            </div>
          </label>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Start seconds</div>
            <input
              type="number"
              step="0.05"
              min="0"
              value={vadSettings.startSecs}
              onChange={(event) => {
                const raw = event.target.value.trim();
                const next = raw === "" ? 0 : Number.parseFloat(raw);
                const value = Number.isFinite(next)
                  ? Math.max(0, next)
                  : vadSettings.startSecs;
                onConfigChange({
                  ...config,
                  turnTaking: {
                    ...config.turnTaking,
                    vad: { ...config.turnTaking.vad, startSecs: value },
                  },
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
            />
            <div style={{ marginTop: "6px", opacity: 0.7 }}>
              How long speech must persist before VAD marks the user speaking.
            </div>
          </label>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Stop seconds</div>
            <input
              type="number"
              step="0.05"
              min="0"
              value={vadSettings.stopSecs}
              onChange={(event) => {
                const raw = event.target.value.trim();
                const next = raw === "" ? 0 : Number.parseFloat(raw);
                const value = Number.isFinite(next)
                  ? Math.max(0, next)
                  : vadSettings.stopSecs;
                onConfigChange({
                  ...config,
                  turnTaking: {
                    ...config.turnTaking,
                    vad: { ...config.turnTaking.vad, stopSecs: value },
                  },
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
            />
            <div style={{ marginTop: "6px", opacity: 0.7 }}>
              How long silence must persist before VAD marks the user stopped.
            </div>
          </label>
          <label style={{ display: "block", marginBottom: "12px" }}>
            <div style={{ marginBottom: "6px" }}>Min volume</div>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={vadSettings.minVolume}
              onChange={(event) => {
                const raw = event.target.value.trim();
                const next = raw === "" ? 0 : Number.parseFloat(raw);
                const value = Number.isFinite(next)
                  ? Math.min(1, Math.max(0, next))
                  : vadSettings.minVolume;
                onConfigChange({
                  ...config,
                  turnTaking: {
                    ...config.turnTaking,
                    vad: { ...config.turnTaking.vad, minVolume: value },
                  },
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
            />
            <div style={{ marginTop: "6px", opacity: 0.7 }}>
              Minimum volume threshold for speech detection. Higher ignores quiet speech.
            </div>
          </label>

          {smartTurnEnabled && (
            <>
              <div
                style={{ marginBottom: "8px", fontSize: "13px", opacity: 0.8 }}
              >
                Smart turn
              </div>
              <label style={{ display: "block", marginBottom: "12px" }}>
                <div style={{ marginBottom: "6px" }}>Stop seconds</div>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={smartTurnSettings.stopSecs}
                  onChange={(event) => {
                    const raw = event.target.value.trim();
                    const next = raw === "" ? 0 : Number.parseFloat(raw);
                    const value = Number.isFinite(next)
                      ? Math.max(0, next)
                      : smartTurnSettings.stopSecs;
                    onConfigChange({
                      ...config,
                      turnTaking: {
                        ...config.turnTaking,
                        smartTurn: {
                          ...config.turnTaking.smartTurn,
                          stopSecs: value,
                        },
                      },
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
                />
                <div style={{ marginTop: "6px", opacity: 0.7 }}>
                  Silence fallback that forces an end-of-turn if the model hasn't decided.
                </div>
              </label>
              <label style={{ display: "block", marginBottom: "12px" }}>
                <div style={{ marginBottom: "6px" }}>Pre-speech ms</div>
                <input
                  type="number"
                  step="10"
                  min="0"
                  value={smartTurnSettings.preSpeechMs}
                  onChange={(event) => {
                    const raw = event.target.value.trim();
                    const next = raw === "" ? 0 : Number.parseFloat(raw);
                    const value = Number.isFinite(next)
                      ? Math.max(0, next)
                      : smartTurnSettings.preSpeechMs;
                    onConfigChange({
                      ...config,
                      turnTaking: {
                        ...config.turnTaking,
                        smartTurn: {
                          ...config.turnTaking.smartTurn,
                          preSpeechMs: value,
                        },
                      },
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
                />
                <div style={{ marginTop: "6px", opacity: 0.7 }}>
                  Audio to include before speech onset when analyzing the turn.
                </div>
              </label>
              <label style={{ display: "block", marginBottom: "12px" }}>
                <div style={{ marginBottom: "6px" }}>Max duration seconds</div>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  value={smartTurnSettings.maxDurationSecs}
                  onChange={(event) => {
                    const raw = event.target.value.trim();
                    const next = raw === "" ? 0 : Number.parseFloat(raw);
                    const value = Number.isFinite(next)
                      ? Math.max(0, next)
                      : smartTurnSettings.maxDurationSecs;
                    onConfigChange({
                      ...config,
                      turnTaking: {
                        ...config.turnTaking,
                        smartTurn: {
                          ...config.turnTaking.smartTurn,
                          maxDurationSecs: value,
                        },
                      },
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
                />
                <div style={{ marginTop: "6px", opacity: 0.7 }}>
                  Caps how much audio the model analyzes for a single turn.
                </div>
              </label>
            </>
          )}
          <p style={{ marginTop: "4px", opacity: 0.7 }}>
            Changes apply on reconnect.
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
                const nextTtsModel = nextModel.startsWith("mlx-community/Qwen3-TTS")
                  ? "mlx-community/Qwen3-TTS"
                  : nextModel;
                onConfigChange({
                  ...config,
                  ttsModel: nextTtsModel,
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

          {isKokoroModel && (
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>
                Voice language filter
              </div>
              <select
                value={config.ttsLanguage}
                onChange={(event) => {
                  const nextLanguage = event.target.value;
                  const voiceForLanguage = KOKORO_VOICES.find(
                    (voice) => voice.language === nextLanguage
                  );
                  const nextVoice =
                    voiceForLanguage?.id ?? config.ttsVoice;
                  onConfigChange({
                    ...config,
                    ttsLanguage: nextLanguage,
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
                {TTS_LANGUAGES.map((language) => (
                  <option key={language.id} value={language.id}>
                    {language.label}
                  </option>
                ))}
              </select>
            </label>
          )}

          {isKokoroModel && (
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

          {isMarvisModel && (
            <>
              <label style={{ display: "block", marginBottom: "12px" }}>
                <div style={{ marginBottom: "6px" }}>Language</div>
                <select
                  value={config.ttsLanguage}
                  onChange={(event) =>
                    onConfigChange({
                      ...config,
                      ttsLanguage: event.target.value,
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
                  {TTS_LANGUAGES.map((language) => (
                    <option key={language.id} value={language.id}>
                      {language.label}
                    </option>
                  ))}
                </select>
              </label>
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
            </>
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

        {isQwenModel && (
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
              Qwen3 TTS (MLX)
            </h2>
            <p style={{ marginBottom: "12px", opacity: 0.7 }}>
              Qwen settings only apply when a Qwen3 model is selected.
            </p>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Mode</div>
              <select
                value={config.qwenTts.mode}
                onChange={(event) =>
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      mode: event.target.value as ConfigValues["qwenTts"]["mode"],
                      model: pickQwenModelForMode(
                        event.target.value as ConfigValues["qwenTts"]["mode"],
                        config.qwenTts.model
                      ),
                    },
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
                <option value="base">Base</option>
                <option value="customVoice">CustomVoice (emotion)</option>
                <option value="voiceDesign">VoiceDesign</option>
                <option value="voiceCloning">Voice Cloning</option>
              </select>
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Chooses the Qwen TTS pipeline: Base uses preset speakers, CustomVoice
                adds emotion prompts, VoiceDesign uses descriptive prompts, and
                Voice Cloning uses reference audio.
              </div>
            </label>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Qwen model</div>
              <select
                value={config.qwenTts.model}
                onChange={(event) =>
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      model: event.target.value,
                      mode:
                        config.qwenTts.mode === "voiceCloning"
                          ? "voiceCloning"
                          : modeFromQwenModel(event.target.value),
                    },
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
                {qwenModelOptions.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Selects the specific weights and size for Qwen TTS. Changing models
                may update the mode automatically.
              </div>
            </label>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Language</div>
              <select
                value={config.qwenTts.language}
                onChange={(event) =>
                  onConfigChange({
                    ...config,
                    qwenTts: { ...config.qwenTts, language: event.target.value },
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
                {QWEN_TTS_LANGUAGES.map((language) => (
                  <option key={language.id} value={language.id}>
                    {language.label}
                  </option>
                ))}
              </select>
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Sets text normalization and pronunciation rules for synthesis.
              </div>
            </label>
            {(config.qwenTts.mode === "base" ||
              config.qwenTts.mode === "customVoice") && (
              <label style={{ display: "block", marginBottom: "12px" }}>
                <div style={{ marginBottom: "6px" }}>Speaker</div>
                <select
                  value={config.qwenTts.speaker}
                  onChange={(event) =>
                    onConfigChange({
                      ...config,
                      qwenTts: {
                        ...config.qwenTts,
                        speaker: event.target.value,
                      },
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
                  {QWEN_TTS_SPEAKERS.map((speaker) => (
                    <option key={speaker.id} value={speaker.id}>
                      {speaker.label}
                    </option>
                  ))}
                </select>
                <div style={{ marginTop: "6px", opacity: 0.7 }}>
                  Picks a preset voice timbre (available in Base/CustomVoice).
                </div>
              </label>
            )}
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Temperature</div>
              <input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={config.qwenTts.temperature}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const nextTemp = raw === "" ? 0 : Number.parseFloat(raw);
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      temperature: Number.isFinite(nextTemp) ? nextTemp : 0,
                    },
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
              />
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Lower values are more consistent. Use 0 for deterministic output.
              </div>
            </label>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Speed</div>
              <input
                type="number"
                step="0.1"
                min="0.5"
                max="2"
                value={config.qwenTts.speed}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const nextSpeed = raw === "" ? 1 : Number.parseFloat(raw);
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      speed: Number.isFinite(nextSpeed) ? nextSpeed : 1,
                    },
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
              />
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                1.0 is default. Higher is faster, lower is slower.
              </div>
            </label>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Top K</div>
              <input
                type="number"
                step="1"
                min="0"
                value={config.qwenTts.topK}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const nextTopK = raw === "" ? 0 : Number.parseInt(raw, 10);
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      topK: Number.isFinite(nextTopK) ? nextTopK : 0,
                    },
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
              />
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Limits sampling to the top K tokens (0 disables the limit).
              </div>
            </label>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Top P</div>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={config.qwenTts.topP}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const nextTopP = raw === "" ? 1 : Number.parseFloat(raw);
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      topP: Number.isFinite(nextTopP) ? nextTopP : 1,
                    },
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
              />
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Nucleus sampling cutoff; 1.0 keeps the full distribution.
              </div>
            </label>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Repetition penalty</div>
              <input
                type="number"
                step="0.05"
                min="0.5"
                max="2"
                value={config.qwenTts.repetitionPenalty}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const nextPenalty = raw === "" ? 1 : Number.parseFloat(raw);
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      repetitionPenalty: Number.isFinite(nextPenalty) ? nextPenalty : 1,
                    },
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
              />
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Higher values reduce repeated tokens; 1.0 disables the penalty.
              </div>
            </label>
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Max tokens</div>
              <input
                type="number"
                step="1"
                min="0"
                value={config.qwenTts.maxTokens}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const nextMax = raw === "" ? 0 : Number.parseInt(raw, 10);
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      maxTokens: Number.isFinite(nextMax) ? nextMax : 0,
                    },
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
              />
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                0 uses the model default.
              </div>
            </label>
            <label
              style={{
                display: "flex",
                gap: "8px",
                alignItems: "center",
                marginBottom: "12px",
              }}
            >
              <input
                type="checkbox"
                checked={config.qwenTts.doSample}
                onChange={(event) =>
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      doSample: event.target.checked,
                    },
                  })
                }
              />
              Enable sampling (do_sample)
            </label>
            <div style={{ marginTop: "-6px", marginBottom: "12px", opacity: 0.7 }}>
              When disabled, decoding is greedy and deterministic (ignores Top K/Top P).
            </div>
            {qwenNeedsInstruct && (
              <label style={{ display: "block", marginBottom: "12px" }}>
                <div style={{ marginBottom: "6px" }}>Instruction</div>
                <input
                  type="text"
                  value={config.qwenTts.instruct}
                  onChange={(event) =>
                    onConfigChange({
                      ...config,
                      qwenTts: { ...config.qwenTts, instruct: event.target.value },
                    })
                  }
                  placeholder={
                    qwenMode === "customVoice"
                      ? "Very happy and excited."
                      : "A cheerful young female voice with high pitch."
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
                <div style={{ marginTop: "6px", opacity: 0.7 }}>
                  Describes the desired voice style, emotion, or character.
                </div>
              </label>
            )}
            {qwenNeedsCloneInputs && (
              <>
                <label style={{ display: "block", marginBottom: "12px" }}>
                  <div style={{ marginBottom: "6px" }}>Reference audio path</div>
                  <input
                    type="text"
                    value={config.qwenTts.refAudioPath}
                    onChange={(event) =>
                      onConfigChange({
                        ...config,
                        qwenTts: {
                          ...config.qwenTts,
                          refAudioPath: event.target.value,
                        },
                      })
                    }
                    placeholder="path/to/sample_audio.wav"
                    style={{
                      width: "100%",
                      padding: "8px 10px",
                      borderRadius: "6px",
                      border: "1px solid rgba(255,255,255,0.2)",
                      background: "rgba(255,255,255,0.04)",
                      color: "inherit",
                    }}
                  />
                  <div style={{ marginTop: "6px", opacity: 0.7 }}>
                    Local path to a clean reference recording for cloning.
                  </div>
                </label>
                {refAudioUrl && (
                  <div style={{ marginBottom: "12px" }}>
                    <div style={{ marginBottom: "6px" }}>
                      Reference audio preview
                    </div>
                    <audio
                      controls
                      key={refAudioUrl}
                      src={refAudioUrl}
                      style={{ width: "100%" }}
                    />
                  </div>
                )}
                <div style={{ marginBottom: "12px" }}>
                  <div style={{ marginBottom: "6px" }}>Record reference audio</div>
                  <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                    <button
                      type="button"
                      onClick={handleStartRecording}
                      disabled={
                        !canRecord ||
                        recordingStatus === "recording" ||
                        recordingStatus === "processing" ||
                        recordingStatus === "saving"
                      }
                      style={{
                        padding: "8px 12px",
                        borderRadius: "6px",
                        border: "1px solid rgba(255,255,255,0.2)",
                        background: "rgba(255,255,255,0.08)",
                        color: "inherit",
                        cursor:
                          !canRecord ||
                          recordingStatus === "recording" ||
                          recordingStatus === "processing" ||
                          recordingStatus === "saving"
                            ? "not-allowed"
                            : "pointer",
                        opacity:
                          !canRecord ||
                          recordingStatus === "recording" ||
                          recordingStatus === "processing" ||
                          recordingStatus === "saving"
                            ? 0.6
                            : 1,
                      }}
                    >
                      {recordingStatus === "recording"
                        ? "Recording..."
                        : "Start recording"}
                    </button>
                    <button
                      type="button"
                      onClick={handleStopRecording}
                      disabled={recordingStatus !== "recording"}
                      style={{
                        padding: "8px 12px",
                        borderRadius: "6px",
                        border: "1px solid rgba(255,255,255,0.2)",
                        background: "rgba(255,255,255,0.08)",
                        color: "inherit",
                        cursor:
                          recordingStatus === "recording" ? "pointer" : "not-allowed",
                        opacity: recordingStatus === "recording" ? 1 : 0.6,
                      }}
                    >
                      Stop
                    </button>
                  </div>
                  <div style={{ marginTop: "6px", opacity: 0.7 }}>
                    Record a short, clean clip. We will save it and fill the path
                    automatically.
                  </div>
                  {recordingStatus === "recording" && (
                    <div style={{ marginTop: "6px", opacity: 0.7 }}>
                      Recording... Speak clearly, then click Stop.
                    </div>
                  )}
                  {recordingStatus === "processing" && (
                    <div style={{ marginTop: "6px", opacity: 0.7 }}>
                      Processing recording...
                    </div>
                  )}
                  {recordingStatus === "saving" && (
                    <div style={{ marginTop: "6px", opacity: 0.7 }}>
                      Saving to project...
                    </div>
                  )}
                  {recordingUrl && (
                    <audio
                      controls
                      src={recordingUrl}
                      style={{ width: "100%", marginTop: "8px" }}
                    />
                  )}
                  {recordingSavedPath && (
                    <div style={{ marginTop: "6px", opacity: 0.7 }}>
                      Saved to {recordingSavedPath}. Save the preset to persist.
                    </div>
                  )}
                  {!canRecord && (
                    <div style={{ marginTop: "6px", color: "#ffdf8a" }}>
                      Recording is not available in this browser.
                    </div>
                  )}
                  {recordingError && (
                    <div style={{ marginTop: "6px", color: "#ffb4b4" }}>
                      {recordingError}
                    </div>
                  )}
                </div>
                <label style={{ display: "block", marginBottom: "12px" }}>
                  <div style={{ marginBottom: "6px" }}>Reference transcript</div>
                  <input
                    type="text"
                    value={config.qwenTts.refText}
                    onChange={(event) =>
                      onConfigChange({
                        ...config,
                        qwenTts: { ...config.qwenTts, refText: event.target.value },
                      })
                    }
                    placeholder="This is what my voice sounds like."
                    style={{
                      width: "100%",
                      padding: "8px 10px",
                      borderRadius: "6px",
                      border: "1px solid rgba(255,255,255,0.2)",
                      background: "rgba(255,255,255,0.04)",
                      color: "inherit",
                    }}
                  />
                  <div style={{ marginTop: "6px", opacity: 0.7 }}>
                    Exact text spoken in the reference audio.
                  </div>
                </label>
                <label
                  style={{
                    display: "flex",
                    gap: "8px",
                    alignItems: "center",
                    marginBottom: "12px",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={config.qwenTts.xVectorOnlyMode}
                    onChange={(event) =>
                      onConfigChange({
                        ...config,
                        qwenTts: {
                          ...config.qwenTts,
                          xVectorOnlyMode: event.target.checked,
                        },
                      })
                    }
                  />
                  Use x-vector only (no transcript required)
                </label>
                <div style={{ marginTop: "-6px", marginBottom: "12px", opacity: 0.7 }}>
                  Builds a speaker embedding without transcript alignment.
                </div>
                <label style={{ display: "block", marginBottom: "12px" }}>
                  <div style={{ marginBottom: "6px" }}>
                    STT model (auto-transcribe reference)
                  </div>
                  <input
                    type="text"
                    value={config.qwenTts.sttModel}
                    onChange={(event) =>
                      onConfigChange({
                        ...config,
                        qwenTts: { ...config.qwenTts, sttModel: event.target.value },
                      })
                    }
                    placeholder="mlx-community/whisper-tiny"
                    style={{
                      width: "100%",
                      padding: "8px 10px",
                      borderRadius: "6px",
                      border: "1px solid rgba(255,255,255,0.2)",
                      background: "rgba(255,255,255,0.04)",
                      color: "inherit",
                    }}
                  />
                  <div style={{ marginTop: "6px", opacity: 0.7 }}>
                    Only used when no reference transcript is provided.
                  </div>
                </label>
              </>
            )}
            <label style={{ display: "block", marginBottom: "12px" }}>
              <div style={{ marginBottom: "6px" }}>Seed (deterministic voice)</div>
              <input
                type="number"
                value={config.qwenTts.seed}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const nextSeed = raw === "" ? 0 : Number.parseInt(raw, 10);
                  onConfigChange({
                    ...config,
                    qwenTts: {
                      ...config.qwenTts,
                      seed: Number.isFinite(nextSeed) ? nextSeed : 0,
                    },
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
              />
              <div style={{ marginTop: "6px", opacity: 0.7 }}>
                Use 0 for randomization; set a fixed value to reproduce results.
              </div>
            </label>
            {qwenNeedsInstruct && !config.qwenTts.instruct && (
              <div style={{ marginTop: "6px", color: "#ffdf8a" }}>
                Instruction is required for this Qwen mode.
              </div>
            )}
            {qwenNeedsCloneInputs &&
              (!config.qwenTts.refAudioPath ||
                (!config.qwenTts.refText && !config.qwenTts.xVectorOnlyMode)) && (
                <div style={{ marginTop: "6px", color: "#ffdf8a" }}>
                  Voice cloning requires reference audio. Add a transcript unless
                  x-vector-only mode is enabled.
                </div>
              )}
          </section>
        )}

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
          {ollamaEnabled && (
            <label
              style={{
                display: "flex",
                gap: "8px",
                alignItems: "center",
                marginBottom: "4px",
              }}
            >
              <input
                type="checkbox"
                checked={config.llmOllamaThink}
                onChange={(event) =>
                  onConfigChange({
                    ...config,
                    llmOllamaThink: event.target.checked,
                  })
                }
              />
              Enable Ollama thinking
            </label>
          )}
          {ollamaEnabled && (
            <p style={{ marginTop: "0px", opacity: 0.7 }}>
              Disable to send think: false for faster responses.
            </p>
          )}
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
