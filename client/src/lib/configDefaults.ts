export type ConfigValues = {
  connectionUrl: string;
  noUserVideo: boolean;
  whisperModel: string;
  whisperLanguage: string;
  ttsModel: string;
  ttsLanguage: string;
  ttsVoice: string;
  ttsSentenceStreaming: boolean;
  llmProvider: "openai-compatible" | "ollama";
  llmBaseUrl: string;
  llmModel: string;
  systemPrompt: string;
};

export type Preset = {
  id: string;
  name: string;
  values: ConfigValues;
  updatedAt: string;
};

export type ConfigStore = {
  version: number;
  activePresetId: string;
  presets: Preset[];
};

export const DEFAULT_PRESET_ID = "default";

export const DEFAULT_CONFIG_VALUES: ConfigValues = {
  connectionUrl: "/api/offer",
  noUserVideo: true,
  whisperModel: "mlx-community/whisper-large-v3-turbo-q4",
  whisperLanguage: "en",
  ttsModel: "mlx-community/Kokoro-82M-bf16",
  ttsLanguage: "en-US",
  ttsVoice: "af_heart",
  ttsSentenceStreaming: false,
  llmProvider: "openai-compatible",
  llmBaseUrl: "http://127.0.0.1:1234/v1",
  llmModel: "gemma-3n-e4b-it-text",
  systemPrompt:
    "You are Pipecat, a friendly, helpful chatbot.\n\nYour input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.\n\nYour output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.\n\nRespond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.\n\nStart the conversation by saying, \"Hello, I'm Pipecat!\" Then stop and wait for the user.",
};

export function sanitizeValues(values: Partial<ConfigValues>): ConfigValues {
  return {
    connectionUrl: values.connectionUrl ?? DEFAULT_CONFIG_VALUES.connectionUrl,
    noUserVideo: values.noUserVideo ?? DEFAULT_CONFIG_VALUES.noUserVideo,
    whisperModel: values.whisperModel ?? DEFAULT_CONFIG_VALUES.whisperModel,
    whisperLanguage: values.whisperLanguage ?? DEFAULT_CONFIG_VALUES.whisperLanguage,
    ttsModel: values.ttsModel ?? DEFAULT_CONFIG_VALUES.ttsModel,
    ttsLanguage: values.ttsLanguage ?? DEFAULT_CONFIG_VALUES.ttsLanguage,
    ttsVoice: values.ttsVoice ?? DEFAULT_CONFIG_VALUES.ttsVoice,
    ttsSentenceStreaming:
      values.ttsSentenceStreaming ?? DEFAULT_CONFIG_VALUES.ttsSentenceStreaming,
    llmProvider: values.llmProvider ?? DEFAULT_CONFIG_VALUES.llmProvider,
    llmBaseUrl: values.llmBaseUrl ?? DEFAULT_CONFIG_VALUES.llmBaseUrl,
    llmModel: values.llmModel ?? DEFAULT_CONFIG_VALUES.llmModel,
    systemPrompt: values.systemPrompt ?? DEFAULT_CONFIG_VALUES.systemPrompt,
  };
}

export function createDefaultStore(now = new Date()): ConfigStore {
  return {
    version: 1,
    activePresetId: DEFAULT_PRESET_ID,
    presets: [
      {
        id: DEFAULT_PRESET_ID,
        name: "Default",
        values: DEFAULT_CONFIG_VALUES,
        updatedAt: now.toISOString(),
      },
    ],
  };
}

export function normalizeConfigStore(input?: Partial<ConfigStore>): ConfigStore {
  const fallback = createDefaultStore();
  if (!input || typeof input !== "object") {
    return fallback;
  }

  const presets = Array.isArray(input.presets) ? input.presets : [];
  const normalizedPresets: Preset[] = presets.map((preset, index) => {
    const id =
      typeof preset?.id === "string" && preset.id.trim().length > 0
        ? preset.id
        : index === 0
        ? DEFAULT_PRESET_ID
        : `preset-${index + 1}`;
    const name =
      typeof preset?.name === "string" && preset.name.trim().length > 0
        ? preset.name
        : id === DEFAULT_PRESET_ID
        ? "Default"
        : "Preset";
    const values = sanitizeValues(preset?.values ?? {});
    const updatedAt =
      typeof preset?.updatedAt === "string" && preset.updatedAt.trim().length > 0
        ? preset.updatedAt
        : new Date().toISOString();
    return { id, name, values, updatedAt };
  });

  if (normalizedPresets.length === 0) {
    return fallback;
  }

  const activePresetId =
    typeof input.activePresetId === "string" &&
    normalizedPresets.some((preset) => preset.id === input.activePresetId)
      ? input.activePresetId
      : normalizedPresets[0].id;

  return {
    version: typeof input.version === "number" ? input.version : 1,
    activePresetId,
    presets: normalizedPresets,
  };
}
