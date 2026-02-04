export type ConfigValues = {
  connectionUrl: string;
  noUserVideo: boolean;
  whisperModel: string;
  whisperLanguage: string;
  ttsModel: string;
  ttsLanguage: string;
  ttsVoice: string;
  ttsSentenceStreaming: boolean;
  qwenTts: {
    mode: "base" | "customVoice" | "voiceDesign" | "voiceCloning";
    model: string;
    language: string;
    speaker: string;
    instruct: string;
    refAudioPath: string;
    refText: string;
    seed: number;
    temperature: number;
    topK: number;
    topP: number;
    repetitionPenalty: number;
    maxTokens: number;
    doSample: boolean;
    speed: number;
    sttModel: string;
    xVectorOnlyMode: boolean;
  };
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
  qwenTts: {
    mode: "base",
    model: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    language: "english",
    speaker: "Ryan",
    instruct: "",
    refAudioPath: "",
    refText: "",
    seed: 1234,
    temperature: 0.0,
    topK: 50,
    topP: 1.0,
    repetitionPenalty: 1.05,
    maxTokens: 0,
    doSample: false,
    speed: 1.0,
    sttModel: "",
    xVectorOnlyMode: false,
  },
  llmProvider: "openai-compatible",
  llmBaseUrl: "http://127.0.0.1:1234/v1",
  llmModel: "gemma-3n-e4b-it-text",
  systemPrompt:
    "You are Pipecat, a friendly, helpful chatbot.\n\nYour input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.\n\nYour output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.\n\nRespond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.\n\nStart the conversation by saying, \"Hello, I'm Pipecat!\" Then stop and wait for the user.",
};

export function sanitizeValues(values: Partial<ConfigValues>): ConfigValues {
  const parseBoolean = (value: unknown, fallback: boolean) => {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (["true", "1", "yes", "on"].includes(normalized)) return true;
      if (["false", "0", "no", "off"].includes(normalized)) return false;
    }
    return fallback;
  };
  const qwenValues = values.qwenTts ?? {};
  const rawTtsModel = values.ttsModel ?? DEFAULT_CONFIG_VALUES.ttsModel;
  const isLegacyQwenModel =
    typeof rawTtsModel === "string" && rawTtsModel.startsWith("mlx-community/Qwen3-TTS-");
  const normalizedTtsModel = isLegacyQwenModel
    ? "mlx-community/Qwen3-TTS"
    : rawTtsModel;
  const qwenModeRaw = qwenValues.mode ?? DEFAULT_CONFIG_VALUES.qwenTts.mode;
  const qwenModelFromMode = (mode: ConfigValues["qwenTts"]["mode"], model: string) => {
    const size = model.includes("1.7B") ? "1.7B" : "0.6B";
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
  const qwenModeFromModel = (model: string): ConfigValues["qwenTts"]["mode"] => {
    if (model.includes("VoiceDesign")) return "voiceDesign";
    if (model.includes("CustomVoice")) return "customVoice";
    return "base";
  };
  const rawQwenLanguage =
    typeof qwenValues.language === "string" ? qwenValues.language : "";
  const qwenLanguageNormalized = rawQwenLanguage
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
  const qwenLanguageAliases: Record<string, string> = {
    en: "english",
    english: "english",
    pt: "portuguese",
    "pt-br": "portuguese",
    pt_br: "portuguese",
    ptbr: "portuguese",
    "portuguese (brazil)": "portuguese",
    portuguese: "portuguese",
    chinese: "chinese",
    zh: "chinese",
    japanese: "japanese",
    ja: "japanese",
    korean: "korean",
    ko: "korean",
    french: "french",
    fr: "french",
    german: "german",
    de: "german",
    italian: "italian",
    it: "italian",
    spanish: "spanish",
    es: "spanish",
    russian: "russian",
    ru: "russian",
    auto: "auto",
  };
  const normalizedQwenLanguage =
    qwenLanguageAliases[qwenLanguageNormalized] ?? qwenLanguageNormalized;
  const seedCandidate = qwenValues.seed ?? DEFAULT_CONFIG_VALUES.qwenTts.seed;
  const seed =
    typeof seedCandidate === "number"
      ? seedCandidate
      : Number.parseInt(String(seedCandidate), 10);
  const temperatureCandidate =
    qwenValues.temperature ?? DEFAULT_CONFIG_VALUES.qwenTts.temperature;
  const temperature =
    typeof temperatureCandidate === "number"
      ? temperatureCandidate
      : Number.parseFloat(String(temperatureCandidate));
  const topKCandidate = qwenValues.topK ?? DEFAULT_CONFIG_VALUES.qwenTts.topK;
  const topK =
    typeof topKCandidate === "number"
      ? topKCandidate
      : Number.parseInt(String(topKCandidate), 10);
  const topPCandidate = qwenValues.topP ?? DEFAULT_CONFIG_VALUES.qwenTts.topP;
  const topP =
    typeof topPCandidate === "number"
      ? topPCandidate
      : Number.parseFloat(String(topPCandidate));
  const repetitionPenaltyCandidate =
    qwenValues.repetitionPenalty ??
    DEFAULT_CONFIG_VALUES.qwenTts.repetitionPenalty;
  const repetitionPenalty =
    typeof repetitionPenaltyCandidate === "number"
      ? repetitionPenaltyCandidate
      : Number.parseFloat(String(repetitionPenaltyCandidate));
  const maxTokensCandidate =
    qwenValues.maxTokens ?? DEFAULT_CONFIG_VALUES.qwenTts.maxTokens;
  const maxTokens =
    typeof maxTokensCandidate === "number"
      ? maxTokensCandidate
      : Number.parseInt(String(maxTokensCandidate), 10);
  const speedCandidate = qwenValues.speed ?? DEFAULT_CONFIG_VALUES.qwenTts.speed;
  const speed =
    typeof speedCandidate === "number"
      ? speedCandidate
      : Number.parseFloat(String(speedCandidate));
  const doSample = parseBoolean(
    qwenValues.doSample,
    DEFAULT_CONFIG_VALUES.qwenTts.doSample
  );
  const sttModel =
    typeof qwenValues.sttModel === "string" ? qwenValues.sttModel : "";
  const xVectorOnlyMode = parseBoolean(
    qwenValues.xVectorOnlyMode,
    DEFAULT_CONFIG_VALUES.qwenTts.xVectorOnlyMode
  );
  const normalizedQwenModel = qwenModelFromMode(
    isLegacyQwenModel ? qwenModeFromModel(rawTtsModel as string) : qwenModeRaw,
    isLegacyQwenModel
      ? (rawTtsModel as string)
      : qwenValues.model ?? DEFAULT_CONFIG_VALUES.qwenTts.model
  );

  return {
    connectionUrl: values.connectionUrl ?? DEFAULT_CONFIG_VALUES.connectionUrl,
    noUserVideo: values.noUserVideo ?? DEFAULT_CONFIG_VALUES.noUserVideo,
    whisperModel: values.whisperModel ?? DEFAULT_CONFIG_VALUES.whisperModel,
    whisperLanguage: values.whisperLanguage ?? DEFAULT_CONFIG_VALUES.whisperLanguage,
    ttsModel: normalizedTtsModel ?? DEFAULT_CONFIG_VALUES.ttsModel,
    ttsLanguage: values.ttsLanguage ?? DEFAULT_CONFIG_VALUES.ttsLanguage,
    ttsVoice: values.ttsVoice ?? DEFAULT_CONFIG_VALUES.ttsVoice,
    ttsSentenceStreaming:
      values.ttsSentenceStreaming ?? DEFAULT_CONFIG_VALUES.ttsSentenceStreaming,
    qwenTts: {
      mode: isLegacyQwenModel
        ? qwenModeFromModel(rawTtsModel as string)
        : qwenModeRaw,
      model: normalizedQwenModel,
      language:
        normalizedQwenLanguage || DEFAULT_CONFIG_VALUES.qwenTts.language,
      speaker: qwenValues.speaker ?? DEFAULT_CONFIG_VALUES.qwenTts.speaker,
      instruct: qwenValues.instruct ?? DEFAULT_CONFIG_VALUES.qwenTts.instruct,
      refAudioPath:
        qwenValues.refAudioPath ?? DEFAULT_CONFIG_VALUES.qwenTts.refAudioPath,
      refText: qwenValues.refText ?? DEFAULT_CONFIG_VALUES.qwenTts.refText,
      seed: Number.isFinite(seed) ? seed : DEFAULT_CONFIG_VALUES.qwenTts.seed,
      temperature: Number.isFinite(temperature)
        ? temperature
        : DEFAULT_CONFIG_VALUES.qwenTts.temperature,
      topK: Number.isFinite(topK) ? topK : DEFAULT_CONFIG_VALUES.qwenTts.topK,
      topP: Number.isFinite(topP) ? topP : DEFAULT_CONFIG_VALUES.qwenTts.topP,
      repetitionPenalty: Number.isFinite(repetitionPenalty)
        ? repetitionPenalty
        : DEFAULT_CONFIG_VALUES.qwenTts.repetitionPenalty,
      maxTokens: Number.isFinite(maxTokens)
        ? Math.max(0, maxTokens)
        : DEFAULT_CONFIG_VALUES.qwenTts.maxTokens,
      doSample,
      speed: Number.isFinite(speed) ? speed : DEFAULT_CONFIG_VALUES.qwenTts.speed,
      sttModel,
      xVectorOnlyMode,
    },
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
