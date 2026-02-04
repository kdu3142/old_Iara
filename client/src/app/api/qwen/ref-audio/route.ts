import { promises as fs } from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CONFIG_DIR = path.join(process.cwd(), ".config");
const RECORDINGS_DIR = path.join(CONFIG_DIR, "qwen-ref-audio");

const AUDIO_CONTENT_TYPES: Record<string, string> = {
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".m4a": "audio/mp4",
  ".aac": "audio/aac",
  ".flac": "audio/flac",
  ".ogg": "audio/ogg",
  ".webm": "audio/webm",
};

const normalizePathParam = (raw: string) => {
  const trimmed = raw.trim();
  if (trimmed.startsWith("file://")) {
    return fileURLToPath(trimmed);
  }
  return trimmed;
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const rawPath = url.searchParams.get("path");
  if (!rawPath) {
    return Response.json({ error: "Missing path parameter." }, { status: 400 });
  }

  let resolvedPath = normalizePathParam(rawPath);
  if (!path.isAbsolute(resolvedPath)) {
    resolvedPath = path.resolve(process.cwd(), resolvedPath);
  }

  const ext = path.extname(resolvedPath).toLowerCase();
  const contentType = AUDIO_CONTENT_TYPES[ext];
  if (!contentType) {
    return Response.json({ error: "Unsupported audio file type." }, { status: 400 });
  }

  try {
    const stats = await fs.stat(resolvedPath);
    if (!stats.isFile()) {
      return Response.json({ error: "Path is not a file." }, { status: 400 });
    }
    const buffer = await fs.readFile(resolvedPath);
    return new Response(buffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : "File not found." },
      { status: 404 }
    );
  }
}

export async function POST(request: Request) {
  const formData = await request.formData();
  const audio = formData.get("audio");

  if (!(audio instanceof File)) {
    return Response.json({ error: "Missing audio file." }, { status: 400 });
  }

  await fs.mkdir(RECORDINGS_DIR, { recursive: true });

  const extFromName = audio.name.split(".").pop()?.toLowerCase();
  const safeExt =
    extFromName && /^[a-z0-9]+$/.test(extFromName) ? extFromName : "wav";
  const fileName = `qwen-ref-${Date.now()}-${crypto.randomUUID().slice(0, 8)}.${safeExt}`;
  const filePath = path.join(RECORDINGS_DIR, fileName);
  const buffer = Buffer.from(await audio.arrayBuffer());

  await fs.writeFile(filePath, buffer);

  return Response.json({ path: filePath, fileName });
}
