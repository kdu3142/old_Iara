import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function normalizeBaseUrl(baseUrl: string) {
  const trimmed = baseUrl.trim();
  if (!trimmed) return "";
  return trimmed.endsWith("/v1") ? trimmed.slice(0, -3) : trimmed;
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { baseUrl?: string };
    const baseUrl = normalizeBaseUrl(body.baseUrl ?? "");
    if (!baseUrl) {
      return NextResponse.json(
        { error: "Missing baseUrl" },
        { status: 400 }
      );
    }

    const response = await fetch(`${baseUrl}/api/tags`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      return NextResponse.json(
        { error: `Ollama returned ${response.status}` },
        { status: 502 }
      );
    }

    const data = (await response.json()) as {
      models?: Array<{ name?: string }>;
    };
    const models =
      data.models?.map((model) => model.name).filter(Boolean) ?? [];

    return NextResponse.json({ models });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
