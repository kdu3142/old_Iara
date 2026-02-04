import { promises as fs } from "fs";
import path from "path";

import {
  ConfigStore,
  createDefaultStore,
  normalizeConfigStore,
} from "@/lib/configDefaults";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CONFIG_DIR = path.join(process.cwd(), ".config");
const CONFIG_PATH = path.join(CONFIG_DIR, "voice-ui-config.json");

async function writeConfigFile(store: ConfigStore) {
  await fs.mkdir(CONFIG_DIR, { recursive: true });
  await fs.writeFile(CONFIG_PATH, JSON.stringify(store, null, 2));
}

async function readConfigFile() {
  try {
    const raw = await fs.readFile(CONFIG_PATH, "utf8");
    const parsed = JSON.parse(raw);
    return normalizeConfigStore(parsed);
  } catch {
    const defaultStore = createDefaultStore();
    await writeConfigFile(defaultStore);
    return defaultStore;
  }
}

export async function GET() {
  const store = await readConfigFile();
  return Response.json(store, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}

export async function POST(request: Request) {
  const body = await request.json();
  const store = normalizeConfigStore(body);
  await writeConfigFile(store);
  return Response.json(store, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
