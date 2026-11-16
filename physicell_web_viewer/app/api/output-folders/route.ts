import { NextResponse } from "next/server";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";

type OutputFolderEntry = {
  name: string;
  exported: boolean;
  description?: string;
};

/** Peek at an exported dataset's manifest.json for its optional `description` field. */
function readDescription(dataRoot: string, name: string): string | undefined {
  try {
    const raw = readFileSync(path.join(dataRoot, name, "manifest.json"), "utf-8");
    const parsed = JSON.parse(raw) as { description?: string };
    return typeof parsed.description === "string" ? parsed.description : undefined;
  } catch {
    return undefined;
  }
}

export async function GET() {
  // Point this at your own PhysiCell simulations root, e.g.:
  //   PHYSICELL_ROOT=/path/to/simulations npm run dev
  // Folders here that also have an exported manifest.json under public/data/
  // are marked "exported"; folders without one show as "(not exported)".
  const physiCellRoot = process.env.PHYSICELL_ROOT ?? path.resolve(process.cwd(), "..");
  const dataRoot = path.resolve(process.cwd(), "public", "data");

  try {
    const physioEntries = readdirSync(physiCellRoot, { withFileTypes: true })
      .filter((d) => d.isDirectory() && d.name.startsWith("output"))
      .map<OutputFolderEntry>((d) => {
        const exported = existsSync(path.join(dataRoot, d.name, "manifest.json"));
        return {
          name: d.name,
          exported,
          description: exported ? readDescription(dataRoot, d.name) : undefined,
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));

    // Also include already-exported datasets even if PhysiCell root is different/unavailable.
    const exportedOnly = readdirSync(dataRoot, { withFileTypes: true })
      .filter((d) => d.isDirectory() && d.name.startsWith("output"))
      .filter((d) => existsSync(path.join(dataRoot, d.name, "manifest.json")))
      .map<OutputFolderEntry>((d) => ({
        name: d.name,
        exported: true,
        description: readDescription(dataRoot, d.name),
      }));

    const byName = new Map<string, OutputFolderEntry>();
    for (const e of [...physioEntries, ...exportedOnly]) byName.set(e.name, e);
    const entries = [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json({ root: physiCellRoot, outputs: entries });
  } catch (error) {
    // Fallback: at least list exported datasets from public/data.
    const entries = readdirSync(dataRoot, { withFileTypes: true })
      .filter((d) => d.isDirectory() && d.name.startsWith("output"))
      .filter((d) => existsSync(path.join(dataRoot, d.name, "manifest.json")))
      .map<OutputFolderEntry>((d) => ({
        name: d.name,
        exported: true,
        description: readDescription(dataRoot, d.name),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
    return NextResponse.json({
      root: physiCellRoot,
      outputs: entries,
      error: error instanceof Error ? error.message : "Failed to list folders",
    });
  }
}
