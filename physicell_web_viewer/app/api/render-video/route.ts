import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";

/**
 * Renders a video directly from a sequence of still frames (JPEG or PNG),
 * one ffmpeg pass, no browser-side real-time capture involved.
 *
 * This replaces the older pipeline of: MediaRecorder captures the canvas in
 * real time (wall-clock bound by playback fps, and at the mercy of dropped
 * frames whenever the browser/GPU falls behind) into a lossy VP8/VP9 WebM,
 * which -- for an MP4 export -- then got uploaded a *second* time and
 * re-encoded losslessly-in-spirit-but-not-in-practice (re-compressing an
 * already-lossy source) via /api/transcode-mp4. Capturing frames
 * deterministically (one real GPU-rendered still per requested frame,
 * waited-for rather than timed) and encoding them here in a single ffmpeg
 * pass is faster (no wall-clock pacing), exact (every requested frame is in
 * the output, never dropped/duplicated), and higher quality (one encode
 * from near-lossless stills instead of two lossy encodes stacked).
 */

async function runFfmpeg(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    p.stderr.on("data", (d) => {
      err += d.toString();
    });
    p.on("error", reject);
    p.on("close", (code) => {
      if (code === 0) resolve(err);
      else reject(new Error(`ffmpeg failed (${code}): ${err.slice(-3000)}`));
    });
  });
}

export async function POST(req: Request) {
  const form = await req.formData();
  const fps = Math.max(1, Math.min(60, Number(form.get("fps") ?? 10)));
  const quality = String(form.get("quality") ?? "high") as "standard" | "high" | "ultra";
  const format = String(form.get("format") ?? "mp4") as "mp4" | "webm";
  const ext = String(form.get("ext") ?? "jpg") as "jpg" | "png";

  const frameFiles: File[] = [];
  for (const [key, value] of form.entries()) {
    if (key.startsWith("frame_") && value instanceof File) {
      frameFiles.push(value);
    }
  }
  if (frameFiles.length === 0) {
    return NextResponse.json({ error: "No frames uploaded." }, { status: 400 });
  }

  const id = randomUUID();
  const tmpDir = path.join(os.tmpdir(), `viz-frames-${id}`);
  const outPath = path.join(os.tmpdir(), `viz-${id}.${format}`);

  try {
    await fs.mkdir(tmpDir, { recursive: true });
    // Frame field names are frame_00000, frame_00001, ... in capture order
    // (set client-side) -- write them out under the same zero-padded index
    // so ffmpeg's image2 demuxer picks them up in the right sequence.
    await Promise.all(
      frameFiles.map(async (file, i) => {
        const buf = Buffer.from(await file.arrayBuffer());
        const name = `frame_${String(i).padStart(5, "0")}.${ext}`;
        await fs.writeFile(path.join(tmpDir, name), buf);
      })
    );

    const crfByQuality = { standard: "20", high: "16", ultra: "13" } as const;
    const presetByQuality = { standard: "medium", high: "slow", ultra: "veryslow" } as const;

    const args = [
      "-y",
      "-framerate",
      String(fps),
      "-i",
      path.join(tmpDir, `frame_%05d.${ext}`),
    ];
    if (format === "mp4") {
      args.push(
        "-c:v",
        "libx264",
        "-preset",
        presetByQuality[quality],
        "-crf",
        crfByQuality[quality],
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart"
      );
    } else {
      // VP9 encoded offline (not real-time) is comfortably higher quality at
      // the same size than the old real-time browser VP8 MediaRecorder path.
      args.push(
        "-c:v",
        "libvpx-vp9",
        "-crf",
        crfByQuality[quality],
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p"
      );
    }
    args.push(outPath);

    await runFfmpeg(args);

    const out = await fs.readFile(outPath);
    return new Response(new Uint8Array(out), {
      headers: {
        "content-type": format === "mp4" ? "video/mp4" : "video/webm",
        "content-disposition": `attachment; filename="visualization.${format}"`,
      },
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Video render failed" },
      { status: 500 }
    );
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    await fs.unlink(outPath).catch(() => {});
  }
}
