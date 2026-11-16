import SimulationViewer from "@/components/SimulationViewer";

export default function Home() {
  return (
    <main className="h-screen w-screen bg-zinc-950 text-zinc-100">
      {/* Placeholder only -- public/data/ ships empty (see its README.md); */}
      {/* SimulationViewer auto-redirects to the first exported dataset it */}
      {/* finds via /api/output-folders, or shows a "run a script" message. */}
      <SimulationViewer dataPrefix="/data/_no_dataset_exported_yet" />
    </main>
  );
}
