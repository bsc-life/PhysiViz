# PhysiViz
Collection of tools for multiscale simulation visualization in 2D and 3D.


## `paraview_converter` 
This is a script that converts the MAT files of a PhysiCell output into ParaView and VTK-compatible formats (.vtu and .pvd), for 3D visualization with these different tools.

## `physicell_web_viewer`
An interactive, in-browser 3D viewer (Next.js/Three.js) for PhysiCell-family
(PhysiCell, PhysiBoSS, PhysiCelldFBA) MultiCellDS output: cells and
microenvironment over time, with playback, a population/node-activity
sparkline, click-to-isolate cell populations, and PNG/SVG/video export. Runs
locally on your own machine — clone the repo, run the three setup scripts,
and open `localhost` in your browser. A Python exporter converts raw
PhysiCell output into the compact bundles the viewer reads. See
[`physicell_web_viewer/README.md`](physicell_web_viewer/README.md) for the
quick start and the exporter CLI reference.

