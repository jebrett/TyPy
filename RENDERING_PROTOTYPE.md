# Smooth rendering comparison

The experimental `prototype_app.py` compares two alternatives to the production viewer's blurred 30 cubed marching-cubes masks. It does not replace the Dash/Plotly entry point or route production rendering through VTK.

## Run

```bash
python -m venv .venv-prototype
source .venv-prototype/bin/activate
python -m pip install -r requirements-prototype.txt
python prototype_app.py
```

The Trame application opens a local browser with synchronized side-by-side VTK views:

- **Direct analytical loft:** Computes the exact plane/ellipsoid ellipse at each of 500 simplexes, clips it with the four barycentric region definitions, and lofts the resulting polygons.
- **VTK Surface Nets:** Evaluates the existing classifier on a denser multi-label grid and extracts shared interfaces with `vtkSurfaceNets3D`.

Use the preset and Surface Nets resolution menus, select **Rebuild comparison**, and inspect the metrics panel. Region colors, visibility, opacity, camera, and wireframe display are shared so the geometry is compared under identical presentation.

## Default benchmark

Measured locally with 500 simplexes, 64 perimeter points per direct cross-section, a 64 cubed Surface Nets grid, and 1,200 deterministic classifier probes:

| Metric | Direct analytical loft | VTK Surface Nets |
| --- | ---: | ---: |
| Geometry generation | 1.21 s | 1.40 s |
| Classifier agreement | 99.33% | 96.67% |
| Unclassified gap probes | 0.42% | 1.33% |
| Overlapping region probes | 0.25% | 1.50% |
| Triangles, all regions | 256,000 | 100,192 |
| Raw mesh arrays | 10.74 MiB | 3.63 MiB |
| Hull overrun | below 2e-15 | below 1e-5 |

Times are machine-dependent. The app reports fresh measurements for every rebuild.

## Evaluation

### Direct analytical loft

**Advantages**

- Smoothness follows the 500 simplex sections rather than a Cartesian voxel grid.
- Closely reproduces the current barycentric classifier.
- Produces watertight region meshes with analytical ellipsoid support-point ends instead of artificial flat caps.
- Geometry quality is independent of ellipsoid rotation relative to world axes.

**Trade-offs**

- Larger meshes at the current 64-point perimeter setting.
- Requires explicit splitting and capping when a region cross-section becomes empty.
- At alpha zero, it intentionally uses the same 0.25-degree limiting classification triangle as the production model.

### VTK Surface Nets

**Advantages**

- Preserves a sampled version of the existing volume classifier.
- Contours all labels together, producing shared interfaces rather than four unrelated marching-cubes passes.
- Produces substantially smaller meshes than the direct prototype.
- Offers a straightforward quality control through 64, 96, and 128 cubed grids.

**Trade-offs**

- Residual quality remains tied to Cartesian grid resolution.
- Higher resolutions increase classification time and memory roughly with the number of grid points.
- Constrained smoothing can move interfaces away from the exact classified boundary.

## Recommendation

Use the **direct analytical loft** as the preferred production direction. It removes the source of the blockiness instead of hiding it with denser voxels, is more accurate against the current classifier, and is faster than a 64 cubed Surface Nets build in the default benchmark.

Retain **Surface Nets as the fallback** for configurations where direct cross-section topology becomes unstable. Before production migration, reduce direct mesh payload by adaptive perimeter sampling and test its topology across a wider randomized parameter sweep.
