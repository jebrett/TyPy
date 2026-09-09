# Three-phase equilibrium viewer

An interactive VTK/Trame viewer for a family of three-phase tie-simplexes inside an ellipsoidal phase boundary. The model is based on the tie-simplex description in Section 6.6 and Figure 16 of *Chemical Reviews* 123 (2023), 8945-8987.

## Run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Open the local URL printed by Trame, normally <http://127.0.0.1:8080>.

## Controls

- **Length, width, height:** Ellipsoid semi-axis radii from 0 to 10. A displayed value of zero is clamped to `0.05` internally so the hull remains numerically valid.
- **Ellipsoid theta and phi:** Shared spherical direction for the ellipsoid's first principal axis and the direction from the origin to its center.
- **Center rho:** Distance from the origin to the ellipsoid center.
- **Z offset:** Independent z-axis intercept added to the directed center vector.
- **Simplex theta:** Selects the initially displayed member of the origin-based simplex family.
- **Simplex phi:** Azimuth shared by the central rays in the family.
- **Alpha window (z1, z2):** Main-vertex z interval over which the simplex opening follows a sine curve.
- **Maximum alpha:** Peak azimuth offset of each leg, constrained below 90 degrees.
- **Main-vertex z:** Browses the reachable simplex family and highlights the nearest member in orange. Its range updates to the monotonic branch of reachable entry vertices anchored at the chosen initial theta, so each z identifies one simplex.
- **Phase regions:** Switches show or hide the green three-phase region, red segregative region, and both blue associative regions. Each region type has a full-spectrum color picker initialized to its default color; the two associative meshes share one switch and color.

Ellipsoid, region appearance, and simplex controls are grouped into collapsible panels. Each render samples exactly 500 reachable simplexes along the active monotonic branch for smoother phase-region boundaries.

The viewer includes labeled Component 1/2/3 axes. Use the toolbar format menu and **Export picture** button to generate an SVG or PDF of the current camera view, then select **Download**. The local viewer is designed for one browser session because multiple views would share the same VTK camera. Each export replaces the previous temporary file, which is removed when the process exits.

## Color key

| Color | Region |
| --- | --- |
| Green | Three-phase coexistence inside the swept triangle |
| Red | Segregative two-phase coexistence beyond the triangle's opposite edge |
| Blue | Associative two-phase coexistence beyond either leg edge |
| Orange | Currently selected simplex |

When alpha is zero, the triangle collapses to a 1-simplex (tie line). The transparent volumes use direct analytical cross-sections lofted through 500 simplexes, eliminating the Cartesian voxel grid. The orange slider snaps to sampled simplexes and updates only the selected simplex actors, preserving the current 3D camera orientation and zoom.

This viewer is a conceptual geometry model rather than a thermodynamic solver.

If a control combination cannot produce four complete analytical regions, the viewer retains the last valid scene and displays the geometry error.

## Legacy Dash viewer

The previous grid/marching-cubes implementation remains available as:

```bash
python legacy_dash_app.py
```

## Rendering comparison

See [RENDERING_PROTOTYPE.md](RENDERING_PROTOTYPE.md) for the side-by-side comparison that informed the production renderer choice.
