# Three-phase equilibrium viewer

An interactive Dash/Plotly viewer for a family of three-phase tie-simplexes inside an ellipsoidal phase boundary. The model is based on the tie-simplex description in Section 6.6 and Figure 16 of *Chemical Reviews* 123 (2023), 8945-8987.

## Run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Open the local URL printed by Dash, normally <http://127.0.0.1:8050>.

## Controls

- **Length, width, height:** Ellipsoid semi-axis radii from 0 to 10. A displayed value of zero is clamped to `0.05` internally so the hull remains numerically valid.
- **Ellipsoid theta and phi:** Shared spherical direction for the ellipsoid's first principal axis and the direction from the origin to its center.
- **Center rho:** Distance from the origin to the ellipsoid center.
- **Simplex theta:** Selects the initially displayed member of the origin-based simplex family.
- **Simplex phi:** Azimuth shared by the central rays in the family.
- **Alpha window (z1, z2):** Main-vertex z interval over which the simplex opening follows a sine curve.
- **Maximum alpha:** Peak azimuth offset of each leg, constrained below 90 degrees.
- **Main-vertex z:** Browses the reachable simplex family and highlights the nearest member in orange. Its range updates to the monotonic branch of reachable entry vertices anchored at the chosen initial theta, so each z identifies one simplex.
- **Phase regions:** Four checkboxes independently show or hide the green, red, blue-left, and blue-right region meshes. Each region also has a color menu initialized to the default color.

Ellipsoid, region appearance, and simplex controls are grouped into collapsible panels. Each render samples exactly 500 reachable simplexes along the active monotonic branch for smoother phase-region boundaries.

## Color key

| Color | Region |
| --- | --- |
| Green | Three-phase coexistence inside the swept triangle |
| Red | Segregative two-phase coexistence beyond the triangle's opposite edge |
| Blue | Associative two-phase coexistence beyond either leg edge |
| Orange | Currently selected simplex |

When alpha is zero, the triangle collapses to a 1-simplex (tie line). The transparent volumes use smoothed marching-cubes meshes of the sampled geometric partitions, clipped to the ellipsoid. The orange slider snaps to sampled simplexes and updates only the selected simplex traces, preserving the current 3D camera orientation and zoom.

This viewer is a conceptual geometry model rather than a thermodynamic solver.
