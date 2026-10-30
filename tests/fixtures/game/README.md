# Frozen Courier scene fixtures

`crystal_courier.json` and `crystal_courier_3d.json` are byte-for-byte copies of
`examples/crystal_courier/scene.json` and `examples/crystal_courier_3d/scene.json`
from the VSim revision recorded in `docs/simulation/reference-manifest.json`.
They intentionally retain legacy script text to verify that import requires
explicit native replacements and never executes that text.

The native factories in `datamunge/game/courier.hpp` implement the grid rules,
3D gameplay, orbit camera and lift. These fixtures exercise exact IDs, script
mapping, geometry, prefabs, real physics and deterministic scene round trips.
