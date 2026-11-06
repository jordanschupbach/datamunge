# CAD port from mep

The port preserves the reusable CAD and polygon modelling functionality in
`~/projects/mep`, with canonical C++ types under `dm::cad`. Editor windows,
mouse tools, Lua/editor RPC routing and subprocess UI sessions are application
integration; their modelling operations are exposed directly to C++ callers.
Mep's FEM solver is a separate subsystem, outside this CAD port.

`cad-port-manifest.json` records the source revision and hashes of the actual
source files copied, including any uncommitted source changes. The MIT notice
is retained in `third_party_licenses/mep-cad-LICENSE`. Source algorithms and
their stated limits are retained; the port does not claim broader STEP/IGES
coverage or general-purpose boolean robustness than the source kernel.

## Feature inventory and acceptance

- [x] Double precision vectors, transforms, tolerances, robust predicates.
- [x] B-spline/NURBS curves and surfaces, analytic geometry and intersections.
- [x] B-rep topology, p-curves, classification, validation, shared-edge tessellation.
- [x] Sketch entities, profiles, constraints and nonlinear sketch solving.
- [x] Feature trees, extrusion/revolve/loft/sweep, persistent names and rebuilds.
- [x] Booleans, fillet/chamfer/shell/draft, patterns and assembly mating.
- [x] Native parametric document and sketch serialization.
- [x] STEP read/write, IGES read, DXF read/write, STL read/write, OBJ/glTF export.
- [x] Drawing views with sampled hidden-line removal and SVG output.
- [x] Polygon editing, procedural primitives, lathe/custom mesh construction.
- [x] OBJ/MTL, glTF/GLB, IQM, VOX, M3D and native Blender import.
- [x] Materials, textures, lights, object transforms and scene export.
- [x] Headless camera, shaded/texture/wireframe rendering, depth and picking.
- [x] Convenient solid construction, operation chaining and file API.
- [x] Independent programs/examples, public API documentation and Python/R bindings.
- [x] Trim-aware body volume, area, centroid and inertia from the source session helper.
- [x] All source regressions plus adapter/rendering/error tests pass.

The [user guide](cad.md) records inherited operation/format limits. The port
adds URDF visual loading and forward kinematics for the requested real robot
example; it does not require Blender or ROS to render that example.

## Acceptance evidence

- All 19 original CAD/polygon/Blender suites and two new API/format suites pass.
  The API suite checks analytic volumes, centroids and inertia, geometry
  ownership, export/import, render pixels/depth/IDs, textures, transparency,
  deterministic threading, URDF kinematics/mimic limits and invalid inputs.
  Dedicated binary-format oracles also check IQM, VOX, M3D and GLB imports,
  explicit glTF matrix placement, and truncated IQM tables rejected before
  native allocations or reads. Four API/format/polygon/Blender suites also pass
  AddressSanitizer and UndefinedBehaviorSanitizer with leak detection disabled.
- Fourteen Python tests and 57 R expectations pass, including the existing
  matrix/algebra/spline checks affected by the generated binding regeneration.
- All six programs in the [spline comparison](mepml/examples/stats/regression/hierarchical_splines.mepml)
  and [DevArm workflow](mepml/examples/cad/seeed_devarm.mepml) execute from their
  report directories. Both C++/Python/R groups produce identical cached
  Markdown and image bytes.
  DevArm uses 34 real mesh parts, 430,912 triangles and seven independent joints;
  the separate exact STEP link tessellates to 12,125 triangles and exports glTF.
- The asset manifest pins and verifies 38 files by SHA-256 and byte size. Large
  source data stay under ignored `build/`; the six compact cached PNGs ship with
  the report. The complete editable C++ program is also a standard examples target.
- The four new facade/render/URDF headers have 132/132 documented definitions;
  five selected clangd definition hovers and 36 complete API documentation
  programs pass, alongside executable installed R help. Original kernel prose
  is retained; it is not claimed as a completed whole-module Doxygen audit.
- The native shared library and installed Python/R core bindings build against
  the checkout. A real setuptools source translation unit verifies that CAD
  uses `-ffp-contract=off`, matching CMake and the R source build rules.
- An independent CMake consumer builds and runs against the installed library,
  including mass properties and rendering. The public headers also compile
  alongside the original mep scene and Blender headers without guard collisions.
- Python and R source archives contain byte-identical copies of all 52 CAD
  translation units, public/private headers, generated core bindings and the
  relevant license/help files. Installed-library consumption is checked
  independently of the source-archive payload checks.

Detailed logs, saved native trees, exports and standalone example renders:
`build/cad-port/`.

## Design

Public kernel headers live in `include/datamunge/cad/`; implementations and
private import/codec adapters live in `src/datamunge/cad/`. Precise CAD geometry
uses doubles; polygon scenes retain the source's float arrays and 32-bit indices.
Floating point contraction stays disabled for CAD sources, as in the source
kernel. Imported image codecs and mesh readers have private namespaces.
The mesh import and primitive adapter owns CPU arrays and texture bytes, with
no window, global renderer cache or GPU round trip. Rendering accepts ordinary
scene values so callers can render several cameras and animate in code.
