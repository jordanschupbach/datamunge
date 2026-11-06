\name{native-cad}
\alias{native-cad}
\alias{CadScene}
\alias{CadScene_from_file}
\alias{CadScene_from_urdf}
\alias{CadSolid_box}
\alias{CadSolid_cylinder}
\alias{CadSolid_sphere}
\alias{CadSolid_torus}
\title{Code-driven CAD modelling, import and offline rendering}
\description{
Owned CAD solids and scenes use the canonical C++ modelling kernel and CPU
renderer, with no editor window or graphics context. The complete kernel is
available to C++ through \code{dm::cad}; these generated R bindings expose
its small scene and solid facade.
}
\details{
\code{CadSolid_box(width, depth, height)} starts at the origin with X/Y/Z
extents. Cylinders use +Z, spheres and tori are centred at the origin.
All dimensions must be positive and finite. Use consistent model units.
Solids offer \code{$translated(x,y,z)}, \code{$rotated(x,y,z,radians)},
\code{$united(other)}, \code{$subtracted(other)}, \code{$intersected(other)},
\code{$filleted(edges,radius)}, \code{$chamfered(edges,setback)} and
\code{$shelled(thickness)}. These return new values and preserve their inputs.
\code{$edges()} and \code{$faces()} return IDs for the modifiers.
\code{$volume(chord)} measures the trimmed tessellated boundary, so curved
geometry has tolerance-dependent accuracy. \code{$valid()} checks the B-rep.
\code{$save(path)} writes STEP, STL, OBJ, glTF or SVG by filename extension.
The source kernel refuses unsupported operations, including general booleans
with coincident faces, with an R error rather than silently altering geometry.

\code{CadScene()} starts empty. \code{$add(solid,color,name)} appends an owned
mesh and returns its object ID; \code{color} is a hex RGB/RGBA string.
\code{$primitive(kind)} generates cube/sphere/cylinder/cone/plane/torus/wedge
polygon meshes. \code{$size(width,height)} and \code{$samples(n)} set the
image size and 1--4 samples per axis. \code{$render(path)} writes a PNG, JPEG
or PPM image with automatic framing. \code{$camera(x,y,z,tx,ty,tz)} sets an
explicit Z-up view and \code{$fit()} resumes automatic framing.
\code{$style(name)} accepts shaded, unlit, wireframe or shaded-wireframe.
\code{$light(x,y,z,color,intensity)} adds a directional light;
\code{$shadows(FALSE)} disables first-light shadows.
\code{$pick_object(x,y)} returns an object ID or -1 for background.
\code{$save_gltf(path)} exports meshes, textures, transforms and lights.

\code{CadScene_from_file(path)} reads STEP/STP, IGES/IGS, STL, OBJ/MTL,
glTF/GLB, IQM, VOX, M3D, native Blender meshes or mepcad documents.
STEP/IGES files with unsupported geometry are rejected by this facade;
C++ readers expose detailed partial-import reports. Exact CAD data are
converted to meshes for the scene.
\code{CadScene_from_urdf(path)} loads a rooted robot visual tree and its assets.
\code{$joint_names()} lists independent movable joints and
\code{$set_joint(name,value)} applies a pose within joint limits, including
mimic relations. Rotations use radians, prismatic joints metres.
Inertial/dynamics data are not evaluated by this visual loader.
All native failures become R errors. Owned solids/scenes release storage
when collected; a scene does not borrow an added solid's lifetime.
}
\examples{
solid <- CadSolid_box(2, 3, 4)
solid$volume() # 24
scene <- CadScene()
invisible(scene$add(solid, "#4080d9", "block"))
invisible(scene$size(160L, 120L))
path <- tempfile(fileext=".png")
invisible(scene$render(path))
file.exists(path) # TRUE
unlink(path)
}
