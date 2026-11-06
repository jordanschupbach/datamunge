#!/usr/bin/env python3
"""Check representative Doxygen descriptions through clangd at source definitions.

Use a temporary compilation database so the check follows the current checkout,
not an installed library or another developer's absolute include paths. This
checks the language server; mep's graphical K integration needs a separate UI check.
"""

import argparse
import json
import os
import re
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import time

## Checkout whose definitions are checked by the language server.
ROOT = Path(__file__).resolve().parents[1]
## Kind, definition file, unique declaration text, symbol token, and expected prose.
TARGETS = (
    ("native-urdf-model", "include/datamunge/sim/articulation.hpp", "class UrdfModel {", "UrdfModel", "independent of rendering and MuJoCo"),
    ("native-urdf-import", "include/datamunge/sim/articulation.hpp", "static UrdfModel load(", "load", "Parse a complete physical URDF"),
    ("native-velocity-servo", "include/datamunge/sim/world.hpp", "void set_joint_motor(", "set_joint_motor", "implicit finite-effort"),
    ("native-mimic-gear", "include/datamunge/sim/world.hpp", "void add_gear(", "add_gear", "dynamic gear constraint"),
    ("native-engine-selection", "include/datamunge/sim/physics_backend.hpp", "enum class ArticulatedBackend", "ArticulatedBackend", "no silent fallback"),
    ("articulated-mode", "include/datamunge/sim/fruit_stream.hpp", "  bool articulated_physics{false};", "articulated_physics", "physical finger contacts"),
    ("articulated-timestep", "include/datamunge/sim/fruit_stream.hpp", "double physics_timestep{1./480};", "physics_timestep", "Internal articulated step"),
    ("measured-robot-joints", "include/datamunge/cad/robot.hpp", "void set_measured_joints(const std::vector", "set_measured_joints", "one transaction"),
    ("rl-native-policy", "include/datamunge/rl/discrete.hpp", "class DiscretePolicy {", "DiscretePolicy", "optimizer/replay/RNG"),
    ("rl-duration-training", "include/datamunge/rl/discrete.hpp", "DiscreteTrainingResult train_discrete(", "train_discrete", "Truncations bootstrap"),
    ("rl-time-limit", "include/datamunge/rl/discrete.hpp", "bool truncated{false};", "truncated", "cut rollout/GAE"),
    ("rl-workcell-features", "include/datamunge/sim/fruit_stream.hpp", "rl::DiscreteObservation learning_observation() const;", "learning_observation", "25 bounded features"),
    ("rl-workcell-curriculum", "include/datamunge/sim/fruit_stream.hpp", "rl::DiscreteEnvironment stream_learning_environment(", "stream_learning_environment", "curriculum"),
    ("fruit-feasible-actions", "include/datamunge/sim/fruit_stream.hpp", "std::array<bool, 5> action_mask() const;", "action_mask", "future arrivals"),
    ("fruit-masked-policy", "include/datamunge/sim/fruit_stream.hpp", "StreamAction action(std::size_t state, const std::array<bool, 5>& allowed) const;", "action", "masked values are ignored"),
    ("fruit-credit-traces", "include/datamunge/sim/fruit_stream.hpp", "double trace_decay{0};", "trace_decay", "Watkins replacing"),
    ("fruit-infeed-watch", "include/datamunge/sim/fruit_stream.hpp", "bool watch_infeed{false};", "watch_infeed", "0.125-second empty"),
    ("workcell-presentation", "include/datamunge/sim/workcell_visuals.hpp", "cad::Scene workcell_presentation(", "workcell_presentation", "calibrated sensor"),
    ("workcell-presentation-options", "include/datamunge/sim/workcell_visuals.hpp", "cad::RenderOptions workcell_presentation_options(", "workcell_presentation_options", "analytic studio reflections"),
    ("studio-environment-control", "include/datamunge/cad/render.hpp", "double studio_environment{0};", "studio_environment", "softbox reflection strength"),
    ("wrist-associated-inspection", "include/datamunge/sim/devarm_camera.hpp", "WristInspection inspect_actors(", "inspect_actors", "total worker budget"),
    ("cad-cuda-renderer", "include/datamunge/cad/render.hpp", "class CudaRenderer {", "CudaRenderer", "Float32 device arithmetic"),
    ("cad-cuda-constructor", "include/datamunge/cad/render.hpp", "explicit CudaRenderer(std::uint32_t device = 0);", "CudaRenderer", "persistent CUDA context"),
    ("ray-bounds-cache", "src/datamunge/cad/detail/ray_box.hpp", "struct BoundsRay {", "BoundsRay", "reciprocal directions"),
    ("ray-bounds-query", "src/datamunge/cad/detail/ray_box.hpp", "box_reciprocal(const Vec3d&", "box_reciprocal", "near-touching"),
    ("compact-triangle-query", "src/datamunge/cad/render.cpp", "struct QueryTriangle {", "QueryTriangle", "leaf-order intersection"),
    ("prepared-scene", "include/datamunge/cad/render.hpp", "class PreparedScene {", "PreparedScene", "immutable transformed geometry"),
    ("prepared-scene-actors", "include/datamunge/cad/render.hpp", "RenderResult render(const Scene& actors,", "render", "mutual shadows"),
    ("video-writer", "include/datamunge/cad/video.hpp", "class VideoWriter {", "VideoWriter", "without intermediate PNGs"),
    ("video-finish", "include/datamunge/cad/video.hpp", "void finish();", "finish", "publish the completed MP4"),
    ("wrist-workers", "include/datamunge/sim/devarm_camera.hpp", "void set_render_threads(int threads);", "set_render_threads", "worker budget"),
    ("workcell-visual-profile", "include/datamunge/sim/workcell_visuals.hpp", "enum class WorkcellVisualStyle", "WorkcellVisualStyle", "collision geometry"),
    ("workcell-render-settings", "include/datamunge/sim/workcell_visuals.hpp", "workcell_render_options(WorkcellVisualStyle style,", "workcell_render_options", "linear-light microfacet"),
    ("cad-text-overlay", "include/datamunge/cad/render.hpp", "void draw_text(", "draw_text", "bitmap text"),
    ("fruit-stream", "include/datamunge/sim/fruit_stream.hpp", "class FruitStreamEnvironment {", "FruitStreamEnvironment", "between-pick reset"),
    ("fruit-stream-learning", "include/datamunge/sim/fruit_stream.hpp", "StreamTrainingResult train_fruit_stream(", "train_fruit_stream", "measured option transitions"),
    ("fruit-domain", "include/datamunge/sim/conveyor.hpp", "struct FruitAppearance {", "FruitAppearance", "shared by both classes"),
    ("wrist-inspection", "include/datamunge/sim/devarm_camera.hpp", "WristInspection inspect(", "inspect", "RGB difference"),
    ("wrist-camera", "include/datamunge/sim/devarm_camera.hpp", "class DevArmWristCamera {", "DevArmWristCamera", "never track fruit coordinates"),
    ("wrist-pose", "include/datamunge/sim/devarm_camera.hpp", "cad::Camera pose(const PickObservation& frame) const;", "pose", "URDF joint pose alone"),
    ("mobile-classifier", "include/datamunge/nn/mobile_classifier.hpp", "class MobileNetClassifier {", "MobileNetClassifier", "all thirteen depthwise/pointwise blocks"),
    ("mobile-rgb-predict", "include/datamunge/nn/mobile_classifier.hpp", "Tensor predict(const Tensor& images);", "predict", "only RGB pixels"),

    ("devarm-learning-environment", "include/datamunge/sim/devarm_pick.hpp", "class DevArmPickEnvironment {", "DevArmPickEnvironment", "caller-selected bin objective"),
    ("devarm-learn", "include/datamunge/sim/devarm_pick.hpp", "PickTrainingResult train_red_bin(", "train_red_bin", "epsilon-greedy tabular Q-learning"),
    ("devarm-policy", "include/datamunge/sim/devarm_pick.hpp", "struct PickPolicy {", "PickPolicy", "portable text checkpoint"),
    ("conveyor-environment", "include/datamunge/sim/conveyor.hpp", "class ConveyorEnvironment {", "ConveyorEnvironment", "headless, single-payload conveyor"),
    ("conveyor-reset", "include/datamunge/sim/conveyor.hpp", "void reset(std::uint64_t seed);", "reset", "seeded random fruit"),
    ("conveyor-surface", "include/datamunge/sim/types.hpp", "Vec3 surface_velocity{0, 0, 0};", "surface_velocity", "tangential contact slip"),
    ("cad-solid-box", "include/datamunge/cad/solid.hpp", "static Solid box(double width, double depth, double height);", "box", "axis-aligned box"),
    ("cad-solid-boolean", "include/datamunge/cad/solid.hpp", "Solid subtracted(const Solid& other) const;", "subtracted", "trimmed faces"),
    ("cad-render", "include/datamunge/cad/render.hpp", "RenderResult render(const Scene& scene", "render", "without a window"),
    ("cad-robot", "include/datamunge/cad/robot.hpp", "static Robot load_urdf(", "load_urdf", "every referenced visual asset"),
    ("cad-view", "include/datamunge/cad/view.hpp", "static View from_urdf(", "from_urdf", "articulated URDF"),
    ("lm-factory", "include/datamunge/datamunge.hpp", "LM lm(Args&&... args)", "lm", "Forwards constructor arguments"),
    ("empty-plot-factory", "include/datamunge/vector.hpp", "inline Plot plot()", "plot", "Create an empty native plot"),
    ("plot-axis-alias", "include/datamunge/plot/plot.hpp", "RPlot& xlim(double min_x, double max_x)", "xlim", "without automatic padding"),
    ("rgba-string-constructor", "include/datamunge/plot/plot.hpp", "RGB(const std::string& color);", "RGB", "Hex alpha is the final"),
    ("rgba-line-color", "include/datamunge/plot/plot.hpp", "using RGBA = RGB;", "RGBA", "numeric channels, RGB values, color names and hex strings"),
    ("vector-expression-evaluation", "include/datamunge/vector.hpp", "Vector<double> eval() const;", "eval", "single pointwise pass"),
    ("vector-expression-constructor", "include/datamunge/vector.hpp", "Vector(const E& expression)", "Vector", "single fused pass"),
    ("vector-expression-sine", "include/datamunge/vector.hpp", "auto sin(E&& v)", "sin", "Deferred double-valued expression"),
    ("sparse-basis-evaluation", "include/datamunge/bspline_regression.hpp", "SpMat<> speval(const std::vector<double>& x) const;", "speval", "compressed sparse rows"),
    ("matrix-lm-constructor", "include/datamunge/datamunge.hpp", "LM(const std::vector<double>& y, const BasicMat<true, T>& predictors", "LM", "CSR predictors"),
    ("matrix-lm-prediction", "include/datamunge/datamunge.hpp", "predict(const BasicMat<S, T>& newdata) const", "predict", "CSR columns in training order"),
    ("mkdir", "include/datamunge/datamunge.hpp", "inline bool mkdir(const std::filesystem::path& path)", "mkdir", "missing parent directories"),
    ("bspline-factory", "include/datamunge/bspline_regression.hpp", "inline Bspline bs(", "bs", "Construct an owned B-spline basis"),
    ("bspline-design", "include/datamunge/bspline_regression.hpp", "class Bspline", "Bspline", "B-spline design factory"),
    ("bspline-design-evaluation", "include/datamunge/bspline_regression.hpp", "Mat<> eval(const std::vector<double>& x) const;", "eval", "training or prediction coordinates"),
    ("matrix-column-names", "include/datamunge/linalg/matrix.hpp", "std::vector<std::string> colnames() const", "colnames", "Default DataFrame column labels"),
    ("matrix-binding", "include/datamunge/linalg/bindings.hpp", "class LinalgMatrix", "LinalgMatrix", "shared, immutable expression operands"),
    ("matrix-binding-product", "include/datamunge/linalg/bindings.hpp", "LinalgMatrix multiply", "multiply", "fusing preceding inverse actions"),
    ("matrix-facade", "include/datamunge/linalg/matrix.hpp", "using Mat = BasicMat<linalg::MatrixStoragePolicy<Storage>::sparse, T>;", "Mat", "lazy transpose/product/solve"),
    ("matrix-dense-policy", "include/datamunge/linalg/matrix.hpp", "using Dense = linalg::DMat;", "Dense", "Dense storage policy"),
    ("matrix-sparse-policy", "include/datamunge/linalg/matrix.hpp", "using Sparse = linalg::SpMat;", "Sparse", "Compressed sparse row storage policy"),
    ("expression-scalar", "include/datamunge/algebra/expression.hpp", "Expr(double value) : node_", "Expr", "implicit conversion"),
    ("expression-add", "include/datamunge/algebra/expression.hpp", "friend Expr operator+(const Expr& left, const Expr& right)", "operator+", "symbolic sum"),
    ("expression-multiply", "include/datamunge/algebra/expression.hpp", "friend Expr operator*(const Expr& left, const Expr& right)", "operator*", "symbolic product"),
    ("expression-power", "include/datamunge/algebra/expression.hpp", "friend Expr operator^(const Expr& left, const Expr& right)", "operator^", "symbolic power"),
    ("expression-negate", "include/datamunge/algebra/expression.hpp", "friend Expr operator-(const Expr& value)", "operator-", "symbolic negation"),
    ("expression-tex", "include/datamunge/algebra/expression.hpp", "Tex to_tex(int precision = 7) const", "to_tex", "composable C++ Tex value"),
    ("polynomial-tex", "include/datamunge/algebra/polynomial.hpp", "Tex to_tex(const std::string& variable = \"x\", int precision = 7) const", "to_tex", "descending powers"),
    ("rational-function-tex", "include/datamunge/algebra/rational_function.hpp", "Tex to_tex(const std::string& variable = \"x\", int precision = 7) const", "to_tex", "composable C++ Tex fraction"),
    ("tex-value", "include/datamunge/tex.hpp", "class Tex {", "Tex", "nonmutating mathematical composition"),
    ("tex-underbrace", "include/datamunge/tex.hpp", "Tex underbrace(std::string_view label) const", "underbrace", "brace and raw math label below"),
    ("tex-underset", "include/datamunge/tex.hpp", "Tex underset(std::string_view label) const", "underset", "raw math label below"),
    ("tex-display", "include/datamunge/tex.hpp", "std::string display() const", "display", "display mathematics"),
    ("matrix-tex", "include/datamunge/linalg/matrix.hpp", "Tex to_tex(const std::string& type = \"bmatrix\") const;", "to_tex", "evaluating a lazy expression once"),
    ("dense-matrix-factory", "include/datamunge/linalg/matrix.hpp", "Mat<Dense, T> mat(std::size_t rows, std::size_t cols)", "mat", "zero-initialised dense matrix"),
    ("sparse-matrix-factory", "include/datamunge/linalg/matrix.hpp", "Mat<Sparse, T> spmat(std::size_t rows, std::size_t cols)", "spmat", "zero-initialised compressed sparse row matrix"),
    ("writable-matrix-coefficient", "include/datamunge/linalg/matrix.hpp", "decltype(auto) operator()(std::size_t row, std::size_t col)", "operator()", "checked, zero-based indices"),
    ("matrix-power", "include/datamunge/linalg/matrix.hpp", "auto operator^(E&& matrix, P exponent)", "operator^", "lazy ordinary matrix power"),
    ("free-matrix-transpose", "include/datamunge/linalg/matrix.hpp", "auto t(E&& matrix)", "t", "lazy matrix transpose"),
    ("matrix-evaluation", "include/datamunge/linalg/matrix.hpp", "auto eval() const;", "eval", "owned matrix snapshot"),
    ("matrix-transpose", "include/datamunge/linalg/matrix.hpp", "auto t() &;", "t", "Lazy transpose"),
    ("matrix-solve", "include/datamunge/linalg/matrix.hpp", "auto solve(E&& coefficient)", "solve", "Defer an inverse action"),
    ("native-diagram", "include/datamunge/plot/diagram.hpp", "class Diagram {", "Diagram", "native glyph outlines"),
    ("diagram-node", "include/datamunge/plot/diagram.hpp", "node(std::string id", "node", "automatically sized label box"),
    ("diagram-edge", "include/datamunge/plot/diagram.hpp", "Diagram& edge(", "edge", "existing node boundaries"),
    ("diagram-svg", "include/datamunge/plot/diagram.hpp", "std::string to_svg() const;", "to_svg", "accessible labels"),
    ("npy-storage-order", "include/datamunge/nn/array_io.hpp", "std::vector<unsigned char> encode_npy(", "encode_npy", "explicitly selected NPY storage order"),
    ("npz-compression", "include/datamunge/nn/array_io.hpp", "std::vector<unsigned char> encode_npz(", "encode_npz", "DEFLATE-compressed NPZ"),
    ("idx-writer", "include/datamunge/nn/array_io.hpp", "void write_idx(", "write_idx", "big-endian IDX"),
    ("native-data-fetch", "include/datamunge/datasets/assets.hpp", "void fetch_file(", "fetch_file", "verify existing and newly downloaded"),
    ("cad-asset-fetch", "include/datamunge/datasets/download.hpp", "std::string fetch(", "fetch", "DevArm verifies the size and SHA-256"),
    ("adaptive-spline-depth", "include/datamunge/fda/adaptive_pspline.hpp", "int max_depth{4};", "max_depth", "0..9"),
    ("adaptive-spline-fixed-fit", "include/datamunge/fda/adaptive_pspline.hpp", "AdaptiveSplineFit fit_hierarchical_pspline(", "fit_hierarchical_pspline", "clamped boundaries"),
    ("adaptive-spline-greedy-fit", "include/datamunge/fda/adaptive_pspline.hpp", "AdaptiveSplineFit fit_adaptive_pspline(", "fit_adaptive_pspline", "parent-local training-MSE drop"),
    ("adaptive-spline-joint-fit", "include/datamunge/fda/adaptive_pspline.hpp", "AdaptiveSplineFit fit_joint_adaptive_pspline(", "fit_joint_adaptive_pspline", "retuning lambda"),
    ("adaptive-spline-derivative", "include/datamunge/fda/adaptive_pspline.hpp", "std::vector<double> derivative(", "derivative", "analytic derivative"),
    ("native-data-extract", "include/datamunge/datasets/assets.hpp", "void extract_archive(", "extract_archive", "Rejects absolute paths"),
    ("native-example-data", "include/datamunge/datasets/assets.hpp", "std::string prepare_example_data(", "prepare_example_data", "original pinned sources"),
    ("compute-context", "include/datamunge/compute/context.hpp", "class Context {", "Context", "persistent workers"),
    ("compute-buffer", "include/datamunge/compute/context.hpp", "class Buffer {", "Buffer", "retaining its creating context"),
    ("cuda-matrix-adapter", "include/datamunge/compute/context.hpp", "void matmul(Buffer& destination", "matmul", "native GPU arithmetic"),
    ("nn-matrix-adapter", "include/datamunge/nn/ops.hpp", "Tensor matmul(const Tensor& a, const Tensor& b", "matmul", "CUDA uses cuBLAS"),
    ("physics-world", "include/datamunge/sim/world.hpp", "class World {", "World", "native physics world"),
    ("physics-batch", "include/datamunge/sim/batch.hpp", "class BatchWorld {", "BatchWorld", "independent native worlds"),
    ("physics-task", "include/datamunge/sim/tasks.hpp", "class TaskBatch {", "TaskBatch", "without a graphics dependency"),
    ("native-behavior", "include/datamunge/game/simulation.hpp", "class Behavior {", "Behavior", "retain state between ticks"),
    ("scene-import", "include/datamunge/game/io.hpp", "struct SceneImport {", "SceneImport", "legacy Lua-bearing"),
    ("native-courier", "include/datamunge/game/courier.hpp", "BehaviorRegistry courier_behaviors();", "courier_behaviors", "native factories"),
    ("native-renderer", "include/datamunge/game/render.hpp", "class Renderer {", "Renderer", "offscreen Vulkan renderer"),
    ("tensor-transfer", "include/datamunge/nn/tensor.hpp", "Tensor to(const compute::Context& context)", "to", "synchronous transfer boundary"),
    ("class", "include/datamunge/nn/tensor.hpp", "class Tensor {", "Tensor", "typed N-d array"),
    ("enumvalue", "include/datamunge/nn/dtype.hpp", "Float32,  //", "Float32", "C++ float elements"),
    ("static", "include/datamunge/nn/dtype.hpp", "static constexpr DType value = DType::Float64", "value", "C++ double element"),
    ("method", "include/datamunge/nn/tensor.hpp", "std::int64_t numel()", "numel", "product of the axis lengths"),
    ("method", "include/datamunge/vector.hpp", "std::size_t size() const noexcept", "size()", "number of elements"),
    ("template-class", "include/datamunge/vector.hpp", "class Vector : public VectorExpressionTag", "Vector", "R-style selection and recycling"),
    ("callable-adapter", "include/datamunge/vector.hpp", "std::decay_t<F> function(F&& callable)", "function", "Owns a callable"),
    ("vector-apply", "include/datamunge/vector.hpp", "auto apply(const Vector<T>& values, F&& callable)", "apply", "once per element in order"),
    ("vector-length", "include/datamunge/vector.hpp", "std::size_t length(const Vector<T>& values)", "length", "number of elements"),
    ("sample-sd", "include/datamunge/vector.hpp", "double sd(const Vector<>& v)", "sd", "Sample standard deviation"),
    ("decimal-round", "include/datamunge/vector.hpp", "double round(double value, int digits = 0)", "round", "tie goes to the even last digit"),
    ("histogram-constructor", "include/datamunge/plot/histogram.hpp", "explicit Histogram(std::vector<double> data", "Histogram", "Bins a copy of finite observations"),
    ("template-method", "include/datamunge/vector.hpp", "Vector select(const std::vector<I>& indices)", "select", "one-based numeric indices"),
    ("plot-layer", "include/datamunge/plot/compose.hpp", "Points with(PointsArgs options)", "with", "leaves this reusable layer unchanged"),
    ("plot-setter", "include/datamunge/plot/plot.hpp", "RPlot& set_title(std::string value)", "set_title", "Sets the main heading"),
    ("color-setter", "include/datamunge/plot/plot.hpp", "RPlot& set_col(std::string color)", "set_col", "Matching is case-insensitive"),
    ("point-setter", "include/datamunge/plot/compose.hpp", "Points& set_col(const Vector<RGB>& colors)", "set_col", "Colors recycle to the point count"),
    ("legend-setter", "include/datamunge/plot/compose.hpp", "Legend& set_labels(const Vector<std::string>& labels)", "set_labels", "in either order"),
    ("namespace-alias", "include/dm/dm.hpp", "namespace dm = datamunge;", "dm", "Short public namespace spelling"),
    ("format-specialization", "include/datamunge/format.hpp", "struct formatter<datamunge::dstruct::DataFrame>", "formatter", "Formats a dataframe"),
    ("column-access", "include/datamunge/dstruct/dataframe.hpp", "const NullableColumn<T>& get_column(", "get_column", "Borrows a read-only typed column"),
    ("constructor", "include/datamunge/nn/autograd.hpp", "NoGradGuard() :", "NoGradGuard", "Saves this thread's recording mode"),
    ("private-field", "include/datamunge/nn/autograd.hpp", "std::uint64_t version_{0};", "version_", "Storage version expected by get"),
    ("optimizer-enum", "include/datamunge/nn/optim.hpp", "  AdamW,", "AdamW", "Adam direction with decoupled weight decay"),
    ("option-field", "include/datamunge/nn/optim.hpp", "double        factor_epsilon{", "factor_epsilon", "Adafactor squared-gradient"),
)


def check_hover(clangd: str, compiler: str, targets=TARGETS) -> list[dict]:
    """Open real headers and require their selected definitions to return descriptive hover text.

    Only the selected compiler is permitted for clangd's system-include query.
    A missing response or server exit raises an error instead of passing a check.
    """
    with tempfile.TemporaryDirectory(prefix="documentation-hover-") as temporary:
        work = Path(temporary)
        headers = sorted({ROOT / target[1] for target in targets})
        commands = [
            {
                "directory": str(ROOT),
                "file": str(header),
                "arguments": [compiler, "-std=c++23", "-x", "c++-header", "-I" + str(ROOT / "include"), "-c", str(header)],
            }
            for header in headers
        ]
        (work / "compile_commands.json").write_text(json.dumps(commands), encoding="utf-8")
        with (work / "clangd.log").open("wb") as log:
            process = subprocess.Popen(
                [clangd, "--enable-config=0", "--background-index=0", "--compile-commands-dir=" + str(work), "--query-driver=" + compiler],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                bufsize=0,
            )
            buffer = bytearray()

            def send(method: str, params: dict | None = None, request_id: int | None = None) -> None:
                """Write one correctly framed JSON-RPC request or notification to clangd."""
                message = {"jsonrpc": "2.0", "method": method}
                if params is not None:
                    message["params"] = params
                if request_id is not None:
                    message["id"] = request_id
                encoded = json.dumps(message).encode("utf-8")
                process.stdin.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode() + encoded)

            def receive() -> dict:
                """Read one JSON-RPC message with a 30-second deadline, retaining buffered messages."""
                deadline = time.monotonic() + 30
                while True:
                    split = buffer.find(b"\r\n\r\n")
                    if split >= 0:
                        header = bytes(buffer[:split]).decode("ascii")
                        length = int(next(line.split(":", 1)[1] for line in header.split("\r\n") if line.lower().startswith("content-length:")))
                        end = split + 4 + length
                        if len(buffer) >= end:
                            message = json.loads(bytes(buffer[split + 4:end]))
                            del buffer[:end]
                            return message
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                        raise TimeoutError("clangd did not respond within 30 seconds")
                    data = os.read(process.stdout.fileno(), 65536)
                    if not data:
                        raise RuntimeError("clangd exited before returning a response")
                    buffer.extend(data)

            def response(request_id: int) -> dict:
                """Wait for a specific reply, ignoring asynchronous notifications; reject RPC errors."""
                while True:
                    message = receive()
                    if message.get("id") == request_id:
                        if "error" in message:
                            raise RuntimeError(f"clangd request failed: {message['error']}")
                        return message

            try:
                send("initialize", {"processId": os.getpid(), "rootUri": ROOT.as_uri(), "capabilities": {"textDocument": {"hover": {"contentFormat": ["markdown"]}}}}, 1)
                response(1)
                send("initialized", {})
                for header in headers:
                    send("textDocument/didOpen", {"textDocument": {"uri": header.as_uri(), "languageId": "cpp", "version": 1, "text": header.read_text(encoding="utf-8")}})
                results = []
                for request_id, (kind, relative, needle, token, expected) in enumerate(targets, 2):
                    header = ROOT / relative
                    lines = header.read_text(encoding="utf-8").splitlines()
                    matches = [i for i, line in enumerate(lines) if needle in line]
                    if len(matches) > 1:
                        matches = [i for i in matches if lines[i].strip() == needle.strip()]
                    if len(matches) != 1:
                        raise ValueError(f"Expected one declaration matching {needle!r} in {relative}")
                    line = matches[0]
                    # Single-letter method names must not match a return type such as auto.
                    occurrences = list(re.finditer(r"(?<!\w)" + re.escape(token) + r"(?!\w)", lines[line]))
                    if not occurrences:
                        raise ValueError(f"No symbol token {token!r} in {relative}:{line + 1}")
                    column = occurrences[0].start() + min(1, len(token) - 1)
                    send("textDocument/hover", {"textDocument": {"uri": header.as_uri()}, "position": {"line": line, "character": column}}, request_id)
                    result = response(request_id).get("result") or {}
                    contents = result.get("contents", {})
                    value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
                    results.append({"kind": kind, "path": relative, "line": line + 1, "symbol": token, "passed": expected in value, "hover": value})
                send("shutdown", request_id=100)
                response(100)
                send("exit")
                process.wait(timeout=10)
                return results
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()


def main() -> int:
    """Run definition-help checks and write evidence; return 1 for failures or 2 for tool errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clangd", default=shutil.which("clangd"), help="clangd executable")
    parser.add_argument("--compiler", default=shutil.which("clang++") or shutil.which("c++"), help="Compiler queried for system includes")
    parser.add_argument("--json", type=Path, default=ROOT / "build/debug/docs/hover.json")
    parser.add_argument("--path", action="append", default=[], help="Only targets under this repo-relative path (repeatable)")
    args = parser.parse_args()
    if not args.clangd or not args.compiler:
        parser.error("clangd and a C++ compiler are required")
    clangd = shutil.which(args.clangd)
    compiler = shutil.which(args.compiler)
    if not clangd or not compiler:
        parser.error("the selected clangd or compiler executable was not found")
    try:
        targets = tuple(target for target in TARGETS if not args.path or any(
            target[1] == path or target[1].startswith(path.rstrip('/') + '/') for path in args.path))
        if not targets:
            parser.error("no hover targets match --path")
        results = check_hover(clangd, compiler, targets)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        print(f"Definition hover check failed: {error}")
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['path']}:{result['line']} {result['kind']} {result['symbol']}")
    return int(any(not result["passed"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
