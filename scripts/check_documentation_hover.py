#!/usr/bin/env python3
"""Check representative Doxygen descriptions through clangd at source definitions.

Use a temporary compilation database so the check follows the current checkout,
not an installed library or another developer's absolute include paths. This
checks the language server; mep's graphical K integration needs a separate UI check.
"""

import argparse
import json
import os
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
    ("npy-storage-order", "include/datamunge/nn/array_io.hpp", "std::vector<unsigned char> encode_npy(", "encode_npy", "explicitly selected NPY storage order"),
    ("npz-compression", "include/datamunge/nn/array_io.hpp", "std::vector<unsigned char> encode_npz(", "encode_npz", "DEFLATE-compressed NPZ"),
    ("idx-writer", "include/datamunge/nn/array_io.hpp", "void write_idx(", "write_idx", "big-endian IDX"),
    ("native-data-fetch", "include/datamunge/datasets/assets.hpp", "void fetch_file(", "fetch_file", "verify existing and newly downloaded"),
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
    ("method", "include/datamunge/vector.hpp", "std::size_t size()", "size()", "number of elements"),
    ("template-class", "include/datamunge/vector.hpp", "class Vector {", "Vector", "R-style selection and recycling"),
    ("callable-adapter", "include/datamunge/vector.hpp", "std::decay_t<F> function(F&& callable)", "function", "Owns a callable"),
    ("vector-apply", "include/datamunge/vector.hpp", "auto apply(const Vector<T>& values, F&& callable)", "apply", "once per element in order"),
    ("vector-length", "include/datamunge/vector.hpp", "std::size_t length(const Vector<T>& values)", "length", "number of elements"),
    ("sample-sd", "include/datamunge/vector.hpp", "double sd(const Vector<>& v)", "sd", "Sample standard deviation"),
    ("decimal-round", "include/datamunge/vector.hpp", "double round(double value, int digits)", "round", "halfway-away-from-zero"),
    ("histogram-constructor", "include/datamunge/plot/histogram.hpp", "explicit Histogram(std::vector<double> data", "Histogram", "Bins a copy of finite observations"),
    ("template-method", "include/datamunge/vector.hpp", "Vector select(const std::vector<I>& indices)", "select", "one-based numeric indices"),
    ("plot-layer", "include/datamunge/plot/compose.hpp", "Points with(PointsArgs options)", "with", "leaves this reusable layer unchanged"),
    ("plot-setter", "include/datamunge/plot/plot.hpp", "RPlot& set_title(std::string value)", "set_title", "Owns the main heading"),
    ("color-setter", "include/datamunge/plot/plot.hpp", "RPlot& set_col(std::string color)", "set_col", "case-insensitive basic color name"),
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
                    if len(matches) != 1:
                        raise ValueError(f"Expected one declaration matching {needle!r} in {relative}")
                    line = matches[0]
                    column = lines[line].index(token) + 1
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
