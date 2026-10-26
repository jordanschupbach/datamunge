"""Examples own their orchestration; common code supplies operations and helpers."""

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
FUNDAMENTALS = "1_convolutional_neural_networks/1.1_fundamentals"


def test_mnist_workflow_is_visible_in_each_language_example():
    cpp = (ROOT / "examples/cpp/nn" / FUNDAMENTALS / "cnn_mnist.cpp").read_text()
    main = cpp[cpp.index("int main(") :]
    assert "mnist::Run(" not in cpp
    assert "nn::BatchOrder" in main
    assert "optimizer.minimize(" in main
    assert "store.save(checkpoint)" in main
    python = ast.parse(
        (ROOT / "examples/python/nn" / FUNDAMENTALS / "cnn_mnist.py").read_text()
    )
    main = next(
        node
        for node in python.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert any(isinstance(node, ast.For) for node in ast.walk(main))
    assert any(isinstance(node, ast.While) for node in ast.walk(main))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "minimize"
        for node in ast.walk(main)
    )
    r = (ROOT / "examples/r/nn" / FUNDAMENTALS / "cnn_mnist.R").read_text()
    main = r[r.index("main <- function") :]
    assert "nn_run_mnist(" not in r
    assert "for (epoch" in main and "repeat {" in main
    assert "optimizer$minimize(" in main


def test_common_modules_do_not_own_end_to_end_workflows():
    for name in (
        "mnist.hpp",
        "image_training.hpp",
        "segmentation.hpp",
        "detection.hpp",
        "seq2seq_example.hpp",
        "marked_sequence_demo.hpp",
        "binary_pixels.hpp",
        "flow_density.hpp",
    ):
        source = (ROOT / "examples/nn/common" / name).read_text()
        assert not re.search(
            r"^(?:inline\s+)?(?:int|void|double|Result|ApResult|SegmentationMetrics)\s+"
            r"(?:Run|Main|RunCifar10|RunMarkedDemo|TrainImages|TrainSegmentation|TrainDetector)\s*\(",
            source,
            re.MULTILINE,
        ), name
    python = ast.parse((ROOT / "src/pydatamunge/nn_workflows.py").read_text())
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("run_")
        for node in python.body
    )
    r = (ROOT / "examples/r/nn/common.R").read_text()
    assert "nn_run_mnist <-" not in r
