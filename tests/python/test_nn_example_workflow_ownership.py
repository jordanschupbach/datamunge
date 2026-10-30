"""Examples own their orchestration; common code supplies operations and helpers."""

import ast
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
FUNDAMENTALS = "1_convolutional_neural_networks/1.1_fundamentals"
CIFAR_WORKFLOWS = (
    "inception_v2_v3", "inception_v4_resnet", "vgg", "preactivation_resnet",
    "pyramidnet", "res2net", "resnet_bottleneck", "resnext", "revnet",
    "stochastic_depth", "wide_resnet", "squeezenet", "eca_net", "non_local",
    "octconv", "repvgg", "senet", "sknet",
)


def test_mnist_workflow_is_visible_in_each_language_example():
    cpp = (ROOT / "examples/cpp/source/nn" / FUNDAMENTALS / "cnn_mnist.cpp").read_text()
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


def load_audit_module():
    spec = importlib.util.spec_from_file_location("nn_example_audit", ROOT / "scripts/nn_example_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_shared_header_owns_a_training_loop():
    report = load_audit_module().audit()
    # Episode rollouts/evaluators of the RL helpers loop over environment steps, not optimizer updates.
    assert report["header_harnesses"] == {}, report["header_harnesses"]


def test_no_generic_training_runner_remains_in_canonical_cpp_examples():
    report = load_audit_module().audit()["examples"]
    assert len(report) == 414
    assert {name: entry["loop_owner"] for name, entry in report.items()
            if not entry["cpp_compliant"]} == {}


def test_final_recurrent_examples_own_normal_and_replay_updates():
    report = load_audit_module().audit()["examples"]
    names = ("seq2seq", "luong", "translation", "pointer_generator", "pointer_network",
             "word_rnn_lm", "memnet", "key_value", "ntm", "dnc")
    for name in names:
        entry = report[name]
        assert entry["loop_owner"] == "main" and not entry["opaque_step_methods"], name
        source = (ROOT / entry["cpp"]).read_text()
        main = source[source.index("int main("):]
        assert "nn::grad(" in main
        assert re.search(r"(?:optimizer|opt)(?:\.|->)step\(", main), name
        assert not re.search(r"\b(?:Train|Run|Main|RunQa)(?:<[^>]*>)?\(", source), name
        assert "confusion" in main and "m.Update(" not in source
        if name in names[:5]:
            assert len(re.findall(r"nn::Optimizer\s+opt\(", source)) == 2, name
        if name == "word_rnn_lm":
            assert len(re.findall(r"nn::Optimizer\s+opt\(", source)) == 2
    common = (ROOT / "examples/nn/common/memory_task.hpp").read_text()
    assert "void Update(" not in common and "RawGradients(" not in common
    assert "optimizer->step(" not in common


def test_memory_ports_own_their_loop_over_a_bound_component_facade():
    report = load_audit_module().audit()["examples"]
    for name in ("ntm", "dnc", "memnet", "key_value"):
        entry = report[name]
        assert entry["binding_adapter"] == "memory_components.hpp" and entry["bindings_complete"]
        python = (ROOT / "examples/python/nn" / Path(entry["cpp"]).relative_to("examples/nn")).with_suffix(".py").read_text()
        tree = ast.parse(python)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        assert any(isinstance(node, ast.For) for node in ast.walk(main))
        assert any(isinstance(node, ast.While) for node in ast.walk(main))
        assert "model.Forward(" in python and "model.Loss(" in python
        assert "nn.grad(" in python and "optimizer.step(" in python
        r = (ROOT / "examples/r/nn" / Path(entry["cpp"]).relative_to("examples/nn")).with_suffix(".R").read_text()
        assert "for (epoch" in r and "repeat {" in r
        assert "model$Forward(" in r and "model$Loss(" in r
        assert "nn$grad(" in r and "optimizer$step(" in r


def test_binding_adapter_must_be_parsed_by_swig(tmp_path, monkeypatch):
    module = load_audit_module()
    interface = tmp_path / "components.i"
    interface.write_text("// binding-adapter: example missing.hpp\n")
    monkeypatch.setattr(module, "COMPONENTS_I", interface)
    try:
        module.binding_adapters()
    except ValueError as error:
        assert "not parsed by SWIG" in str(error)
    else:
        raise AssertionError("unparsed adapters must not count as bound")


def test_workflow_ownership_never_regresses():
    """Every example recorded as owning its loop in examples/nn/example_audit.json stays that way."""
    recorded = json.loads((ROOT / "examples/nn/example_audit.json").read_text())["examples"]
    current = load_audit_module().audit()["examples"]
    assert set(current) == set(recorded)
    regressions = sorted(name for name, entry in recorded.items()
                         if entry["cpp_compliant"] and not current[name]["cpp_compliant"])
    assert regressions == []
    # The pasted CIFAR harness is gone for good.
    for name, entry in current.items():
        assert entry["loop_owner"] != "pasted_harness" or name in {
            n for n, e in recorded.items() if e["loop_owner"] == "pasted_harness"}, name


def test_component_bindings_cover_the_resnet_basic_example():
    current = load_audit_module().audit()["examples"]
    for name in ("resnet_basic", "cnn_cifar10", "cnn_mnist", "alexnet", "xception", "densenet",
                 "ghostnet", "mobilenet_v2", "mnasnet", "convnext", "efficientnet", "efficientnet_v2"):
        assert current[name]["cpp_compliant"], name
        assert current[name]["unbound_headers"] == [], (name, current[name]["unbound_headers"])


def test_vision_and_competitive_workflows_use_concrete_loops_and_visible_steps():
    report = load_audit_module().audit()["examples"]
    names = ("pointnet", "pointnet_plus_plus", "flownets", "pwcnet", "factorized_video", "i3d",
             "capsule_networks", "spatial_transformer", "k_sparse_autoencoder", "winner_take_all_autoencoder")
    for name in names:
        entry = report[name]
        assert entry["cpp_compliant"] and not entry["opaque_step_methods"], name
        source = (ROOT / entry["cpp"]).read_text()
        assert "TrainVideoModel" not in source and "TrainOpticalFlow" not in source
        assert "TrainCompetitiveAutoencoder" not in source and "image_training::TrainImages" not in source
        assert "for (int epoch" in source
        assert re.search(r"optimizer(?:\.|->)(?:minimize|step)\(", source), name
    for name in ("k_sparse_autoencoder", "winner_take_all_autoencoder"):
        assert report[name]["bindings_complete"], name
    # The host helper must call the canonical C++ model, including discrete ranking.
    python = (ROOT / "examples/python/nn/common/competitive_autoencoder.py").read_text()
    r = (ROOT / "examples/r/nn/common/competitive_autoencoder.R").read_text()
    assert "competitive_autoencoder = _core.CompetitiveAutoencoder" in python
    assert "competitive_autoencoder_forward <- nn_components$ae$CompetitiveAutoencoder" in r


def test_binary_density_convlstm_and_entropy_workflows_have_visible_updates():
    report = load_audit_module().audit()["examples"]
    for name in ("pixel_cnn", "pixel_rnn", "gated_pixel_cnn", "made_density", "conv_lstm", "entropy_exploration"):
        entry = report[name]
        assert entry["loop_owner"] == "main" and not entry["opaque_step_methods"], name
        source = (ROOT / entry["cpp"]).read_text()
        main = source[source.index("int main("):]
        assert "nn::grad(" in main
        assert re.search(r"optimizer(?:\.|->)step\(", main), name
        assert not re.search(r"\b(?:bpix::)?Run\(", main)
        assert "model->Update(" not in main and "model->Gradient(" not in main
    conv = (ROOT / report["conv_lstm"]["cpp"]).read_text()
    assert "confusion.add(target_pixels, predicted_pixels)" in conv
    for header in ("point_training.hpp", "optical_flow.hpp", "binary_pixels.hpp"):
        shared = (ROOT / "examples/nn/common" / header).read_text()
        assert not re.search(r"optimizer->(?:step|minimize)\(", shared), header


def test_cifar_examples_own_steps_evaluation_and_explicit_epoch_budgets():
    report = load_audit_module().audit()["examples"]
    for name in CIFAR_WORKFLOWS:
        entry = report[name]
        assert entry["cpp_compliant"], (name, entry["loop_owner"])
        source = (ROOT / entry["cpp"]).read_text()
        assert "image_training::RunCifar10(" not in source
        assert "optimizer.minimize(objective)" in source
        assert "auto evaluate =" in source
        assert re.search(r"if\s*\(!explicit_epochs\)\s+cfg\.epochs", source)
        assert "params.save(params_path)" in source and "params.load(params_path)" in source


def test_unrolled_recurrent_examples_own_their_updates():
    report = load_audit_module().audit()["examples"]
    for name in ("bidirectional_rnn", "clockwork_rnn", "indrnn", "mgu", "phased_lstm",
                 "qrnn", "sru", "stacked_rnn", "vanilla_rnn", "gru", "lstm",
                 "peephole_lstm", "layer_norm_lstm", "gradient_clipping",
                 "recurrent_dropout", "truncated_bptt"):
        entry = report[name]
        assert entry["loop_owner"] == "main", name
        source = (ROOT / entry["cpp"]).read_text()
        assert "RunMarkedDemo(" not in source and "MarkedDemoMain(" not in source
        assert "model->Update(" not in source
        assert "model->optimizer->step(" in source
        assert "confusion.add(test.labels" in source


def test_audit_flags_pointer_calls_to_bundled_training_steps(tmp_path, monkeypatch):
    audit = load_audit_module()
    source = tmp_path / "example.cpp"
    source.write_text("int main() { nn::ParameterStore p; for (int epoch=0; epoch<3; ++epoch) model->Update(batch); }")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "LEDGER", tmp_path / "missing.json")
    monkeypatch.setattr(audit, "sources", lambda: [(source, Path("1_demo/example.cpp"))])
    monkeypatch.setattr(audit, "header_harnesses", lambda: {})
    entry = audit.audit()["examples"]["example"]
    assert entry["loop_owner"] == "main"
    assert entry["step_owner"] == "header_method" and entry["opaque_step_methods"]


def test_flow_workflows_live_in_their_cpp_examples():
    audit = load_audit_module()
    report = audit.audit()["examples"]
    for name in ("cnf_density", "flow_density_2d", "glow_density", "iaf_density", "maf_density",
                 "nice_density", "planar_radial_density", "real_nvp_density", "spline_density"):
        entry = report[name]
        assert entry["loop_owner"] == "main", name
        source = (ROOT / entry["cpp"]).read_text()
        assert "ParseOptions(argc, argv" in source
        assert not any(function[0] in ("Run", "Main") for function in audit.function_bodies(source)), name


def test_main_loop_does_not_hide_a_second_generic_workflow(tmp_path, monkeypatch):
    audit = load_audit_module()
    source = tmp_path / "example.cpp"
    source.write_text("""void Train(const LogitsBuilder& model) {
  for (int epoch=0; epoch<3; ++epoch) optimizer.minimize(model());
}
int main() {
  nn::ParameterStore p;
  for (int epoch=0; epoch<3; ++epoch) optimizer.minimize(loss);
  Train(callback);
}
""")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "LEDGER", tmp_path / "missing.json")
    monkeypatch.setattr(audit, "sources", lambda: [(source, Path("1_demo/example.cpp"))])
    monkeypatch.setattr(audit, "header_harnesses", lambda: {})
    entry = audit.audit()["examples"]["example"]
    assert entry["loop_owner"] == "pasted_harness" and not entry["cpp_compliant"]
