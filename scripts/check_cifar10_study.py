#!/usr/bin/env python3
"""Optional independent NumPy/PyTorch oracle for the frozen CIFAR-10 report.

This development check does not acquire data, train, select a model or write
report inputs. Run in an existing environment providing numpy and torch.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main() -> None:
    """Compare full-test inference, proper scores, Grad-CAM and integrated gradients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("docs/mepml/examples/datasets/cifar10/artifacts"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/cifar10-report/oracle.json"))
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(8)
    root = args.artifacts
    prep = np.load(root / "preprocessing.npz")
    evidence = np.load(root / "evaluation.npz")
    explanations = np.load(root / "explanations.npz")
    weights = {k: torch.from_numpy(v.copy()) for k, v in np.load(root / "best_parameters.npz").items()}
    state = {k: torch.from_numpy(v.copy()) for k, v in np.load(root / "best_bn.npz").items()}
    mean = torch.from_numpy(prep["mean"]).reshape(1, 3, 1, 1)
    sd = torch.from_numpy(prep["sd"]).reshape(1, 3, 1, 1)
    data = args.data or Path(os.environ.get("DATAMUNGE_DATA_DIR", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache"))) / "datamunge/datasets"))) / "cifar10"

    def conv(x, name, stride=1, relu=True):
        """Independent NCHW convolution and frozen batch normalization."""
        kernel = weights[name + "/W"].permute(3, 2, 0, 1).contiguous()
        z = F.conv2d(x, kernel, stride=stride, padding=kernel.shape[2] // 2)
        z = F.batch_norm(z, state[name + "/bn/moving_mean"], state[name + "/bn/moving_var"],
                         weights[name + "/bn/gamma"], weights[name + "/bn/beta"], training=False, eps=1e-5)
        return F.relu(z) if relu else z

    def model(raw):
        """Evaluate exactly the reported four-stage compact ResNet-18."""
        x = conv((raw.permute(0, 3, 1, 2)-mean)/sd, "stem")
        for stage in range(1, 5):
            for block in range(1, 3):
                name = f"stage{stage}/block{block}"
                stride = 2 if stage > 1 and block == 1 else 1
                shortcut = conv(x, name + "/shortcut", stride, False) if stride == 2 else x
                residual = conv(conv(x, name + "/conv1", stride), name + "/conv2", relu=False)
                x = F.relu(shortcut + residual)
        logits = x.mean((2, 3)) @ weights["output/W"] + weights["output/b"]
        return logits, x

    batch = np.fromfile(data / "test_batch.bin", dtype=np.uint8).reshape(10000, 3073)
    assert np.array_equal(batch[:, 0], evidence["test_labels"])
    raw = torch.from_numpy(batch[:, 1:].copy().reshape(10000, 3, 32, 32)).float().permute(0, 2, 3, 1) / 255
    with torch.no_grad():
        actual = torch.cat([model(raw[i:i+128])[0] for i in range(0, 10000, 128)]).numpy()
    reference = evidence["test_logits"]
    result = {"torch_version": torch.__version__, "numpy_version": np.__version__,
              "test_logit_max_abs": float(np.max(np.abs(actual-reference))),
              "test_argmax_disagreements": int(np.sum(actual.argmax(1) != reference.argmax(1)))}
    assert result["test_logit_max_abs"] < .002, result
    assert result["test_argmax_disagreements"] == 0, result
    expected_confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(expected_confusion, (batch[:, 0], actual.argmax(1)), 1)
    assert np.array_equal(expected_confusion, evidence["confusion"])

    def scores(z, y, temperature):
        """Independent stable proper scores and 15-bin confidence calibration."""
        z = z.astype(np.float64)/temperature
        z = z-z.max(1, keepdims=True)
        log_prob = z-np.log(np.exp(z).sum(1, keepdims=True))
        prob = np.exp(log_prob)
        confidence, correct = prob.max(1), prob.argmax(1) == y
        bins = np.minimum((confidence*15).astype(int), 14)
        ece = sum(abs(correct[bins==b].sum()-confidence[bins==b].sum())/len(y) for b in range(15))
        return np.array([correct.mean(), -log_prob[np.arange(len(y)), y].mean(),
                         np.square(prob-np.eye(10)[y]).sum(1).mean(), ece])

    # NLL is convex in inverse temperature: independently solve its derivative root.
    vz = evidence["validation_logits"].astype(np.float64)
    vy = evidence["validation_labels"]
    def inverse_temperature_derivative(beta):
        """Mean softmax-weighted logit minus the true-class logit."""
        scaled = beta*vz
        scaled -= scaled.max(1,keepdims=True)
        prob = np.exp(scaled)
        prob /= prob.sum(1,keepdims=True)
        return float(((prob*vz).sum(1)-vz[np.arange(len(vy)),vy]).mean())
    lo,hi = .1,20.
    if inverse_temperature_derivative(lo)>=0:
        expected_temperature=1/lo
    elif inverse_temperature_derivative(hi)<=0:
        expected_temperature=1/hi
    else:
        for _ in range(80):
            mid=(lo+hi)/2
            if inverse_temperature_derivative(mid)<0: lo=mid
            else: hi=mid
        expected_temperature=2/(lo+hi)
    result["temperature_abs_error"]=abs(expected_temperature-float(evidence["temperature"]))
    assert result["temperature_abs_error"]<2e-5, result

    rows = np.genfromtxt(root / "scores.tsv", delimiter="\t", names=True, dtype=None, encoding="utf8")
    max_score_error = 0.
    for name, temperature in [("test_raw",1.),("test_calibrated",float(evidence["temperature"]))]:
        row = rows[rows["split"]==name][0]
        recorded = np.array([row["accuracy"],row["nll"],row["brier"],row["ece15"]])
        max_score_error = max(max_score_error,float(np.max(np.abs(scores(reference,batch[:,0],temperature)-recorded))))
    result["proper_score_max_abs"] = max_score_error
    assert max_score_error < 1e-9, result

    selected = torch.from_numpy(explanations["raw_images"].copy()).requires_grad_()
    target = torch.from_numpy(explanations["targets"].astype(np.int64))
    logits, activation = model(selected)
    derivative = torch.autograd.grad(logits.gather(1,target[:,None]).sum(),activation)[0]
    cam = F.relu((activation*derivative.mean((2,3),keepdim=True)).sum(1)).detach().numpy()
    result["grad_cam_max_abs"] = float(np.max(np.abs(cam-explanations["cam_raw"])))
    assert result["grad_cam_max_abs"] < .002, result

    def integrated(baseline, steps=512):
        """Independent midpoint quadrature with Float64 path coordinates and accumulation."""
        raw_np = explanations["raw_images"].astype(np.float64)
        base_np = np.broadcast_to(baseline, raw_np.shape).astype(np.float64)
        total = np.zeros_like(raw_np)
        for start in range(0, steps, 4):
            alpha = (np.arange(start,min(start+4,steps),dtype=np.float64)+.5)/steps
            points = base_np[None]+alpha[:,None,None,None,None]*(raw_np-base_np)[None]
            x = torch.from_numpy(points.reshape(-1,32,32,3).astype(np.float32)).requires_grad_()
            fixed = target.repeat(len(alpha))
            z = model(x)[0].gather(1,fixed[:,None]).sum()
            gradient = torch.autograd.grad(z,x)[0].detach().numpy().astype(np.float64)
            total += gradient.reshape(len(alpha),len(raw_np),32,32,3).sum(0)
        return ((raw_np-base_np)*total/steps).astype(np.float32)

    actual_ig = integrated(prep["mean"].reshape(1,1,1,3))
    difference = actual_ig-explanations["ig512"]
    result["integrated_gradients_max_abs"] = float(np.max(np.abs(difference)))
    result["integrated_gradients_rmse"] = float(np.sqrt(np.square(difference.astype(np.float64)).mean()))
    assert result["integrated_gradients_rmse"] < .001, result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
