#!/usr/bin/env python3
"""Optional NumPy oracle for the native contractive-autoencoder report's artifacts.

Reads C++-generated arrays only; never downloads/prepares data or fits a model.
Checks complete first-batch chain-rule gradients, the Float32 first Adam update,
all held-out predictions, every retained Jacobian entry and original split IDs.
Example: python3 scripts/check_contractive_autoencoder.py --directory
docs/mepml/examples/nn/2_autoencoders/2.1_deterministic_autoencoders/build
--report build/nn-report-conversion/contractive/oracle.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def arrays(path):
    """Own each numeric array and close its archive immediately; pickle is disabled."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def sigmoid(value):
    """Evaluate the logistic function in Float64 without overflowing either tail."""
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def check(directory):
    """Check the three preserved source variants; return scalar evidence or raise."""
    data = arrays(directory / "contractive_results.npz")
    records = []
    for name in ("plain", "contractive", "strong"):
        prefix = directory / ("contractive_" + name)
        initial = arrays(Path(str(prefix) + "_initial.npz"))
        gradients = arrays(Path(str(prefix) + "_first_gradients.npz"))
        updated = arrays(Path(str(prefix) + "_first_update.npz"))
        best = arrays(Path(str(prefix) + "_best.npz"))
        coefficient = float(gradients["weight"])
        x = gradients["input"].astype(np.float64)
        w, b, v, c = (initial[key].astype(np.float64) for key in
                        ("encoder/W", "encoder/b", "decoder/W", "decoder/b"))
        h = sigmoid(x @ w + b)
        prediction = sigmoid(h @ v + c)
        slope = h * (1 - h)
        column_norm = (w * w).sum(axis=0)
        decoder_delta = 2 * (prediction - x) * prediction * (1 - prediction) / x.size
        encoder_delta = (decoder_delta @ v.T) * slope
        penalty_delta = 2 / len(x) * slope**2 * (1 - 2 * h) * column_norm
        expected = {
            "encoder/W": x.T @ encoder_delta + coefficient *
                (x.T @ penalty_delta + 2 / len(x) * w * (slope**2).sum(axis=0)),
            "encoder/b": encoder_delta.sum(axis=0) + coefficient * penalty_delta.sum(axis=0),
            "decoder/W": h.T @ decoder_delta,
            "decoder/b": decoder_delta.sum(axis=0),
        }
        gradient_errors = {}
        for key, value in expected.items():
            np.testing.assert_allclose(value, gradients[key], atol=2e-8, rtol=2e-5)
            gradient_errors[key] = float(np.max(np.abs(value - gradients[key])))
            # Isolate optimizer rounding from the separate widened gradient check.
            # Moment updates and correction factors are each rounded to Float32.
            g = gradients[key]
            moment = g * np.float32(.1)
            variance = (g * g) * np.float32(.001)
            direction = (moment / np.float32(.1)) / (
                np.sqrt(variance / np.float32(.001)) + np.float32(1e-8))
            next_value = initial[key] - direction * np.float32(.003)
            np.testing.assert_array_equal(next_value, updated[key])
        test = data["test_input"].astype(np.float64)
        code = sigmoid(test @ best["encoder/W"].astype(np.float64) + best["encoder/b"])
        reconstruction = sigmoid(code @ best["decoder/W"].astype(np.float64) + best["decoder/b"])
        penalty = ((code * (1 - code))**2 *
                   (best["encoder/W"].astype(np.float64)**2).sum(axis=0)).sum(axis=1)
        np.testing.assert_allclose(code, data[name + "_code"], atol=5e-7, rtol=2e-6)
        np.testing.assert_allclose(reconstruction, data[name + "_reconstruction"], atol=5e-7, rtol=2e-6)
        np.testing.assert_allclose(penalty, data[name + "_penalty"], atol=2e-6, rtol=2e-5)
        probe = data[name + "_probe_input"].astype(np.float64)
        probe_code = sigmoid(probe @ best["encoder/W"].astype(np.float64) + best["encoder/b"])
        jacobian = (probe_code * (1 - probe_code))[:, :, None] * best["encoder/W"].T[None, :, :]
        np.testing.assert_allclose(jacobian, data[name + "_probe_jacobian"], atol=2e-6, rtol=2e-5)
        records.append({"variant": name, "weight": coefficient,
                        "first_gradient_errors": gradient_errors, "Float32_Adam_step_bit_identical": True,
                        "test_mse": float(np.mean((reconstruction - test)**2)),
                        "mean_squared_Jacobian": float(penalty.mean()),
                        "maximum_full_Jacobian_error": float(np.max(np.abs(jacobian - data[name + "_probe_jacobian"]))),
                        "saturated_fraction": float(np.mean((code < .01) | (code > .99))),
                        "mean_code_std": float(np.std(code, axis=0).mean())})
    for key, count in (("fit_ids", 1000), ("validation_ids", 200), ("test_ids", 200)):
        if len(np.unique(data[key])) != count or data[key].shape != (count,):
            raise AssertionError(f"{key}: expected {count} distinct original IDs")
        lower, upper = (60000, 70000) if key == "test_ids" else (0, 60000)
        if np.any((data[key] < lower) | (data[key] >= upper)):
            raise AssertionError(f"{key}: IDs outside the official source split")
    for left, right in (("fit_ids", "validation_ids"), ("fit_ids", "test_ids"), ("validation_ids", "test_ids")):
        if np.intersect1d(data[left], data[right]).size:
            raise AssertionError(f"{left} and {right} overlap")
    return {"passed": True, "reconstruction_pixels": 3 * int(data["test_input"].size),
            "full_Jacobian_entries": 3 * int(data["plain_probe_jacobian"].size), "variants": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="native report's local build directory")
    parser.add_argument("--report", type=Path, required=True, help="output scalar verification JSON")
    args = parser.parse_args()
    result = check(args.directory)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checked {result['reconstruction_pixels']} pixels and {result['full_Jacobian_entries']} Jacobian entries across three variants")


if __name__ == "__main__":
    main()
