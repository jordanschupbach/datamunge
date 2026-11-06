#!/usr/bin/env python3
"""Build and measure the native recorded-workcell benchmark without training again."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    """Run one checked subprocess; never continue using an old binary after a build failure."""
    return subprocess.run(command, check=True, cwd=ROOT, **kwargs)


def summarize(work, output, encoding, environment):
    """Retain all raw samples alongside grouped timings and a stratified full-minute estimate."""
    with (work / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["stage"])].append(float(row["milliseconds"]))
    stats = {}
    for (suite, stage), values in grouped.items():
        stats.setdefault(suite, {})[stage] = dict(
            count=len(values), mean_ms=statistics.mean(values),
            median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
            stdev_ms=statistics.stdev(values) if len(values) > 1 else 0,
        )
    count = stats["pipeline"]["total"]["count"]
    pipeline = {stage: sum(values) / count for (suite, stage), values in grouped.items()
                if suite == "pipeline" and stage != "total"}
    pipeline["wrist_total"] = pipeline.get("wrist_capture", 0) + pipeline.get("wrist_inspect", 0)
    baseline = ROOT / "benchmarks/results/fruit_render/baseline.json"
    historical_seconds = (json.loads(baseline.read_text())["historical_replay_seconds"] if baseline.exists()
                          else json.loads((ROOT / "build/playground/run.json").read_text())["seconds"])
    data = ROOT / "docs/mepml/examples/build/fruit_stream_realistic"
    expected = json.loads((data / "metrics.json").read_text())
    with (work / "simulation.csv").open() as stream:
        simulation = list(csv.DictReader(stream))
    scene = json.loads((work / "scene.json").read_text())
    if len(simulation) != scene["repetitions"] or count != scene["repetitions"] * 12:
        raise ValueError("Incomplete benchmark sample set")
    for row in simulation:
        if float(row["seconds"]) != 60 or int(row["frames"]) != 901:
            raise ValueError("Simulation horizon changed")
        for key in ["arrivals", "correct", "wrong", "missed", "pending"]:
            if int(row[key]) != expected.get("simulation_"+key,expected["video_" + key]):
                raise ValueError("Simulation outcome differs from the frozen report: " + key)
    result = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), environment=environment,
        methodology="12 equal-width temporal strata, three repetitions by default; warmed assets/page cache; serial cases. 1x1 rendering is a scene-validation/transform/BVH-build proxy, not an exact internal profiler. Pipeline excludes simulation, compilation, checkpoint I/O and final encoding; wrist inspection includes localization/cropping but no inference. HUD is a representative inset blit/text, not an exact replay of every annotation.",
        historical_replay_seconds=historical_seconds, scene=scene,
        simulation_outcomes=simulation, simulation_matches_frozen_report=True,
        statistics=stats, pipeline_mean_ms_per_frame=pipeline,
        estimated_900_frame_pipeline_seconds=stats["pipeline"]["total"]["mean_ms"] * .9,
        encoding=encoding, raw_samples=rows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Historical completed replay: %.2f s" % result["historical_replay_seconds"])
    print("Sampled pipeline estimate for 900 frames: %.2f s" % result["estimated_900_frame_pipeline_seconds"])
    print("Mean milliseconds per sampled frame:")
    for stage, value in pipeline.items():
        print("  %-24s %9.3f" % (stage, value))
    for suite in ["render_control", "component", "simulation"]:
        print(suite)
        for stage, value in stats[suite].items():
            print("  %-30s %9.3f ms (median)" % (stage, value["median_ms"]))
    print("Results:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads",type=int,default=0,help="Overview/wrist worker cap, 0 keeps the renderer default")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/fruit-render-bench")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/fruit-render-benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/fruit_render/current.json")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--reuse-samples", action="store_true", help="Summarize an already completed native run")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        parser.error("--repetitions must be 1..20")
    if not 0<=args.threads<=256:
        parser.error("--threads must be 0..256")
    args.build_dir = args.build_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    data = ROOT / "docs/mepml/examples/build/fruit_stream_realistic"
    for path in [data / "trajectory.csv", data / "fruit_trajectory.csv", data / "mobile.npz", data / "policy.txt", data / "frames/00899.png"]:
        if not path.is_file():
            parser.error("Missing %s; first run DATAMUNGE_VIDEO_START=0 DATAMUNGE_VIDEO_SECONDS=60 DATAMUNGE_STREAM_KEEP_FRAMES=1 just playground" % path)
    executable = args.build_dir / "fruit_render_bench"
    environment = dict(platform=platform.platform(), machine=platform.machine(),
                       native_samples_reused=args.reuse_samples,
                       load_snapshot_scope="summary/encoding pass" if args.reuse_samples else "native benchmark and encoding pass",
                       logical_cpus=os.cpu_count(), compiler=subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
                       load_before=Path("/proc/loadavg").read_text().strip(),
                       cpu_info=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown"),
                       core_pkg_config=subprocess.check_output(["pkg-config", "--cflags", "--libs", "datamunge"], text=True).strip())
    if not args.reuse_samples:
        if not args.no_build:
            run(["cmake", "-S", "benchmarks/cad", "-B", str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release"])
            run(["cmake", "--build", str(args.build_dir), "--config", "Release", "-j", "4"])
        run([str(executable), str(ROOT), str(args.work_dir), str(args.repetitions),str(args.threads)])
    environment["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    cache = args.build_dir / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:"):
                actual_compiler = line.split("=", 1)[1]
                environment["compiler"] = subprocess.check_output([actual_compiler,"--version"],text=True).splitlines()[0]
                environment["compiler_path"] = actual_compiler
    environment["linked_libraries"] = subprocess.check_output(["ldd", str(executable)], text=True)
    for line in environment["linked_libraries"].splitlines():
        if "libdatamunge" in line and "=>" in line:
            library = Path(line.split("=>", 1)[1].strip().split()[0])
            environment["core_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    environment["benchmark_source_sha256"] = hashlib.sha256((ROOT / "benchmarks/cad/fruit_render.cpp").read_bytes()).hexdigest()
    for name in ["trajectory.csv", "fruit_trajectory.csv", "mobile.npz", "policy.txt", "metrics.json"]:
        environment[name + "_sha256"] = hashlib.sha256((data / name).read_bytes()).hexdigest()
    encoding = []
    # Encode the complete existing sequence; no rendering, simulation or training overlaps this case.
    for repetition in range(args.repetitions):
        destination = args.work_dir / "encode.mp4"
        begin = time.perf_counter()
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(data / "frames/%05d.png"), "-frames:v", "900", "-an", "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1", "-bf", "0", "-g", "30", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)])
        encoding.append(dict(repetition=repetition, seconds=time.perf_counter()-begin, bytes=destination.stat().st_size))
    environment["load_after"] = Path("/proc/loadavg").read_text().strip()
    summarize(args.work_dir, args.output.resolve(), encoding, environment)


if __name__ == "__main__":
    main()
