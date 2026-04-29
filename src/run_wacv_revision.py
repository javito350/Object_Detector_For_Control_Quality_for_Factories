"""
run_wacv_revision.py
====================
Master orchestrator for the complete WACV revision experimental suite.

Runs experiments in dependency-aware order:
    Phase 1 (fast):      EXP-A (latency), FIX-4 (calibration)
    Phase 2 (medium):    EXP-B (2×2 ablation)
    Phase 3 (overnight): EXP-C (quant-EVT), FIX-1 (multi-seed PatchCore)
    Phase 4 (overnight): FIX-3 (VisA expansion)

Usage:
    python src/run_wacv_revision.py                    # Run everything
    python src/run_wacv_revision.py --phase 1          # Run Phase 1 only
    python src/run_wacv_revision.py --phase 1 2        # Run Phases 1 and 2
    python src/run_wacv_revision.py --skip-concurrency # Skip EXP-A concurrency test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENTS = {
    1: [
        {
            "name": "EXP-A: End-to-End Latency",
            "script": "wacv_exp_a_e2e_latency.py",
            "args": [],
            "estimated_minutes": 30,
        },
        {
            "name": "FIX-4: EVT Calibration Verification",
            "script": "wacv_fix4_evt_calibration.py",
            "args": [],
            "estimated_minutes": 90,
        },
    ],
    2: [
        {
            "name": "EXP-B: 2×2 Symmetry vs Retrieval Ablation",
            "script": "wacv_exp_b_2x2_ablation.py",
            "args": [],
            "estimated_minutes": 120,
        },
    ],
    3: [
        {
            "name": "EXP-C: Quantization–EVT Interaction",
            "script": "wacv_exp_c_quant_evt.py",
            "args": [],
            "estimated_minutes": 480,
        },
        {
            "name": "FIX-1: Multi-Seed PatchCore Baseline",
            "script": "wacv_fix1_multiseed_patchcore.py",
            "args": [],
            "estimated_minutes": 360,
        },
    ],
    4: [
        {
            "name": "FIX-3: VisA Evaluation Expansion",
            "script": "wacv_fix3_visa_expansion.py",
            "args": [],
            "estimated_minutes": 600,
        },
    ],
}


def run_experiment(exp: dict) -> bool:
    script_path = SCRIPT_DIR / exp["script"]
    if not script_path.exists():
        print(f"  ✗ Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)] + exp["args"]
    print(f"  Running: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ✓ Completed in {elapsed / 60:.1f} min")
        return True
    else:
        print(f"  ✗ Failed (exit code {result.returncode}) after {elapsed / 60:.1f} min")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="WACV Revision Master Runner")
    parser.add_argument(
        "--phase", type=int, nargs="+", default=[1, 2, 3, 4],
        help="Which phases to run (default: all)",
    )
    parser.add_argument(
        "--skip-concurrency", action="store_true",
        help="Pass --skip-concurrency to EXP-A",
    )
    args = parser.parse_args()

    if args.skip_concurrency:
        for exp in EXPERIMENTS.get(1, []):
            if "exp_a" in exp["script"]:
                exp["args"].append("--skip-concurrency")

    total_minutes = sum(
        exp["estimated_minutes"]
        for phase in args.phase
        for exp in EXPERIMENTS.get(phase, [])
    )

    print("=" * 70)
    print("WACV REVISION — EXPERIMENTAL SUITE")
    print(f"Phases: {args.phase}")
    print(f"Estimated total runtime: ~{total_minutes / 60:.1f} hours")
    print("=" * 70)

    successes, failures = 0, 0

    for phase in sorted(args.phase):
        experiments = EXPERIMENTS.get(phase, [])
        if not experiments:
            print(f"\nPhase {phase}: No experiments.")
            continue

        print(f"\n{'='*70}")
        print(f"PHASE {phase}")
        print(f"{'='*70}")

        for exp in experiments:
            print(f"\n▶ {exp['name']} (~{exp['estimated_minutes']} min)")
            if run_experiment(exp):
                successes += 1
            else:
                failures += 1

    print(f"\n{'='*70}")
    print(f"WACV REVISION COMPLETE: {successes} passed, {failures} failed")
    print(f"{'='*70}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
