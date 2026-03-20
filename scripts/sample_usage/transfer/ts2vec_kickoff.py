#!/usr/bin/env python3
"""Kickoff TS2Vec pretraining and downstream probe runs in YAIB.

This script does not build corpora. It assumes the pretrain corpus already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class StepResult:
    phase: str
    name: str
    command: list[str]
    return_code: int
    duration_seconds: float
    started_at: str
    ended_at: str
    skipped: bool = False


@dataclass
class ProbeJob:
    task: str
    model: str
    data_dir: str
    name: str
    extra_hparams: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def parse_probe_job(raw: str) -> ProbeJob:
    # Format: task:model:data_dir:name[:hp1,hp2,...]
    # The fifth field is optional and contains comma-separated hparams.
    parts = raw.split(":")
    if len(parts) < 4:
        raise ValueError(
            f"Invalid probe job '{raw}'. Expected at least 4 parts: task:model:data_dir:name "
            "[optional :hp1,hp2,...]"
        )
    task, model, data_dir, name = parts[:4]
    extra_hparams = []
    if len(parts) > 4 and parts[4]:
        extra_hparams = [x for x in parts[4].split(",") if x]
    return ProbeJob(task=task, model=model, data_dir=data_dir, name=name, extra_hparams=extra_hparams)


def read_probe_jobs(args: argparse.Namespace) -> list[ProbeJob]:
    jobs: list[ProbeJob] = []
    if args.probe_jobs:
        jobs.extend(parse_probe_job(raw) for raw in args.probe_jobs)
    if args.probe_jobs_file:
        data = json.loads(Path(args.probe_jobs_file).read_text(encoding="utf-8"))
        for item in data:
            jobs.append(
                ProbeJob(
                    task=item["task"],
                    model=item.get("model", "TS2VecProbe"),
                    data_dir=item["data_dir"],
                    name=item["name"],
                    extra_hparams=item.get("extra_hparams", []),
                )
            )
    if args.smoke and not jobs:
        jobs = [
            ProbeJob(
                task="BinaryClassification",
                model="TS2VecProbe",
                data_dir="demo_data/mortality24/eicu_demo",
                name="eicu_demo_mortality_probe",
                extra_hparams=["train_common.weight=''", "execute_repeated_cv.cv_repetitions_to_train=1", "execute_repeated_cv.cv_folds_to_train=1"],
            ),
            ProbeJob(
                task="Regression",
                model="TS2VecProbe",
                data_dir="demo_data/kidney_function/eicu_demo",
                name="eicu_demo_kf_probe",
                extra_hparams=["execute_repeated_cv.cv_repetitions_to_train=1", "execute_repeated_cv.cv_folds_to_train=1"],
            ),
        ]
    return jobs


def run_step(cmd: list[str], phase: str, name: str, dry_run: bool, cwd: Path, mplconfigdir: Path) -> StepResult:
    started = now_iso()
    started_ts = time.time()
    print(f"[{phase}] {name}")
    print(f"  $ {format_cmd(cmd)}")
    if dry_run:
        ended = now_iso()
        return StepResult(
            phase=phase,
            name=name,
            command=cmd,
            return_code=0,
            duration_seconds=0.0,
            started_at=started,
            ended_at=ended,
            skipped=True,
        )
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mplconfigdir))
    proc = subprocess.run(cmd, check=False, cwd=str(cwd), env=env)
    ended_ts = time.time()
    ended = now_iso()
    return StepResult(
        phase=phase,
        name=name,
        command=cmd,
        return_code=proc.returncode,
        duration_seconds=ended_ts - started_ts,
        started_at=started,
        ended_at=ended,
    )


def probe_experiment_for_task(task_gin: str) -> str | None:
    """YAIB experiment gin so TS2VecProbe matches pretrain (preprocess.use_static=False)."""
    return {
        "BinaryClassification": "ProbeClassification",
        "Regression": "ProbeRegression",
        "RegressionLoS": "ProbeRegressionLoS",
    }.get(task_gin)


def find_latest_pretrain_checkpoint(pretrain_log_dir: Path, run_name: str) -> Path:
    base = pretrain_log_dir / run_name / "Pretrain" / "TS2Vec"
    if not base.exists():
        raise FileNotFoundError(f"Pretrain log path not found: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No timestamped pretrain runs in: {base}")
    latest = runs[-1]
    fold_dir = latest / "repetition_0" / "fold_0"
    preferred = fold_dir / "last.ckpt"
    fallback = fold_dir / "model.ckpt"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No checkpoint found in {fold_dir} (expected last.ckpt or model.ckpt)")


def build_pretrain_cmd(args: argparse.Namespace, yaib_root: Path, pretrain_log_dir: Path) -> list[str]:
    cmd = [
        args.python_bin,
        "-m",
        "icu_benchmarks.run",
        "-d",
        args.pretrain_data_dir,
        "-t",
        "Pretrain",
        "-m",
        "TS2Vec",
        "--complete-train",
        "--log-dir",
        str(pretrain_log_dir),
        "-n",
        args.pretrain_name,
        "-s",
        str(args.seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    if args.debug:
        cmd.append("--debug")
    if args.generate_cache:
        cmd.append("-gc")
    if args.load_cache:
        cmd.append("-lc")
    return cmd


def build_probe_cmd(
    args: argparse.Namespace,
    job: ProbeJob,
    probe_log_dir: Path,
    checkpoint: Path,
) -> list[str]:
    hp_values = [f"TS2VecProbe.pretrained_encoder_path='{checkpoint}'"]
    hp_values.extend(args.global_hparams)
    hp_values.extend(job.extra_hparams)
    cmd = [
        args.python_bin,
        "-m",
        "icu_benchmarks.run",
        "-d",
        job.data_dir,
        "-t",
        job.task,
        "-m",
        job.model,
        "--log-dir",
        str(probe_log_dir),
        "-n",
        job.name,
        "-s",
        str(args.seed),
    ]
    if job.model == "TS2VecProbe":
        exp = probe_experiment_for_task(job.task)
        if exp:
            cmd.extend(["-e", exp])
    if args.cpu:
        cmd.append("--cpu")
    if args.debug:
        cmd.append("--debug")
    if args.generate_cache:
        cmd.append("-gc")
    if args.load_cache:
        cmd.append("-lc")
    if hp_values:
        cmd.append("-hp")
        cmd.extend(hp_values)
    return cmd


def write_json_log(
    out_path: Path,
    args: argparse.Namespace,
    steps: list[StepResult],
    checkpoint: Path | None,
    probe_jobs: list[ProbeJob],
) -> None:
    payload = {
        "created_at": now_iso(),
        "args": vars(args),
        "resolved_checkpoint": str(checkpoint) if checkpoint else None,
        "probe_jobs": [asdict(job) for job in probe_jobs],
        "steps": [asdict(step) for step in steps],
        "success": all(step.return_code == 0 for step in steps if not step.skipped),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[log] Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kick off YAIB TS2Vec pretraining + probe runs.")
    parser.add_argument("--yaib-root", default=".", help="Path to YAIB repo root.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to run YAIB with.")
    parser.add_argument("--pretrain-data-dir", required=True, help="Existing corpus path for TS2Vec pretraining.")
    parser.add_argument("--pretrain-name", default=None, help="Run name for pretraining. Defaults to corpus folder name.")
    parser.add_argument("--probe-jobs", nargs="*", default=[], help="Probe jobs as task:model:data_dir:name[:hp1,hp2].")
    parser.add_argument("--probe-jobs-file", default=None, help="JSON file with probe job objects.")
    parser.add_argument("--log-root", default="../yaib_logs_ts2vec_kickoff", help="Root log folder for kickoff artifacts.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed passed to YAIB.")
    parser.add_argument("--cpu", action="store_true", help="Pass --cpu to YAIB runs.")
    parser.add_argument("--debug", action="store_true", help="Pass --debug to YAIB runs.")
    parser.add_argument("--generate-cache", action="store_true", help="Pass -gc to YAIB runs.")
    parser.add_argument("--load-cache", action="store_true", help="Pass -lc to YAIB runs.")
    parser.add_argument("--global-hparams", nargs="*", default=[], help="Additional -hp values applied to all probe jobs.")
    parser.add_argument("--continue-on-probe-error", action="store_true", help="Continue if a probe run fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--smoke", action="store_true", help="Use built-in two-probe smoke jobs if none provided.")
    parser.add_argument("--mplconfigdir", default=None, help="Optional MPLCONFIGDIR. Defaults to <yaib-root>/.mplconfig.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    yaib_root = Path(args.yaib_root).resolve()
    if not args.pretrain_name:
        args.pretrain_name = Path(args.pretrain_data_dir).name

    probe_jobs = read_probe_jobs(args)
    if not probe_jobs:
        parser.error("No probe jobs provided. Use --probe-jobs, --probe-jobs-file, or --smoke.")

    log_root = Path(args.log_root).resolve()
    pretrain_log_dir = log_root / "pretrain"
    probe_log_dir = log_root / "probe"
    run_log_path = log_root / "kickoff_run.json"
    mplconfigdir = Path(args.mplconfigdir).resolve() if args.mplconfigdir else (yaib_root / ".mplconfig")
    if not args.dry_run:
        mplconfigdir.mkdir(parents=True, exist_ok=True)

    steps: list[StepResult] = []
    resolved_ckpt: Path | None = None

    pretrain_cmd = build_pretrain_cmd(args, yaib_root, pretrain_log_dir)
    step = run_step(
        pretrain_cmd,
        phase="pretrain",
        name=args.pretrain_name,
        dry_run=args.dry_run,
        cwd=yaib_root,
        mplconfigdir=mplconfigdir,
    )
    steps.append(step)
    if step.return_code != 0:
        write_json_log(run_log_path, args, steps, None, probe_jobs)
        return step.return_code

    if args.dry_run:
        resolved_ckpt = pretrain_log_dir / args.pretrain_name / "Pretrain" / "TS2Vec" / "<latest>" / "repetition_0" / "fold_0" / "last.ckpt"
    else:
        resolved_ckpt = find_latest_pretrain_checkpoint(pretrain_log_dir, args.pretrain_name)
        print(f"[pretrain] resolved checkpoint: {resolved_ckpt}")

    for idx, job in enumerate(probe_jobs):
        probe_cmd = build_probe_cmd(args, job, probe_log_dir, resolved_ckpt)
        probe_step = run_step(
            probe_cmd,
            phase="probe",
            name=f"{idx+1}:{job.name}",
            dry_run=args.dry_run,
            cwd=yaib_root,
            mplconfigdir=mplconfigdir,
        )
        steps.append(probe_step)
        if probe_step.return_code != 0 and not args.continue_on_probe_error:
            write_json_log(run_log_path, args, steps, resolved_ckpt, probe_jobs)
            return probe_step.return_code

    write_json_log(run_log_path, args, steps, resolved_ckpt, probe_jobs)
    return 0 if all(step.return_code == 0 for step in steps if not step.skipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
