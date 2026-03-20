from argparse import Namespace
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import wandb

_CV_FOLD_ARTIFACT_TYPE = "cv-fold-state"
_CV_FOLD_JSON_NAMES = ("durations.json", "test_metrics.json", "val_metrics.json")


def _get_resume_run_id() -> Optional[str]:
    """Return run ID to resume if WANDB_RUN_ID and WANDB_RESUME=must are set."""
    run_id = os.environ.get("WANDB_RUN_ID")
    resume = os.environ.get("WANDB_RESUME", "").lower()
    if run_id and resume == "must":
        return run_id
    return None


def apply_wandb_resume(run_id: str, args: Namespace) -> Namespace:
    """Load config from an existing run and init wandb with resume=\"must\".

    Use when re-running a crashed/failed run so it continues as the same W&B run.
    See: https://docs.wandb.ai/guides/runs/resuming
    """
    api = wandb.Api()
    # run_path: "entity/project/run_id" (full) or run_id only (then use env for entity/project)
    if "/" in run_id and run_id.count("/") >= 2:
        run_path = run_id
        parts = run_id.split("/")
        entity = os.environ.get("WANDB_ENTITY", parts[0] if len(parts) >= 3 else "")
        project = os.environ.get("WANDB_PROJECT", parts[1] if len(parts) >= 3 else "")
    else:
        entity = os.environ.get("WANDB_ENTITY", api.default_entity or "")
        project = os.environ.get("WANDB_PROJECT", "")
        run_path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(run_path)
    except Exception as e:
        logging.warning("Could not fetch run %s for resume: %s", run_path, e)
        raise
    config = dict(run.config)
    # Apply sweep config to args (same keys as agent would set)
    for key in ("data_dir", "task", "model", "seed", "name", "use_pretrained_imputation", "experiment"):
        if key in config:
            val = config[key]
            if key == "data_dir" and isinstance(val, str):
                val = Path(val)
            setattr(args, key, val)
    if args.hyperparams is None:
        args.hyperparams = []
    for key, value in config.items():
        if key.startswith("_") or key == "run-name":
            continue
        if key in ("data_dir", "task", "model", "seed", "name", "use_pretrained_imputation", "experiment"):
            continue
        args.hyperparams.append(f"{key}=" + (("'" + str(value) + "'") if isinstance(value, str) else str(value)))
    logging.info("Resuming run %s with config (data_dir=%s, task=%s, model=%s)", run_id, args.data_dir, args.task, args.model)
    wandb.init(
        entity=entity or None,
        project=project or None,
        id=run_id,
        resume="must",
        allow_val_change=True,
        dir=args.log_dir,
    )
    return args


def wandb_running() -> bool:
    """Check if wandb is running."""
    return wandb.run is not None


def mark_wandb_run_cleanup_exit(exit_code: int = 1) -> None:
    """Mark the current W&B run as failed (e.g. after SIGTERM/SIGINT cleanup).

    Call this when exiting due to timeout/cancel so the run is not reported as
    \"finished\". Uses wandb.finish(exit_code=...) so the run state becomes
    \"failed\" and is excluded from finished_metrics.json / finished-only reports.
    """
    if wandb_running():
        try:
            wandb.finish(exit_code=exit_code)
        except Exception as e:
            logging.warning("Could not mark W&B run as cleanup exit: %s", e)


def update_wandb_config(config: dict) -> None:
    """updates wandb config if wandb is running

    Args:
        config (dict): config to set
    """
    logging.debug(f"Updating Wandb config: {config}")
    if wandb_running():
        wandb.config.update(config, allow_val_change=True)


def apply_wandb_sweep(args: Namespace) -> Namespace:
    """Applies the wandb sweep configuration to the namespace object.

    If WANDB_RUN_ID and WANDB_RESUME=must are set, resumes that run instead
    (loads config from API and inits with resume=\"must\").
    """
    resume_run_id = _get_resume_run_id()
    if resume_run_id:
        return apply_wandb_resume(resume_run_id, args)
    wandb.init(allow_val_change=True, dir=args.log_dir)
    sweep_config = wandb.config
    args.__dict__.update(sweep_config)
    if args.hyperparams is None:
        args.hyperparams = []
    for key, value in sweep_config.items():
        if key == "experiment":
            continue
        args.hyperparams.append(f"{key}=" + (("'" + value + "'") if isinstance(value, str) else str(value)))
    logging.info(f"hyperparams after loading sweep config: {args.hyperparams}")
    return args


def wandb_log(log_dict):
    """logs metrics to wandb

    Args:
        log_dict (dict): metric dict to log
    """
    if wandb_running():
        wandb.log(log_dict)


def set_wandb_experiment_name(args, mode):
    """stores the run name in wandb config

    Args:
        args (Namespace): parsed arguments
        mode (RunMode): run mode
    """
    if args.name is None:
        data_dir = Path(args.data_dir)
        args.name = data_dir.name
    run_name = f"{mode}_{args.model}_{args.name}"
    if args.modalities:
        run_name += f"_mods_{args.modalities}"
    if args.fine_tune:
        run_name += f"_source_{args.source_name}_fine-tune_{args.fine_tune}_samples"
    elif args.eval:
        run_name += f"_source_{args.source_name}"
    elif args.samples:
        run_name += f"_train_size_{args.samples}_samples"
    elif args.complete_train:
        run_name += "_complete_training"

    if wandb_running():
        wandb.config.update({"run-name": run_name})
        wandb.run.name = run_name


def fetch_optuna_db_from_current_run(
    download_dir: Path,
    db_filename: str = "hyperparameter_tuning_logs.db",
) -> Optional[Path]:
    """Download the Optuna DB from the current (resumed) run so tuning can continue.

    Used when resuming a crashed/failed run (WANDB_RUN_ID + WANDB_RESUME=must).
    Prefers versioned W&B artifact optuna-db-{run.id}, then run file upload.

    Returns the local path to the downloaded DB, or None if not found.
    """
    if not wandb_running():
        return None

    api = wandb.Api()
    run_id = wandb.run.id
    entity = wandb.run.entity or api.default_entity
    project = wandb.run.project

    # Try artifact first
    artifact_name = f"{entity}/{project}/optuna-db-{run_id}:latest"
    try:
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download(root=str(download_dir))
        candidate = Path(artifact_dir) / db_filename
        if candidate.exists():
            local_path = download_dir / db_filename
            if candidate != local_path:
                candidate.rename(local_path)
            n_trials = artifact.metadata.get("n_trials", "?")
            logging.info(
                "Downloaded Optuna DB from current run %s (%s trials) to %s",
                run_id, n_trials, local_path,
            )
            return local_path
    except Exception:
        pass

    # Fall back to flat file uploaded to this run
    try:
        run_files = {f.name for f in wandb.run.files()}
    except Exception:
        return None
    if db_filename not in run_files:
        return None
    local_path = download_dir / db_filename
    try:
        wandb.run.file(db_filename).download(root=str(download_dir), replace=True)
        logging.info("Downloaded Optuna DB (flat file) from current run %s to %s", run_id, local_path)
        return local_path
    except Exception as exc:
        logging.warning("Failed to download Optuna DB from current run %s: %s", run_id, exc)
        return None


def _iter_cv_fold_resume_files(log_dir: Path):
    """Yield (absolute_path, artifact_relative_posix) for JSON files in completed folds under log_dir."""
    if not log_dir.is_dir():
        return
    for rep_dir in sorted(log_dir.glob("repetition_*")):
        if not rep_dir.is_dir():
            continue
        for fold_dir in sorted(rep_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            if not (fold_dir / "durations.json").is_file():
                continue
            rel = fold_dir.relative_to(log_dir).as_posix()
            for fname in _CV_FOLD_JSON_NAMES:
                path = fold_dir / fname
                if path.is_file():
                    yield path, f"{rel}/{fname}"


def upload_cv_fold_state_incremental(
    log_dir: Path,
    repetition: int,
    fold_index: int,
    cv_repetitions_to_train: Optional[int] = None,
    cv_folds_to_train: Optional[int] = None,
) -> None:
    """Upload all completed-fold JSON state to W&B so resume can rehydrate a fresh run_dir.

    Only call from final training (trial is None); tuning uses a temp log_dir and must not
    upload. Each log creates a new artifact version with the full set of completed folds.
    """
    if not wandb_running():
        return

    paths = list(_iter_cv_fold_resume_files(log_dir))
    if not paths:
        return

    run_id = wandb.run.id
    artifact = wandb.Artifact(
        f"cv-folds-{run_id}",
        type=_CV_FOLD_ARTIFACT_TYPE,
        metadata={
            "repetition": repetition,
            "fold_index": fold_index,
            "cv_repetitions_to_train": cv_repetitions_to_train,
            "cv_folds_to_train": cv_folds_to_train,
            "n_fold_files": len(paths),
        },
    )
    for local_path, name in paths:
        artifact.add_file(str(local_path), name=name)
    try:
        wandb.log_artifact(artifact)
        logging.info(
            "Uploaded CV fold state to W&B (%s files, last completed rep=%s fold=%s)",
            len(paths),
            repetition,
            fold_index,
        )
    except Exception as exc:
        logging.warning("Failed to upload CV fold state artifact: %s", exc)


def fetch_cv_fold_state_from_current_run(download_dir: Path) -> bool:
    """Download cv-folds-{run.id} from W&B and merge repetition_*/fold_* JSON into download_dir.

    Used when resuming (WANDB_RESUME=must) so skip-completed-fold logic sees prior folds under
    the new timestamped run_dir. Returns True if any files were merged.
    """
    if not wandb_running():
        return False

    api = wandb.Api()
    run_id = wandb.run.id
    entity = wandb.run.entity or api.default_entity
    project = wandb.run.project
    artifact_name = f"{entity}/{project}/cv-folds-{run_id}:latest"

    try:
        artifact = api.artifact(artifact_name)
    except Exception:
        logging.warning(
            "No CV fold state artifact for run %s (%s); starting with empty run_dir.",
            run_id,
            artifact_name,
        )
        return False

    merged = 0
    with tempfile.TemporaryDirectory(prefix="cv_folds_wandb_") as tmp:
        artifact_dir = Path(artifact.download(root=tmp))
        for rep_dir in sorted(artifact_dir.glob("repetition_*")):
            if not rep_dir.is_dir():
                continue
            for fold_dir in sorted(rep_dir.glob("fold_*")):
                if not fold_dir.is_dir():
                    continue
                dest_fold = download_dir / rep_dir.name / fold_dir.name
                dest_fold.mkdir(parents=True, exist_ok=True)
                for fname in _CV_FOLD_JSON_NAMES:
                    src = fold_dir / fname
                    if src.is_file():
                        shutil.copy2(src, dest_fold / fname)
                        merged += 1

    if merged:
        logging.info(
            "Merged CV fold state from W&B into %s (%s JSON file(s))",
            download_dir,
            merged,
        )
    else:
        logging.warning(
            "Downloaded CV fold artifact for run %s but found no fold JSON to merge.",
            run_id,
        )
    return merged > 0
