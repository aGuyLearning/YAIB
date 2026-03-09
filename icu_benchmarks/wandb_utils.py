from argparse import Namespace
import logging
import os
from pathlib import Path
from typing import Optional

import wandb


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
    for key in ("data_dir", "task", "model", "seed", "name", "use_pretrained_imputation"):
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
        if key in ("data_dir", "task", "model", "seed", "name", "use_pretrained_imputation"):
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


def fetch_optuna_db_from_sibling_runs(
    download_dir: Path,
    db_filename: str = "hyperparameter_tuning_logs.db",
) -> Optional[Path]:
    """Download the Optuna DB from a crashed/failed sibling run of the current sweep.

    Prefers versioned W&B artifacts (logged by the updated tune_step_callback)
    and falls back to flat files uploaded via wandb.save.

    Returns the local path to the downloaded DB, or None if nothing was found.
    """
    if not wandb_running() or wandb.run.sweep_id is None:
        return None

    api = wandb.Api()
    sweep_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}"
    try:
        sweep = api.sweep(sweep_path)
    except Exception as exc:
        logging.warning(f"Could not fetch sweep {sweep_path}: {exc}")
        return None

    current_run_id = wandb.run.id
    current_config = dict(wandb.config)

    def configs_match(run_config: dict) -> bool:
        for key in current_config:
            if key.startswith("_") or key == "run-name":
                continue
            if run_config.get(key) != current_config.get(key):
                return False
        return True

    best_n_trials = 0
    best_local_path: Optional[Path] = None

    for run in sweep.runs:
        if run.id == current_run_id:
            continue
        if run.state not in ("crashed", "failed"):
            continue
        if not configs_match(dict(run.config)):
            continue

        # Try artifact first (has trial count in metadata for picking the best)
        artifact_name = f"{wandb.run.entity}/{wandb.run.project}/optuna-db-{run.id}:latest"
        try:
            artifact = api.artifact(artifact_name)
            n_trials = artifact.metadata.get("n_trials", 0)
            if n_trials > best_n_trials:
                artifact_dir = artifact.download(root=str(download_dir))
                candidate = Path(artifact_dir) / db_filename
                if candidate.exists():
                    best_n_trials = n_trials
                    best_local_path = download_dir / db_filename
                    if candidate != best_local_path:
                        candidate.rename(best_local_path)
                    logging.info(
                        f"Downloaded Optuna artifact from sibling run {run.id} "
                        f"({n_trials} trials) to {best_local_path}"
                    )
            continue
        except Exception:
            pass

        # Fall back to flat file
        if best_local_path is not None:
            continue
        try:
            run_files = {f.name for f in run.files()}
        except Exception:
            continue
        if db_filename not in run_files:
            continue
        local_path = download_dir / db_filename
        try:
            run.file(db_filename).download(root=str(download_dir), replace=True)
            logging.info(f"Downloaded Optuna DB (flat file) from sibling run {run.id} to {local_path}")
            best_local_path = local_path
        except Exception as exc:
            logging.warning(f"Failed to download DB from run {run.id}: {exc}")
            continue

    if best_local_path is None:
        logging.info("No sibling runs with an Optuna DB found for this sweep combination.")
    return best_local_path
