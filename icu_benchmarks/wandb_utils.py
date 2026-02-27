from argparse import Namespace
import logging
from pathlib import Path

import wandb

WANDB_RUN_ID_FILE = "wandb_run_id.txt"


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
    """applies the wandb sweep configuration to the namespace object

    Args:
        args (Namespace): parsed arguments

    Returns:
        Namespace: arguments with sweep configuration applied (some are applied via hyperparams)
    """
    wandb.init(allow_val_change=True, dir=args.log_dir)
    sweep_config = wandb.config
    args.__dict__.update(sweep_config)
    if args.hyperparams is None:
        args.hyperparams = []
    for key, value in sweep_config.items():
        args.hyperparams.append(f"{key}=" + (("'" + value + "'") if isinstance(value, str) else str(value)))
    logging.info(f"hyperparams after loading sweep config: {args.hyperparams}")
    return args


def maybe_resume_wandb_run(args: Namespace, run_dir: Path, resume_requested: bool = False) -> None:
    """Persists and optionally resumes a wandb run id for a YAIB run directory."""
    if not wandb_running():
        return

    run_id_path = run_dir / WANDB_RUN_ID_FILE
    saved_run_id = run_id_path.read_text(encoding="utf-8").strip() if run_id_path.is_file() else None
    current_run_id = wandb.run.id

    if resume_requested and saved_run_id and saved_run_id != current_run_id:
        logging.info(f"Switching wandb run from {current_run_id} to saved run id {saved_run_id}.")
        try:
            wandb.finish()
            wandb.init(allow_val_change=True, dir=args.log_dir, id=saved_run_id, resume="allow")
            current_run_id = wandb.run.id
        except Exception as exception:
            logging.warning(f"Could not resume wandb run id {saved_run_id}: {exception}")
            return

    if saved_run_id is None or (not resume_requested and saved_run_id != current_run_id):
        run_id_path.write_text(str(current_run_id), encoding="utf-8")
        logging.info(f"Persisted wandb run id {current_run_id} to {run_id_path}.")


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
