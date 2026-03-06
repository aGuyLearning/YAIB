import json
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import gin
from pytorch_lightning import seed_everything

from icu_benchmarks.cache_utils import clean_run_cache
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.run_utils import aggregate_results, log_full_line
from icu_benchmarks.wandb_utils import wandb_log

if TYPE_CHECKING:
    import optuna


@gin.configurable
def execute_repeated_cv(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    eval_only: bool = False,
    train_size: Optional[int] = None,
    load_weights: bool = False,
    source_dir: Path = Path(""),
    cv_repetitions: int = 5,
    cv_repetitions_to_train: Optional[int] = None,
    cv_folds: int = 5,
    cv_folds_to_train: Optional[int] = None,
    reproducible: bool = True,
    debug: bool = False,
    generate_cache: bool = False,
    load_cache: bool = False,
    test_on: str = "test",
    mode: RunMode = RunMode.classification,
    pretrained_imputation_model: Optional[str] = None,
    cpu: bool = False,
    verbose: bool = False,
    wandb: bool = False,
    complete_train: bool = False,
    cache_dir: Optional[Path] = None,
    trial: Optional["optuna.trial.BaseTrial"] = None,
) -> float:
    """Preprocesses data and trains a model for each fold.

    Args:

        complete_train: Use the full data for training instead of held out test splits.
        wandb: Use wandb for logging.
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        eval_only: Whether to only evaluate the model.
        train_size: Fixed size of train split (including validation data).
        load_weights: Whether to load weights from source_dir.
        source_dir: Path to the source directory.
        cv_folds: Number of folds for cross validation.
        cv_folds_to_train: Number of folds to use during training. If None, all folds are trained on.
        cv_repetitions: Amount of cross validation repetitions.
        cv_repetitions_to_train: Amount of training repetitions. If None, all repetitions are trained on.
        reproducible: Whether to make torch reproducible.
        debug: Whether to load less data and enable more logging.
        generate_cache: Whether to generate and save cache.
        load_cache: Whether to load previously cached data.
        test_on: Dataset to test on. Can be "test" or "val" (e.g. for hyperparameter tuning).
        mode: Run mode. Can be one of the values of RunMode
        pretrained_imputation_model: Use a pretrained imputation model.
        cpu: Whether to run on CPU.
        verbose: Enable detailed logging.
    Returns:
        The average loss of all folds.
    """
    if not cv_repetitions_to_train:
        cv_repetitions_to_train = cv_repetitions
    if not cv_folds_to_train:
        cv_folds_to_train = cv_folds

    effective_cache_dir = cache_dir or log_dir
    prev_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def _cleanup_on_signal(signum, frame):
        logging.warning("Received signal %s, cleaning up cache before exit.", signum)
        clean_run_cache(effective_cache_dir)
        prev = prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        else:
            raise SystemExit(128 + (signum if signum is not None else 0))

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _cleanup_on_signal)
        except (ValueError, OSError):
            pass

    try:
        agg_loss = 0
        seed_everything(seed, reproducible)
        if complete_train:
            logging.info("Will train full model without cross validation.")
            cv_repetitions_to_train = 1
            cv_folds_to_train = 1

        else:
            logging.info(f"Starting nested CV with {cv_repetitions_to_train} repetitions of {cv_folds_to_train} folds.")
        # Train model for each repetition (a manner of splitting the folds)
        for repetition in range(cv_repetitions_to_train):
            # Train model for each fold configuration (i.e, one fold is test fold and the rest are train/val folds)
            for fold_index in range(cv_folds_to_train):
                repetition_fold_dir = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}"
                repetition_fold_dir.mkdir(parents=True, exist_ok=True)

                start_time = datetime.now()
                data = preprocess_data(
                    data_dir,
                    seed=seed,
                    debug=debug,
                    load_cache=load_cache,
                    generate_cache=generate_cache,
                    cv_repetitions=cv_repetitions,
                    repetition_index=repetition,
                    train_size=train_size,
                    cv_folds=cv_folds,
                    fold_index=fold_index,
                    pretrained_imputation_model=pretrained_imputation_model,
                    runmode=mode,
                    complete_train=complete_train,
                    cache_dir=effective_cache_dir / "cache",
                )
                preprocess_time = datetime.now() - start_time
                start_time = datetime.now()
                agg_loss += train_common(
                    data,
                    log_dir=repetition_fold_dir,
                    eval_only=eval_only,
                    load_weights=load_weights,
                    source_dir=source_dir,
                    reproducible=reproducible,
                    test_on=test_on,
                    mode=mode,
                    cpu=cpu,
                    verbose=verbose,
                    use_wandb=wandb,
                    train_only=complete_train,
                )
                train_time = datetime.now() - start_time

                log_full_line(
                    f"FINISHED FOLD {fold_index}| PREPROCESSING DURATION {preprocess_time}| PROCEDURE DURATION {train_time}",
                    level=logging.INFO,
                )
                durations = {"preprocessing_duration": preprocess_time, "train_duration": train_time}

                with open(repetition_fold_dir / "durations.json", "w") as f:
                    json.dump(durations, f, cls=JsonResultLoggingEncoder)
                if wandb:
                    wandb_log({"Iteration": repetition * cv_folds_to_train + fold_index})
                if repetition * cv_folds_to_train + fold_index > 1 and mode != RunMode.pretrain:
                    try:
                        aggregate_results(log_dir)
                    except Exception as e:
                        logging.error(f"Failed to aggregate results: {e}")

                if trial is not None:
                    import optuna

                    step = repetition * cv_folds_to_train + fold_index
                    running_avg = agg_loss / (step + 1)
                    trial.report(running_avg, step)
                    logging.info(f"Reported running avg loss {running_avg:.4f} to Optuna (step {step}).")
                    if trial.should_prune():
                        logging.info(
                            f"Trial pruned after fold {fold_index} (rep {repetition}): "
                            f"running avg loss = {running_avg:.4f}"
                        )
                        clean_run_cache(effective_cache_dir)
                        raise optuna.TrialPruned(
                            f"Pruned at step {step} with running avg loss {running_avg:.4f}"
                        )
            log_full_line(f"FINISHED CV REPETITION {repetition}", level=logging.INFO, char="=", num_newlines=3)

        clean_run_cache(effective_cache_dir)
        return agg_loss / (cv_repetitions_to_train * cv_folds_to_train)
    finally:
        for sig, prev in prev_handlers.items():
            if callable(prev):
                try:
                    signal.signal(sig, prev)
                except (ValueError, OSError):
                    pass


@gin.configurable
def execute_pretrain_loop(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    reproducible: bool = True,
    debug: bool = False,
    generate_cache: bool = False,
    load_cache: bool = False,
    cpu: bool = False,
    verbose: bool = False,
    wandb: bool = False,
    complete_train: bool = False,
) -> float:
    """Executes TS2Vec-style self-supervised pretraining loop in YAIB."""
    return execute_repeated_cv(
        data_dir=data_dir,
        log_dir=log_dir,
        seed=seed,
        eval_only=False,
        train_size=None,
        load_weights=False,
        source_dir=Path(""),
        reproducible=reproducible,
        debug=debug,
        generate_cache=generate_cache,
        load_cache=load_cache,
        mode=RunMode.pretrain,
        pretrained_imputation_model=None,
        cpu=cpu,
        verbose=verbose,
        wandb=wandb,
        complete_train=complete_train,
    )
