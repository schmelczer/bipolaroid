from pathlib import Path


def get_next_run_name(path: Path, prefix: str = "run") -> str:
    run_ids = [int(run.stem.split("_")[1]) for run in path.glob(f"{prefix}_*")]
    next_run_id = max(run_ids, default=-1) + 1
    return f"{prefix}_{next_run_id}"
