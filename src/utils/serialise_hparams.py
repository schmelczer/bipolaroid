from typing import Any, Dict


def serialise_hparams(hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    return {k: str(v) if isinstance(v, list) else v for k, v in hyperparameters.items()}
