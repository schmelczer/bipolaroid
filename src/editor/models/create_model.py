from .v1 import HistogramRestorationNet as v1


def create_model(type: str, bin_count: int):
    return {"v1": v1}[type](bin_count)
