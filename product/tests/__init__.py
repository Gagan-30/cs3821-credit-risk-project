from importlib import import_module
from typing import Any, List

__all__ = [
    "generate_linear_clusters",
    "generate_quadrant_clusters",
    "generate_concentric_rings",
    "load_stratified_credit_sample",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".synthetic_data", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)