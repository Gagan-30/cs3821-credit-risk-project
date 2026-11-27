from importlib import import_module
from typing import Any, List

__all__ = [
    "inspect_tree_structure",
    "profile_decision_tree_recursion",
    "profile_knn_distance_scaling",
    "run_all_profiles",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".profiler", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)