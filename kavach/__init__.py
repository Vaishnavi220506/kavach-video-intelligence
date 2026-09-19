"""KAVACH reusable application components."""

__all__ = [
    "Detection",
    "EventDatabase",
    "EvidenceReplay",
    "GroundedAssistant",
    "Incident",
    "IncidentManager",
    "OllamaClient",
    "RiskAssessment",
    "RiskEngine",
    "WarehouseDetector",
    "draw_detections",
]


def __getattr__(name: str):
    """Load optional/heavy components only when callers request them.

    Dataset inspection and configuration tools should work without importing
    Torch/OpenCV model libraries. The public root-level imports remain
    compatible, while model-backed components are loaded on demand.
    """

    modules = {
        "Detection": (".perception", "Detection"),
        "WarehouseDetector": (".perception", "WarehouseDetector"),
        "draw_detections": (".perception", "draw_detections"),
        "EvidenceReplay": (".incidents", "EvidenceReplay"),
        "Incident": (".incidents", "Incident"),
        "IncidentManager": (".incidents", "IncidentManager"),
        "GroundedAssistant": (".assistant", "GroundedAssistant"),
        "OllamaClient": (".assistant", "OllamaClient"),
        "RiskAssessment": (".risk", "RiskAssessment"),
        "RiskEngine": (".risk", "RiskEngine"),
        "EventDatabase": (".storage", "EventDatabase"),
    }
    try:
        module_name, attribute = modules[name]
    except KeyError as exc:
        raise AttributeError(f"module 'kavach' has no attribute {name!r}") from exc
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
