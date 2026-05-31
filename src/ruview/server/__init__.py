"""Local research server components."""

from ruview.server.schemas import (
    SENSING_UPDATE_TYPE,
    ClassificationSummary,
    FeatureSummary,
    NodeInfo,
    SensingUpdate,
    SignalFieldSummary,
    VitalSignsSummary,
    sensing_update_to_dict,
)
from ruview.server.sources import (
    LatestState,
    ReplaySensingSource,
    SensingSource,
    SimulatedSensingSource,
    UdpSensingSource,
    sensing_update_from_record,
    sensing_update_from_window,
)

_APP_EXPORTS = {"ServerConfig", "create_app", "empty_update", "poll_once", "source_from_config"}


def __getattr__(name: str):
    if name in _APP_EXPORTS:
        from ruview.server import app as app_module

        return getattr(app_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SENSING_UPDATE_TYPE",
    "ClassificationSummary",
    "FeatureSummary",
    "LatestState",
    "NodeInfo",
    "ReplaySensingSource",
    "SensingSource",
    "SensingUpdate",
    "SignalFieldSummary",
    "SimulatedSensingSource",
    "UdpSensingSource",
    "VitalSignsSummary",
    "ServerConfig",
    "create_app",
    "empty_update",
    "poll_once",
    "source_from_config",
    "sensing_update_from_record",
    "sensing_update_from_window",
    "sensing_update_to_dict",
]
