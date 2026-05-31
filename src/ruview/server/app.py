"""FastAPI app factory for the local research sensing server."""

import asyncio
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruview.server.schemas import SensingUpdate
from ruview.server.sources import (
    LatestState,
    ReplaySensingSource,
    SensingSource,
    SimulatedSensingSource,
    UdpSensingSource,
)


@dataclass(frozen=True)
class ServerConfig:
    """Local-only server settings for the research app."""

    source: str = "simulated"
    bind_host: str = "127.0.0.1"
    tick_ms: int = 100
    replay_path: str | Path | None = None
    udp_host: str = "0.0.0.0"
    udp_port: int = 5005
    udp_timeout: float = 0.1

    def __post_init__(self) -> None:
        if self.tick_ms < 0:
            raise ValueError("tick_ms must be non-negative")
        if self.udp_port <= 0:
            raise ValueError("udp_port must be positive")
        if self.udp_timeout < 0.0:
            raise ValueError("udp_timeout must be non-negative")

    @property
    def local_only(self) -> bool:
        return self.bind_host in {"127.0.0.1", "localhost", "::1"}


def create_app(
    *,
    source: SensingSource | None = None,
    state: LatestState | None = None,
    config: ServerConfig | None = None,
):
    """Create a FastAPI app exposing Milestone 7 sensing endpoints."""

    try:
        from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the research extra to run the server: uv sync --extra research") from exc

    server_config = config or ServerConfig()
    sensing_source = source or source_from_config(server_config)
    latest_state = state or LatestState()
    app = FastAPI(title="RuView Python Research Sensing Server", version="0.1.0")

    app.state.ruview_source = sensing_source
    app.state.ruview_latest = latest_state
    app.state.ruview_config = server_config

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "source": sensing_source.source,
            "local_only": server_config.local_only,
        }

    @app.get("/api/v1/sensing/latest")
    def latest() -> dict[str, Any]:
        update = poll_once(sensing_source, latest_state)
        if update is None:
            current = latest_state.latest_dict()
            return current if current is not None else empty_update(sensing_source.source)
        return update.to_dict()

    @app.get("/api/v1/vital-signs")
    def vital_signs() -> dict[str, Any]:
        update = latest_state.latest
        if update is None:
            update = poll_once(sensing_source, latest_state)
        if update is None:
            return {"source": sensing_source.source, "tick": 0, "vital_signs": {}, "status": "unavailable"}

        payload = update.to_dict()
        vital_payload = dict(payload.get("vital_signs", {}))
        vital_payload.setdefault("source", payload["source"])
        vital_payload.setdefault("tick", payload["tick"])
        vital_payload.setdefault("status", "available" if payload.get("vital_signs") else "not_estimated")
        return vital_payload

    @app.websocket("/ws/sensing")
    async def ws_sensing(
        websocket: WebSocket,
        limit: int | None = Query(default=None, ge=1),
        tick_ms: int | None = Query(default=None, ge=0),
    ) -> None:
        await websocket.accept()
        sent = 0
        delay = (server_config.tick_ms if tick_ms is None else tick_ms) / 1000.0
        try:
            while limit is None or sent < limit:
                update = poll_once(sensing_source, latest_state)
                if update is None:
                    await asyncio.sleep(delay)
                    continue
                await websocket.send_json(update.to_dict())
                sent += 1
                if limit is None and delay > 0.0:
                    await asyncio.sleep(delay)
        except WebSocketDisconnect:
            return
        finally:
            await websocket.close()

    return app


def source_from_config(config: ServerConfig) -> SensingSource:
    """Create a sensing source from local server config."""

    source_kind = config.source.strip().lower()
    if source_kind == "simulated":
        return SimulatedSensingSource(source="simulated")
    if source_kind == "replay":
        if config.replay_path is None:
            raise ValueError("replay source requires replay_path")
        return ReplaySensingSource(config.replay_path)
    if source_kind in {"esp32", "udp"}:
        return UdpSensingSource(
            config.udp_host,
            config.udp_port,
            timeout=config.udp_timeout,
            source="esp32",
        )
    raise ValueError("source must be one of: simulated, replay, esp32")


def poll_once(source: SensingSource, state: LatestState) -> SensingUpdate | None:
    """Poll one source update and store it as latest."""

    update = source.next_update()
    if update is None:
        return None
    return state.update(update)


def empty_update(source: str) -> dict[str, Any]:
    """Return the plan-shaped empty update used before any source data exists."""

    return {
        "type": "sensing_update",
        "source": source,
        "tick": 0,
        "nodes": [],
        "features": {},
        "classification": {},
        "signal_field": {},
        "vital_signs": {},
        "estimated_persons": 0,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the local research server with uvicorn."""

    parser = argparse.ArgumentParser(description="Run the RuView Python research sensing server.")
    parser.add_argument("--source", choices=("simulated", "replay", "esp32"), default="simulated")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host; defaults to local-only")
    parser.add_argument("--port", type=int, default=8080, help="HTTP bind port")
    parser.add_argument("--tick-ms", type=int, default=100, help="WebSocket tick delay")
    parser.add_argument("--replay", help="JSONL replay file when --source replay is used")
    parser.add_argument("--udp-host", default="0.0.0.0", help="UDP bind host for --source esp32")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP bind port for --source esp32")
    parser.add_argument("--udp-timeout", type=float, default=0.1, help="UDP receive timeout in seconds")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("Install the research extra to run the server: uv sync --extra research") from exc

    config = ServerConfig(
        source=args.source,
        bind_host=args.host,
        tick_ms=args.tick_ms,
        replay_path=args.replay,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        udp_timeout=args.udp_timeout,
    )
    uvicorn.run(create_app(config=config), host=args.host, port=args.port)


__all__ = ["ServerConfig", "create_app", "empty_update", "main", "poll_once", "source_from_config"]


if __name__ == "__main__":
    main()
