"""Deterministic mesh-topology view for simulated swarm nodes."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Sequence, TypeAlias, TypeVar

from ruview.swarm import DroneState, NodeId, coerce_node_id

T = TypeVar("T")


@dataclass
class MeshTopology:
    """Hierarchical mesh state with deterministic node ordering."""

    nodes: dict[NodeId, DroneState] = field(default_factory=dict)
    cluster_head: NodeId | None = None

    def update_node(self, state: DroneState) -> None:
        """Upsert a drone state by its node id."""

        self.nodes[state.id] = state

    def update(self, state: DroneState) -> None:
        """Compatibility alias for :meth:`update_node`."""

        self.update_node(state)

    def remove_node(self, node_id: NodeId | int) -> bool:
        """Remove a node and clear it as cluster head if necessary."""

        nid = coerce_node_id(node_id)
        removed = self.nodes.pop(nid, None) is not None
        if self.cluster_head == nid:
            self.cluster_head = None
        return removed

    def remove(self, node_id: NodeId | int) -> bool:
        """Compatibility alias for :meth:`remove_node`."""

        return self.remove_node(node_id)

    def active_nodes(self) -> list[DroneState]:
        """Return active drone states sorted by node id."""

        return sorted(self.nodes.values(), key=lambda state: int(state.id))

    def nearest_k(self, from_node: NodeId | int, k: int) -> list[NodeId]:
        """Return the ``k`` nearest node ids to ``from_node`` by 3-D distance."""

        if k <= 0:
            return []
        origin_id = coerce_node_id(from_node)
        origin = self.nodes.get(origin_id)
        if origin is None:
            return []

        distances: list[tuple[float, int, NodeId]] = []
        for node_id, state in self.nodes.items():
            if node_id == origin_id:
                continue
            distances.append((origin.position.distance_to(state.position), int(node_id), node_id))
        distances.sort(key=lambda item: (item[0], item[1]))
        return [node_id for _, _, node_id in distances[:k]]


@dataclass
class GossipState(Generic[T]):
    """Versioned value for deterministic gossip-style state dissemination."""

    value: T
    origin: NodeId
    timestamp_ms: int
    version: int = 1

    def __post_init__(self) -> None:
        self.origin = coerce_node_id(self.origin)
        self.timestamp_ms = max(0, int(self.timestamp_ms))
        self.version = max(0, int(self.version))

    @classmethod
    def new(cls, value: T, origin: NodeId | int, timestamp_ms: int) -> "GossipState[T]":
        return cls(value=value, origin=coerce_node_id(origin), timestamp_ms=timestamp_ms, version=1)

    @staticmethod
    def merge(a: "GossipState[T]", b: "GossipState[T]") -> "GossipState[T]":
        """Last-write-wins merge: higher version, then higher origin id."""

        if a.version > b.version:
            return a
        if b.version > a.version:
            return b
        return a if int(a.origin) >= int(b.origin) else b

    def bump(self) -> None:
        self.version += 1

    def spread(
        self,
        fanout: int,
        all_peers: Sequence[NodeId | int],
        local_id: NodeId | int,
        *,
        seed: int | None = None,
    ) -> list[NodeId]:
        """Select peers for gossip, excluding local and origin nodes."""

        local = coerce_node_id(local_id)
        candidates = [
            coerce_node_id(peer)
            for peer in all_peers
            if coerce_node_id(peer) not in {local, self.origin}
        ]
        if seed is None:
            candidates.sort()
        else:
            random.Random(seed).shuffle(candidates)
        return candidates[: max(0, int(fanout))]


@dataclass(frozen=True)
class RaftConfig:
    """Configuration for deterministic in-memory Raft election simulation."""

    election_timeout_ms: int = 300
    heartbeat_ms: int = 100
    min_battery_pct: float = 20.0
    min_link_quality: float = 0.4


class RaftRole(str, Enum):
    """Role within a simulated Raft cluster."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass(frozen=True)
class LogEntry:
    """A deterministic Raft log entry."""

    term: int
    data: bytes = b""


@dataclass(frozen=True)
class RequestVote:
    term: int
    candidate_id: NodeId
    last_log_index: int = 0
    last_log_term: int = 0


@dataclass(frozen=True)
class VoteGranted:
    term: int
    voter_id: NodeId
    granted: bool


@dataclass(frozen=True)
class AppendEntries:
    term: int
    leader_id: NodeId
    prev_log_index: int = 0
    prev_log_term: int = 0
    entries: tuple[LogEntry, ...] = ()
    leader_commit: int = 0


@dataclass(frozen=True)
class AppendEntriesAck:
    term: int
    follower_id: NodeId
    success: bool
    match_index: int


RaftMessage: TypeAlias = RequestVote | VoteGranted | AppendEntries | AppendEntriesAck


@dataclass
class RaftNode:
    """Pure in-memory Raft node used for cluster-head election simulation."""

    id: NodeId
    config: RaftConfig = field(default_factory=RaftConfig)
    role: RaftRole = RaftRole.FOLLOWER
    current_term: int = 0
    voted_for: NodeId | None = None
    log: list[LogEntry] = field(default_factory=list)
    commit_index: int = 0
    votes_received: int = 0
    elapsed_since_last_event_ms: int = 0

    def __post_init__(self) -> None:
        self.id = coerce_node_id(self.id)

    @staticmethod
    def is_eligible_leader(state: DroneState, config: RaftConfig | None = None) -> bool:
        cfg = config or RaftConfig()
        return state.battery_pct >= cfg.min_battery_pct and state.link_quality >= cfg.min_link_quality

    def tick(self, elapsed_ms: int | float, peers: Sequence[DroneState]) -> RaftMessage | None:
        """Advance timers and return a broadcast message when an event fires."""

        self.elapsed_since_last_event_ms += max(0, int(elapsed_ms))
        if self.role == RaftRole.LEADER:
            if self.elapsed_since_last_event_ms < self.config.heartbeat_ms:
                return None
            self.elapsed_since_last_event_ms = 0
            return AppendEntries(
                term=self.current_term,
                leader_id=self.id,
                prev_log_index=len(self.log),
                prev_log_term=self.log[-1].term if self.log else 0,
                entries=(),
                leader_commit=self.commit_index,
            )

        if self.elapsed_since_last_event_ms < self.config.election_timeout_ms:
            return None
        self.elapsed_since_last_event_ms = 0
        self.current_term += 1
        self.role = RaftRole.CANDIDATE
        self.voted_for = self.id
        self.votes_received = 1
        quorum = len(peers) // 2 + 1
        if quorum <= 1:
            self.role = RaftRole.LEADER
        return RequestVote(
            term=self.current_term,
            candidate_id=self.id,
            last_log_index=len(self.log),
            last_log_term=self.log[-1].term if self.log else 0,
        )

    def handle_message(self, message: RaftMessage) -> RaftMessage | None:
        """Process a simulated Raft message and optionally produce a reply."""

        if isinstance(message, RequestVote):
            if message.term > self.current_term:
                self.current_term = message.term
                self.role = RaftRole.FOLLOWER
                self.voted_for = None
            granted = message.term >= self.current_term and (
                self.voted_for is None or self.voted_for == message.candidate_id
            )
            if granted:
                self.voted_for = message.candidate_id
                self.elapsed_since_last_event_ms = 0
            return VoteGranted(term=self.current_term, voter_id=self.id, granted=granted)

        if isinstance(message, VoteGranted):
            if message.term == self.current_term and self.role == RaftRole.CANDIDATE and message.granted:
                self.votes_received += 1
            return None

        if isinstance(message, AppendEntries):
            if message.term >= self.current_term:
                self.current_term = message.term
                self.role = RaftRole.FOLLOWER
                self.voted_for = None
                self.elapsed_since_last_event_ms = 0
                self.log.extend(message.entries)
                if message.leader_commit > self.commit_index:
                    self.commit_index = min(message.leader_commit, len(self.log))
                return AppendEntriesAck(
                    term=self.current_term,
                    follower_id=self.id,
                    success=True,
                    match_index=len(self.log),
                )
            return AppendEntriesAck(
                term=self.current_term,
                follower_id=self.id,
                success=False,
                match_index=len(self.log),
            )

        return None

    def try_promote(self, cluster_size: int) -> None:
        if self.role == RaftRole.CANDIDATE and self.votes_received >= int(cluster_size) // 2 + 1:
            self.role = RaftRole.LEADER


__all__ = [
    "AppendEntries",
    "AppendEntriesAck",
    "GossipState",
    "LogEntry",
    "MeshTopology",
    "RaftConfig",
    "RaftMessage",
    "RaftNode",
    "RaftRole",
    "RequestVote",
    "VoteGranted",
]
