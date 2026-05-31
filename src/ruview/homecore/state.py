"""Deterministic HOMECORE state primitives for research tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Self
from uuid import uuid4


class EntityIdError(ValueError):
    """Raised when an entity id is outside the HOMECORE ASCII subset."""


@dataclass(frozen=True, order=True, slots=True)
class EntityId:
    """Validated ``domain.name`` entity identifier.

    The research subset mirrors the Rust HOMECORE P1 rule:
    ``[a-z0-9_]+.[a-z0-9_]+`` with ASCII characters only.
    """

    value: str

    def __post_init__(self) -> None:
        value = str(self.value)
        domain, sep, name = value.partition(".")
        if not sep:
            raise EntityIdError(f"entity_id {value!r} is missing '.' between domain and name")
        if not domain:
            raise EntityIdError(f"entity_id {value!r} has an empty domain segment")
        if not name:
            raise EntityIdError(f"entity_id {value!r} has an empty name segment")
        for char in domain + name:
            if not (char.isascii() and (char.islower() or char.isdigit() or char == "_")):
                raise EntityIdError(
                    f"entity_id {value!r} contains invalid character {char!r}; "
                    "only [a-z0-9_] are allowed"
                )
        object.__setattr__(self, "value", value)

    @classmethod
    def parse(cls, value: EntityId | str) -> EntityId:
        """Return ``value`` as a validated :class:`EntityId`."""

        if isinstance(value, EntityId):
            return value
        return cls(str(value))

    @classmethod
    def new(cls, value: str) -> EntityId:
        """Alias matching the Rust constructor name."""

        return cls.parse(value)

    @property
    def domain(self) -> str:
        return self.value.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.value.split(".", 1)[1]

    def as_str(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"EntityId({self.value})"


@dataclass(frozen=True, slots=True)
class Context:
    """Causality context for state changes, events, and service calls."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    parent_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        if self.user_id is not None:
            object.__setattr__(self, "user_id", str(self.user_id))
        if self.parent_id is not None:
            object.__setattr__(self, "parent_id", str(self.parent_id))

    @classmethod
    def new(cls) -> Context:
        return cls()

    @classmethod
    def with_user(cls, user_id: str) -> Context:
        return cls(user_id=str(user_id))

    @classmethod
    def child_of(cls, parent: Context) -> Context:
        return cls(user_id=parent.user_id, parent_id=parent.id)


class EventType:
    """Well-known HOMECORE event type names."""

    STATE_CHANGED: ClassVar[str] = "state_changed"
    SERVICE_REGISTERED: ClassVar[str] = "service_registered"
    SERVICE_REMOVED: ClassVar[str] = "service_removed"
    CALL_SERVICE: ClassVar[str] = "call_service"
    HOMECORE_START: ClassVar[str] = "homeassistant_start"
    HOMECORE_STARTED: ClassVar[str] = "homeassistant_started"
    HOMECORE_STOP: ClassVar[str] = "homeassistant_stop"


class EventOrigin(str, Enum):
    """Origin of a domain event."""

    LOCAL = "local"
    REMOTE = "remote"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: datetime | None) -> datetime:
    if value is None:
        return _now_utc()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _empty_mapping() -> Mapping[str, Any]:
    return MappingProxyType({})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(val) for key, val in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return _empty_mapping()
    frozen = _freeze_value(value)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"attributes must be a mapping, got {type(value).__name__}")
    return frozen


_MISSING = object()


@dataclass(frozen=True, slots=True)
class State:
    """Immutable snapshot for one entity at one point in time."""

    entity_id: EntityId | str
    state: str
    attributes: Mapping[str, Any] = field(default_factory=_empty_mapping)
    context: Context = field(default_factory=Context.new)
    last_changed: datetime | None = None
    last_updated: datetime | None = None

    def __post_init__(self) -> None:
        last_updated = _coerce_datetime(self.last_updated)
        last_changed = _coerce_datetime(self.last_changed) if self.last_changed is not None else last_updated
        object.__setattr__(self, "entity_id", EntityId.parse(self.entity_id))
        object.__setattr__(self, "state", str(self.state))
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))
        object.__setattr__(self, "context", self.context if isinstance(self.context, Context) else Context())
        object.__setattr__(self, "last_changed", last_changed)
        object.__setattr__(self, "last_updated", last_updated)

    @classmethod
    def new(
        cls,
        entity_id: EntityId | str,
        state: str,
        attributes: Mapping[str, Any] | None = None,
        context: Context | None = None,
        *,
        now: datetime | None = None,
    ) -> State:
        timestamp = _coerce_datetime(now)
        return cls(
            entity_id=entity_id,
            state=state,
            attributes=attributes or {},
            context=context or Context.new(),
            last_changed=timestamp,
            last_updated=timestamp,
        )

    def next(
        self,
        state: str,
        attributes: Mapping[str, Any] | object = _MISSING,
        context: Context | None = None,
        *,
        now: datetime | None = None,
    ) -> State:
        """Return the next immutable snapshot.

        ``last_changed`` is preserved when the state string is unchanged;
        ``last_updated`` is bumped for every write, including no-op writes.
        """

        new_state = str(state)
        timestamp = _coerce_datetime(now)
        attrs = self.attributes if attributes is _MISSING else attributes
        last_changed = self.last_changed if new_state == self.state else timestamp
        return State(
            entity_id=self.entity_id,
            state=new_state,
            attributes=attrs,  # type: ignore[arg-type]
            context=context or Context.new(),
            last_changed=last_changed,
            last_updated=timestamp,
        )


@dataclass(frozen=True, slots=True)
class StateChangedEvent:
    """State-change event carrying old and new snapshots."""

    entity_id: EntityId | str
    old_state: State | None
    new_state: State | None
    fired_at: datetime = field(default_factory=_now_utc)
    context: Context = field(default_factory=Context.new)

    event_type: ClassVar[str] = EventType.STATE_CHANGED

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", EntityId.parse(self.entity_id))
        object.__setattr__(self, "fired_at", _coerce_datetime(self.fired_at))


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Untyped integration event for automation event triggers."""

    event_type: str
    event_data: Mapping[str, Any] = field(default_factory=_empty_mapping)
    origin: EventOrigin = EventOrigin.LOCAL
    context: Context = field(default_factory=Context.new)
    fired_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", str(self.event_type))
        object.__setattr__(self, "event_data", _freeze_mapping(self.event_data))
        object.__setattr__(self, "origin", EventOrigin(self.origin))
        object.__setattr__(self, "fired_at", _coerce_datetime(self.fired_at))


@dataclass(frozen=True, order=True, slots=True)
class ServiceName:
    """Service name within a domain, such as ``light.turn_on``."""

    domain: str
    service: str

    def __post_init__(self) -> None:
        domain = str(self.domain)
        service = str(self.service)
        if not domain:
            raise ValueError("service domain must be non-empty")
        if not service:
            raise ValueError("service name must be non-empty")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "service", service)

    @classmethod
    def parse(cls, value: ServiceName | str, service: str | None = None) -> ServiceName:
        if isinstance(value, ServiceName):
            return value
        if service is not None:
            return cls(str(value), service)
        domain, sep, service_name = str(value).partition(".")
        if not sep:
            raise ValueError(f"service name {value!r} is missing '.' between domain and service")
        return cls(domain, service_name)

    def as_str(self) -> str:
        return f"{self.domain}.{self.service}"

    def __str__(self) -> str:
        return self.as_str()


@dataclass(frozen=True, slots=True)
class ServiceCall:
    """Recorded local service call payload."""

    name: ServiceName
    data: Mapping[str, Any] = field(default_factory=_empty_mapping)
    context: Context = field(default_factory=Context.new)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", ServiceName.parse(self.name))
        object.__setattr__(self, "data", _freeze_mapping(self.data))


class EventSubscriber:
    """Synchronous in-memory state-event subscriber."""

    def __init__(
        self,
        callback: Callable[[StateChangedEvent], object] | None = None,
        owner: StateMachine | None = None,
    ) -> None:
        self._callback = callback
        self._owner = owner
        self._events: list[StateChangedEvent] = []

    @property
    def events(self) -> tuple[StateChangedEvent, ...]:
        return tuple(self._events)

    def notify(self, event: StateChangedEvent) -> None:
        self._events.append(event)
        if self._callback is not None:
            self._callback(event)

    def drain(self) -> list[StateChangedEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def clear(self) -> None:
        self._events.clear()

    def unsubscribe(self) -> None:
        if self._owner is not None:
            self._owner._unsubscribe(self)
            self._owner = None

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[StateChangedEvent]:
        return iter(self.events)


class StateMachine:
    """Synchronous HOMECORE state machine with deterministic event logging."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or _now_utc
        self._states: dict[EntityId, State] = {}
        self._event_log: list[StateChangedEvent] = []
        self._subscribers: list[EventSubscriber] = []

    def _timestamp(self, now: datetime | None = None) -> datetime:
        return _coerce_datetime(now if now is not None else self._clock())

    def subscribe(
        self,
        callback: Callable[[StateChangedEvent], object] | None = None,
    ) -> EventSubscriber:
        subscriber = EventSubscriber(callback, owner=self)
        self._subscribers.append(subscriber)
        return subscriber

    def _unsubscribe(self, subscriber: EventSubscriber) -> None:
        self._subscribers = [existing for existing in self._subscribers if existing is not subscriber]

    def _publish(self, event: StateChangedEvent) -> None:
        self._event_log.append(event)
        for subscriber in tuple(self._subscribers):
            subscriber.notify(event)

    @property
    def event_log(self) -> tuple[StateChangedEvent, ...]:
        return tuple(self._event_log)

    @property
    def events(self) -> tuple[StateChangedEvent, ...]:
        return self.event_log

    def clear_event_log(self) -> None:
        self._event_log.clear()

    def get(self, entity_id: EntityId | str) -> State | None:
        return self._states.get(EntityId.parse(entity_id))

    def set(
        self,
        entity_id: EntityId | str,
        state: str,
        attributes: Mapping[str, Any] | None = None,
        context: Context | None = None,
        *,
        now: datetime | None = None,
    ) -> State:
        entity = EntityId.parse(entity_id)
        timestamp = self._timestamp(now)
        ctx = context or Context.new()
        old = self._states.get(entity)
        new_state = str(state)
        attrs = attributes or {}
        next_state = (
            old.next(new_state, attrs, ctx, now=timestamp)
            if old is not None
            else State.new(entity, new_state, attrs, ctx, now=timestamp)
        )
        is_noop = old is not None and old.state == new_state and old.attributes == next_state.attributes

        self._states[entity] = next_state
        if not is_noop:
            self._publish(StateChangedEvent(entity, old, next_state, fired_at=timestamp, context=ctx))
        return next_state

    def remove(
        self,
        entity_id: EntityId | str,
        context: Context | None = None,
        *,
        now: datetime | None = None,
    ) -> State | None:
        entity = EntityId.parse(entity_id)
        old = self._states.pop(entity, None)
        if old is None:
            return None
        timestamp = self._timestamp(now)
        ctx = context or Context.new()
        self._publish(StateChangedEvent(entity, old, None, fired_at=timestamp, context=ctx))
        return old

    def all(self) -> list[State]:
        return [self._states[entity] for entity in sorted(self._states)]

    def all_by_domain(self, domain: str) -> list[State]:
        domain_str = str(domain)
        return [state for state in self.all() if state.entity_id.domain == domain_str]

    def len(self) -> int:
        return len(self._states)

    def is_empty(self) -> bool:
        return not self._states

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, entity_id: object) -> bool:
        try:
            entity = EntityId.parse(entity_id)  # type: ignore[arg-type]
        except (EntityIdError, TypeError, ValueError):
            return False
        return entity in self._states

    def __iter__(self) -> Iterator[State]:
        return iter(self.all())


__all__ = [
    "Context",
    "DomainEvent",
    "EntityId",
    "EntityIdError",
    "EventOrigin",
    "EventSubscriber",
    "EventType",
    "ServiceCall",
    "ServiceName",
    "State",
    "StateChangedEvent",
    "StateMachine",
]
