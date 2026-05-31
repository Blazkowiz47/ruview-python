"""Deterministic HOMECORE automation primitives for research tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .state import (
    Context,
    DomainEvent,
    EntityId,
    EventOrigin,
    EventSubscriber,
    ServiceCall,
    ServiceName,
    State,
    StateChangedEvent,
    StateMachine,
    _coerce_datetime,
    _empty_mapping,
    _freeze_mapping,
    _now_utc,
)


class RunMode(str, Enum):
    """Script run mode names used by Home Assistant automations."""

    SINGLE = "single"
    RESTART = "restart"
    QUEUED = "queued"
    PARALLEL = "parallel"
    IGNORE_FIRST = "ignore_first"


@dataclass(frozen=True, slots=True)
class TriggerContext:
    """Context produced by a matching state or event trigger."""

    platform: str
    entity_id: EntityId | None = None
    to_state: State | None = None
    from_state: State | None = None
    fired_at: datetime = field(default_factory=_now_utc)
    event_type: str | None = None
    event: StateChangedEvent | DomainEvent | None = None

    def __post_init__(self) -> None:
        entity = EntityId.parse(self.entity_id) if self.entity_id is not None else None
        object.__setattr__(self, "platform", str(self.platform))
        object.__setattr__(self, "entity_id", entity)
        object.__setattr__(self, "fired_at", _coerce_datetime(self.fired_at))
        if self.event_type is not None:
            object.__setattr__(self, "event_type", str(self.event_type))

    @classmethod
    def state_changed(cls, event: StateChangedEvent) -> TriggerContext:
        return cls(
            platform="state",
            entity_id=event.entity_id,
            to_state=event.new_state,
            from_state=event.old_state,
            fired_at=event.fired_at,
            event=event,
        )

    @classmethod
    def domain_event(cls, event: DomainEvent) -> TriggerContext:
        return cls(
            platform="event",
            fired_at=event.fired_at,
            event_type=event.event_type,
            event=event,
        )


@dataclass(frozen=True, slots=True, init=False)
class StateTrigger:
    """Fire when an entity state change matches optional from/to values."""

    entity_id: EntityId
    from_state: str | None = None
    to_state: str | None = None

    def __init__(
        self,
        entity_id: EntityId | str,
        from_state: str | None = None,
        to_state: str | None = None,
        *,
        from_: str | None = None,
        to: str | None = None,
    ) -> None:
        if from_ is not None:
            from_state = from_
        if to is not None:
            to_state = to
        object.__setattr__(self, "entity_id", EntityId.parse(entity_id))
        object.__setattr__(self, "from_state", None if from_state is None else str(from_state))
        object.__setattr__(self, "to_state", None if to_state is None else str(to_state))


@dataclass(frozen=True, slots=True)
class NumericStateTrigger:
    """Fire when an entity's numeric state satisfies threshold bounds."""

    entity_id: EntityId | str
    above: float | None = None
    below: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", EntityId.parse(self.entity_id))
        if self.above is not None:
            object.__setattr__(self, "above", float(self.above))
        if self.below is not None:
            object.__setattr__(self, "below", float(self.below))


@dataclass(frozen=True, slots=True)
class EventTrigger:
    """Fire when a named domain event is published."""

    event_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", str(self.event_type))


@dataclass(frozen=True, slots=True)
class TimeTrigger:
    """Stored time trigger configuration.

    The research engine does not run timers; callers can inspect this
    dataclass or evaluate time externally.
    """

    at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", str(self.at))


Trigger = StateTrigger | NumericStateTrigger | EventTrigger | TimeTrigger


@dataclass(frozen=True, slots=True)
class StateCondition:
    """Pass when an entity's current state equals ``state``."""

    entity_id: EntityId | str
    state: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", EntityId.parse(self.entity_id))
        object.__setattr__(self, "state", str(self.state))


@dataclass(frozen=True, slots=True)
class NumericStateCondition:
    """Pass when an entity's current numeric state is within bounds."""

    entity_id: EntityId | str
    above: float | None = None
    below: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", EntityId.parse(self.entity_id))
        if self.above is not None:
            object.__setattr__(self, "above", float(self.above))
        if self.below is not None:
            object.__setattr__(self, "below", float(self.below))


@dataclass(frozen=True, slots=True)
class TemplateCondition:
    """Tiny stdlib-only truthy literal condition placeholder."""

    value_template: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_template", str(self.value_template))


@dataclass(frozen=True, slots=True)
class AndCondition:
    conditions: tuple[Condition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", _tuple_conditions(self.conditions))


@dataclass(frozen=True, slots=True)
class OrCondition:
    conditions: tuple[Condition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", _tuple_conditions(self.conditions))


@dataclass(frozen=True, slots=True)
class NotCondition:
    conditions: tuple[Condition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", _tuple_conditions(self.conditions))


Condition = StateCondition | NumericStateCondition | TemplateCondition | AndCondition | OrCondition | NotCondition


@dataclass(frozen=True, slots=True)
class ServiceAction:
    """Record and optionally dispatch a local service call."""

    domain: str
    service: str
    data: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", str(self.domain))
        object.__setattr__(self, "service", str(self.service))
        object.__setattr__(self, "data", _freeze_mapping(self.data))


@dataclass(frozen=True, slots=True)
class SceneAction:
    """Activate a scene through ``homeassistant.turn_on``."""

    scene: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene", str(self.scene))


@dataclass(frozen=True, slots=True)
class CallbackAction:
    """Execute a registered local callback or callable."""

    callback: str | Callable[[ActionContext], object]
    data: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _freeze_mapping(self.data))


@dataclass(frozen=True, slots=True)
class DelayAction:
    """Record a deterministic delay without sleeping."""

    seconds: float

    def __post_init__(self) -> None:
        if float(self.seconds) < 0.0:
            raise ValueError("delay seconds must be non-negative")
        object.__setattr__(self, "seconds", float(self.seconds))


@dataclass(frozen=True, slots=True)
class FireEventAction:
    """Fire a local domain event from an automation action."""

    event_type: str
    event_data: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", str(self.event_type))
        object.__setattr__(self, "event_data", _freeze_mapping(self.event_data))


Action = ServiceAction | SceneAction | CallbackAction | DelayAction | FireEventAction


@dataclass(frozen=True, slots=True)
class Automation:
    """Parsed automation definition used by the deterministic engine."""

    id: str
    trigger: tuple[Trigger, ...] = field(default_factory=tuple)
    action: tuple[Action, ...] = field(default_factory=tuple)
    condition: tuple[Condition, ...] = field(default_factory=tuple)
    alias: str | None = None
    description: str | None = None
    enabled: bool = True
    mode: RunMode = RunMode.SINGLE
    max: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "trigger", _tuple_items(self.trigger, _TRIGGER_TYPES))
        object.__setattr__(self, "action", _tuple_items(self.action, _ACTION_TYPES))
        object.__setattr__(self, "condition", _tuple_items(self.condition, _CONDITION_TYPES))
        object.__setattr__(self, "mode", RunMode(self.mode))
        if self.alias is not None:
            object.__setattr__(self, "alias", str(self.alias))
        if self.description is not None:
            object.__setattr__(self, "description", str(self.description))
        if self.max is not None:
            object.__setattr__(self, "max", int(self.max))

    @classmethod
    def new(cls, id: str, triggers: Sequence[Trigger], actions: Sequence[Action]) -> Automation:
        return cls(id=id, trigger=tuple(triggers), action=tuple(actions))


@dataclass(frozen=True, slots=True)
class ActionContext:
    """Runtime context passed to callback actions."""

    engine: AutomationEngine
    automation_id: str
    trigger: TriggerContext
    context: Context
    data: Mapping[str, Any] = field(default_factory=_empty_mapping)


@dataclass(frozen=True, slots=True)
class CallbackRecord:
    """Deterministic record of a callback action execution."""

    name: str
    data: Mapping[str, Any]
    context: Context
    automation_id: str
    trigger: TriggerContext
    result: object = None


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Deterministic record of one executed action."""

    automation_id: str
    action: Action
    context: Context
    result: object = None


@dataclass(frozen=True, slots=True)
class AutomationRun:
    """Deterministic record of one automation run."""

    automation_id: str
    trigger: TriggerContext
    context: Context
    actions: tuple[ActionRecord, ...]


_TRIGGER_TYPES = (StateTrigger, NumericStateTrigger, EventTrigger, TimeTrigger)
_CONDITION_TYPES = (
    StateCondition,
    NumericStateCondition,
    TemplateCondition,
    AndCondition,
    OrCondition,
    NotCondition,
)
_ACTION_TYPES = (ServiceAction, SceneAction, CallbackAction, DelayAction, FireEventAction)


def _tuple_items(value: object, allowed: tuple[type[object], ...]) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, allowed):
        return (value,)
    return tuple(value)  # type: ignore[arg-type]


def _tuple_conditions(value: object) -> tuple[Condition, ...]:
    return _tuple_items(value, _CONDITION_TYPES)


def trigger_matches(trigger: Trigger, context: TriggerContext) -> bool:
    """Return whether ``trigger`` matches ``context``."""

    if isinstance(trigger, StateTrigger):
        if context.platform != "state" or context.entity_id != trigger.entity_id:
            return False
        if trigger.from_state is not None and _state_value(context.from_state) != trigger.from_state:
            return False
        if trigger.to_state is not None and _state_value(context.to_state) != trigger.to_state:
            return False
        return True
    if isinstance(trigger, NumericStateTrigger):
        if context.platform != "state" or context.entity_id != trigger.entity_id or context.to_state is None:
            return False
        value = _parse_float(context.to_state.state)
        if value is None:
            return False
        return (trigger.above is None or value > trigger.above) and (
            trigger.below is None or value < trigger.below
        )
    if isinstance(trigger, EventTrigger):
        return context.platform == "event" and context.event_type == trigger.event_type
    if isinstance(trigger, TimeTrigger):
        return False
    raise TypeError(f"unknown trigger type {type(trigger).__name__}")


def condition_passes(condition: Condition, states: StateMachine) -> bool:
    """Evaluate a condition against the current state machine snapshot."""

    if isinstance(condition, StateCondition):
        state = states.get(condition.entity_id)
        return state is not None and state.state == condition.state
    if isinstance(condition, NumericStateCondition):
        state = states.get(condition.entity_id)
        value = _parse_float(state.state) if state is not None else None
        if value is None:
            return False
        return (condition.above is None or value > condition.above) and (
            condition.below is None or value < condition.below
        )
    if isinstance(condition, TemplateCondition):
        return condition.value_template.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(condition, AndCondition):
        return all(condition_passes(child, states) for child in condition.conditions)
    if isinstance(condition, OrCondition):
        return any(condition_passes(child, states) for child in condition.conditions)
    if isinstance(condition, NotCondition):
        return not any(condition_passes(child, states) for child in condition.conditions)
    raise TypeError(f"unknown condition type {type(condition).__name__}")


def _state_value(state: State | None) -> str:
    return state.state if state is not None else "unavailable"


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


class AutomationEngine:
    """Synchronous automation engine for state and domain events."""

    def __init__(
        self,
        states: StateMachine | None = None,
        *,
        auto_start: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.states = states if states is not None else StateMachine(clock=clock)
        self.automations: list[Automation] = []
        self.service_calls: list[ServiceCall] = []
        self.callback_records: list[CallbackRecord] = []
        self.action_records: list[ActionRecord] = []
        self.run_log: list[AutomationRun] = []
        self.event_log: list[DomainEvent] = []
        self._clock = clock or _now_utc
        self._subscription: EventSubscriber | None = None
        self._service_handlers: dict[ServiceName, Callable[[ServiceCall], object] | None] = {}
        self._callbacks: dict[str, Callable[[ActionContext], object]] = {}
        self._ignore_first_seen: set[str] = set()
        if auto_start:
            self.start()

    def _timestamp(self, now: datetime | None = None) -> datetime:
        return _coerce_datetime(now if now is not None else self._clock())

    def start(self) -> AutomationEngine:
        if self._subscription is None:
            self._subscription = self.states.subscribe(self.handle_state_changed)
        return self

    def stop(self) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None

    def register(self, automation: Automation) -> Automation:
        self.automations.append(automation)
        return automation

    def register_service(
        self,
        domain: ServiceName | str,
        service: str | None = None,
        callback: Callable[[ServiceCall], object] | None = None,
    ) -> ServiceName:
        name = ServiceName.parse(domain, service)
        self._service_handlers[name] = callback
        return name

    def register_callback(self, name: str, callback: Callable[[ActionContext], object]) -> str:
        callback_name = str(name)
        self._callbacks[callback_name] = callback
        return callback_name

    def handle_state_changed(self, event: StateChangedEvent) -> list[AutomationRun]:
        return self._process_trigger_context(TriggerContext.state_changed(event))

    def fire_event(
        self,
        event_type: str,
        event_data: Mapping[str, Any] | None = None,
        context: Context | None = None,
        *,
        origin: EventOrigin = EventOrigin.LOCAL,
        now: datetime | None = None,
    ) -> DomainEvent:
        timestamp = self._timestamp(now)
        event = DomainEvent(
            event_type=str(event_type),
            event_data=event_data or {},
            origin=origin,
            context=context or Context.new(),
            fired_at=timestamp,
        )
        self.event_log.append(event)
        self._process_trigger_context(TriggerContext.domain_event(event))
        return event

    def _process_trigger_context(self, context: TriggerContext) -> list[AutomationRun]:
        runs: list[AutomationRun] = []
        for automation in tuple(self.automations):
            run = self._maybe_run(automation, context)
            if run is not None:
                runs.append(run)
        return runs

    def _maybe_run(self, automation: Automation, trigger_context: TriggerContext) -> AutomationRun | None:
        if not automation.enabled:
            return None
        if not any(trigger_matches(trigger, trigger_context) for trigger in automation.trigger):
            return None
        if automation.mode is RunMode.IGNORE_FIRST and automation.id not in self._ignore_first_seen:
            self._ignore_first_seen.add(automation.id)
            return None
        if not all(condition_passes(condition, self.states) for condition in automation.condition):
            return None

        context = self._child_context(trigger_context)
        action_records = [self._execute_action(automation, action, trigger_context, context) for action in automation.action]
        run = AutomationRun(
            automation_id=automation.id,
            trigger=trigger_context,
            context=context,
            actions=tuple(action_records),
        )
        self.run_log.append(run)
        return run

    def _child_context(self, trigger_context: TriggerContext) -> Context:
        if isinstance(trigger_context.event, DomainEvent):
            return Context.child_of(trigger_context.event.context)
        if trigger_context.to_state is not None:
            return Context.child_of(trigger_context.to_state.context)
        if trigger_context.from_state is not None:
            return Context.child_of(trigger_context.from_state.context)
        return Context.new()

    def _execute_action(
        self,
        automation: Automation,
        action: Action,
        trigger_context: TriggerContext,
        context: Context,
    ) -> ActionRecord:
        if isinstance(action, ServiceAction):
            result = self._call_service(ServiceName(action.domain, action.service), action.data, context)
        elif isinstance(action, SceneAction):
            result = self._call_service(
                ServiceName("homeassistant", "turn_on"),
                {"entity_id": action.scene},
                context,
            )
        elif isinstance(action, CallbackAction):
            result = self._call_callback(automation, action, trigger_context, context)
        elif isinstance(action, DelayAction):
            result = {"delay_seconds": action.seconds}
        elif isinstance(action, FireEventAction):
            result = self.fire_event(action.event_type, action.event_data, context=context)
        else:
            raise TypeError(f"unknown action type {type(action).__name__}")

        record = ActionRecord(automation.id, action, context, result)
        self.action_records.append(record)
        return record

    def _call_service(self, name: ServiceName, data: Mapping[str, Any], context: Context) -> object:
        call = ServiceCall(name=name, data=data, context=context)
        self.service_calls.append(call)
        handler = self._service_handlers.get(name)
        return handler(call) if handler is not None else call

    def _call_callback(
        self,
        automation: Automation,
        action: CallbackAction,
        trigger_context: TriggerContext,
        context: Context,
    ) -> object:
        action_context = ActionContext(
            engine=self,
            automation_id=automation.id,
            trigger=trigger_context,
            context=context,
            data=action.data,
        )
        if isinstance(action.callback, str):
            name = action.callback
            callback = self._callbacks.get(name)
        else:
            callback = action.callback
            name = getattr(callback, "__name__", "callback")
        result = callback(action_context) if callback is not None else None
        record = CallbackRecord(name, action.data, context, automation.id, trigger_context, result)
        self.callback_records.append(record)
        return result


__all__ = [
    "Action",
    "ActionContext",
    "ActionRecord",
    "AndCondition",
    "Automation",
    "AutomationEngine",
    "AutomationRun",
    "CallbackAction",
    "CallbackRecord",
    "Condition",
    "DelayAction",
    "EventTrigger",
    "FireEventAction",
    "NotCondition",
    "NumericStateCondition",
    "NumericStateTrigger",
    "OrCondition",
    "RunMode",
    "SceneAction",
    "ServiceAction",
    "StateCondition",
    "StateTrigger",
    "TemplateCondition",
    "TimeTrigger",
    "Trigger",
    "TriggerContext",
    "condition_passes",
    "trigger_matches",
]
