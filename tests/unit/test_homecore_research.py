from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ruview.homecore import (
    Automation,
    AutomationEngine,
    CallbackAction,
    Context,
    EntityId,
    EntityIdError,
    EventTrigger,
    RunMode,
    ServiceAction,
    StateCondition,
    StateMachine,
    StateTrigger,
)


BASE = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)


def at(seconds: int) -> datetime:
    return BASE + timedelta(seconds=seconds)


def test_entity_id_validation_ascii_domain_name_subset() -> None:
    entity_id = EntityId.parse("light.living_room")

    assert entity_id.domain == "light"
    assert entity_id.name == "living_room"
    assert entity_id.as_str() == "light.living_room"
    assert str(entity_id) == "light.living_room"

    for invalid in ("light_living_room", ".kitchen", "light.", "light.Kitchen", "light.küche", "light.a.b"):
        with pytest.raises(EntityIdError):
            EntityId.parse(invalid)


def test_state_timestamps_and_noop_event_suppression() -> None:
    states = StateMachine()
    subscriber = states.subscribe()

    first = states.set("sensor.temperature", "20", {"unit": "C"}, now=at(0))
    changed_attrs = states.set("sensor.temperature", "20", {"unit": "F"}, now=at(1))
    noop = states.set("sensor.temperature", "20", {"unit": "F"}, now=at(2))

    assert first.last_changed == at(0)
    assert first.last_updated == at(0)
    assert changed_attrs.last_changed == first.last_changed
    assert changed_attrs.last_updated == at(1)
    assert noop.last_changed == first.last_changed
    assert noop.last_updated == at(2)
    assert states.get("sensor.temperature") == noop

    assert len(states.event_log) == 2
    assert len(subscriber.events) == 2
    assert states.event_log[0].old_state is None
    assert states.event_log[1].old_state == first
    assert states.event_log[1].new_state == changed_attrs


def test_domain_filtering_and_removal_events_are_deterministic() -> None:
    states = StateMachine()
    states.set("light.bedroom", "off", now=at(0))
    states.set("sensor.temperature", "21", now=at(1))
    states.set("light.kitchen", "on", now=at(2))

    assert [state.entity_id.as_str() for state in states.all()] == [
        "light.bedroom",
        "light.kitchen",
        "sensor.temperature",
    ]
    assert [state.entity_id.as_str() for state in states.all_by_domain("light")] == [
        "light.bedroom",
        "light.kitchen",
    ]

    removed = states.remove("light.kitchen", now=at(3))

    assert removed is not None
    assert removed.state == "on"
    assert len(states) == 2
    assert states.remove("light.unknown", now=at(4)) is None
    assert len(states.event_log) == 4
    assert states.event_log[-1].entity_id == EntityId.parse("light.kitchen")
    assert states.event_log[-1].old_state == removed
    assert states.event_log[-1].new_state is None
    assert states.event_log[-1].fired_at == at(3)


def test_automation_state_and_event_triggers_conditions_and_actions() -> None:
    states = StateMachine()
    engine = AutomationEngine(states)
    service_payloads: list[dict[str, object]] = []
    callback_hits: list[tuple[str, str, str]] = []

    engine.register_service(
        "light",
        "turn_on",
        lambda call: service_payloads.append(dict(call.data)) or "service-ok",
    )
    engine.register_callback(
        "mark",
        lambda ctx: callback_hits.append(
            (ctx.automation_id, ctx.trigger.entity_id.as_str(), str(ctx.data["tag"]))
        )
        or "callback-ok",
    )
    states.set("sensor.ready", "yes", now=at(0))

    engine.register(
        Automation(
            "auto.living",
            trigger=[StateTrigger("switch.living", to="on")],
            condition=[StateCondition("sensor.ready", "yes")],
            action=[
                ServiceAction("light", "turn_on", {"brightness": 100}),
                CallbackAction("mark", {"tag": "ran"}),
            ],
        )
    )
    parent_context = Context.with_user("alice")
    states.set("switch.living", "on", context=parent_context, now=at(1))

    assert [call.name.as_str() for call in engine.service_calls] == ["light.turn_on"]
    assert engine.service_calls[0].data == {"brightness": 100}
    assert engine.service_calls[0].context.parent_id == parent_context.id
    assert service_payloads == [{"brightness": 100}]
    assert callback_hits == [("auto.living", "switch.living", "ran")]
    assert len(engine.run_log) == 1

    engine.register(
        Automation(
            "auto.event",
            trigger=[EventTrigger("homecore_boot")],
            action=[ServiceAction("notify", "create", {"message": "booted"})],
        )
    )
    engine.fire_event("homecore_boot", {"source": "test"}, now=at(2))

    assert engine.event_log[-1].event_type == "homecore_boot"
    assert [call.name.as_str() for call in engine.service_calls] == [
        "light.turn_on",
        "notify.create",
    ]
    assert engine.service_calls[-1].data == {"message": "booted"}
    assert len(engine.run_log) == 2


def test_disabled_automations_do_not_run() -> None:
    states = StateMachine()
    engine = AutomationEngine(states)
    engine.register(
        Automation(
            "auto.disabled",
            trigger=[StateTrigger("switch.living", to="on")],
            action=[ServiceAction("light", "turn_on")],
            enabled=False,
        )
    )

    states.set("switch.living", "on", now=at(0))

    assert engine.service_calls == []
    assert engine.run_log == []


def test_run_mode_defaults_to_single() -> None:
    automation = Automation.new("auto.default", [EventTrigger("tick")], [])
    restarted = Automation("auto.restart", trigger=EventTrigger("tick"), action=ServiceAction("a", "b"), mode="restart")

    assert automation.enabled is True
    assert automation.mode is RunMode.SINGLE
    assert restarted.mode is RunMode.RESTART
    assert restarted.trigger == (EventTrigger("tick"),)
    assert restarted.action == (ServiceAction("a", "b"),)
