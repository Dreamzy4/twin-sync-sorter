"""Debounced overheat / overload state machine tests.

The hysteresis logic in ``Telemetry._update_hysteresis`` is what stops
dashboard warning badges from flickering when a measured signal oscillates
across the threshold. We exercise it directly without instantiating the
real Telemetry (which spawns a daemon thread and opens an HTTP session) by
binding the unbound method to a minimal stub that carries only the three
state fields the method touches.
"""

import telemetry


class _HysteresisStub:
    """Stand-in self for the unbound _update_hysteresis call.

    Mirrors the three private attributes the method reads/writes for a
    single named state machine (here: 'overheat').
    """

    _update_hysteresis = telemetry.Telemetry._update_hysteresis

    def __init__(self):
        self._overheat_hits = 0
        self._overheat_active = False
        self._overheat_cool = 0


def test_overheat_requires_three_consecutive_hits():
    s = _HysteresisStub()
    assert s._update_hysteresis("overheat", True, hits_n=3, cool_n=2) is False
    assert s._update_hysteresis("overheat", True, hits_n=3, cool_n=2) is False
    assert s._update_hysteresis("overheat", True, hits_n=3, cool_n=2) is True
    assert s._overheat_active is True


def test_overheat_resets_on_cool_break():
    s = _HysteresisStub()
    s._update_hysteresis("overheat", True, hits_n=3, cool_n=2)
    s._update_hysteresis("overheat", True, hits_n=3, cool_n=2)
    # A single cool tick before reaching hits_n must reset the counter.
    s._update_hysteresis("overheat", False, hits_n=3, cool_n=2)
    assert s._overheat_hits == 0
    assert s._overheat_active is False
    # Two more hot ticks should not yet trigger - we are back at 0.
    assert s._update_hysteresis("overheat", True, hits_n=3, cool_n=2) is False
    assert s._update_hysteresis("overheat", True, hits_n=3, cool_n=2) is False
    assert s._overheat_active is False


def test_overheat_clears_after_cool_n_ticks():
    s = _HysteresisStub()
    # Trigger.
    for _ in range(3):
        s._update_hysteresis("overheat", True, hits_n=3, cool_n=2)
    assert s._overheat_active is True

    # First cool tick: still active (cool=1, threshold 2).
    assert s._update_hysteresis("overheat", False, hits_n=3, cool_n=2) is True
    # Second cool tick: clears.
    assert s._update_hysteresis("overheat", False, hits_n=3, cool_n=2) is False
    assert s._overheat_active is False
    assert s._overheat_hits == 0
    assert s._overheat_cool == 0


def test_overheat_cool_counter_resets_on_brief_spike():
    s = _HysteresisStub()
    for _ in range(3):
        s._update_hysteresis("overheat", True, hits_n=3, cool_n=2)
    assert s._overheat_active is True

    # One cool tick (cool=1), then a hot spike must reset the cool counter
    # so the alert stays active rather than flickering off.
    s._update_hysteresis("overheat", False, hits_n=3, cool_n=2)
    assert s._update_hysteresis("overheat", True, hits_n=3, cool_n=2) is True
    assert s._overheat_cool == 0
    assert s._overheat_active is True
