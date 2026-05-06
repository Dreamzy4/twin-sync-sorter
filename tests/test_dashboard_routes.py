"""Route-level smoke tests for the Flask dashboard server.

Uses the in-process Flask test client (no real HTTP socket). Verifies that
the public REST contract used by the Isaac runtime and the browser front-end
stays stable:

* GET  /api/cv/config exposes scene-derived UI reference values.
* POST /api/cv/update accepts a twin-sync sample and persists it.
* GET  /api/cv exposes the twin-sync history that the browser dashboard polls.
"""


def test_cv_config_returns_scene_constants(flask_client):
    rv = flask_client.get("/api/cv/config")
    assert rv.status_code == 200, rv.data
    data = rv.get_json()
    for key in (
        "pickup_zone",
        "workspace_high",
        "workspace_low",
        "containers",
        "pickup_colors",
        "usd_yaw_tolerance",
        "usd_match_max_distance",
    ):
        assert key in data, f"missing key {key!r} in /api/cv/config response"
    assert "red" in data["containers"] and "blue" in data["containers"]


def test_cv_update_persists_twin_sync(flask_client, cv_state_clean):
    payload = {"twin_sync_mm": 4.2, "cycle": 1, "status": "test"}
    rv = flask_client.post("/api/cv/update", json=payload)
    assert rv.status_code == 200
    # Response carries 'ok'; the reset_requested flag piggybacks here too,
    # so this is the loose form rather than strict equality.
    assert rv.get_json()["ok"] is True

    rv = flask_client.get("/api/cv")
    assert rv.status_code == 200
    history = rv.get_json()["twin_sync_history"]
    assert history == [4.2], f"expected [4.2], got {history!r}"


def test_clear_history_signals_reset_to_orchestrator(flask_client, cv_state_clean):
    """CLEAR must surface a reset_requested flag in the next /api/cv/update
    response, then clear it. CVPoster on the orchestrator side reads this
    and triggers _reset_runtime_state() so the dashboard CLEAR also wipes
    the in-process processed-cube memos.
    """
    flask_client.post("/api/cv/clear-history")

    # First update after CLEAR carries the flag.
    rv1 = flask_client.post("/api/cv/update", json={"cycle": 1})
    assert rv1.get_json().get("reset_requested") is True

    # Subsequent updates do not - the flag is consumed once.
    rv2 = flask_client.post("/api/cv/update", json={"cycle": 2})
    assert rv2.get_json().get("reset_requested") is None


def test_cv_update_appends_multiple_twin_sync(flask_client, cv_state_clean):
    for delta in (1.1, 2.2, 3.3):
        flask_client.post("/api/cv/update", json={"twin_sync_mm": delta})
    history = flask_client.get("/api/cv").get_json()["twin_sync_history"]
    assert history == [1.1, 2.2, 3.3]


def test_cv_clear_history_drops_twin_sync(flask_client, cv_state_clean):
    flask_client.post("/api/cv/update", json={"twin_sync_mm": 9.9})
    flask_client.post("/api/cv/clear-history")
    history = flask_client.get("/api/cv").get_json()["twin_sync_history"]
    assert history == []


def test_cv_clear_history_resets_sorted_state(flask_client, cv_state_clean):
    """CLEAR must also reset the sorted-cube history visible elsewhere on the
    dashboard: container counts, the per-cycle phase-timings log, the cycle-
    duration stats, and the cycle counter. Otherwise the user sees stale
    'Performance Stats' / 'Containers' counters after pressing CLEAR.
    """
    flask_client.post(
        "/api/cv/update",
        json={
            "twin_sync_mm": 4.4,
            "twin_sync_pre_bias_mm": 26.2,
            "containers": {"red": 3, "blue": 2},
            "cycle": 7,
            "cycle_time": 35.0,
            "phase_timings": {
                "cycle": 7,
                "color": "red",
                "phases": {"pre_align": 5.0, "descent": 6.0, "place": 4.0},
            },
        },
    )

    flask_client.post("/api/cv/clear-history")

    snapshot = flask_client.get("/api/cv").get_json()
    assert snapshot["twin_sync_history"] == []
    assert snapshot["twin_sync_pre_bias_history"] == []
    assert snapshot["containers"] == {"red": 0, "blue": 0}
    assert snapshot["phase_timings_log"] == []
    assert snapshot["last_cycle_time"] is None
    assert snapshot["avg_cycle_time"] is None
    assert snapshot["min_cycle_time"] is None
    assert snapshot["max_cycle_time"] is None
    assert snapshot["sorted_total"] == 0
    assert snapshot["cycle"] == 0


def test_telemetry_update_roundtrip(flask_client):
    sample = {"tick": 42, "status": "OK", "joint_angles": [0.0] * 9}
    rv = flask_client.post("/api/update", json=sample)
    assert rv.status_code == 200

    rv = flask_client.get("/api/telemetry")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data.get("tick") == 42
    assert data.get("status") == "OK"
