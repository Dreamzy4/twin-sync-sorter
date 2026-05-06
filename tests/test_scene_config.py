"""Geometric invariants for scene_config constants.

These checks guard against silent regressions when porting to a new scene:
the pickup zone must be reachable, containers must not be inside the pickup
zone (otherwise the robot would try to pick up cubes it just dropped), and
CV bias must stay within the per-cycle clip limit so a stale config does
not push detections out of the workspace on first cycle.
"""

import numpy as np

import scene_config as cfg


def _zone_subset(inner, outer):
    """True iff inner XY rectangle is fully contained in outer XY rectangle."""
    return (
        inner["x_min"] >= outer["x_min"]
        and inner["x_max"] <= outer["x_max"]
        and inner["y_min"] >= outer["y_min"]
        and inner["y_max"] <= outer["y_max"]
    )


def test_pickup_zone_within_workspace_low():
    assert _zone_subset(cfg.PICKUP_ZONE, cfg.WORKSPACE_LOW), (
        "PICKUP_ZONE must be a subset of WORKSPACE_LOW: cubes outside the "
        "low-Z workspace cannot be reached by the gripper at table level."
    )


def test_containers_outside_pickup_zone():
    pz = cfg.PICKUP_ZONE
    for color, xyz in cfg.CONTAINERS.items():
        x, y, _ = xyz
        in_zone = pz["x_min"] <= x <= pz["x_max"] and pz["y_min"] <= y <= pz["y_max"]
        assert not in_zone, (
            f"Container {color} at ({x}, {y}) lies inside PICKUP_ZONE - "
            f"the robot would re-pick cubes it just dropped."
        )


def test_workspace_low_z_below_high():
    assert cfg.WORKSPACE_LOW["z_probe"] < cfg.WORKSPACE_HIGH["z_probe"], (
        "Low-Z workspace probe must sit below high-Z probe: the dual-Z model "
        "expects table-level pick at z=low and transit at z=high."
    )
    assert cfg.WORKSPACE_LOW["z_min"] <= cfg.WORKSPACE_HIGH["z_min"], (
        "Low-Z floor cannot be higher than high-Z floor."
    )


def test_cv_world_bias_in_bounds():
    bias = np.asarray(cfg.CV_WORLD_BIAS)
    assert bias.shape == (3,), "CV_WORLD_BIAS must be a 3-vector (x, y, z)."
    assert np.all(np.abs(bias) < 0.05), (
        "CV_WORLD_BIAS components must stay under ±5 cm absolute - the "
        "per-cycle EMA clip enforces this at runtime; the static config "
        "must respect it too."
    )


def test_pickup_colors_subset_of_containers():
    for color in cfg.PICKUP_COLORS:
        assert color in cfg.CONTAINERS, (
            f"PICKUP_COLORS contains '{color}' but CONTAINERS has no drop-off "
            f"defined for it - sort cycle would have nowhere to place the cube."
        )


def test_usd_yaw_tolerance_reasonable():
    assert 0 < cfg.USD_YAW_TOLERANCE <= 90, (
        "USD_YAW_TOLERANCE outside (0, 90]° - too tight or wider than "
        "physically meaningful for a 4-fold-symmetric cube."
    )
    assert 0 < cfg.USD_MATCH_MAX_DISTANCE <= 1.0, (
        "USD_MATCH_MAX_DISTANCE outside (0, 1.0] m - matches across the room "
        "are not ground-truth."
    )
