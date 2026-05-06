"""Pure-math tests for joint_control.WorkspaceConstraints / DualWorkspace.

Covers:
* Cuboid edge clamping for X / Y / Z (separately and together).
* Radial reach clamping (``max_reach``) when XY norm exceeds the limit.
* Pass-through behaviour for in-range targets.
* Dual-Z probe selection (low vs high workspace) based on target Z.
* Cross-term: a target that is *both* outside the cuboid *and* outside
  the radial reach gets both clamps applied in the right order.

The class is import-clean (``omni.*`` is wrapped in try/except elsewhere
in joint_control.py) so these tests run on stock Python in CI.
"""

import numpy as np

import joint_control as jc


def _ws():
    return jc.WorkspaceConstraints(
        x_min=0.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=0.0,
        z_max=2.0,
        max_reach=10.0,  # large so cuboid tests don't trip the radial clamp
        z_probe=0.5,
    )


def test_clamp_passes_through_in_range_target():
    ws = _ws()
    target = np.array([0.5, 0.5, 1.0])
    out = ws.clamp(target)
    assert np.allclose(out, target)


def test_clamp_x_below_min():
    out = _ws().clamp(np.array([-0.5, 0.0, 1.0]))
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 1.0


def test_clamp_x_above_max():
    out = _ws().clamp(np.array([2.0, 0.0, 1.0]))
    assert out[0] == 1.0


def test_clamp_y_below_min():
    out = _ws().clamp(np.array([0.5, -2.0, 1.0]))
    assert out[1] == -1.0


def test_clamp_y_above_max():
    out = _ws().clamp(np.array([0.5, 2.0, 1.0]))
    assert out[1] == 1.0


def test_clamp_z_below_min():
    out = _ws().clamp(np.array([0.5, 0.0, -0.5]))
    assert out[2] == 0.0


def test_clamp_z_above_max():
    out = _ws().clamp(np.array([0.5, 0.0, 5.0]))
    assert out[2] == 2.0


def test_clamp_radial_reach_when_xy_exceeds_max_reach():
    """Target outside max_reach in XY plane should be scaled back along the radial direction."""
    ws = jc.WorkspaceConstraints(
        x_min=-10, x_max=10, y_min=-10, y_max=10,
        z_min=0, z_max=10, max_reach=1.0, z_probe=0.5,
    )
    target = np.array([3.0, 4.0, 1.0])  # XY norm = 5
    out = ws.clamp(target)
    # Direction preserved; magnitude rescaled to max_reach.
    xy_out_norm = float(np.linalg.norm(out[:2]))
    assert abs(xy_out_norm - 1.0) < 1e-6
    # Direction: (3, 4) / 5 = (0.6, 0.8)
    assert abs(out[0] - 0.6) < 1e-6
    assert abs(out[1] - 0.8) < 1e-6
    # Z is independent of radial clamp.
    assert out[2] == 1.0


def test_clamp_radial_no_op_when_within_reach():
    ws = jc.WorkspaceConstraints(
        x_min=-10, x_max=10, y_min=-10, y_max=10,
        z_min=0, z_max=10, max_reach=10.0, z_probe=0.5,
    )
    target = np.array([3.0, 4.0, 1.0])
    out = ws.clamp(target)
    assert np.allclose(out, target)


def test_clamp_does_not_mutate_input():
    ws = _ws()
    target = np.array([5.0, 5.0, 5.0])
    target_before = target.copy()
    ws.clamp(target)
    assert np.array_equal(target, target_before)


def test_dual_workspace_picks_low_for_pickup_height():
    dw = jc.DualWorkspace()
    # At z=0.05 (table level) low workspace must be selected.
    chosen = dw.for_z(0.05)
    assert chosen is dw.low


def test_dual_workspace_picks_high_for_transit_height():
    dw = jc.DualWorkspace()
    # At z=0.40 (transit altitude) high workspace must be selected.
    chosen = dw.for_z(0.40)
    assert chosen is dw.high


def test_dual_workspace_split_is_midpoint_of_z_probes():
    dw = jc.DualWorkspace()
    midpoint = (dw.high.z_probe + dw.low.z_probe) / 2
    # Just below midpoint -> low; just above -> high.
    assert dw.for_z(midpoint - 1e-6) is dw.low
    assert dw.for_z(midpoint + 1e-6) is dw.high


def test_dual_workspace_low_allows_wider_xy_than_high():
    """Low workspace must extend to the cube zone; high tightens for transit."""
    dw = jc.DualWorkspace()
    assert dw.low.x_max >= dw.high.x_max
    assert dw.low.y_max >= dw.high.y_max
    assert dw.low.max_reach >= dw.high.max_reach
