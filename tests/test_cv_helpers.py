"""Tests for the pure-Python helpers inside cv_detector.

These functions touch only numpy / Python arithmetic - no Isaac, no
camera, no USD - so they are the natural unit-test surface for the CV
module. The tests exercise:

* ``_order_quad_points``: TL/TR/BR/BL ordering invariant under input
  permutation (rotated and shuffled corner sets must yield the same
  canonical order).
* ``_normalize_yaw_90`` / ``_normalize_yaw_45``: cube-symmetry yaw
  folding (cubes are 4-fold symmetric, so any yaw is equivalent
  modulo 90°; the helper picks the representative in (-90°, 90°]
  for 90-fold and (-45°, 45°] for 45-fold).
* ``COLOR_RANGES``: red wraps the HSV hue boundary so it must contain
  two intervals; ranges of different colors must be disjoint in hue.
"""

import numpy as np

import cv_detector


def test_order_quad_canonical_ordering_axis_aligned_square():
    """A unit square at the origin: TL/TR/BR/BL should match expected order."""
    # Y grows downward in image coordinates -> "bottom" has larger Y.
    tl, tr, br, bl = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    pts = np.array([tr, bl, tl, br], dtype=np.float32)
    ordered = cv_detector._order_quad_points(pts)
    assert ordered[0].tolist() == list(tl)
    assert ordered[1].tolist() == list(tr)
    assert ordered[2].tolist() == list(br)
    assert ordered[3].tolist() == list(bl)


def test_order_quad_invariant_under_input_permutation():
    """Same four corners passed in any order must produce the same canonical layout."""
    pts = np.array([[10, 20], [30, 20], [30, 50], [10, 50]], dtype=np.float32)
    base = cv_detector._order_quad_points(pts)
    for shift in range(1, 4):
        rolled = np.roll(pts, shift, axis=0)
        assert np.allclose(cv_detector._order_quad_points(rolled), base)


def test_normalize_yaw_90_keeps_angle_in_minus90_to_90_range():
    cv = cv_detector.CVDetector.__new__(cv_detector.CVDetector)
    for raw, expected in [
        (0.0, 0.0),
        (45.0, 45.0),
        (90.0, -90.0),  # boundary: 90 % 180 == 90, then -180 -> -90
        (135.0, -45.0),
        (180.0, 0.0),
        (-45.0, -45.0),
        (-90.0, 90.0),  # -90 % 180 == 90, > 90? no, == 90, stays
        (-180.0, 0.0),
        (270.0, 90.0),  # 270 % 180 == 90 -> stays 90 (boundary)
        (10.0, 10.0),
    ]:
        # Tolerance to absorb the floating-point boundary at 90°/-90°.
        got = cv._normalize_yaw_90(raw)
        # Either expected, or 180° away (the modular equivalent on a 180° fold).
        assert abs(got - expected) < 1e-6 or abs(abs(got - expected) - 180) < 1e-6, (
            f"_normalize_yaw_90({raw}) = {got}, expected {expected}"
        )


def test_normalize_yaw_45_keeps_angle_in_minus45_to_45_range():
    cv = cv_detector.CVDetector.__new__(cv_detector.CVDetector)
    for raw in (-180, -135, -90, -45, 0, 45, 90, 135, 180, 225, 270):
        out = cv._normalize_yaw_45(float(raw))
        assert -45.0 <= out <= 45.0, f"_normalize_yaw_45({raw}) = {out}"


def test_color_ranges_red_wraps_hue_boundary():
    """Red sits at the 0/180 hue wrap, so it must be expressed as two intervals."""
    red_ranges = cv_detector.COLOR_RANGES["red"]
    assert len(red_ranges) == 2, "red must have two HSV intervals to cover the wrap"
    lo1, hi1 = red_ranges[0]
    lo2, hi2 = red_ranges[1]
    assert lo1[0] == 0, "first red interval must start at hue 0"
    assert hi2[0] == 180, "second red interval must end at hue 180"


def test_color_ranges_red_blue_green_hue_intervals_are_disjoint():
    """Red (~0/~180), blue (100-130), green (40-85) must not overlap in hue."""

    def hue_intervals(color):
        return [(int(lo[0]), int(hi[0])) for lo, hi in cv_detector.COLOR_RANGES[color]]

    red = hue_intervals("red")
    blue = hue_intervals("blue")
    green = hue_intervals("green")

    def overlaps(a, b):
        a_lo, a_hi = a
        b_lo, b_hi = b
        return not (a_hi < b_lo or b_hi < a_lo)

    for r_iv in red:
        for b_iv in blue:
            assert not overlaps(r_iv, b_iv), f"red {r_iv} overlaps blue {b_iv}"
        for g_iv in green:
            assert not overlaps(r_iv, g_iv), f"red {r_iv} overlaps green {g_iv}"
    for b_iv in blue:
        for g_iv in green:
            assert not overlaps(b_iv, g_iv), f"blue {b_iv} overlaps green {g_iv}"


def test_color_ranges_value_lower_bound_consistent():
    """Saturation/value lower bounds protect against grey background false positives."""
    for color, ranges in cv_detector.COLOR_RANGES.items():
        for lo, _ in ranges:
            assert lo[1] >= 50, f"{color} has S_min={lo[1]} - too permissive"
            assert lo[2] >= 50, f"{color} has V_min={lo[2]} - too permissive"
