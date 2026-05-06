"""Smoke test: every project module imports without the Isaac Sim runtime.

The CI runs on plain ``ubuntu-latest`` Python; ``omni.*`` and ``pxr`` are not
pip-installable. Each module that uses those packages wraps its imports in
``try: ... except ImportError: ...`` so the module body itself loads even
without Isaac, and Isaac-only code paths fail at *call* time rather than at
*import* time.

This test makes that contract part of CI: if a future commit drops a guard
or adds a new top-level Isaac import, this test will fail before any of the
unit tests get to run.
"""

import importlib

PROJECT_MODULES = [
    "async_logger",
    "scene_config",
    "telemetry",
    "dashboard_server",
    "cv_detector",
    "joint_control",
    "robot_motion",
]


def test_project_modules_import_without_isaac():
    """All seven project modules must import in a pure-Python environment."""
    for name in PROJECT_MODULES:
        importlib.import_module(name)


def test_isaac_symbols_degrade_to_none():
    """Without Isaac, the wrapped Isaac symbols evaluate to None.

    Guards against accidental ``import *`` or future imports that would
    raise ImportError instead of degrading gracefully.
    """
    import cv_detector
    import joint_control

    # cv_detector wraps Camera and pxr.
    assert cv_detector.Camera is None
    # joint_control wraps the motion-generation stack and pxr.
    assert joint_control.RmpFlow is None
    assert joint_control.ArticulationMotionPolicy is None
    assert joint_control.euler_angles_to_quat is None
    assert joint_control.load_supported_motion_policy_config is None
