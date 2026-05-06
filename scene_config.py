"""Scene-specific calibration constants for the Franka pickup demo.

Single source of truth for everything that depends on the physical/virtual
scene layout: USD prim paths, CV->world bias offsets, RmpFlow finger correction,
workspace bounds, container positions, USD fallback tuning.

When porting to a new scene, this is the only file that needs editing -
``robot_motion``, ``joint_control`` and ``cv_detector`` import named constants
from here. The dashboard server also imports a subset for UI reference values
(via ``GET /api/cv/config``).
"""

import numpy as np

# Camera prim paths within the loaded Isaac scene.
COLOR_PRIM = "/World/realsense_d455/RSD455/Camera_OmniVision_OV9782_Color"
DEPTH_PRIM = "/World/realsense_d455/RSD455/Camera_Pseudo_Depth"

# Per-cycle search loop: tries up to N stable detections before giving up.
MAX_SEARCH_RETRIES = 3

# Colors actually picked up by the robot. Cubes of any other color (e.g. 'green')
# are detected and shown on the dashboard but skipped by the sort logic.
PICKUP_COLORS = ["red", "blue"]

# Systematic CV->world projection offset (meters), refined per-cycle by EMA from
# USD ground truth. Clipped to ±2cm/step against anomalies (recalibrate_bias).
CV_WORLD_BIAS = np.array([0.0053, 0.0284, 0.0033])

# RmpFlow corrects to flange origin, but we want finger-tip placement: this
# offset is added to every move target. Scene-dependent (depends on gripper
# attachment in the USD scene).
RMPFLOW_OFFSET = np.array([-0.0025, 0.0448, 0.006])

# Cubes whose world XY falls outside this rectangle are ignored at detection
# time - keeps the robot from grabbing anything outside the working area.
PICKUP_ZONE = {
    "x_min": 0.23,
    "x_max": 0.82,
    "y_min": -0.67,
    "y_max": 0.62,
}

# Robot reach bounds for the dashboard UI overlay. Real motion clamping lives
# in joint_control.DualWorkspace (these are reference copies served via
# GET /api/cv/config). Two profiles because RmpFlow's reach envelope tightens
# at high Z (transit) compared to low Z (table-level pick / place).
WORKSPACE_HIGH = {
    "x_min": 0.23,
    "x_max": 0.77,
    "y_min": -0.67,
    "y_max": 0.57,
    "z_min": 0.03,
    "z_max": 0.82,
    "max_reach": 0.72,
    "z_probe": 0.40,
}
WORKSPACE_LOW = {
    "x_min": 0.23,
    "x_max": 0.82,
    "y_min": -0.67,
    "y_max": 0.62,
    "z_min": 0.005,
    "z_max": 0.82,
    "max_reach": 0.77,
    "z_probe": 0.05,
}

# Container drop-off coordinates. This is the single source of truth -
# joint_control.CONTAINER_RED / CONTAINER_BLUE are np.asarray views over
# these values, and the dashboard renders the diamond markers from the
# same dict served via GET /api/cv/config.
CONTAINERS = {
    "red": [0.0000, 0.4499, 0.0154],
    "blue": [0.0000, -0.5501, 0.0154],
}

# USD ground-truth fallback tuning. When CV yaw confidence is low (|CV-USD|
# > tolerance), we use the USD prim's yaw instead. Match also requires the
# prim to be within MAX_DISTANCE of the CV-detected world position.
USD_YAW_TOLERANCE = 20.0  # degrees
USD_MATCH_MAX_DISTANCE = 0.25  # meters

# Known USD cube prim paths (fast prior lookup). The auto-traverse fallback in
# _iter_usd_cube_prims will additionally pick up new sibling cubes added to the
# same parent (/World/scen/Cubes/), but explicitly listing them here avoids a
# full stage walk on every cycle.
USD_CUBE_PRIM_PATHS = [
    "/World/scen/Cubes/Cube_11",
    "/World/scen/Cubes/Cube_20",
    "/World/scen/Cubes/Cube_01",
    "/World/scen/Cubes/Cube_21",
    "/World/scen/Cubes/Cube_15",
    "/World/scen/Cubes/Cube_17",
    "/World/scen/Cubes/Cube_22",
    "/World/scen/Cubes/Cube_16",
    "/World/scen/Cubes/Cube_10",
]
