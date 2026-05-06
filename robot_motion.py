"""Real-time CV-driven robotic sorting orchestrator for Isaac Sim Franka digital twin.

Coordinates the sorting cycle: searches for colored cubes via the CV pipeline,
validates against USD ground truth, picks them up with RmpFlow motion planning,
and places into colored containers. Pushes telemetry and CV state to the Flask
dashboard process running on localhost:5000.

Entry point: ``main()``. The module is import-clean: importing it does not
mutate ``sys.path`` or replace ``builtins.print``; both side effects (and the
asyncio loop bring-up) live in ``main()``. See README's Run section for
launch syntax.

Twin validation metric (Δ CV-USD) reported per cycle in the [Target] log line -
shows how closely the CV detection matches USD ground-truth position.
"""

import asyncio
import os
import time

import numpy as np

# Isaac Sim modules are only available inside the Isaac Sim Python environment.
# Wrap the imports so this module can be imported (e.g. by linters, CI, or
# Sphinx) without omni.* installed; the runtime entry points still require
# Isaac and will fail clearly at call time when omni is None.
try:
    import omni.kit.app  # type: ignore[import-not-found]
except ImportError:
    omni = None  # type: ignore[assignment]

try:
    from omni.isaac.core import World  # type: ignore[import-not-found]
    from omni.isaac.core.robots import Robot  # type: ignore[import-not-found]
except ImportError:
    World = Robot = None  # type: ignore[assignment,misc]

import scene_config as _cfg
from async_logger import flush_async_print, install_async_print
from cv_detector import CVDetector
from joint_control import (
    CONTAINER_BLUE,
    CONTAINER_RED,
    CONTAINER_SIZE_M,
    WORKSPACE,
    JointController,
)
from telemetry import CVPoster, Telemetry

COLOR_PRIM = _cfg.COLOR_PRIM
DEPTH_PRIM = _cfg.DEPTH_PRIM
MAX_SEARCH_RETRIES = _cfg.MAX_SEARCH_RETRIES
PICKUP_COLORS = _cfg.PICKUP_COLORS
CV_WORLD_BIAS = _cfg.CV_WORLD_BIAS
RMPFLOW_OFFSET = _cfg.RMPFLOW_OFFSET
PICKUP_ZONE = _cfg.PICKUP_ZONE
USD_YAW_TOLERANCE = _cfg.USD_YAW_TOLERANCE
USD_MATCH_MAX_DISTANCE = _cfg.USD_MATCH_MAX_DISTANCE
USD_CUBE_PRIM_PATHS = _cfg.USD_CUBE_PRIM_PATHS


PROCESSED_USD_PRIMS = set()
PROCESSED_PICKUP_POINTS = []
PROCESSED_PICKUP_RADIUS = 0.08
ACTIVE_TARGET = None
PLACED_COUNTS = {"red": 0, "blue": 0}

CONTAINER_SCAN_RADIUS = CONTAINER_SIZE_M / 2 + 0.08

# Module-level runtime state. Set by initialize_system() once Isaac is ready,
# then shared across cycle helpers. Globals keep the orchestration thin and
# avoid passing the same five objects through every helper signature.

world = None
robot = None
cv = None
ctl = None
tel = None
CV_STATE = None

# Single-run guard. Set to True while main_loop is iterating; checked at the
# top of main_loop() so a second call to main() while the first loop is still
# alive (e.g. user re-issues 'import robot_motion; robot_motion.main()' before
# the previous run has exited) signals the first to stop and waits for it,
# preventing two main_loops from racing on PROCESSED_USD_PRIMS.
_MAIN_LOOP_ACTIVE = False
_MAIN_LOOP_SHOULD_EXIT = False

CV_SNAPSHOT_MIN_INTERVAL = 0.14
CV_MOTION_SNAPSHOT_MIN_INTERVAL = 0.30
DEPTH_MOTION_SNAPSHOT_MIN_INTERVAL = CV_MOTION_SNAPSHOT_MIN_INTERVAL * 2.0
_LAST_CV_SNAPSHOT = 0.0
_LAST_MOTION_DEPTH_SNAPSHOT = 0.0
TELEMETRY_MIN_INTERVAL = 0.10
_LAST_TELEMETRY_PUSH = 0.0
_LAST_TELEMETRY_DATA = None

# Verbose log toggle: set DTCV_VERBOSE=1 for extra detail (motion steps,
# bias updates, target detail, slot calculations). Default produces a
# concise summary (~5 lines per cycle).
VERBOSE = os.environ.get("DTCV_VERBOSE") == "1"


def _dbg(msg: str):
    """Print only when verbose mode is enabled."""
    if VERBOSE:
        print(msg)


def initialize_system():
    """Create Isaac/CV runtime objects explicitly instead of during import."""
    global world, robot, cv, ctl, tel, CV_STATE

    if cv is not None and ctl is not None and tel is not None and CV_STATE is not None:
        return world, robot, cv, ctl, tel

    world = World.instance()
    if world is None:
        world = World()

    try:
        robot = world.scene.add(
            Robot(prim_path="/World/panda_instanceable", name="panda_instanceable")
        )
        _dbg("[Init] robot added to scene")
    except Exception as e:
        robot = world.scene.get_object("panda_instanceable")
        if robot is None:
            raise RuntimeError("Cannot create or get panda_instanceable robot") from e
        _dbg("[Init] robot found in scene")

    cv = CVDetector(color_camera_path=COLOR_PRIM, depth_camera_path=DEPTH_PRIM)
    ctl = JointController(robot)
    tel = Telemetry(robot)
    CV_STATE = CVPoster()

    print("[Init] systems ready (Isaac + CV + Motion + Telemetry)")
    return world, robot, cv, ctl, tel


# Single-line target summary printed once per cycle in normal mode, plus
# detail lines gated behind _dbg() for verbose investigations.


def _log_target_detail(det: dict, world_xyz: np.ndarray):
    color = det.get("color", "?").upper()
    px, py = det.get("pixel_x", 0), det.get("pixel_y", 0)
    area = det.get("area", 0)
    bbox = det.get("bbox_px", (0, 0, 0, 0))
    w_m = det.get("width_m", 0.0)
    h_m = det.get("height_m", 0.0)
    cv_yaw = det.get("world_yaw", 0.0)
    usd_yaw = det.get("usd_yaw")
    grip_yaw = det.get("angle_deg", 0.0)
    yaw_src = det.get("yaw_source", "cv")
    yaw_diff = det.get("yaw_diff_deg")
    prim = det.get("usd_prim_path") or "N/A"
    usd_xyz = det.get("usd_xyz")
    bias = cv.get_world_bias()

    # One-line target summary (key info for normal mode).
    # Twin-Sync is split into two figures:
    #  - Δ_post: ||world_xyz - usd_xyz|| -> residual after EMA bias correction
    #  - Δ_pre:  ||(world_xyz - bias) - usd_xyz|| -> raw projection error
    # Reporting both makes the metric defensible: Δ_post can converge by
    # construction (the bias is fitted against USD), so Δ_pre is the honest
    # measure of perception quality and Δ_post is the convergence indicator.
    prim_short = prim.rsplit("/", 1)[-1] if prim != "N/A" else "no-USD"
    if world_xyz is not None and usd_xyz is not None:
        usd_arr = np.array(usd_xyz, dtype=float)
        d_post = world_xyz - usd_arr
        d_pre = d_post - bias
        delta_post_mm = float(np.linalg.norm(d_post) * 1000)
        delta_pre_mm = float(np.linalg.norm(d_pre) * 1000)
        sync = f"Δ_pre={delta_pre_mm:.1f} Δ_post={delta_post_mm:.1f}mm"
        if CV_STATE is not None:
            CV_STATE.update(
                twin_sync_mm=delta_post_mm,
                twin_sync_pre_bias_mm=delta_pre_mm,
            )
    else:
        sync = "Δ_twin=N/A"
    print(
        f"[Target] {color} {prim_short} px=({px},{py}) "
        f"xyz=({world_xyz[0]:.3f},{world_xyz[1]:.3f},{world_xyz[2]:.3f}) "
        f"yaw={grip_yaw:+.0f}°({yaw_src}) {sync}"
    )

    # Verbose-only details: bbox, full CV/USD coords, yaw verdict, bias.
    _dbg(
        f"[Target/dbg] bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) "
        f"w={w_m * 100:.1f}cm h={h_m * 100:.1f}cm area={area:.0f}"
    )
    if usd_xyz is not None and world_xyz is not None:
        ua = np.array(usd_xyz, dtype=float)
        d = world_xyz - ua
        _dbg(f"[Target/dbg] CV={world_xyz} USD={ua} Δ=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})")
    if usd_yaw is not None:
        diff_str = f"{yaw_diff:.1f}°" if yaw_diff is not None else "?"
        verdict = "CV-trusted" if yaw_src == "cv" else "USD-fallback"
        _dbg(
            f"[Target/dbg] yaw CV={cv_yaw:+.1f}° USD={usd_yaw:+.1f}° "
            f"diff={diff_str} tol={cv.usd_yaw_tolerance}° -> {verdict}"
        )
    _dbg(f"[Target/dbg] bias=({bias[0]:+.4f},{bias[1]:+.4f},{bias[2]:+.4f})")


# Per-cycle CV bias calibration. Compares CV-detected world XYZ against the
# matched USD prim's ground-truth XYZ, then nudges _world_bias via EMA so
# subsequent detections are pre-corrected. Heart of the twin-validation loop.


async def recalibrate_bias(stable_detection, world_xyz=None):
    """Smoothly update _world_bias via exponential moving average.

    - Per-cycle correction step is clipped to ±2cm.
    - Global bounds: each bias component is kept within ±5cm.
    - Uses USD ground-truth position as the reference but adjusts the bias
      gradually rather than snapping to a single observation.
    """
    try:
        pose = cv.get_usd_pose_for_detection(stable_detection, world_xyz)
        if pose is None:
            return

        usd_xyz = np.array(pose["top_xyz"], dtype=float)

        # Temporarily zero the bias so pixel_to_world returns raw CV coords -
        # we need the un-corrected reading to compute the new correction.
        old_bias = cv.get_world_bias().copy()
        try:
            cv.set_world_bias(np.zeros(3))
            raw_xyz = cv.pixel_to_world(stable_detection["pixel_x"], stable_detection["pixel_y"])
        finally:
            cv.set_world_bias(old_bias)

        if raw_xyz is None:
            return

        diff = usd_xyz - raw_xyz  # current measured CV->USD offset
        # IMPORTANT: clip applies to the target (diff), not the step
        # (diff - old_bias). This looks like a bug ("2cm/cycle step" from the
        # nearby comment isn't actually implemented), but it's intentional:
        # per-cube diff ranges 0.025..0.07m due to position-dependent camera
        # non-linearity; a scalar bias can't track all cubes perfectly.
        # Clip(diff, ±0.02) keeps the bias at a conservative average, giving
        # the minimum worst-case error. Attempting to clip the step instead
        # (diff - old_bias) caused bias overshoot to ±0.05 and false positives
        # in _is_processed_detection's radius check. DO NOT change without
        # moving to a per-position bias model (polynomial / lookup table).
        diff_clipped = np.clip(diff, -0.02, 0.02)  # at most 2cm per step

        # EMA smoothing (α=0.3) - converges in a few cycles without jumps.
        alpha = 0.3
        new_bias = (1 - alpha) * old_bias + alpha * diff_clipped
        new_bias = np.clip(new_bias, -0.05, 0.05)  # global hard cap

        cv.set_world_bias(new_bias)

        _dbg(
            f"[CAL] bias updated: {np.round(new_bias, 4)} "
            f"(USD={np.round(usd_xyz, 3)} raw={np.round(raw_xyz, 3)} "
            f"diff={np.round(diff, 3)} step={np.round(diff_clipped, 3)})"
        )
    except Exception as e:
        print(f"[CAL] recalibrate_bias error: {e}")


def _center_xyz_from_matched_cv(stable_detection: dict, world_xyz: np.ndarray) -> np.ndarray:
    """Keep CV selection, but correct large projection drift to the matched cube center."""
    if not stable_detection:
        return world_xyz

    usd_xyz = stable_detection.get("usd_xyz")
    prim_path = stable_detection.get("usd_prim_path")
    if usd_xyz is None or not prim_path:
        return world_xyz

    cv_xyz = np.array(world_xyz, dtype=float)
    usd_top = np.array(usd_xyz, dtype=float)
    delta = usd_top - cv_xyz
    xy_err = float(np.linalg.norm(delta[:2]))
    z_err = abs(float(delta[2]))

    # Drift-correction band, tuned empirically on the demo scene:
    #   xy_err < 1.8 cm AND z_err < 1.2 cm: CV is close enough; below gripper
    #     grasp tolerance so a USD snap would not change pickup outcome.
    #   xy_err > 9.0 cm: too far - either USD match is wrong, or CV projection
    #     is so off that snapping to USD would teleport the target. Keep raw
    #     CV and let the verify-after-place step fail loudly instead of hiding.
    #   in between: snap XY (and optionally Z) to USD top, below.
    if xy_err < 0.018 and z_err < 0.012:
        return cv_xyz
    if xy_err > 0.090:
        _dbg(f"[CAL] CV/USD drift {xy_err * 100:.1f}cm > tol; keeping CV XYZ")
        return cv_xyz

    corrected = cv_xyz.copy()
    corrected[:2] = usd_top[:2]
    if z_err >= 0.012:
        corrected[2] = usd_top[2]

    stable_detection["world_xyz"] = [float(v) for v in corrected]
    _dbg(
        f"[CAL] target center correction: raw={np.round(cv_xyz, 3)} "
        f"center={np.round(corrected, 3)} "
        f"dxy={xy_err * 100:.1f}cm dz={delta[2] * 100:.1f}cm"
    )
    return corrected


def _container_center_for_color(color: str) -> np.ndarray:
    if color == "red":
        return CONTAINER_RED.copy()
    elif color == "blue":
        return CONTAINER_BLUE.copy()
    raise ValueError(f"Unknown placement color: {color}")


def _container_slot_offset(index: int, cube_size: float, container_color: str = None) -> np.ndarray:
    # Fixed slot pitch sized for the largest expected cube (7cm + 1cm gap),
    # so the placement grid stays stable regardless of which cube goes there.
    spacing = 0.080
    xs = (-spacing / 2.0, +spacing / 2.0)
    # First slots go to the center row + the side nearer the robot. For the
    # BLUE bin (Y=-0.55) "nearer" means larger Y; for RED (Y=+0.45) - smaller Y.
    near_dir = +1 if container_color == "blue" else -1
    ys = (0.0, +spacing * near_dir, -spacing * near_dir)
    grid = [(x, y) for y in ys for x in xs]
    dx, dy = grid[index % len(grid)]
    return np.array([dx, dy, 0.0])


def _place_xyz_for_color(color: str, cube_size: float = 0.05) -> np.ndarray:
    center = _container_center_for_color(color)
    index = PLACED_COUNTS.get(color, 0)
    place = center + _container_slot_offset(index, cube_size, container_color=color)
    _dbg(
        f"[Place/SLOT] {color} slot={index} center={np.round(center, 3)} "
        f"offset={np.round(place - center, 3)} target={np.round(place, 3)}"
    )
    return place


def _det_world_xyz(det: dict) -> "np.ndarray | None":
    xyz = det.get("world_xyz") if det else None
    if xyz is None:
        return None
    try:
        arr = np.array(xyz, dtype=float)
        return arr if arr.shape[0] >= 3 and np.all(np.isfinite(arr[:3])) else None
    except Exception:
        return None


def _is_processed_detection(det: dict) -> bool:
    if not det:
        return False

    prim_path = det.get("usd_prim_path")
    if prim_path:
        return prim_path in PROCESSED_USD_PRIMS

    xyz = _det_world_xyz(det)
    if xyz is None:
        return False

    if _in_any_container_zone(xyz) is not None:
        return True

    for old_xyz in PROCESSED_PICKUP_POINTS:
        if float(np.linalg.norm(xyz[:2] - old_xyz[:2])) <= PROCESSED_PICKUP_RADIUS:
            return True
    return False


def _filter_unprocessed(dets: list, log: bool = True) -> list:
    fresh = []
    skipped = []
    for det in dets:
        if _is_processed_detection(det):
            skipped.append(det)
        else:
            fresh.append(det)
    if skipped and log:
        labels = []
        for det in skipped:
            label = (
                det.get("usd_prim_path")
                or f"{det.get('color')}@{det.get('pixel_x')},{det.get('pixel_y')}"
            )
            labels.append(label)
        _dbg(f"[Select] ignoring already-processed: {labels}")
    return fresh


def _distance_to_hand(det: dict) -> float:
    xyz = _det_world_xyz(det)
    if xyz is not None:
        return float(np.linalg.norm(xyz[:2]))
    return 1e6 - float(det.get("area", 0.0)) * 1e-6


def _sort_by_hand_distance(dets: list) -> list:
    return sorted(dets, key=_distance_to_hand)


def _mark_processed(stable: dict, world_xyz: np.ndarray):
    prim_path = stable.get("usd_prim_path") if stable else None
    if prim_path:
        PROCESSED_USD_PRIMS.add(prim_path)
    if world_xyz is not None:
        PROCESSED_PICKUP_POINTS.append(np.array(world_xyz, dtype=float))
    _dbg(
        f"[Select] marked processed: prim={prim_path} "
        f"pickup={np.round(world_xyz, 3) if world_xyz is not None else None}"
    )


def _cube_visible_at_source(stable: dict, original_xyz: np.ndarray, radius: float = 0.06) -> bool:
    color = stable.get("color")
    if not color:
        return False

    frame = cv.get_frame()
    depth = cv.get_depth_frame()
    if frame is None:
        return False

    saved_x, saved_y = cv.pickup_zone_x, cv.pickup_zone_y
    old_colors = cv.active_colors
    cv.pickup_zone_x = cv.pickup_zone_y = None
    cv.active_colors = [color]
    try:
        dets = cv.detect_colored_objects(
            frame, depth_frame=depth, with_usd=False, estimate_yaw=False
        )
    finally:
        cv.active_colors = old_colors
        cv.pickup_zone_x, cv.pickup_zone_y = saved_x, saved_y

    for det in dets:
        world_pt = cv.pixel_to_world(det["pixel_x"], det["pixel_y"], depth_frame=depth)
        if world_pt is None:
            continue
        dist = float(np.linalg.norm(world_pt[:2] - original_xyz[:2]))
        if dist <= radius:
            print(f"[Verify] {color} cube still at source (dist={dist * 100:.1f}cm) - failed")
            return True

    _dbg(f"[Verify] {color} cube not at source - success")
    return False


def _in_any_container_zone(world_pt: np.ndarray) -> "str | None":
    if np.linalg.norm(world_pt[:2] - CONTAINER_RED[:2]) <= CONTAINER_SCAN_RADIUS:
        return "red"
    if np.linalg.norm(world_pt[:2] - CONTAINER_BLUE[:2]) <= CONTAINER_SCAN_RADIUS:
        return "blue"
    return None


def _scan_containers() -> dict:
    all_cubes = cv.get_usd_cube_states(x_min=-1.0, x_max=1.0)

    counts = {"red": 0, "blue": 0}
    container_dets = []
    labels = {"red": [], "blue": []}

    for cube in all_cubes:
        xyz = np.array(cube["xyz"])
        zone = _in_any_container_zone(xyz)
        if zone is None:
            continue
        counts[zone] += 1
        labels[zone].append(cube["path"].rsplit("/", 1)[-1])
        px_py = cv.world_to_pixel(xyz, use_bias=False)
        if px_py is not None:
            container_dets.append(
                {
                    "pixel_x": px_py[0],
                    "pixel_y": px_py[1],
                    "color": cube["color"],
                    "container": zone,
                }
            )

    for color in ("red", "blue"):
        info = ", ".join(labels[color]) if labels[color] else "-"
        print(f"[Containers] {color.upper()}: {counts[color]} -> {info}")

    for color in ("red", "blue"):
        PLACED_COUNTS[color] = counts[color]

    _push_cube_states(counts, container_dets=container_dets, cube_states=all_cubes)
    return counts


def _push_cube_states(
    containers: dict = None, container_dets: list = None, cube_states: list = None
):
    # Wide scan: report cubes inside the pickup zone for the dashboard cube
    # table, plus cubes inside containers for the diamond markers overlay.
    raw = cube_states if cube_states is not None else cv.get_usd_cube_states(x_min=-1.0, x_max=1.0)
    cubes = []
    auto_container_dets = []
    for c in raw:
        x = float(c["xyz"][0])
        if 0.20 <= x <= 0.80:
            cubes.append(
                {
                    "path": c["path"],
                    "label": c["path"].rsplit("/", 1)[-1],
                    "color": c["color"],
                    "xyz": c["xyz"],
                    "yaw_deg": c["yaw_deg"],
                    "processed": c["path"] in PROCESSED_USD_PRIMS,
                }
            )
            continue
        # Outside the pickup zone - check container membership for diamond markers.
        xyz = np.array(c["xyz"])
        zone = _in_any_container_zone(xyz)
        if zone is None:
            continue
        px_py = cv.world_to_pixel(xyz, use_bias=False)
        if px_py is not None:
            auto_container_dets.append(
                {
                    "pixel_x": px_py[0],
                    "pixel_y": px_py[1],
                    "color": c["color"],
                    "container": zone,
                }
            )
    # If the caller passed container_dets explicitly (from _scan_containers),
    # use those. Otherwise compute them from the current wide scan so the
    # diamond markers refresh every cycle - without this, the startup snapshot
    # would freeze forever as cubes get sorted.
    if container_dets is None:
        container_dets = auto_container_dets
    # Pass original pickup XYs of already-processed cubes so the dashboard
    # can suppress stale detection markers on the RGB stream. Detections
    # linger in the list until the next detect_fn - especially on the
    # prefetch path where detect_fn doesn't run on every cycle.
    processed_xy = [[float(p[0]), float(p[1])] for p in PROCESSED_PICKUP_POINTS]
    CV_STATE.update(
        containers=containers, cubes=cubes, container_dets=container_dets, processed_xy=processed_xy
    )


def _set_active_target(det: dict, world_xyz: "np.ndarray | None" = None):
    global ACTIVE_TARGET
    if det is None:
        ACTIVE_TARGET = None
        return
    target = dict(det)
    target.pop("contour", None)
    if world_xyz is not None:
        target["world_xyz"] = [float(v) for v in world_xyz]
    target["target_locked"] = True
    ACTIVE_TARGET = target
    _dbg(
        f"[Select] active target: color={target.get('color')} "
        f"prim={target.get('usd_prim_path')} "
        f"pixel=({target.get('pixel_x')},{target.get('pixel_y')})"
    )


def _clear_active_target():
    global ACTIVE_TARGET
    if ACTIVE_TARGET is not None:
        _dbg(f"[Select] active target cleared: prim={ACTIVE_TARGET.get('usd_prim_path')}")
    ACTIVE_TARGET = None


def _clear_dashboard_target(status: str = None, cycle: int = None, log_msg: str = None):
    CV_STATE.update(
        detection=None,
        world_xyz=None,
        status=status,
        cycle=cycle,
        log_msg=log_msg,
    )


def _push_telemetry_now(detection=None, force: bool = False):
    global _LAST_TELEMETRY_PUSH, _LAST_TELEMETRY_DATA
    if tel is None:
        return None
    now = time.time()
    if not force and now - _LAST_TELEMETRY_PUSH < TELEMETRY_MIN_INTERVAL:
        return _LAST_TELEMETRY_DATA
    det = detection if detection is not None else ACTIVE_TARGET
    try:
        _LAST_TELEMETRY_PUSH = now
        _LAST_TELEMETRY_DATA = tel.collect(detection=det)
        return _LAST_TELEMETRY_DATA
    except Exception as e:
        print(f"[Tel] push error: {e}")
        return None


def _push_cv_snapshot(
    select_fallback: bool = False,
    force: bool = False,
    run_detection: bool = False,
    send_depth: bool = False,
    min_interval: float = None,
):
    global _LAST_CV_SNAPSHOT
    if cv is None or CV_STATE is None:
        return None
    now = time.time()
    interval = CV_SNAPSHOT_MIN_INTERVAL if min_interval is None else float(min_interval)
    if not force and now - _LAST_CV_SNAPSHOT < interval:
        return ACTIVE_TARGET
    _LAST_CV_SNAPSHOT = now

    frame = cv.get_frame()
    depth = cv.get_depth_frame() if send_depth or run_detection else None
    if frame is None:
        return None

    det = ACTIVE_TARGET
    available = []
    if run_detection:
        try:
            dets = cv.detect_colored_objects(
                frame,
                depth_frame=depth,
                with_usd=False,
                estimate_yaw=False,
            )
            available = _sort_by_hand_distance(_filter_unprocessed(dets, log=False))
            det = _select_active_from_detections(dets)
            if det is None and select_fallback and available:
                det = available[0]
        except Exception as e:
            print(f"[CV] snapshot error: {e}")

    CV_STATE.update(
        frame=frame,
        depth=depth,
        detection=det,
        detections=available if run_detection else None,
        send_depth=send_depth,
    )
    return det


def _motion_depth_due() -> bool:
    global _LAST_MOTION_DEPTH_SNAPSHOT
    now = time.time()
    if now - _LAST_MOTION_DEPTH_SNAPSHOT < DEPTH_MOTION_SNAPSHOT_MIN_INTERVAL:
        return False
    _LAST_MOTION_DEPTH_SNAPSHOT = now
    return True


async def _telemetry_heartbeat(app, detection=None, frames: int = 1, flush: bool = False):
    data = None
    count = max(1, int(frames))
    for _ in range(count):
        cv_det = _push_cv_snapshot(
            select_fallback=False,
            force=flush,
            send_depth=flush,
            min_interval=CV_SNAPSHOT_MIN_INTERVAL,
        )
        data = _push_telemetry_now(
            detection=detection if detection is not None else cv_det,
            force=flush,
        )
        if app is not None:
            await app.next_update_async()
    if flush and tel is not None:
        tel.flush(timeout=0.01)
    return data


async def _heartbeat_clear(app):
    """Shortcut for the frequent pattern: heartbeat with cleared detection + flush.

    Compresses ``_telemetry_heartbeat(app, detection=None, flush=True)``
    to ``_heartbeat_clear(app)`` at the call sites.
    """
    await _telemetry_heartbeat(app, detection=None, flush=True)


async def _retry_search(app, label: str, message: str):
    """Log message + heartbeat for retry-localize branches.

    Caller is expected to follow up with ``detection = None; continue``.
    """
    print(f"[{label}] {message}")
    await _heartbeat_clear(app)


def _set_active_destination(color: str, place_xyz: np.ndarray):
    if ACTIVE_TARGET is None:
        return
    ACTIVE_TARGET["place_color"] = color
    ACTIVE_TARGET["place_xyz"] = [float(v) for v in place_xyz]


def _copy_active_metadata(det: dict) -> dict:
    locked = dict(det)
    for key in ("target_locked", "place_color", "place_xyz"):
        if ACTIVE_TARGET is not None and key in ACTIVE_TARGET:
            locked[key] = ACTIVE_TARGET[key]
    locked["target_locked"] = True
    return locked


def _select_active_from_detections(dets: list) -> "dict | None":
    if ACTIVE_TARGET is None:
        return None

    active_prim = ACTIVE_TARGET.get("usd_prim_path")
    if active_prim:
        for det in dets:
            if det.get("usd_prim_path") == active_prim:
                return _copy_active_metadata(det)
        return ACTIVE_TARGET

    active_px = np.array(
        [
            ACTIVE_TARGET.get("pixel_x", 0),
            ACTIVE_TARGET.get("pixel_y", 0),
        ],
        dtype=float,
    )
    active_world = _det_world_xyz(ACTIVE_TARGET)
    active_color = ACTIVE_TARGET.get("color")

    close_candidates = []
    for det in dets:
        if det.get("color") != active_color:
            continue

        det_px = np.array([det.get("pixel_x", 0), det.get("pixel_y", 0)], dtype=float)
        px_dist = float(np.linalg.norm(det_px - active_px))
        world_dist = None
        det_world = _det_world_xyz(det)
        if active_world is not None and det_world is not None:
            world_dist = float(np.linalg.norm(det_world[:2] - active_world[:2]))

        pixel_ok = px_dist <= 18.0
        world_ok = world_dist is not None and world_dist <= 0.035
        if pixel_ok and (active_world is None or world_ok):
            close_candidates.append((px_dist, world_dist if world_dist is not None else 999.0, det))

    if close_candidates:
        close_candidates.sort(key=lambda item: (item[0], item[1]))
        return _copy_active_metadata(close_candidates[0][2])

    return ACTIVE_TARGET


# Async prefetch: kick off the next-cube detection while the previous cycle's
# arm is still over the container (post-release lift), saving 3-5s per cycle.


def _run_color_detection():
    """Shared core for detect_fn / _prefetch: grabs frame+depth, swaps the
    active_colors list, runs detection, restores active_colors.

    Returns (frame, depth, dets) or None if CV is unavailable / no frame.
    """
    if cv is None:
        return None
    frame = cv.get_frame()
    depth = cv.get_depth_frame()
    if frame is None:
        return None
    old = cv.active_colors
    cv.active_colors = PICKUP_COLORS + ["green"]
    try:
        dets = cv.detect_colored_objects(
            frame, depth_frame=depth, with_usd=False, estimate_yaw=False
        )
    finally:
        cv.active_colors = old
    return frame, depth, dets


async def _prefetch_next_detection():
    try:
        result = _run_color_detection()
        if result is None:
            return None
        _, _, dets = result
        available = _sort_by_hand_distance(_filter_unprocessed(dets, log=False))
        actionable = _sort_by_hand_distance([d for d in available if d["color"] in PICKUP_COLORS])
        return actionable[0] if actionable else None
    except Exception as e:
        print(f"[Prefetch] error: {e}")
        return None


# Single-cube cycle: search -> localize -> calibrate bias -> pick -> place -> verify.
# Returns 'red'/'blue'/'green' on successful sort, None on retryable failure.


async def run_one_cycle(app, cycle_num: int) -> "str | None":
    print(f"\n[Cycle {cycle_num}] start")
    _cycle_t0 = time.time()
    _clear_active_target()
    _clear_dashboard_target(status=f"Cycle {cycle_num} - searching...", cycle=cycle_num)
    await _heartbeat_clear(app)

    detection = None
    world_xyz = None

    # Consume the prefetched detection from the previous cycle, if any and
    # still valid (cube not already processed, color in pickup list).
    prefetched = None
    if ctl is not None and getattr(ctl, "prefetch_task", None) is not None:
        try:
            prefetched = await ctl.prefetch_task
        except Exception as e:
            print(f"[Prefetch] await error: {e}")
            prefetched = None
        ctl.prefetch_task = None
        if prefetched is not None and _is_processed_detection(prefetched):
            prefetched = None
        if prefetched is not None:
            color = prefetched.get("color")
            if color not in PICKUP_COLORS:
                prefetched = None
        if prefetched is not None:
            _dbg(
                f"[Prefetch] ✓ using: color={prefetched.get('color')} "
                f"pixel=({prefetched.get('pixel_x')},{prefetched.get('pixel_y')})"
            )

    for search_attempt in range(1, MAX_SEARCH_RETRIES + 1):
        _dbg(f"[Search] attempt {search_attempt}/{MAX_SEARCH_RETRIES}")

        def detect_fn():
            result = _run_color_detection()
            if result is None:
                return None
            frame, depth, dets = result
            available = _sort_by_hand_distance(_filter_unprocessed(dets))
            det = available[0] if available else None
            CV_STATE.update(
                frame=frame,
                depth=depth,
                detection=det,
                detections=available,
                status=f"Cycle {cycle_num} - scanning...",
                cycle=cycle_num,
            )
            _push_telemetry_now(det)
            actionable = _sort_by_hand_distance(
                [d for d in available if d["color"] in PICKUP_COLORS]
            )
            if actionable:
                top = actionable[0]
                _dbg(
                    f"[Select] nearest: {top.get('color')} "
                    f"dist={_distance_to_hand(top):.3f}m "
                    f"xyz={np.round(_det_world_xyz(top), 3) if _det_world_xyz(top) is not None else None}"
                )
            return actionable[0] if actionable else None

        if search_attempt == 1 and prefetched is not None:
            detection = prefetched
            prefetched = None
        else:
            detection = await ctl.scan_for_object(detect_fn)

        if detection is None:
            await _retry_search(app, "Search", "no cube found, retrying...")
            continue

        cube_color = detection.get("color", "unknown")
        # Color is included in the [Target] one-liner below - skip a duplicate log.

        if cube_color == "green":
            print("[Sort] green cube - skipping (not in pickup list)")
            _clear_dashboard_target(status="Green cube - skipped", cycle=cycle_num)
            await _heartbeat_clear(app)
            return "green"

        if cube_color not in PICKUP_COLORS:
            _clear_dashboard_target(status="Unknown color - skipped", cycle=cycle_num)
            print(f"[Sort] unknown color {cube_color} - skipping")
            await _heartbeat_clear(app)
            return None

        stable = cv.detect_stable(
            n_frames=2,
            max_jitter=20,
            color=cube_color,
            target_pixel=(detection["pixel_x"], detection["pixel_y"]),
        )
        if stable is None:
            _clear_dashboard_target(status="Unstable detection - retrying search", cycle=cycle_num)
            detection = None
            await _retry_search(app, "Localize", "unstable detection, retrying")
            continue
        if _is_processed_detection(stable):
            _clear_dashboard_target(
                status="Target already sorted - retrying search", cycle=cycle_num
            )
            detection = None
            await _retry_search(app, "Localize", "cube already processed, retrying")
            continue

        # Single depth snapshot - bias is already EMA-stable at this point.
        _depth_snap = cv.get_depth_frame()

        world_xyz = cv.pixel_to_world(stable["pixel_x"], stable["pixel_y"], depth_frame=_depth_snap)
        if world_xyz is None:
            _clear_dashboard_target(status="No 3D coordinates - retrying search", cycle=cycle_num)
            detection = None
            await _retry_search(app, "Localize", "no 3D coordinates, retrying")
            continue

        # Smooth bias calibration: USD ground truth used only as training signal.
        await recalibrate_bias(stable, world_xyz)

        # Re-measure after bias correction (reuses the same depth snapshot).
        world_xyz = cv.pixel_to_world(stable["pixel_x"], stable["pixel_y"], depth_frame=_depth_snap)
        if world_xyz is None:
            _clear_dashboard_target(status="No 3D coordinates after calibration", cycle=cycle_num)
            detection = None
            await _retry_search(app, "Localize", "no 3D coords after calibration, retrying")
            continue

        stable = cv.apply_usd_yaw_fallback(stable, world_xyz)
        world_xyz = _center_xyz_from_matched_cv(stable, world_xyz)

        if _is_processed_detection(stable):
            _clear_dashboard_target(status="Target already sorted after USD match", cycle=cycle_num)
            detection = None
            await _retry_search(
                app, "Localize", "point/prim already processed after USD match, retrying"
            )
            continue

        _log_target_detail(stable, world_xyz)
        stable["world_xyz"] = [float(v) for v in world_xyz]
        _set_active_target(stable, world_xyz)

        CV_STATE.update(
            world_xyz=world_xyz,
            detection=stable,
            status=f"Cycle {cycle_num} - {cube_color} localized",
            log_msg=(
                f"xyz={np.round(world_xyz, 3)} yaw={stable.get('angle_deg', 0):.1f}° "
                f"[{stable.get('yaw_source', 'cv')}]"
            ),
        )
        await _telemetry_heartbeat(app, detection=stable)

        if not cv.in_pickup_zone(world_xyz):
            _clear_active_target()
            _clear_dashboard_target(status="Out of pickup zone", cycle=cycle_num)
            CV_STATE.update(status="⚠ Out of pickup zone")
            detection = None
            await _retry_search(app, "WS", "cube outside pickup zone, skipping")
            continue

        ws = WORKSPACE.for_z(float(world_xyz[2]))
        x, y, _ = world_xyz
        reach = float(np.linalg.norm(world_xyz[:2]))
        out = []
        if x < ws.x_min:
            out.append(f"X={x:.3f} < {ws.x_min}")
        if x > ws.x_max:
            out.append(f"X={x:.3f} > {ws.x_max}")
        if y < ws.y_min:
            out.append(f"Y={y:.3f} < {ws.y_min}")
        if y > ws.y_max:
            out.append(f"Y={y:.3f} > {ws.y_max}")
        if reach > ws.max_reach:
            out.append(f"reach={reach:.3f} > {ws.max_reach}")
        if out:
            _clear_active_target()
            _clear_dashboard_target(status="Out of workspace", cycle=cycle_num)
            CV_STATE.update(status="⚠ Out of workspace", log_msg=", ".join(out))
            detection = None
            await _retry_search(app, "WS", f"outside workspace: {', '.join(out)}")
            continue

        _dbg(
            f"[Localize] {cube_color} xyz={np.round(world_xyz, 3)} "
            f"width={stable['width_m'] * 100:.1f}cm"
        )
        break

    if detection is None or world_xyz is None:
        _clear_active_target()
        _clear_dashboard_target(status="Cube not found or out of range", cycle=cycle_num)
        print("[Cycle] ✗ no pickup target found")
        await _heartbeat_clear(app)
        return None

    cube_color = stable["color"]
    cube_size = float(np.clip(stable.get("width_m", 0.05), 0.02, 0.08))
    cube_angle = float(stable.get("angle_deg", 0.0))
    cube_height = float(np.clip(stable.get("size_m", cube_size), 0.02, 0.15))
    if abs(cube_height - cube_size) > 0.03:
        cube_height = cube_size

    place_xyz = _place_xyz_for_color(cube_color, cube_size)
    _set_active_destination(cube_color, place_xyz)
    stable["target_locked"] = True
    stable["place_color"] = cube_color
    stable["place_xyz"] = [float(v) for v in place_xyz]

    print(f"[Sort] {cube_color.upper()} -> container {np.round(place_xyz, 3)}")
    _dbg(
        f"[Localize/dbg] h={cube_height * 100:.1f}cm w={cube_size * 100:.1f}cm  "
        f"yaw={cube_angle:.1f}° source={stable.get('yaw_source', 'cv')} "
        f"prim={stable.get('usd_prim_path')}"
    )

    CV_STATE.update(
        status=f"Cycle {cycle_num} - {cube_color} to container",
        detection=stable,
        log_msg=(
            f"place={np.round(place_xyz, 2)} angle={cube_angle:.1f}° "
            f"[{stable.get('yaw_source', 'cv')}]"
        ),
    )

    CV_STATE.update(status=f"Picking up {cube_color}...")
    success = await ctl.pick_and_place(
        target_xyz=world_xyz,
        place_xyz=place_xyz,
        cube_size=cube_size,
        cube_height=cube_height,
        cube_angle_deg=cube_angle,
        max_pick_attempts=2,
        prefetch_scan_fn=_prefetch_next_detection,
    )

    if not success:
        _clear_dashboard_target(status=f"Pickup failed for {cube_color}", cycle=cycle_num)
        print(f"[Cycle] ✗ grasp failed for {cube_color}")
        _clear_active_target()
        await ctl.retreat_to_safe_height()
        await _heartbeat_clear(app)
        return None

    # Phase breakdown + sim performance metrics for the dashboard.
    # Both are populated by pick_and_place and read off the controller here.
    try:
        sim_perf = ctl.get_sim_perf()
        CV_STATE.update(
            phase_timings={
                "cycle": cycle_num,
                "color": cube_color,
                "phases": dict(getattr(ctl, "phase_timings", {}) or {}),
            },
            sim_fps=sim_perf.get("sim_fps"),
            rtf=sim_perf.get("rtf"),
        )
    except Exception as e:
        print(f"[Dash] phase push error: {e}")

    CV_STATE.update(status=f"{cube_color.upper()} placed - verifying...")

    if _cube_visible_at_source(stable, world_xyz):
        _clear_dashboard_target(status="Cube was not moved", cycle=cycle_num)
        print("[Cycle] ✗ cube still at source - not marking processed")
        _clear_active_target()
        await ctl.retreat_to_safe_height()
        await _heartbeat_clear(app)
        return None

    _mark_processed(stable, world_xyz)
    PLACED_COUNTS[cube_color] = PLACED_COUNTS.get(cube_color, 0) + 1
    _clear_active_target()
    data = await _telemetry_heartbeat(app, detection=stable)
    if data:
        print(
            f"[Tel] T={data['motor_temps'][0]:.1f}°C  "
            f"L={data['motor_loads'][0]:.1f}%  {data['status']}"
        )

    _cycle_dt = round(time.time() - _cycle_t0, 1)
    _clear_dashboard_target(cycle=cycle_num)
    CV_STATE.update(
        status=f"✓ {cube_color.upper()} placed  (cycle {cycle_num}, {_cycle_dt}s)",
        cycle=cycle_num,
        containers=dict(PLACED_COUNTS),
        cycle_time=_cycle_dt,
    )
    return cube_color


def _reset_runtime_state():
    """Wipe per-session runtime state so a re-run starts clean.

    The orchestrator tracks 'already processed' cubes via two module-level
    globals (PROCESSED_USD_PRIMS, PROCESSED_PICKUP_POINTS) and a per-color
    placement counter (PLACED_COUNTS). When the user resets the Isaac scene
    and re-issues 'import robot_motion; robot_motion.main()', these globals
    must be cleared - otherwise main_loop() considers every cube already
    sorted and exits immediately. ACTIVE_TARGET is reset for the same reason.
    Listed in Limitations as the price of keeping the orchestrator state at
    module level rather than wrapping it in a Runtime dataclass.
    """
    global ACTIVE_TARGET, _LAST_CV_SNAPSHOT, _LAST_MOTION_DEPTH_SNAPSHOT
    global _LAST_TELEMETRY_PUSH, _LAST_TELEMETRY_DATA
    PROCESSED_USD_PRIMS.clear()
    PROCESSED_PICKUP_POINTS.clear()
    PLACED_COUNTS["red"] = 0
    PLACED_COUNTS["blue"] = 0
    ACTIVE_TARGET = None
    _LAST_CV_SNAPSHOT = 0.0
    _LAST_MOTION_DEPTH_SNAPSHOT = 0.0
    _LAST_TELEMETRY_PUSH = 0.0
    _LAST_TELEMETRY_DATA = None


async def main_loop():
    global _MAIN_LOOP_ACTIVE, _MAIN_LOOP_SHOULD_EXIT

    if _MAIN_LOOP_ACTIVE:
        # Previous main_loop is still running (typically paused inside an
        # await app.next_update_async() after the user pressed Stop in
        # Isaac). Signal it to exit and wait until it does, otherwise both
        # loops would mutate PROCESSED_USD_PRIMS in parallel and the new
        # run would see cubes re-marked 'processed' by the old one's
        # post-Play resume.
        print("[Main] previous main_loop still active - signalling exit")
        _MAIN_LOOP_SHOULD_EXIT = True
        for _ in range(100):
            if not _MAIN_LOOP_ACTIVE:
                break
            await asyncio.sleep(0.1)
        else:
            print("[Main] warning: previous main_loop did not stop within 10s")

    _MAIN_LOOP_ACTIVE = True
    _MAIN_LOOP_SHOULD_EXIT = False
    _reset_runtime_state()

    try:
        initialize_system()
        app = omni.kit.app.get_app()

        print("[Init] waiting for simulation...")
        for _ in range(30):
            await app.next_update_async()

        robot.initialize()
        for _ in range(10):
            await app.next_update_async()

        pos = robot.get_joint_positions()
        if pos is None:
            print("[Init] ✗ press Play in Isaac Sim!")
            return
        print("[Init] ✓ robot ready")

        frame = cv.get_frame()
        if frame is None:
            try:
                cv.color_camera.resume()
                cv.depth_camera.resume()
            except Exception as e:
                _dbg(f"[Init] camera resume() failed: {e}")
            for _ in range(15):
                await app.next_update_async()
            frame = cv.get_frame()
            if frame is None:
                print("[Init] ✗ camera not responding")
                return
        print(f"[Init] ✓ camera ready: {frame.shape}")
        CV_STATE.update(status="Configuring CV...")

        cv.set_world_bias(CV_WORLD_BIAS)
        cv.active_colors = PICKUP_COLORS + ["green"]
        cv.usd_cube_prim_paths = USD_CUBE_PRIM_PATHS
        cv.usd_yaw_tolerance = USD_YAW_TOLERANCE
        cv.usd_match_max_distance = USD_MATCH_MAX_DISTANCE
        print(f"[Init] CV world-bias initialized: {np.round(CV_WORLD_BIAS, 4)}")
        cv.debug_usd_cubes()

        cv.set_pickup_zone(**PICKUP_ZONE)

        ctl.rmpflow_offset = RMPFLOW_OFFSET.copy()

        cv.usd_quiet = True

        def _tick_push():
            det = _push_cv_snapshot(
                select_fallback=True,
                send_depth=_motion_depth_due(),
                min_interval=CV_MOTION_SNAPSHOT_MIN_INTERVAL,
            )
            _push_telemetry_now(det)

        ctl.on_tick = _tick_push

        await ctl.retreat_to_safe_height(max_steps=55)

        WORKSPACE.summary()

        _scan_containers()
        CV_STATE.update(status="Ready - starting sorting", cycle=0)
        await _telemetry_heartbeat(app, detection=None, frames=2, flush=True)

        stats = {"red": 0, "blue": 0, "green": 0, "failed": 0}
        cycle = 0
        empty_strikes = 0
        MAX_EMPTY_STRIKES = 3
        sorting_complete = False

        while True:
            # Single-run guard: a fresh main() call signalled us to stop so
            # that a new main_loop can take over without two loops racing
            # on PROCESSED_USD_PRIMS.
            if _MAIN_LOOP_SHOULD_EXIT:
                print("[Main] exit signal received - stopping")
                break

            # Honour CLEAR pressed on the dashboard mid-run: CVPoster sets the
            # flag from the dashboard's POST response. Reset our processed-cube
            # memos so the next cycle re-evaluates every cube as fresh.
            if CV_STATE is not None and getattr(CV_STATE, "reset_requested", False):
                print("[Main] dashboard CLEAR received - resetting processed-cube memos")
                _reset_runtime_state()
                CV_STATE.reset_requested = False
                stats = {"red": 0, "blue": 0, "green": 0, "failed": 0}
                cycle = 0
                empty_strikes = 0

            cycle += 1
            # Refresh cube positions and `processed` flags for dashboard each cycle
            # (otherwise the workspace map and cube table show stale init-time data).
            _push_cube_states()
            result = await run_one_cycle(app, cycle)

            _ts = time.strftime("%H:%M:%S")
            if result == "red":
                stats["red"] += 1
                empty_strikes = 0
                CV_STATE.update(
                    cycle_log_entry={"cycle": cycle, "color": "red", "result": "✓ red", "ts": _ts}
                )
            elif result == "blue":
                stats["blue"] += 1
                empty_strikes = 0
                CV_STATE.update(
                    cycle_log_entry={"cycle": cycle, "color": "blue", "result": "✓ blue", "ts": _ts}
                )
            elif result == "green":
                stats["green"] += 1
                empty_strikes = 0
                CV_STATE.update(
                    cycle_log_entry={"cycle": cycle, "color": "green", "result": "⊘ skip", "ts": _ts}
                )
            else:
                stats["failed"] += 1
                empty_strikes += 1
                CV_STATE.update(
                    cycle_log_entry={"cycle": cycle, "color": None, "result": "✗ fail", "ts": _ts}
                )
                usd_cubes = cv.get_usd_cube_states(x_min=0.20, x_max=0.80)
                actionable = [
                    c
                    for c in usd_cubes
                    if c.get("color") in PICKUP_COLORS and 0.20 <= float(c["xyz"][0]) <= 0.80
                ]
                if not actionable:
                    sorting_complete = True
                    print("[Main] pickup zone empty - stopping")
                    break
                if empty_strikes >= MAX_EMPTY_STRIKES:
                    print(f"[Main] {MAX_EMPTY_STRIKES} consecutive failures - stopping")
                    break
                _dbg(f"[Main] {len(actionable)} cubes in zone, retrying...")

        if sorting_complete:
            CV_STATE.update(status="Sorting complete - returning home")
            await _heartbeat_clear(app)
            print("[Main] all cubes sorted - returning to home")
            await ctl.move_to_home(steps=60)
            CV_STATE.update(status="Sorting complete - home")
            await _heartbeat_clear(app)

        tel.save()
        print(
            f"\n[Result] {cycle} cycles | red={stats['red']} blue={stats['blue']} "
            f"green-skipped={stats['green']} failed={stats['failed']} | "
            f"dashboard: http://localhost:5000"
        )
        flush_async_print(timeout=1.0)
    finally:
        # Always release the single-run guard so a subsequent main() call
        # can start a fresh main_loop, even if the body raised or returned
        # early (e.g. robot init failed).
        _MAIN_LOOP_ACTIVE = False


def main():
    """Entry point invoked by ``python robot_motion.py`` and by Isaac's
    Script Editor exec().

    sys.path setup happens at module top so the project imports resolve.
    install_async_print() is gated here so that a third party who simply
    `import robot_motion` for inspection or testing does not get
    builtins.print silently replaced.
    """
    install_async_print()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.create_task(main_loop())
        return loop.run_until_complete(main_loop())
    except RuntimeError:
        return asyncio.run(main_loop())


if __name__ == "__main__":
    main()
