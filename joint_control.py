"""Robot motion planning, gripper control and pick-and-place orchestration.

Provides :class:`JointController` wrapping Isaac Sim's RmpFlow + Articulation
motion policies, plus :class:`DualWorkspace` for height-aware reach constraints.
The full :meth:`pick_and_place` cycle handles pre-align -> descent -> grip -> lift
-> pre-place -> place -> retreat with phase-level timing for the dashboard.

Container coordinates and workspace bounds are scene-specific (Franka Panda on
a table); see ``scene_config.py`` for reference values.
"""

import asyncio
import os
import time

import numpy as np

# Isaac Sim modules are only available inside the Isaac Sim Python environment.
# Wrap the imports so this module is importable in pure-Python contexts; any
# code path that actually uses RmpFlow / Articulation policies still requires
# Isaac and will fail clearly at instantiation when the symbols are None.
try:
    import omni.kit.app  # type: ignore[import-not-found]
    from omni.isaac.core.utils.rotations import euler_angles_to_quat  # type: ignore[import-not-found]
    from omni.isaac.motion_generation import (  # type: ignore[import-not-found]
        ArticulationMotionPolicy,
        RmpFlow,
    )
    from omni.isaac.motion_generation.interface_config_loader import (  # type: ignore[import-not-found]
        load_supported_motion_policy_config,
    )
except ImportError:
    omni = None  # type: ignore[assignment]
    euler_angles_to_quat = None  # type: ignore[assignment,misc]
    ArticulationMotionPolicy = RmpFlow = None  # type: ignore[assignment,misc]
    load_supported_motion_policy_config = None  # type: ignore[assignment,misc]

try:
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
except ImportError:
    UsdGeom = Usd = None  # type: ignore[assignment,misc]

import scene_config as _scene_config

# Verbose log toggle (env var DTCV_VERBOSE=1). Mirrored across modules so each
# can independently gate its detail output without a shared import.
VERBOSE = os.environ.get("DTCV_VERBOSE") == "1"


def _dbg(msg: str):
    """Print only when verbose mode is enabled."""
    if VERBOSE:
        print(msg)


HOME_JOINT_POSITIONS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

FINGER_TIP_OFFSET_Z = 0.1034  # flange-to-fingertip distance (meters)
GRIPPER_OPEN_EXTRA = 0.018  # extra clearance per finger before approaching the cube
GRIPPER_OPEN_MAX = 0.045  # slightly wider than the 4cm default to clear cube edges
GRIPPER_SQUEEZE_MAX = 0.023  # cap on per-finger squeeze: don't trust inflated CV widths
DEBUG_PICK_POSE = False


CONTAINER_SIZE_M = 0.28

# Container drop-off centers in world coordinates (meters). scene_config
# holds the canonical definition; the np.array conversion here is just a
# typing convenience for the motion code that does vector arithmetic.
CONTAINER_RED = np.asarray(_scene_config.CONTAINERS["red"], dtype=float)
CONTAINER_BLUE = np.asarray(_scene_config.CONTAINERS["blue"], dtype=float)


# Workspace constraints for Franka Panda with gripper pointing down.
class WorkspaceConstraints:
    """Cuboid + radial-reach workspace bounds for the Franka end-effector.

    Used by ``clamp()`` to bring out-of-range targets back into reach before
    handing them to RmpFlow. Keeps z_probe so :class:`DualWorkspace` can pick
    high vs low constraint based on target Z.
    """

    def __init__(
        self,
        x_min: float = 0.23,
        x_max: float = 0.77,
        y_min: float = -0.67,
        y_max: float = 0.57,
        z_min: float = 0.03,
        z_max: float = 0.82,
        max_reach: float = 0.72,
        z_probe: float = 0.40,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.max_reach = max_reach
        self.z_probe = z_probe

    def clamp(self, xyz: np.ndarray) -> np.ndarray:
        clamped = np.array(
            [
                np.clip(xyz[0], self.x_min, self.x_max),
                np.clip(xyz[1], self.y_min, self.y_max),
                np.clip(xyz[2], self.z_min, self.z_max),
            ]
        )
        reach = float(np.linalg.norm(clamped[:2]))
        if reach > self.max_reach:
            scale = self.max_reach / reach
            clamped[0] *= scale
            clamped[1] *= scale
        if not np.allclose(clamped, xyz, atol=1e-4):
            _dbg(f"[WS] clamp: {np.round(xyz, 3)} -> {np.round(clamped, 3)}")
        return clamped

    def __repr__(self):
        return (
            f"WS(z_probe={self.z_probe} "
            f"X=[{self.x_min},{self.x_max}] "
            f"Y=[{self.y_min},{self.y_max}] "
            f"Z=[{self.z_min},{self.z_max}] "
            f"reach={self.max_reach})"
        )


class DualWorkspace:
    """Height-aware workspace selector: low (cube level) vs high (transit).

    Picks the appropriate :class:`WorkspaceConstraints` based on the target's
    Z coordinate. Low workspace allows wider X/Y reach near the table; high
    workspace tightens bounds to stay inside RmpFlow's reach envelope during
    fast transit.
    """

    def __init__(self):
        # Bounds come from scene_config so the dashboard reach overlay and the
        # motion clamp share one definition. WORKSPACE_HIGH / WORKSPACE_LOW
        # there are dicts; unpack them as kwargs to the constraints class.
        self.high = WorkspaceConstraints(**_scene_config.WORKSPACE_HIGH)
        self.low = WorkspaceConstraints(**_scene_config.WORKSPACE_LOW)

    def for_z(self, z: float) -> WorkspaceConstraints:
        if z < (self.high.z_probe + self.low.z_probe) / 2:
            return self.low
        return self.high

    def clamp(self, xyz: np.ndarray) -> np.ndarray:
        return self.for_z(float(xyz[2])).clamp(xyz)

    def summary(self):
        z_split = (self.high.z_probe + self.low.z_probe) / 2
        print(f"[Init] workspace high(z≥{z_split:.2f}m): {self.high}")
        print(f"[Init] workspace low (z< {z_split:.2f}m): {self.low}")
        print(
            f"[Init] containers: RED={np.round(CONTAINER_RED, 3)} "
            f"BLUE={np.round(CONTAINER_BLUE, 3)}"
        )


WORKSPACE = DualWorkspace()


class JointController:
    """RmpFlow-based motion controller with pick-and-place orchestration.

    Wraps three :class:`ArticulationMotionPolicy` profiles (normal / fast /
    aggressive) for different motion phases, plus the gripper actions and
    the full ``pick_and_place()`` cycle. Reports phase-level timings to the
    dashboard for performance breakdown.
    """

    def __init__(
        self, robot, dt: float = 1 / 35, dt_fast: float = 1 / 18, dt_aggressive: float = 1 / 8
    ):
        self.robot = robot
        self.app = omni.kit.app.get_app()
        self.grip_down = euler_angles_to_quat(np.array([0, np.pi, 0]))
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmpflow = RmpFlow(**rmp_config)
        self.art_rmp = ArticulationMotionPolicy(robot, self.rmpflow, dt)
        self.art_rmp_fast = ArticulationMotionPolicy(robot, self.rmpflow, dt_fast)
        self.art_rmp_aggressive = ArticulationMotionPolicy(robot, self.rmpflow, dt_aggressive)

        self.rmpflow_offset = np.zeros(3)
        self.on_tick = None
        self._tick_counter = 0
        self._tick_interval = 8
        self._min_settle_steps = 2
        self.prefetch_task = None
        self.phase_timings = {}
        # Real-Time Factor / sim-FPS counters reported to the dashboard.
        self._sim_tick_count = 0
        self._sim_wall_t0 = time.time()
        # Cache absolute prim_path so we can re-resolve hand_prim through the
        # stage on every call. Required because self.robot.prim becomes expired
        # after Isaac stop+play, which would otherwise spam thousands of errors.
        self._robot_path = getattr(robot, "prim_path", None)
        self._last_ee_err_log = 0.0

    def _apply_offset(self, target: np.ndarray) -> np.ndarray:
        return target + self.rmpflow_offset

    async def _step(self, run_tick: bool = True):
        await self.app.next_update_async()
        self._sim_tick_count += 1
        if run_tick and self.on_tick is not None:
            self._tick_counter += 1
            if self._tick_counter >= self._tick_interval:
                self._tick_counter = 0
                try:
                    self.on_tick()
                except Exception as e:
                    _dbg(f"[Tick] on_tick callback raised: {e}")

    def get_sim_perf(self, physics_dt: float = 1 / 60) -> dict:
        """Real-Time Factor + simulated FPS - reported to the dashboard."""
        wall_dt = max(time.time() - self._sim_wall_t0, 1e-3)
        ticks = max(self._sim_tick_count, 1)
        sim_fps = ticks / wall_dt
        rtf = sim_fps * physics_dt
        return {"sim_fps": round(sim_fps, 1), "rtf": round(rtf, 3)}

    async def _wait(self, n: int, run_tick: bool = True):
        for _ in range(n):
            await self._step(run_tick=run_tick)

    async def move_to_pose(
        self,
        position: np.ndarray,
        max_steps: int = 180,
        clamp: bool = True,
        tol: float = 0.05,
        fast: bool = False,
        aggressive: bool = False,
        tag: str = "move",
    ) -> bool:
        if clamp:
            position = WORKSPACE.clamp(position)
        command_position = self._apply_offset(position)
        if aggressive:
            art = self.art_rmp_aggressive
        elif fast:
            art = self.art_rmp_fast
        else:
            art = self.art_rmp
        self.rmpflow.set_end_effector_target(
            target_position=command_position.astype(np.float64), target_orientation=self.grip_down
        )
        for i in range(max_steps):
            self.rmpflow.update_world()
            action = art.get_next_articulation_action()
            self.robot.apply_action(action)
            await self._step()
            if i > self._min_settle_steps and self._reached(position, tol=tol):
                _dbg(f"[Steps] {tag}: {i + 1}/{max_steps} reached=YES")
                return True
        pos = self.get_ee_position()
        err = float(np.linalg.norm(pos - position)) if pos is not None else float("inf")
        reached = bool(pos is not None and err <= tol)
        _dbg(
            f"[Steps] {tag}: {max_steps}/{max_steps} reached={'YES' if reached else 'NO'} err={err:.3f}"
        )
        if not reached:
            _dbg(
                f"[Move] {tag} target={np.round(position, 3)} cmd={np.round(command_position, 3)} err={err:.3f} tol={tol:.3f}"
            )
        return reached

    async def move_to_pose_oriented_fast(
        self,
        position: np.ndarray,
        yaw_deg: float = 0.0,
        max_steps: int = 90,
        tol: float = 0.025,
        xy_tol: float = None,
        z_tol: float = None,
        run_tick: bool = False,
        aggressive: bool = False,
        tag: str = "move/fast",
    ) -> bool:
        position = WORKSPACE.clamp(position)
        command_position = self._apply_offset(position)
        yaw_rad = np.deg2rad(yaw_deg)
        orientation = euler_angles_to_quat(np.array([0.0, np.pi, yaw_rad]))
        art = self.art_rmp_aggressive if aggressive else self.art_rmp_fast
        self.rmpflow.set_end_effector_target(
            target_position=command_position.astype(np.float64), target_orientation=orientation
        )
        for i in range(max_steps):
            self.rmpflow.update_world()
            action = art.get_next_articulation_action()
            self.robot.apply_action(action)
            await self._step(run_tick=run_tick)
            if self._reached_pose(position, tol=tol, xy_tol=xy_tol, z_tol=z_tol):
                _dbg(f"[Steps] {tag}: {i + 1}/{max_steps} reached=YES")
                return True
        pos, err, xy_err, z_err = self._pose_error(position)
        reached = self._reached_pose(position, tol=tol, xy_tol=xy_tol, z_tol=z_tol)
        _dbg(
            f"[Steps] {tag}: {max_steps}/{max_steps} reached={'YES' if reached else 'NO'} "
            f"err={err:.3f} xy={xy_err:.3f} z_err={z_err * 1000:+.1f}mm"
        )
        if not reached:
            _dbg(
                f"[Move/Fast] {tag} target={np.round(position, 3)} cmd={np.round(command_position, 3)} yaw={yaw_deg:.1f} tol={tol:.3f}"
            )
        return reached

    async def move_through_oriented_waypoints(
        self,
        waypoints: list,
        yaw_deg: float = 0.0,
        run_tick: bool = False,
        open_gripper_width: float = None,
        open_gripper_steps: int = 0,
    ) -> bool:
        yaw_rad = np.deg2rad(yaw_deg)
        orientation = euler_angles_to_quat(np.array([0.0, np.pi, yaw_rad]))
        final_ok = True
        open_target = None
        open_start_l = open_start_r = 0.0
        open_step = 0
        open_steps = max(1, int(open_gripper_steps))
        if open_gripper_width is not None:
            pos = self.robot.get_joint_positions()
            if pos is not None:
                open_start_l = float(pos[-2])
                open_start_r = float(pos[-1])
                open_target = float(
                    np.clip(
                        open_gripper_width / 2.0 + GRIPPER_OPEN_EXTRA,
                        0.005,
                        GRIPPER_OPEN_MAX,
                    )
                )

        for idx, wp in enumerate(waypoints):
            position = np.asarray(wp["position"], dtype=float)
            target = WORKSPACE.clamp(position) if wp.get("clamp", True) else position
            command_target = self._apply_offset(target)
            max_steps = int(wp.get("max_steps", 80))
            min_steps = int(wp.get("min_steps", 0))
            tol = float(wp.get("tol", 0.04))
            xy_tol = wp.get("xy_tol")
            z_tol = wp.get("z_tol")
            required = bool(wp.get("required", idx == len(waypoints) - 1))
            if wp.get("aggressive"):
                art = self.art_rmp_aggressive
            elif wp.get("fast"):
                art = self.art_rmp_fast
            else:
                art = self.art_rmp

            self.rmpflow.set_end_effector_target(
                target_position=command_target.astype(np.float64),
                target_orientation=orientation,
            )

            reached = False
            steps_used = max_steps
            for step in range(max_steps):
                self.rmpflow.update_world()
                action = art.get_next_articulation_action()
                self.robot.apply_action(action)
                if open_target is not None and open_step < open_steps:
                    cur = self.robot.get_joint_positions()
                    if cur is not None:
                        t = (open_step + 1) / open_steps
                        cur[-2] = open_start_l + (open_target - open_start_l) * t
                        cur[-1] = open_start_r + (open_target - open_start_r) * t
                        self.robot.set_joint_positions(cur)
                    open_step += 1
                await self._step(run_tick=run_tick)
                if step + 1 >= min_steps and self._reached_pose(
                    target, tol=tol, xy_tol=xy_tol, z_tol=z_tol
                ):
                    reached = True
                    steps_used = step + 1
                    break

            wp_name = wp.get("name", idx)
            _dbg(
                f"[Steps] path/{wp_name}: {steps_used}/{max_steps} "
                f"reached={'YES' if reached else 'NO'}"
            )

            if required and not reached:
                _, err, xy_err, z_err = self._pose_error(target)
                _dbg(
                    f"[Move/Path] {wp_name} err={err:.3f} xy={xy_err:.3f} "
                    f"z_err={z_err * 1000:+.1f}mm"
                )
                final_ok = False

        return final_ok

    def get_ee_position(self) -> "np.ndarray | None":
        # Re-resolve hand_prim through the stage on every call: self.robot.prim
        # becomes expired after Isaac stop+play, which would spam thousands of
        # error logs without the absolute-path lookup approach.
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None or not self._robot_path:
                return None
            hand_prim = stage.GetPrimAtPath(f"{self._robot_path}/panda_hand")
            if not hand_prim.IsValid():
                hand_prim = stage.GetPrimAtPath(f"{self._robot_path}/panda_hand_0")
            if not hand_prim.IsValid():
                return None
            xf = UsdGeom.Xformable(hand_prim)
            mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos = np.array([mat[3][0], mat[3][1], mat[3][2]])
            scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
            hand_pos = pos * scale
            return hand_pos - np.array([0.0, 0.0, FINGER_TIP_OFFSET_Z])
        except Exception as e:
            now = time.monotonic()
            if now - self._last_ee_err_log > 5.0:
                print(f"[Pick] failed to get EE position: {e}")
                self._last_ee_err_log = now
        return None

    def _reached(self, target: np.ndarray, tol: float = 0.015) -> bool:
        pos = self.get_ee_position()
        if pos is not None:
            return float(np.linalg.norm(pos - target)) < tol
        return False

    def _pose_error(self, target: np.ndarray):
        pos = self.get_ee_position()
        if pos is None:
            return None, float("inf"), float("inf"), float("inf")
        delta = pos - target
        err = float(np.linalg.norm(delta))
        xy_err = float(np.linalg.norm(delta[:2]))
        z_err = float(delta[2])
        return pos, err, xy_err, z_err

    def _reached_pose(
        self, target: np.ndarray, tol: float, xy_tol: float = None, z_tol: float = None
    ) -> bool:
        _, err, xy_err, z_err = self._pose_error(target)
        if err > tol:
            return False
        if xy_tol is not None and xy_err > xy_tol:
            return False
        if z_tol is not None and abs(z_err) > z_tol:
            return False
        return True

    def _zero_joint_velocities(self):
        try:
            cur = self.robot.get_joint_positions()
            if cur is not None and hasattr(self.robot, "set_joint_velocities"):
                self.robot.set_joint_velocities(np.zeros_like(cur))
        except Exception as e:
            _dbg(f"[Stop] zero-velocities call failed: {e}")

    async def _stabilize(self, frames: int = 5, run_tick: bool = True):
        try:
            self.rmpflow.reset()
        except Exception as e:
            _dbg(f"[Stabilize] rmpflow.reset() failed: {e}")
        self._zero_joint_velocities()
        cur = self.robot.get_joint_positions()
        if cur is not None:
            for _ in range(frames):
                self.robot.set_joint_positions(cur)
                self._zero_joint_velocities()
                await self._step(run_tick=run_tick)

    async def move_to_home(self, steps: int = 48) -> None:
        cur = self.robot.get_joint_positions()
        if cur is None:
            target = HOME_JOINT_POSITIONS.copy()
            target[-2:] = 0.0
            self.robot.set_joint_positions(target)
            await self._wait(6)
        else:
            target = HOME_JOINT_POSITIONS.copy()
            if cur.size >= 2:
                target[-2:] = cur[-2:]
            delta = target - cur
            if float(np.max(np.abs(delta))) < 1e-4:
                try:
                    self.rmpflow.reset()
                except Exception:
                    pass
                _dbg("[IK] home position reached")
                return
            for i in range(1, steps + 1):
                t = 0.5 * (1.0 - np.cos(np.pi * i / steps))
                self.robot.set_joint_positions(cur + t * delta)
                await self._step()
        try:
            self.rmpflow.reset()
        except Exception:
            pass
        _dbg("[IK] home position reached")

    async def close_gripper_smooth(
        self,
        cube_width: "float | None" = None,
        steps: int = 42,
        hold_arm: bool = False,
    ) -> bool:
        pos = self.robot.get_joint_positions()
        if pos is None:
            print("[Grip] cannot close: joint_positions=None")
            return False
        hold_pos = pos.copy()
        start = float(pos[-2])

        if cube_width is not None:
            raw_safe_min = max(cube_width / 2.0 - 0.008, 0.002)
            safe_min = min(raw_safe_min, GRIPPER_SQUEEZE_MAX)
        else:
            raw_safe_min = 0.002
            safe_min = 0.002

        contact_detected = False
        final_commanded = start
        _dbg(
            f"[Grip] closing: start={start * 1000:.1f}mm "
            f"safe_min={safe_min * 1000:.1f}mm "
            f"(raw={raw_safe_min * 1000:.1f}mm)  steps={steps}"
        )

        for i in range(1, steps + 1):
            t = i / steps
            commanded = max(start * (1.0 - t), safe_min)
            final_commanded = commanded

            if hold_arm:
                cur = hold_pos.copy()
            else:
                cur = self.robot.get_joint_positions()
                if cur is None:
                    break
            cur[-2] = commanded
            cur[-1] = commanded
            self.robot.set_joint_positions(cur)
            if hold_arm:
                self._zero_joint_velocities()
            await self._step(run_tick=not hold_arm)

            try:
                art = self.robot._articulation_view
                if art is not None:
                    real = art.get_joint_positions()
                    if real is not None:
                        real_opening = float(real[0, -2])
                        error = real_opening - commanded
                        if i > max(4, steps // 3) and error > 0.001:
                            _dbg(
                                f"[Grip] contact at step {i}: "
                                f"commanded={commanded * 1000:.1f}mm  "
                                f"actual={real_opening * 1000:.1f}mm  "
                                f"error={error * 1000:.1f}mm"
                            )
                            contact_detected = True
                            break
            except Exception as e:
                _dbg(f"[Grip] contact-detect probe failed: {e}")

            if commanded <= safe_min + 0.0005:
                _dbg(f"[Grip] safe minimum: {commanded * 1000:.1f}mm")
                break

        if not contact_detected:
            _dbg("[Grip] no contact detected")

        if hold_arm:
            hold_pos[-2] = final_commanded
            hold_pos[-1] = final_commanded
            for _ in range(1):
                self.robot.set_joint_positions(hold_pos)
                self._zero_joint_velocities()
                await self._step(run_tick=False)
        else:
            await self._wait(8)
        return contact_detected

    async def retreat_to_safe_height(self, z: float = 0.34, max_steps: int = 42) -> bool:
        ee_now = self.get_ee_position()
        if ee_now is None:
            return False
        target = np.array([ee_now[0], ee_now[1], max(float(z), ee_now[2] + 0.04)])
        return await self.move_to_pose_oriented_fast(
            target,
            yaw_deg=0.0,
            max_steps=max_steps,
            tol=0.090,
            xy_tol=0.090,
            z_tol=0.025,
            run_tick=True,
        )

    def is_grasping(self, min_gap: float = 0.015) -> bool:
        gap = None

        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None and self._robot_path:
                for fn in [
                    "panda_leftfinger",
                    "panda_rightfinger",
                    "panda_leftfinger_0",
                    "panda_rightfinger_0",
                ]:
                    prim = stage.GetPrimAtPath(f"{self._robot_path}/{fn}")
                    if prim.IsValid():
                        attr = prim.GetAttribute("physics:jointPosition")
                        if attr.IsValid():
                            val = attr.Get()
                            if val is not None:
                                gap = float(val) * 2
                                break
        except Exception as e:
            print(f"[Grip] USD: {e}")

        if gap is None:
            try:
                art = self.robot._articulation_view
                if art is not None:
                    positions = art.get_joint_positions()
                    if positions is not None:
                        gap = float(positions[0, -2]) + float(positions[0, -1])
            except Exception as e:
                print(f"[Grip] ArtView: {e}")

        if gap is None:
            pos = self.robot.get_joint_positions()
            if pos is None:
                return False
            gap = float(pos[-2]) + float(pos[-1])

        result = gap > min_gap
        # Per-cycle gripper telemetry. Verbose-only so the normal log stays
        # at the documented '~5 lines per cycle' budget; the dashboard's
        # gripper indicator is the user-facing surface.
        _dbg(
            f"[Grip] gap={gap * 1000:.1f}mm | threshold={min_gap * 1000:.0f}mm -> "
            f"{'HOLDING' if result else 'EMPTY'}"
        )
        return result

    async def scan_for_object(self, detect_fn) -> "dict | None":
        for _ in range(2):
            await self._wait(1)
            d1 = detect_fn()
            if d1 is None:
                continue
            await self._wait(1)
            d2 = detect_fn()
            if d2 is None:
                continue
            if (
                d1.get("color") == d2.get("color")
                and abs(d1["pixel_x"] - d2["pixel_x"]) < 30
                and abs(d1["pixel_y"] - d2["pixel_y"]) < 30
            ):
                _dbg(
                    f"[Scan] confirmed: color={d2.get('color')} "
                    f"pixel=({d2['pixel_x']}, {d2['pixel_y']})"
                )
                return d2
        _dbg("[Scan] no object found")
        return None

    async def pick_and_place(
        self,
        target_xyz: np.ndarray,
        place_xyz: np.ndarray,
        cube_size: "float | None" = None,
        cube_height: "float | None" = None,
        cube_angle_deg: float = 0.0,
        max_pick_attempts: int = 3,
        prefetch_scan_fn=None,
    ) -> bool:
        cube_height = float(cube_height) if cube_height else 0.05
        cube_height = float(np.clip(cube_height, 0.02, 0.30))
        top_z = float(target_xyz[2])

        # Cube geometry: top_z is the detected top surface, floor is the table
        # contact point (top - height), grasp center is the cube midpoint.
        FLOOR_Z = top_z - cube_height
        if FLOOR_Z < 0.01:  # cube sits on the floor (or below table reference)
            FLOOR_Z = 0.0
        measured_height = float(np.clip(top_z - FLOOR_Z, 0.02, cube_height + 0.01))
        effective_height = float(np.clip(cube_height, 0.02, max(measured_height, cube_height)))
        center_of_cube = top_z - effective_height / 2.0  # geometric mid-height

        grip_yaw = cube_angle_deg
        _dbg(f"[Pick] h={cube_height * 100:.1f}cm yaw={cube_angle_deg:.1f}°")
        _dbg(
            f"[Pick] top_z={top_z:.3f} height={effective_height:.3f} floor_z={FLOOR_Z:.3f} center_z={center_of_cube:.3f}"
        )
        _dbg(f"[Pick] rmpflow_offset={np.round(self.rmpflow_offset, 4)}")

        grasp_z = max(center_of_cube, FLOOR_Z + 0.015)
        base_grasp_z = grasp_z
        _dbg(f"[Pick] grasp center_z={grasp_z:.3f} (cube center)")

        # 10 cm above the cube top: enough that the gripper fingertips clear
        # neighbouring cubes during the descent, small enough that the descent
        # itself stays inside one motion-policy step at the 'fast' dt.
        approach_clearance = 0.10
        above_z = max(top_z + approach_clearance, grasp_z + 0.04)
        entry_z = max(top_z + 0.055, grasp_z + 0.030)
        transit_z = max(above_z + 0.12, 0.28)
        grasp_xyz = np.array([target_xyz[0], target_xyz[1], grasp_z])
        entry_xyz = np.array([target_xyz[0], target_xyz[1], entry_z])
        above_xyz = np.array([target_xyz[0], target_xyz[1], above_z])
        transit_xyz = np.array([target_xyz[0], target_xyz[1], transit_z])
        place_base_z = float(place_xyz[2])
        release_clearance = 0.0
        place_release_z = place_base_z + cube_height / 2.0 + FINGER_TIP_OFFSET_Z + release_clearance
        place_xyz_adj = np.array([place_xyz[0], place_xyz[1], place_release_z])
        pre_place = np.array([place_xyz[0], place_xyz[1], place_release_z + 0.16])

        _dbg(
            f"[Pick] grasp_xyz={np.round(grasp_xyz, 3)} above_xyz={np.round(above_xyz, 3)} "
            f"transit={np.round(transit_xyz, 3)} "
            f"(top+{approach_clearance * 100:.0f}cm)"
        )
        _dbg(
            f"[Place] container_center={np.round(place_xyz, 3)} "
            f"base_z={place_base_z:.3f} clearance={release_clearance:.3f} "
            f"release_xyz={np.round(place_xyz_adj, 3)} "
            f"(no rmpflow_offset)"
        )

        # Start cycle timer; phase boundaries are timestamped below for the
        # dashboard's stacked-bar phase breakdown.
        self.phase_timings = {}
        _t_cycle_start = time.time()

        for attempt in range(1, max_pick_attempts + 1):
            _dbg(f"[Pick] attempt {attempt}/{max_pick_attempts}")

            try:
                self.rmpflow.reset()
            except Exception:
                pass

            # Two-stage approach: position above the cube first (gripper still
            # at default yaw=0), then rotate to grasp yaw in place. Combining
            # translation+rotation in one move can cause RmpFlow to overshoot.
            # Stage 1: fast vertical-axis approach to the safe height above cube.
            # 0.32 m floor is the empirical minimum at which the workspace-high
            # constraint still admits the full XY pickup region without Lula
            # joint-limit clipping. transit_xyz[2] is usually higher (~0.4 m).
            safe_z = max(transit_xyz[2], 0.32)
            pre_align_xyz = np.array([grasp_xyz[0], grasp_xyz[1], safe_z])

            # Move above the cube with default orientation (no rotation yet).
            _dbg(f"[Pick] pre-align {np.round(pre_align_xyz, 3)}")
            pre_ok = await self.move_to_pose_oriented_fast(
                pre_align_xyz,
                yaw_deg=0.0,
                max_steps=130,
                tol=0.080,
                xy_tol=0.070,
                z_tol=0.070,
                run_tick=True,
                aggressive=True,
                tag="pick/pre_align",
            )
            pre_precise = self._reached_pose(pre_align_xyz, tol=0.045, xy_tol=0.020, z_tol=0.040)
            if not pre_ok or not pre_precise:
                _dbg("[Pick] pre-align imprecise, refining position")
                pre_ok = await self.move_to_pose_oriented_fast(
                    pre_align_xyz,
                    yaw_deg=0.0,
                    max_steps=27,
                    tol=0.045,
                    xy_tol=0.020,
                    z_tol=0.040,
                    run_tick=True,
                    tag="pick/pre_align_retry",
                )
                pre_precise = self._reached_pose(
                    pre_align_xyz, tol=0.045, xy_tol=0.020, z_tol=0.040
                )
            if not pre_ok or not pre_precise:
                _dbg("[Pick] pre-align retry failed - abort descent")
                await self.move_to_pose(
                    transit_xyz,
                    max_steps=35,
                    clamp=False,
                    tol=0.100,
                    fast=True,
                    tag="pick/abort_transit",
                )
                continue

            # Stage 2: in-place rotation to the grasp yaw before descent.
            _dbg(f"[Pick] rotate to yaw={grip_yaw:.1f}°")
            if abs(float(grip_yaw)) > 1.0:
                rotate_ok = await self.move_to_pose_oriented_fast(
                    pre_align_xyz,
                    yaw_deg=grip_yaw,
                    max_steps=2,
                    tol=0.040,
                    xy_tol=0.030,
                    z_tol=0.040,
                    run_tick=False,
                    tag="pick/rotate",
                )
                if not rotate_ok:
                    _dbg("[Pick] yaw rotation imprecise, refining")
                    rotate_ok = await self.move_to_pose_oriented_fast(
                        pre_align_xyz,
                        yaw_deg=grip_yaw,
                        max_steps=18,
                        tol=0.040,
                        xy_tol=0.030,
                        z_tol=0.040,
                        run_tick=False,
                        tag="pick/rotate_retry",
                    )
                if not rotate_ok:
                    _dbg("[Pick] yaw refinement failed - abort descent")
                    await self.move_to_pose(
                        transit_xyz,
                        max_steps=35,
                        clamp=False,
                        tol=0.100,
                        fast=True,
                        tag="pick/abort_transit",
                    )
                    continue
            else:
                _dbg("[Pick] rotate skip: yaw≈0°")

            # Open fingers now that we're above the cube with correct yaw,
            # then perform a precise vertical descent to the grasp point.
            _t_pre_align_done = time.time()
            _dbg(f"[Pick] descent entry={np.round(entry_xyz, 3)} grasp={np.round(grasp_xyz, 3)}")
            path_ok = await self.move_through_oriented_waypoints(
                [
                    {
                        "name": "entry",
                        "position": entry_xyz,
                        "max_steps": 60,
                        "tol": 0.024,
                        "xy_tol": 0.012,
                        "z_tol": 0.020,
                        "required": True,
                        "fast": True,
                    },
                    {
                        "name": "grasp",
                        "position": grasp_xyz,
                        "max_steps": 55,
                        "tol": 0.022,
                        "xy_tol": 0.010,
                        "z_tol": 0.010,
                        "required": True,
                    },
                ],
                yaw_deg=grip_yaw,
                run_tick=True,
                open_gripper_width=cube_size,
                open_gripper_steps=6,
            )
            if not path_ok:
                _dbg("[Pick] final-align before grip")
                path_ok = await self.move_to_pose_oriented_fast(
                    grasp_xyz,
                    yaw_deg=grip_yaw,
                    max_steps=9,
                    tol=0.020,
                    xy_tol=0.008,
                    z_tol=0.008,
                    run_tick=True,
                    tag="pick/final_align",
                )

            if not path_ok:
                _dbg("[Pick] failed to reach cube center - retrying approach")
                await self.move_to_pose(
                    transit_xyz,
                    max_steps=22,
                    clamp=False,
                    tol=0.100,
                    fast=True,
                    tag="pick/abort_transit",
                )
                continue

            self._zero_joint_velocities()

            if DEBUG_PICK_POSE:
                ee_pre = self.get_ee_position()
                if ee_pre is not None:
                    d_xy = float(np.linalg.norm(ee_pre[:2] - grasp_xyz[:2]))
                    d_z = ee_pre[2] - grasp_z
                    print(f"[Pick] EE before grasp: {np.round(ee_pre, 4)}")
                    print(f"[Pick] grasp target:    {np.round(grasp_xyz, 4)}")
                    print(f"[Pick] deviation XY={d_xy * 1000:.1f}mm dZ={d_z * 1000:+.1f}mm")

            _t_descent_done = time.time()
            _dbg("[Pick] closing gripper")
            await self.close_gripper_smooth(cube_width=cube_size, steps=5, hold_arm=True)
            await self._stabilize(frames=1, run_tick=False)

            min_g = float(np.clip(cube_size * 0.35, 0.008, 0.025)) if cube_size else 0.015
            if self.is_grasping(min_gap=min_g):
                _t_grip_done = time.time()
                _dbg("[Pick] grasped, transporting")

                # Strictly vertical lift to transit height - avoids brushing
                # neighbouring cubes and clears the working plane before the
                # diagonal move to the placement point.
                lift_z = max(float(above_xyz[2]), float(pre_place[2]), 0.36)
                lift_xyz = np.array([float(above_xyz[0]), float(above_xyz[1]), lift_z])
                await self.move_to_pose(
                    lift_xyz,
                    max_steps=53,
                    clamp=False,
                    tol=0.060,
                    aggressive=True,
                    tag="place/lift",
                )
                _t_lift_done = time.time()

                pre_place_cmd = pre_place
                place_cmd = place_xyz_adj

                _dbg(
                    f"[Place] pre_place={np.round(pre_place, 3)} "
                    f"release={np.round(place_xyz_adj, 3)}"
                )

                # Direct diagonal move to pre_place; RmpFlow finds the shortest
                # path through all three axes simultaneously.
                pre_ok = await self.move_to_pose(
                    pre_place_cmd,
                    max_steps=140,
                    clamp=False,
                    tol=0.045,
                    aggressive=True,
                    tag="place/pre_place",
                )
                _t_preplace_done = time.time()
                if not pre_ok:
                    ee = self.get_ee_position()
                    print(
                        f"[Place] ⚠ pre-place not reached precisely "
                        f"EE={np.round(ee, 3) if ee is not None else 'N/A'}  "
                        f"continuing..."
                    )

                _dbg("[Place] descending into container")
                # Use slow RmpFlow profile (art_rmp, dt=1/35) for precise descent
                # into the slot center. The fast profile (dt=1/18) oscillates and
                # cannot converge within 2cm - only the slow one settles cleanly.
                place_ok = await self.move_to_pose(
                    place_cmd, max_steps=83, clamp=False, tol=0.022, fast=False, tag="place/descent"
                )
                _t_place_done = time.time()

                release_ready = place_ok or self._reached_pose(
                    place_cmd, tol=0.030, xy_tol=0.020, z_tol=0.030
                )

                if not release_ready:
                    _dbg("[Place] refining release-point before drop")
                    place_command = self._apply_offset(place_cmd)
                    self.rmpflow.set_end_effector_target(
                        target_position=place_command.astype(np.float64),
                        target_orientation=self.grip_down,
                    )
                    _release_max = 120
                    _release_used = _release_max
                    for _i in range(_release_max):
                        self.rmpflow.update_world()
                        action = self.art_rmp.get_next_articulation_action()
                        self.robot.apply_action(action)
                        await self._step(run_tick=False)
                        if self._reached_pose(place_cmd, tol=0.025, xy_tol=0.018, z_tol=0.030):
                            release_ready = True
                            _release_used = _i + 1
                            break
                    _dbg(
                        f"[Steps] place/release_align: {_release_used}/{_release_max} "
                        f"reached={'YES' if release_ready else 'NO'}"
                    )

                if not release_ready:
                    ee = self.get_ee_position()
                    slot_xy_ok = (
                        ee is not None and float(np.linalg.norm(ee[:2] - place_cmd[:2])) <= 0.030
                    )
                    slot_z_ok = ee is not None and abs(float(ee[2] - place_cmd[2])) <= 0.045
                    zone_ok = bool(slot_xy_ok and slot_z_ok)
                    if zone_ok:
                        _dbg(
                            f"[Place] release-point imprecise but inside container: "
                            f"EE={np.round(ee, 3)} target={np.round(place_cmd, 3)}"
                        )
                        release_ready = True
                    else:
                        print(
                            f"[Place] release-point not reached; cube not released. "
                            f"EE={np.round(ee, 3) if ee is not None else 'N/A'} "
                            f"target={np.round(place_cmd, 3)}"
                        )
                        return False

                # Open fingers in parallel with the lift: cube is already at the
                # bottom, fingers travel strictly upward - no risk to neighbours.
                post_release = np.array(
                    [
                        float(place_cmd[0]),
                        float(place_cmd[1]),
                        max(0.34, float(place_cmd[2]) + 0.18),
                    ]
                )
                # Kick off the next-target detection async: arm is above the
                # container so it doesn't occlude the camera's view of the
                # remaining cubes - saves ~3-5s on the next cycle.
                if prefetch_scan_fn is not None:
                    try:
                        self.prefetch_task = asyncio.ensure_future(prefetch_scan_fn())
                    except Exception as e:
                        print(f"[Prefetch] spawn error: {e}")
                        self.prefetch_task = None
                await self.move_through_oriented_waypoints(
                    [
                        {
                            "name": "post_release_lift",
                            "position": post_release,
                            "max_steps": 38,
                            "tol": 0.060,
                            "required": False,
                            "fast": True,
                            "clamp": False,
                        },
                    ],
                    yaw_deg=0.0,
                    run_tick=False,
                    open_gripper_width=0.10,
                    open_gripper_steps=6,
                )
                _t_retreat_done = time.time()
                # Per-phase timings (seconds) for the dashboard's stacked bar.
                self.phase_timings = {
                    "pre_align": round(_t_pre_align_done - _t_cycle_start, 2),
                    "descent": round(_t_descent_done - _t_pre_align_done, 2),
                    "grip": round(_t_grip_done - _t_descent_done, 2),
                    "lift": round(_t_lift_done - _t_grip_done, 2),
                    "pre_place": round(_t_preplace_done - _t_lift_done, 2),
                    "place": round(_t_place_done - _t_preplace_done, 2),
                    "retreat": round(_t_retreat_done - _t_place_done, 2),
                    "total": round(_t_retreat_done - _t_cycle_start, 2),
                }
                pt = self.phase_timings
                print(
                    f"[Pick&Place] ✓ done in {pt['total']:.1f}s | "
                    f"align={pt['pre_align']:.1f} descent={pt['descent']:.1f} "
                    f"grip={pt['grip']:.2f} lift={pt['lift']:.1f} "
                    f"preplace={pt['pre_place']:.1f} place={pt['place']:.1f} "
                    f"retreat={pt['retreat']:.1f}"
                )
                return True

            _dbg(f"[Pick] attempt {attempt} failed")
            grasp_z = base_grasp_z
            above_z = max(top_z + approach_clearance, grasp_z + 0.04)
            entry_z = max(top_z + 0.055, grasp_z + 0.030)
            transit_z = max(above_z + 0.12, 0.28)
            grasp_xyz[:] = np.array([target_xyz[0], target_xyz[1], grasp_z])
            entry_xyz[:] = np.array([target_xyz[0], target_xyz[1], entry_z])
            above_xyz[:] = np.array([target_xyz[0], target_xyz[1], above_z])
            transit_xyz[:] = np.array([target_xyz[0], target_xyz[1], transit_z])
            await self.move_to_pose(transit_xyz, max_steps=35, fast=True, tag="pick/retry_transit")

        print("[Pick&Place] ✗ all attempts failed")
        return False
