"""Robot telemetry sampling and dashboard dispatch.

Hosts two daemon-thread pushers to the Flask dashboard process:

* :class:`Telemetry` - joint angles, motor temperatures (simulated EMA), loads,
  torques. Hysteresis-gated overheat / overload warnings to avoid flicker.
  Bounded log (~10k ticks) saved to ``logs/telemetry_log.json`` on shutdown.
* :class:`CVPoster` - frame/depth images and CV state (detections, containers,
  active target) encoded as base64 JPEG; rate-limited at ~5 Hz.

Both run independently of Isaac runtime - only :mod:`numpy`, :mod:`cv2` and
:mod:`requests` are required, so this module is testable without omni.* stack.
"""

import base64
import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep
from time import time as _time

import cv2
import numpy as np
import requests

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DASHBOARD_URL = "http://localhost:5000/api/update"
CV_DASHBOARD_URL = "http://localhost:5000/api/cv/update"
LOG_PATH = _HERE / "logs" / "telemetry_log.json"

_MISSING = object()


def _strip_det(d: dict) -> dict:
    d = dict(d)
    d.pop("contour", None)
    return d


class CVPoster:
    """Pushes CV state to /api/cv/update from a daemon thread.

    The dashboard's response can carry a ``reset_requested`` flag (set by the
    CLEAR button on the Completed Cycles card). When seen, this poster flips
    its public ``reset_requested`` attribute; the orchestrator's main loop
    reads it before each cycle and triggers ``_reset_runtime_state()`` so the
    in-process processed-cube memos clear in sync with the dashboard.
    """

    def __init__(self):
        self._q = {}
        self._lock = threading.Lock()
        self._last_error_log = 0.0
        self.reset_requested = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def update(
        self,
        frame=None,
        depth=None,
        detection=_MISSING,
        detections=None,
        world_xyz=_MISSING,
        status=None,
        log_msg=None,
        cycle=None,
        containers=None,
        cubes=None,
        container_dets=None,
        send_depth: bool = True,
        cycle_log_entry=None,
        **extra,
    ):
        payload = {}
        if status is not None:
            payload["status"] = status
        if cycle is not None:
            payload["cycle"] = int(cycle)
        if containers is not None:
            payload["containers"] = containers
        if cubes is not None:
            payload["cubes"] = cubes
        if container_dets is not None:
            payload["container_dets"] = container_dets
        if cycle_log_entry is not None:
            payload["cycle_log_entry"] = cycle_log_entry
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        if detection is not _MISSING:
            payload["detection"] = None if detection is None else _strip_det(detection)
            if detection is None and world_xyz is _MISSING:
                payload["world_xyz"] = None
        if detections is not None:
            payload["detections"] = [_strip_det(d) for d in detections]
        if world_xyz is not _MISSING:
            payload["world_xyz"] = None if world_xyz is None else list(world_xyz)
        if log_msg is not None:
            payload["log_msg"] = str(log_msg)
        if frame is not None:
            payload["_frame"] = frame
        if send_depth and depth is not None:
            payload["_depth"] = depth
        with self._lock:
            if "log_msg" in payload and "log_msg" in self._q:
                payload["log_msg"] = self._q["log_msg"] + "\n" + payload["log_msg"]
            self._q.update(payload)

    def _encode_payload(self, payload: dict) -> dict:
        frame = payload.pop("_frame", None)
        depth = payload.pop("_depth", None)

        if frame is not None:
            try:
                arr = np.ascontiguousarray(frame)
                if float(arr.mean()) > 5.0:
                    _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    payload["frame_b64"] = base64.b64encode(buf.tobytes()).decode()
            except Exception as e:
                print(f"[CVPost] Frame encode: {e}")

        if depth is not None:
            try:
                d2 = np.asarray(depth, dtype=np.float32).copy()
                d2[~np.isfinite(d2)] = 0
                payload["depth_b64"] = base64.b64encode(d2.tobytes()).decode()
                payload["depth_shape"] = list(d2.shape)
            except Exception as e:
                print(f"[CVPost] Depth encode: {e}")

        return payload

    def _loop(self):
        while True:
            sleep(0.20)
            with self._lock:
                if not self._q:
                    continue
                payload, self._q = self._q, {}
            payload = self._encode_payload(payload)
            if not payload:
                continue
            try:
                # 600 ms timeout: the payload can carry an 80% JPEG plus a
                # 640x480 float32 depth blob, ~700 KB total. localhost POST is
                # usually <30 ms, but a Werkzeug worker re-render or a browser
                # GC pause can spike this to 200-400 ms - 200 ms was clipping
                # those frames silently.
                resp = requests.post(CV_DASHBOARD_URL, json=payload, timeout=0.6)
                # Dashboard sets reset_requested=True in the response right
                # after the user clicks CLEAR; surface it to the orchestrator.
                if resp.ok:
                    try:
                        if resp.json().get("reset_requested"):
                            self.reset_requested = True
                    except Exception:
                        pass
            except Exception as e:
                now = _time()
                if now - self._last_error_log > 5.0:
                    print(f"[CVPost] Dashboard update failed: {e}")
                    self._last_error_log = now


class Telemetry:
    """Periodic robot telemetry sampler with hysteresis-gated alerts.

    Reads joint positions / velocities each tick, simulates motor temperature
    and load (EMA-smoothed), tracks overheat / overload conditions through a
    debounced state machine. Pushes the latest sample to the Flask dashboard
    in a daemon thread; persists a bounded session log to disk on shutdown.
    """

    def __init__(self, robot):
        self.robot = robot
        # Bound the telemetry log so long sessions don't grow unboundedly.
        # ~10k ticks ≈ 5-10 MB JSON at save time; older ticks are dropped.
        self.log = deque(maxlen=10000)
        self.tick = 0
        self._last_push_error = 0.0
        self._pending = None
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._sent_tick = 0
        self._sent_cond = threading.Condition()
        self._session = requests.Session()
        self._temp_ema = None
        self._overheat_hits = 0
        self._overload_hits = 0
        self._overheat_active = False
        self._overload_active = False
        self._overheat_cool = 0
        self._overload_cool = 0
        self._sender = threading.Thread(target=self._push_loop, daemon=True)
        self._sender.start()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _update_hysteresis(
        self, name: str, condition: bool, hits_n: int = 3, cool_n: int = 5
    ) -> bool:
        active = getattr(self, f"_{name}_active")
        hits = getattr(self, f"_{name}_hits")
        cool = getattr(self, f"_{name}_cool")
        if active:
            cool = 0 if condition else cool + 1
            if cool >= cool_n:
                active = False
                hits = 0
                cool = 0
        else:
            hits = hits + 1 if condition else 0
            if hits >= hits_n:
                active = True
                cool = 0
        setattr(self, f"_{name}_active", active)
        setattr(self, f"_{name}_hits", hits)
        setattr(self, f"_{name}_cool", cool)
        return active

    def collect(self, detection: "dict | None" = None) -> dict:
        self.tick += 1
        joints = self.robot.get_joint_positions()
        vels = self.robot.get_joint_velocities()

        warnings = []
        if joints is None:
            n = len(vels) if vels is not None else 9
            joints = np.zeros(n, dtype=float)
            warnings.append("NO_JOINT_DATA")
        else:
            joints = np.asarray(joints, dtype=float).reshape(-1)

        if vels is None:
            vels = np.zeros_like(joints, dtype=float)
            warnings.append("NO_VEL_DATA")
        else:
            vels = np.asarray(vels, dtype=float).reshape(-1)

        if joints.size == 0:
            joints = np.zeros(9, dtype=float)
            warnings.append("EMPTY_JOINT_DATA")
        if vels.size == 0:
            vels = np.zeros_like(joints, dtype=float)
            warnings.append("EMPTY_VEL_DATA")
        if joints.size != vels.size:
            n = max(joints.size, vels.size)
            joints = np.pad(joints, (0, n - joints.size))
            vels = np.pad(vels, (0, n - vels.size))
            warnings.append("JOINT_VEL_SIZE_MISMATCH")

        # Simulated motor metrics: real Franka in Isaac doesn't expose physical
        # temp/load/torque sensors, so we synthesize them for dashboard demo.
        # Temperature model: 40°C base + 8°C/(rad/s) heating + ±0.8°C noise,
        # smoothed with α=0.15 EMA so brief velocity spikes don't trigger alerts.
        speed = np.clip(np.abs(vels), 0.0, 2.4)
        raw_temps = 40.0 + speed * 8.0 + np.random.uniform(-0.8, 0.8, size=vels.size)
        if self._temp_ema is None or len(self._temp_ema) != len(raw_temps):
            self._temp_ema = raw_temps
        else:
            self._temp_ema = 0.85 * self._temp_ema + 0.15 * raw_temps
        temps = [round(float(t), 2) for t in self._temp_ema]
        # Load: weighted sum of joint angle (carrying torque proxy) and velocity.
        loads = [
            round(min(100, abs(j) * 16 + abs(v) * 6 + np.random.uniform(0, 3)), 2)
            for j, v in zip(joints, vels, strict=False)
        ]
        # Torque: scaled by joint angle magnitude (gravity holding cost proxy).
        torques = [round(abs(j) * 2.8 + np.random.uniform(-0.1, 0.1), 3) for j in joints]

        status = "OK"
        if warnings:
            status = "WARNING"
        # Debounced overheat alert: needs 3 consecutive hot ticks to trigger and
        # 2 cool ticks to clear. At ~1.74 Hz tick rate, that's ~1.7s on / ~1.1s
        # off - prevents flicker when values oscillate around the threshold.
        if self._update_hysteresis("overheat", any(t > 60 for t in temps), cool_n=2):
            status = "WARNING"
            warnings.append("OVERHEAT")
        if self._update_hysteresis("overload", any(load > 85 for load in loads)):
            status = "WARNING"
            warnings.append("OVERLOAD")

        det = detection if isinstance(detection, dict) else None
        data = {
            "tick": self.tick,
            "timestamp": datetime.now().isoformat(),
            "joint_angles": [round(float(j), 4) for j in joints],
            "joint_vels": [round(float(v), 4) for v in vels],
            "motor_temps": temps,
            "motor_loads": loads,
            "torques": torques,
            "status": status,
            "warnings": warnings,
            # CV detection cross-reference (optional; None when robot is idle).
            "target_detected": det is not None,
            "target_pixel": [det.get("pixel_x"), det.get("pixel_y")] if det else None,
            "target_area": det.get("area") if det else None,
        }

        self.log.append(data)
        self._queue_push(data)
        return data

    def _queue_push(self, data: dict):
        with self._pending_lock:
            self._pending = data
            self._pending_event.set()

    def _push_loop(self):
        while True:
            self._pending_event.wait()
            with self._pending_lock:
                data = self._pending
                self._pending = None
                self._pending_event.clear()
            if data is None:
                continue

            if self._send(data):
                tick = int(data.get("tick", 0))
                with self._sent_cond:
                    self._sent_tick = max(self._sent_tick, tick)
                    self._sent_cond.notify_all()
            else:
                with self._pending_lock:
                    if self._pending is None:
                        self._pending = data
                        self._pending_event.set()
                sleep(0.15)

    def _send(self, data: dict) -> bool:
        try:
            response = self._session.post(DASHBOARD_URL, json=data, timeout=0.6)
            response.raise_for_status()
            return True
        except Exception as e:
            now = monotonic()
            if now - self._last_push_error > 5.0:
                print(f"[Telemetry] Dashboard push failed: {e}")
                self._last_push_error = now
            return False

    def flush(self, timeout: float = 1.0) -> bool:
        target_tick = self.tick
        deadline = monotonic() + timeout
        with self._sent_cond:
            while self._sent_tick < target_tick:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._sent_cond.wait(remaining)
        return True

    def save(self) -> None:
        self.flush(timeout=1.0)
        with LOG_PATH.open("w", encoding="utf-8") as f:
            json.dump(list(self.log), f, indent=2)
        print(f"[Telemetry] saved {len(self.log)} records -> {LOG_PATH}")
