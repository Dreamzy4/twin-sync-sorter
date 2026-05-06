"""Flask dashboard server for real-time robot + CV telemetry visualization.

Runs as a separate process from Isaac Sim. Receives JSON telemetry / CV state
from the Isaac runtime via HTTP POST, decodes base64-encoded frames, renders
overlay markers (detection boxes, container diamonds, processed pickup points)
and serves MJPEG streams + telemetry JSON to the browser dashboard.

Routes:
    GET  /                  - main dashboard HTML
    POST /api/update        - robot telemetry (joints, motors, status)
    POST /api/cv/update     - CV state (detections, frames)
    GET  /api/telemetry     - latest telemetry JSON
    GET  /api/cv            - CV metadata (detection list, containers)
    GET  /api/cv/config     - scene constants (workspace, containers, USD config)
    GET  /stream/rgb        - annotated RGB MJPEG stream
    GET  /stream/depth      - depth map MJPEG stream

Run: ``python dashboard_server.py`` then open http://localhost:5000
"""

import base64
import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

# Scene constants for UI reference values. Loaded optionally - if the module
# is missing (e.g. running the server outside the project tree), the
# /api/cv/config endpoint will return HTTP 503 instead of crashing import.
try:
    import scene_config
except ImportError:
    scene_config = None

_HERE = Path(__file__).resolve().parent
LOG_PATH = _HERE / "logs" / "telemetry_log.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
# Template auto-reload is a dev convenience (lets dashboard.html edits show
# up without restarting the server). Off by default; enable with
# DTCV_TEMPLATE_RELOAD=1 when iterating on the UI.
app.config["TEMPLATES_AUTO_RELOAD"] = os.environ.get("DTCV_TEMPLATE_RELOAD") == "1"
latest_data = {}
TELEMETRY_LOCK = threading.Lock()
PINK = (180, 105, 255)

# Werkzeug's default access log floods stdout with one line per CV update
# (~5 Hz). Suppress info/debug; warnings and errors still get through.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Frame buffers - two independent MJPEG streams (RGB overlay + depth heatmap).


class FrameBuffer:
    """Thread-safe single-slot buffer for the latest JPEG frame + ready event."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def push(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes
        self._event.set()

    def get(self) -> bytes | None:
        with self._lock:
            return self._frame

    def wait(self, timeout=0.5):
        self._event.wait(timeout)
        self._event.clear()


RGB_BUF = FrameBuffer()
DEPTH_BUF = FrameBuffer()

# CV state - non-image metadata (detections, containers, log buffer).


class CVState:
    """Server-side accumulator for incoming CV updates from the Isaac runtime.

    Holds the latest detection list, container counts, processed pickup points,
    a bounded log ring buffer (40 entries), and rendered MJPEG frames.
    Background render thread re-encodes RGB / depth on push.
    """

    def __init__(self):
        self.detection = None
        self.detections = []
        self.world_xyz = None
        self.log = deque(maxlen=40)
        self.cycle_num = 0
        self.status = "Waiting..."
        self.containers = {"red": 0, "blue": 0}
        self.cubes = []
        self.container_dets = []  # cubes detected inside container zones
        self.processed_xy = []  # original pickup-point XYs of already-processed cubes
        self.cycle_log = deque(maxlen=50)
        self.last_cycle_time = None
        self.cycle_times = deque(maxlen=50)
        # Phase timings per cycle: [{cycle, color, phases: {pre_align,...,total}}].
        self.phase_timings_log = deque(maxlen=50)
        # Twin-sync metric: per-cycle |CV - USD| in millimetres.
        # Two parallel series:
        #   - twin_sync_history: residual after EMA bias correction (Δ_post)
        #   - twin_sync_pre_bias_history: raw projection error (Δ_pre)
        # Reporting both stops the post-bias series from being self-referential
        # (the bias is fitted against USD, so Δ_post can converge by construction).
        self.twin_sync_history = deque(maxlen=50)
        self.twin_sync_pre_bias_history = deque(maxlen=50)
        # Set by clear_history(); the next /api/cv/update response carries this
        # flag so the orchestrator's CVPoster sees it and triggers an in-process
        # _reset_runtime_state(). Cleared once consumed.
        self.reset_requested = False
        # Simulation performance metrics: real-time factor + simulated FPS.
        self.sim_fps = None
        self.rtf = None
        # Server start timestamp - anchor for throughput (cycles/min) calc.
        self.server_start_t = time.time()
        self.lock = threading.Lock()
        self._render_lock = threading.Lock()
        self._render_event = threading.Event()
        self._pending_frame_b64 = None
        self._pending_depth_b64 = None
        self._pending_depth_shape = [480, 640]
        self._render_snapshot = None
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def update(self, d: dict):
        frame_b64 = d.get("frame_b64")
        depth_b64 = d.get("depth_b64")
        depth_shape = d.get("depth_shape", [480, 640])
        snapshot = None

        with self.lock:
            if "status" in d:
                self.status = d["status"]
            if "cycle" in d:
                self.cycle_num = d["cycle"]
            if "detection" in d:
                self.detection = d["detection"]
            if "detections" in d:
                self.detections = d["detections"] or []
            if "containers" in d and d["containers"] is not None:
                self.containers = d["containers"]
            if "cubes" in d and d["cubes"] is not None:
                self.cubes = d["cubes"]
            if "container_dets" in d and d["container_dets"] is not None:
                self.container_dets = d["container_dets"]
            if "processed_xy" in d and d["processed_xy"] is not None:
                self.processed_xy = d["processed_xy"]
            if "world_xyz" in d:
                self.world_xyz = d["world_xyz"]
            if d.get("log_msg"):
                self.log.append(f"[{time.strftime('%H:%M:%S')}] {d['log_msg']}")
            if "cycle_log_entry" in d:
                self.cycle_log.append(d["cycle_log_entry"])
            if "cycle_time" in d:
                self.last_cycle_time = d["cycle_time"]
                self.cycle_times.append(d["cycle_time"])
            # Portfolio-friendly metrics propagated from the Isaac runtime.
            pt = d.get("phase_timings")
            if pt and pt.get("phases"):
                self.phase_timings_log.append(pt)
            if "sim_fps" in d and d["sim_fps"] is not None:
                self.sim_fps = d["sim_fps"]
            if "rtf" in d and d["rtf"] is not None:
                self.rtf = d["rtf"]
            ts = d.get("twin_sync_mm")
            if ts is not None:
                try:
                    self.twin_sync_history.append(float(ts))
                except (TypeError, ValueError):
                    pass
            ts_pre = d.get("twin_sync_pre_bias_mm")
            if ts_pre is not None:
                try:
                    self.twin_sync_pre_bias_history.append(float(ts_pre))
                except (TypeError, ValueError):
                    pass

            if frame_b64 or depth_b64:
                snapshot = self._snapshot_unlocked()

        if frame_b64 or depth_b64:
            with self._render_lock:
                if frame_b64:
                    self._pending_frame_b64 = frame_b64
                if depth_b64:
                    self._pending_depth_b64 = depth_b64
                    self._pending_depth_shape = depth_shape
                self._render_snapshot = snapshot
                self._render_event.set()

    def _snapshot_unlocked(self):
        return (
            self.detection,
            self.world_xyz,
            list(self.detections),
            list(self.container_dets),
            dict(self.containers),
            list(self.processed_xy),
        )

    def _render_loop(self):
        while True:
            self._render_event.wait()
            with self._render_lock:
                frame_b64 = self._pending_frame_b64
                depth_b64 = self._pending_depth_b64
                depth_shape = self._pending_depth_shape
                snapshot = self._render_snapshot
                self._pending_frame_b64 = None
                self._pending_depth_b64 = None
                self._render_event.clear()

            if frame_b64:
                self._decode_rgb_frame(frame_b64, snapshot)
            if depth_b64:
                self._decode_depth_frame(depth_b64, depth_shape)

    def _decode_rgb_frame(self, frame_b64: str, snapshot):
        try:
            det, xyz, detections, container_dets, containers, processed_xy = snapshot
            buf = base64.b64decode(frame_b64)
            arr = np.frombuffer(buf, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None or float(frame.mean()) < 5.0:
                raise ValueError("blank frame")
            rendered = _render_rgb(
                frame, det, xyz, detections, container_dets, containers, processed_xy
            )
            _, jpeg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
            RGB_BUF.push(jpeg.tobytes())
        except Exception as e:
            print(f"[CV/Server] RGB decode error: {e}")

    def _decode_depth_frame(self, depth_b64: str, depth_shape):
        try:
            buf = base64.b64decode(depth_b64)
            arr = np.frombuffer(buf, dtype=np.float32)
            h, w = depth_shape
            h, w = int(h), int(w)
            if arr.size != h * w:
                raise ValueError(f"depth size {arr.size} != {h}x{w}")
            depth = arr.reshape(h, w)
            rendered = _render_depth(depth)
            _, jpeg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 70])
            DEPTH_BUF.push(jpeg.tobytes())
        except Exception as e:
            print(f"[CV/Server] depth decode error: {e}")

    def get_meta(self) -> dict:
        with self.lock:
            ct = list(self.cycle_times)
            # Throughput in cycles/min, averaged over recently-completed cycles.
            throughput = None
            if len(ct) >= 2:
                avg = sum(ct) / len(ct)
                throughput = round(60.0 / avg, 2) if avg > 0 else None
            sorted_total = self.containers.get("red", 0) + self.containers.get("blue", 0)
            return {
                "detection": self.detection,
                "detections": self.detections,
                "world_xyz": self.world_xyz,
                "status": self.status,
                "cycle": self.cycle_num,
                "log": list(self.log),
                "containers": dict(self.containers),
                "cubes": list(self.cubes),
                "container_dets": list(self.container_dets),
                "cycle_log": list(self.cycle_log),
                "last_cycle_time": self.last_cycle_time,
                "avg_cycle_time": round(sum(ct) / len(ct), 1) if ct else None,
                "min_cycle_time": round(min(ct), 1) if ct else None,
                "max_cycle_time": round(max(ct), 1) if ct else None,
                "throughput_cpm": throughput,
                "sorted_total": sorted_total,
                "phase_timings_log": list(self.phase_timings_log),
                "twin_sync_history": list(self.twin_sync_history),
                "twin_sync_pre_bias_history": list(self.twin_sync_pre_bias_history),
                "sim_fps": self.sim_fps,
                "rtf": self.rtf,
            }

    def clear_history(self):
        """Reset everything accumulated by the sorting cycles.

        Clears the per-cycle log, both Twin-Sync series, the per-cycle phase
        timing breakdown, the cycle-duration stats, and the per-color sorted
        counters. Live state (current detection, cube positions reported by
        the orchestrator each tick) is untouched - it refreshes on the next
        telemetry push.
        """
        with self.lock:
            self.cycle_log.clear()
            self.twin_sync_history.clear()
            self.twin_sync_pre_bias_history.clear()
            self.phase_timings_log.clear()
            self.cycle_times.clear()
            self.last_cycle_time = None
            self.containers = {"red": 0, "blue": 0}
            self.processed_xy = []
            self.cycle_num = 0
            # Cube-level views (Sorting Status table + Workspace Map markers)
            # are populated from these lists; without clearing them, CLEAR
            # leaves stale 'already processed' badges on every cube even after
            # the scene was reset. They repopulate from the next telemetry
            # push when the orchestrator runs again.
            self.cubes = []
            self.container_dets = []
            self.detection = None
            self.detections = []
            self.world_xyz = None
            # Tell the orchestrator (via the next /api/cv/update response) to
            # also reset its module-level processed-cube memos. Otherwise
            # CLEAR fixes only the dashboard side and the arm sits idle thinking
            # 'every cube already sorted'.
            self.reset_requested = True


CV_STATE = CVState()


# Server-side frame rendering: overlays detection boxes, container markers,
# and processed-pickup diamonds onto incoming RGB frames before MJPEG-encoding.


def _det_color(det):
    color = (det or {}).get("color", "unknown")
    return {
        "red": (0, 0, 255),
        "blue": (255, 80, 0),
        "green": (0, 220, 0),
    }.get(color, (0, 255, 0))


def _same_detection(a, b):
    if not a or not b:
        return False
    prim_a = a.get("usd_prim_path")
    prim_b = b.get("usd_prim_path")
    if prim_a and prim_b and prim_a == prim_b:
        return True
    return (
        abs(int(a.get("pixel_x", -9999)) - int(b.get("pixel_x", 9999))) <= 2
        and abs(int(a.get("pixel_y", -9999)) - int(b.get("pixel_y", 9999))) <= 2
        and a.get("color") == b.get("color")
    )


def _draw_center_marker(vis, px, py, bgr, radius, cross_half):
    cv2.circle(vis, (px, py), radius, bgr, 1)
    cv2.line(vis, (px - cross_half, py), (px + cross_half, py), bgr, 1)
    cv2.line(vis, (px, py - cross_half), (px, py + cross_half), bgr, 1)


def _target_box(det, shape):
    h, w = shape[:2]
    px = int(det.get("pixel_x", 0))
    py = int(det.get("pixel_y", 0))
    bx, by, bw, bh = det.get("bbox_px", (0, 0, 0, 0))
    try:
        bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
    except Exception:
        bx, by, bw, bh = px - 18, py - 18, 36, 36

    if bw <= 0 or bh <= 0:
        side = 24
    else:
        side = int(np.clip(max(bw, bh), 16, 36))
        if max(bw / max(bh, 1), bh / max(bw, 1)) > 1.45:
            side = int(np.clip(min(bw, bh), 16, 28))

    half = max(8, side // 2)
    x0 = max(0, min(w - 1, px - half))
    y0 = max(0, min(h - 1, py - half))
    x1 = max(0, min(w - 1, px + half))
    y1 = max(0, min(h - 1, py + half))
    return x0, y0, x1, y1


def _render_rgb(
    frame, det, xyz, detections=None, container_dets=None, containers=None, processed_xy=None
):
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    vis = frame.copy()

    # Stale detection filter: skip drawing markers at the original pickup
    # position of an already-processed cube. Such detections linger in the
    # list when the prefetch path is used (detect_fn doesn't re-run on every
    # cycle, so the previous detection's coords stay until the next scan).
    STALE_RADIUS_M = 0.06
    stale_xy = [(float(p[0]), float(p[1])) for p in (processed_xy or []) if len(p) >= 2]

    for other in detections or []:
        if _same_detection(other, det):
            continue
        ot_xyz = other.get("world_xyz")
        if ot_xyz and len(ot_xyz) >= 2 and stale_xy:
            ox, oy = float(ot_xyz[0]), float(ot_xyz[1])
            if any((px - ox) ** 2 + (py - oy) ** 2 < STALE_RADIUS_M**2 for px, py in stale_xy):
                continue
        px = int(other.get("pixel_x", 0))
        py = int(other.get("pixel_y", 0))
        bgr = _det_color(other)
        _draw_center_marker(vis, px, py, bgr, radius=6, cross_half=9)
        cv2.circle(vis, (px, py), 1, bgr, -1)
        cv2.putText(
            vis,
            other.get("color", "?")[:1].upper(),
            (px + 9, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.24,
            bgr,
            1,
        )

    if det:
        px, py = det.get("pixel_x", 0), det.get("pixel_y", 0)
        target_color = (0, 255, 0)
        _draw_center_marker(vis, px, py, target_color, radius=8, cross_half=13)
        cv2.circle(vis, (px, py), 2, (0, 255, 0), -1)
        x0, y0, x1, y1 = _target_box(det, vis.shape)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 1)
        yaw = det.get("angle_deg", 0)
        ax = int(px + 42 * math.cos(math.radians(yaw)))
        ay = int(py - 42 * math.sin(math.radians(yaw)))
        cv2.arrowedLine(vis, (px, py), (ax, ay), (0, 100, 255), 1, tipLength=0.3)
        px2 = px + 24 if px < 430 else px - 132
        usd = det.get("usd_yaw")
        place_color = det.get("place_color")
        place_xyz = det.get("place_xyz")
        lines = [
            "TARGET" if det.get("target_locked") else "selected",
            f"to={str(place_color).upper()}" if place_color else "to=?",
            f"grip={yaw:+.1f}",
            f"cv  ={det.get('world_yaw', 0):+.1f}",
            f"usd ={usd:+.1f}" if usd is not None else "usd =N/A",
            f"W={det.get('width_m', 0) * 100:.1f}cm",
        ]
        if place_xyz and len(place_xyz) >= 3:
            lines += [f"P=({place_xyz[0]:+.2f},{place_xyz[1]:+.2f},{place_xyz[2]:+.2f})"]
        if xyz:
            lines += [f"X={xyz[0]:+.3f}", f"Y={xyz[1]:+.3f}", f"Z={xyz[2]:+.3f}"]
        for i, line in enumerate(lines):
            ty = py - 8 + i * 13
            cv2.rectangle(vis, (px2 - 2, ty - 10), (px2 + 150, ty + 3), (0, 0, 0), -1)
            col = (0, 255, 128) if i in (0, 1) else (180, 255, 200)
            cv2.putText(vis, line, (px2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)
    else:
        cv2.putText(vis, "No detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
    # Diamond markers showing cubes already deposited into containers.
    for cd in container_dets or []:
        cx, cy = int(cd.get("pixel_x", 0)), int(cd.get("pixel_y", 0))
        sz = 7
        pts = np.array([[cx, cy - sz], [cx + sz, cy], [cx, cy + sz], [cx - sz, cy]], np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=PINK, thickness=1)
        label = cd.get("container", "?")[:1].upper() + "v"
        cv2.putText(vis, label, (cx + 9, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, PINK, 1)

    # Top-left panel: live container counts (red / blue) overlay.
    if containers is not None:
        r_count = containers.get("red", 0)
        b_count = containers.get("blue", 0)
        panel_y = vis.shape[0] - 10
        cv2.rectangle(vis, (0, panel_y - 16), (180, panel_y + 4), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"RED:{r_count}  BLUE:{b_count}",
            (6, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 255),
            1,
        )

    # Bottom-right timestamp + frame source FPS for the operator.
    cv2.putText(
        vis,
        time.strftime("%H:%M:%S"),
        (vis.shape[1] - 80, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 80),
        1,
    )
    return vis


def _render_depth(depth):
    global _DEPTH_COLORBAR
    if depth is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if _DEPTH_COLORBAR is None:
        ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
        mapped = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)  # (256, 1, 3)
        bar = np.zeros((480, 10, 3), dtype=np.uint8)
        for i in range(256):
            bar[i * 480 // 256 : (i + 1) * 480 // 256] = mapped[i, 0]
        _DEPTH_COLORBAR = bar
    v = depth.copy()
    v[~np.isfinite(v)] = 0
    v = np.clip(v, 0, 5.0)
    norm = (v / v.max() * 255).astype(np.uint8) if v.max() > 0 else np.zeros_like(v, dtype=np.uint8)
    col = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    col[:, -14:-4] = _DEPTH_COLORBAR
    cv2.putText(
        col, "5m", (col.shape[1] - 32, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )
    cv2.putText(
        col, "0m", (col.shape[1] - 32, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )
    cv2.putText(col, "DEPTH", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return col


# MJPEG generator - yields multipart frames for the browser stream.

BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
EMPTY_JPEG = None  # lazy-initialized placeholder JPEG
_DEPTH_COLORBAR = None  # (480, 10, 3) - built once on first use


def _get_empty_jpeg():
    global EMPTY_JPEG
    if EMPTY_JPEG is None:
        blank = np.full((480, 640, 3), 30, dtype=np.uint8)
        cv2.putText(
            blank,
            "Waiting for stream...",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (120, 200, 120),
            2,
        )
        _, buf = cv2.imencode(".jpg", blank)
        EMPTY_JPEG = buf.tobytes()
    return EMPTY_JPEG


def _mjpeg_gen(buf: FrameBuffer):
    while True:
        buf.wait(timeout=0.5)
        frame = buf.get() or _get_empty_jpeg()
        yield BOUNDARY + frame + b"\r\n"


def _make_stream(buf: FrameBuffer):
    return Response(
        stream_with_context(_mjpeg_gen(buf)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Routes: telemetry endpoints (robot motors, joints, status).


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/update", methods=["POST"])
def update():
    global latest_data
    data = request.get_json() or {}
    with TELEMETRY_LOCK:
        latest_data = data
    return jsonify({"ok": True})


@app.route("/api/telemetry")
def get_latest():
    with TELEMETRY_LOCK:
        data = dict(latest_data)
    return jsonify(data)


@app.route("/api/log")
def get_log():
    try:
        with LOG_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data[-100:])
    except Exception as e:
        return jsonify({"error": str(e)})


# Routes: CV state, scene config, MJPEG streams.


@app.route("/api/cv/update", methods=["POST"])
def cv_update():
    CV_STATE.update(request.get_json() or {})
    # Consume reset_requested under the same lock that clear_history sets it,
    # so a CLEAR is delivered to exactly one orchestrator update response.
    response = {"ok": True}
    with CV_STATE.lock:
        if CV_STATE.reset_requested:
            response["reset_requested"] = True
            CV_STATE.reset_requested = False
    return jsonify(response)


@app.route("/api/cv")
def cv_api():
    """CV metadata only - frames go through the MJPEG endpoints."""
    return jsonify(CV_STATE.get_meta())


@app.route("/api/cv/clear-history", methods=["POST"])
def cv_clear_history():
    CV_STATE.clear_history()
    return jsonify({"ok": True})


@app.route("/api/cv/config")
def cv_config():
    """Scene config (workspace bounds, containers, etc.) for UI render."""
    if scene_config is None:
        return jsonify({"error": "scene_config unavailable"}), 503
    return jsonify(
        {
            "pickup_zone": scene_config.PICKUP_ZONE,
            "workspace_high": scene_config.WORKSPACE_HIGH,
            "workspace_low": scene_config.WORKSPACE_LOW,
            "containers": scene_config.CONTAINERS,
            "pickup_colors": scene_config.PICKUP_COLORS,
            "usd_yaw_tolerance": scene_config.USD_YAW_TOLERANCE,
            "usd_match_max_distance": scene_config.USD_MATCH_MAX_DISTANCE,
        }
    )


@app.route("/stream/rgb")
def stream_rgb():
    """MJPEG stream of RGB frames with detection overlay."""
    return _make_stream(RGB_BUF)


@app.route("/stream/depth")
def stream_depth():
    """MJPEG stream of the depth map (heatmap-encoded)."""
    return _make_stream(DEPTH_BUF)


# Entry point - Flask dev server. For production, run behind gunicorn or uwsgi.

if __name__ == "__main__":
    print("=" * 50)
    print("  Digital Twin Dashboard + CV Stream")
    print("  http://localhost:5000")
    print("  RGB stream:   /stream/rgb")
    print("  Depth stream: /stream/depth")
    print("  CV meta:      /api/cv")
    print("  Telemetry:    /api/telemetry")
    print("=" * 50)
    # Bind to loopback explicitly. Flask 3 defaults to 127.0.0.1 already, but
    # making it explicit keeps the dashboard off conference Wi-Fi if a future
    # version flips the default or the user copy-pastes app.run() elsewhere.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
