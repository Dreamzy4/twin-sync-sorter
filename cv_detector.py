"""Computer vision pipeline for cube detection in Isaac Sim with USD ground-truth fallback.

Provides :class:`CVDetector` with an HSV-based color segmentation pipeline plus
optional USD prim transform fallback for robust yaw estimation. Bias-calibrated
pixel->world projection compensates for systematic CV errors against USD truth.

Pipeline per detection:
    HSV mask -> contour -> centroid -> depth lookup -> world XYZ -> yaw (PnP/PCA)
    -> optional USD match validation (twin sync metric Δ CV-USD).

Used by :mod:`robot_motion` as ``cv`` global; runs in Isaac Sim runtime.
"""

import os

import cv2
import numpy as np

# Isaac Sim modules are only available inside the Isaac Sim Python environment.
# Wrapping the imports lets this module load in pure-Python contexts (linters,
# CI, tests) that don't need to instantiate the actual hardware abstractions.
try:
    from omni.isaac.sensor import Camera  # type: ignore[import-not-found]
except ImportError:
    Camera = None  # type: ignore[assignment,misc]

try:
    from pxr import Usd, UsdGeom, UsdShade  # type: ignore[import-not-found]
except ImportError:
    UsdGeom = Usd = UsdShade = None  # type: ignore[assignment,misc]

# Verbose log toggle (env var DTCV_VERBOSE=1) - same convention as
# robot_motion / joint_control modules. Each gates its detail output.
VERBOSE = os.environ.get("DTCV_VERBOSE") == "1"


def _dbg(msg: str):
    """Print only when verbose mode is enabled."""
    if VERBOSE:
        print(msg)


# Quadrilateral helpers (originally in quad.py - inlined to keep the module
# self-contained without a tiny extra file).


def _order_quad_points(pts):
    """Order 4 quadrilateral corners as TL->TR->BR->BL."""
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# HSV color ranges for cube segmentation. Red wraps around the hue boundary
# (0/180), so it gets two ranges. Tuned visually for the demo lighting.

COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 80]), np.array([12, 255, 255])),
        (np.array([158, 100, 80]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([100, 100, 60]), np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([40, 80, 60]), np.array([85, 255, 255])),
    ],
}

_MORPH_KERNEL = np.ones((5, 5), np.uint8)


class CVDetector:
    """HSV-based cube detection with USD ground-truth fallback for twin sync.

    Pipeline: HSV color mask -> contour -> centroid -> depth lookup -> world XYZ
    (bias-corrected) -> yaw via PnP/PCA. Falls back to matched USD prim
    transform when CV yaw confidence is low (|CV - USD| > tolerance).

    Caches USD color classification by prim path (color is static at runtime).
    Caches camera-to-world matrix to avoid repeated USD reads per frame.
    """

    def __init__(self, color_camera_path: str, depth_camera_path: str):
        self.color_camera = Camera(prim_path=color_camera_path, resolution=(640, 480))
        self.color_camera.initialize()
        self._enable_attr(self.color_camera, ["camera:enabled", "enabled"], "Color")
        _dbg(f"[CV] color camera: {color_camera_path}")

        self.depth_camera = Camera(prim_path=depth_camera_path, resolution=(640, 480))
        self.depth_camera.initialize()
        self._enable_attr(self.depth_camera, ["camera:depthEnabled", "depthEnabled"], "Depth")
        _dbg(f"[CV] depth camera: {depth_camera_path}")

        self._rep_depth_ann = None
        self._rep_render_prod = None
        self._setup_replicator_depth()

        self.min_area = 50

        # Colors actively searched per frame. Any subset of red/blue/green; the
        # caller can swap this list at runtime (e.g. to skip green on this run).
        self.active_colors = ["red", "blue", "green"]

        # World-space pickup zone (X/Y rectangle). Detections outside it are
        # filtered out so the robot never tries to grab cubes off the table or
        # outside its safe reach. Defaults are sized for the demo scene; call
        # set_pickup_zone() to override or set to None for no filtering.
        self.pickup_zone_x = (0.23, 0.82)  # (x_min, x_max) in meters
        self.pickup_zone_y = (-0.67, 0.62)  # (y_min, y_max) in meters

        self.fx = self.fy = self.cx = self.cy = None
        self.intrinsics = None
        self._get_intrinsics()
        self._depth_scale = None

        self._world_bias = np.zeros(3)
        self._cam_matrix_cache = None
        self._cam_rotation_cache = None
        # USD prim color is static at runtime, so cache the classification
        # result by prim path. Saves 5-15ms/cycle (avoids the material-binding
        # shader walk on every frame). Stale entries are harmless: if a prim
        # is removed from the stage, _iter_usd_cube_prims won't yield it again.
        self._usd_color_cache: dict = {}

        # USD yaw fallback configuration: when CV yaw confidence is low,
        # fall back to the matched USD prim's yaw (set by robot_motion init
        # from scene_config.USD_YAW_TOLERANCE / USD_MATCH_MAX_DISTANCE).
        self.usd_cube_prim_paths = None  # list/tuple, or dict {color: paths}
        self.usd_yaw_tolerance = 20.0
        self.usd_match_max_distance = 0.25
        self.usd_quiet = False  # suppress "No match" logs (e.g. for tick-push)

        # PnP solver configuration: assumes a 5cm cube (default) with no lens
        # distortion (zeros). Used by estimate_yaw_pnp() for orientation.
        self.cube_size_m = 0.05
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Configuration setters / public-API accessors.

    def set_pickup_zone(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        """Set the pickup zone in world coordinates (meters).

        Cubes outside this rectangle are filtered out at detection time.

        Example (the demo scene's zone):
            cv.set_pickup_zone(x_min=0.23, x_max=0.82, y_min=-0.67, y_max=0.62)
        """
        self.pickup_zone_x = (x_min, x_max)
        self.pickup_zone_y = (y_min, y_max)
        print(f"[Init] pickup zone: X=[{x_min:.2f}..{x_max:.2f}] Y=[{y_min:.2f}..{y_max:.2f}]")

    def _in_pickup_zone(self, world_xyz: np.ndarray) -> bool:
        """Return True if the world XY falls inside the configured pickup zone."""
        if self.pickup_zone_x is None or self.pickup_zone_y is None:
            return True
        x, y = float(world_xyz[0]), float(world_xyz[1])
        ok = (
            self.pickup_zone_x[0] <= x <= self.pickup_zone_x[1]
            and self.pickup_zone_y[0] <= y <= self.pickup_zone_y[1]
        )
        return ok

    # Public API: thin wrappers over private state for external consumers.
    def in_pickup_zone(self, world_xyz: np.ndarray) -> bool:
        """Public alias for _in_pickup_zone."""
        return self._in_pickup_zone(world_xyz)

    def get_world_bias(self) -> np.ndarray:
        """Current CV->world projection bias (meters, EMA-calibrated)."""
        return self._world_bias

    def set_world_bias(self, bias: np.ndarray) -> None:
        """Replace the bias vector. Caller is responsible for clip semantics."""
        self._world_bias = np.array(bias, dtype=float)

    def _setup_replicator_depth(self):
        try:
            import omni.replicator.core as rep

            rp = rep.create.render_product(self.depth_camera.prim_path, (640, 480))
            self._rep_render_prod = rp
            ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            ann.attach([rp])
            self._rep_depth_ann = ann
            _dbg("[CV] replicator depth annotator attached")
        except Exception as e:
            _dbg(f"[CV] replicator unavailable ({e}), using get_depth()")

    def _enable_attr(self, camera, attr_names, label):
        prim = camera.prim
        for name in attr_names:
            attr = prim.GetAttribute(name)
            if attr.IsValid():
                attr.Set(True)
                _dbg(f"[CV] {label}: {name}=True")
                return
        _dbg(f"[CV] {label}: enable attribute not found")

    def _get_intrinsics(self):
        try:
            self.intrinsics = self.depth_camera.get_intrinsics_matrix()
            self.fx, self.fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
            self.cx, self.cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
            print(
                f"[Init] camera intrinsics fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}"
            )
            return
        except AttributeError as e:
            _dbg(f"[Init] camera attribute lookup failed, falling back: {e}")
        try:
            hfov = self.depth_camera.prim.GetAttribute("camera:horizontalFieldOfView").Get()
            W, H = 640, 480
            self.fx = W / (2 * np.tan(np.deg2rad(hfov) / 2))
            self.fy = H / (2 * np.tan(np.deg2rad(hfov * H / W) / 2))
            self.cx, self.cy = W / 2, H / 2
        except Exception:
            self.fx, self.fy = 617.4, 617.3
            self.cx, self.cy = 316.5, 242.1
        _dbg("[CV] using fallback D455 intrinsics")
        self.intrinsics = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    # Frame acquisition: RGB from color camera, depth via replicator (preferred)
    # or sensor get_depth() fallback. Both return numpy arrays or None.

    def get_frame(self) -> "np.ndarray | None":
        try:
            if hasattr(self.color_camera, "update_frame"):
                self.color_camera.update_frame()
        except Exception as e:
            _dbg(f"[CV] color_camera.update_frame() failed: {e}")
        rgba = self.color_camera.get_rgba()
        if rgba is None:
            return None
        return cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)

    def get_depth_frame(self) -> "np.ndarray | None":
        arr = None

        if self._rep_depth_ann is not None:
            try:
                data = self._rep_depth_ann.get_data()
                raw = data.get("data") if isinstance(data, dict) else data
                if raw is not None and hasattr(raw, "__len__") and len(raw) > 0:
                    arr = np.array(raw, dtype=np.float32)
                    if arr.ndim == 1:
                        if arr.size == 480 * 640:
                            arr = arr.reshape(480, 640)
                        else:
                            _dbg(f"[CV] depth/rep: unexpected size {arr.size}")
                            arr = None
                    if arr is not None and ((arr.ndim == 1) or (arr > 0).sum() == 0):
                        _dbg("[CV] depth/rep: no valid pixels")
                        arr = None
            except Exception as e:
                _dbg(f"[CV] replicator get_data() error: {e}")
                arr = None

        if arr is None:
            try:
                if hasattr(self.depth_camera, "update_frame"):
                    self.depth_camera.update_frame()
            except Exception as e:
                _dbg(f"[CV] depth_camera.update_frame() failed: {e}")
            depth = self.depth_camera.get_depth()
            if depth is None:
                print("[CV] ✗ both depth paths returned None")
                return None
            arr = np.array(depth, dtype=np.float32)

        if self._depth_scale is None:
            valid = arr[arr > 0]
            if len(valid) > 0:
                median_val = float(np.median(valid))
                self._depth_scale = 0.001 if median_val > 100.0 else 1.0
                unit = "mm->m" if self._depth_scale == 0.001 else "m"
                _dbg(f"[CV] depth: median={median_val:.3f} units={unit}")
            else:
                # Empty depth frame on Isaac warm-up - skip the unit-detection log
                # this cycle. Next call retries; keeping _depth_scale None keeps
                # the auto-detect branch active.
                _dbg("[CV] depth: warm-up frame all zero, deferring unit detect")

        if self._depth_scale is not None and self._depth_scale != 1.0:
            arr = arr * self._depth_scale

        return arr

    # Multi-color detection: HSV mask -> contour -> centroid -> 3D world position
    # for every active color, optionally cross-validated against USD ground truth.

    def _build_color_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """Build an HSV mask for the given color name."""
        ranges = COLOR_RANGES.get(color, [])
        if not ranges:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
        mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
        for lo, hi in ranges[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        return mask

    def detect_colored_objects(
        self,
        frame: "np.ndarray | None" = None,
        depth_frame: "np.ndarray | None" = None,
        with_usd: bool = True,
        estimate_yaw: bool = True,
    ) -> list:
        """Detect every colored cube in the current frame.

        Returns a list of detections, each with a ``color`` field
        ('red'/'blue'/'green'). Sorted by contour area, largest first.
        Cubes outside the pickup zone are filtered out.
        """
        if frame is None:
            frame = self.get_frame()
        if frame is None:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if depth_frame is None:
            depth_frame = self.get_depth_frame()
        results = []

        for color in self.active_colors:
            mask = self._build_color_mask(hsv, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                px = int(M["m10"] / M["m00"])
                py = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)

                z_center = self._sample_depth(px, py, depth_frame, radii=(5,))
                if z_center and self.fx:
                    width_m = (w * z_center) / self.fx
                    height_m = (h * z_center) / self.fy
                else:
                    width_m = height_m = 0.05

                size_m = float(np.clip(min(width_m, height_m), 0.02, 0.10))

                cv_yaw = (
                    self._yaw_pca_with_pnp(contour, depth_frame, px, py, size_m)
                    if estimate_yaw
                    else 0.0
                )
                grip_yaw = self._normalize_yaw_90(cv_yaw)

                detection = {
                    "color": color,
                    "pixel_x": px,
                    "pixel_y": py,
                    "area": area,
                    "size_m": size_m,
                    "width_m": width_m,
                    "height_m": height_m,
                    "depth_m": float(z_center) if z_center else 0.0,
                    "bbox_px": (x, y, w, h),
                    "angle_deg": grip_yaw,
                    "world_yaw": cv_yaw,
                    "usd_yaw": None,
                    "usd_prim_path": None,
                    "usd_match_dist_m": None,
                    "yaw_source": "cv",
                    "yaw_diff_deg": None,
                    "contour": contour,
                }
                results.append(detection)

        # Sort by area, largest first - bigger contour usually = closer cube.
        results.sort(key=lambda d: d["area"], reverse=True)

        # Filter by pickup zone (only if we have depth to compute world XYZ).
        if depth_frame is not None:
            filtered = []
            for det in results:
                world_pt = self.pixel_to_world(
                    det["pixel_x"], det["pixel_y"], depth_frame=depth_frame
                )
                if world_pt is not None and self._in_pickup_zone(world_pt):
                    det["world_xyz"] = [float(v) for v in world_pt]
                    if with_usd:
                        self.apply_usd_yaw_fallback(det, world_pt)
                    filtered.append(det)
                elif world_pt is None:
                    # No depth - keep the detection; world-space filter will
                    # run later in robot_motion once depth becomes available.
                    if with_usd:
                        self.apply_usd_yaw_fallback(det, None)
                    filtered.append(det)
            results = filtered
        else:
            if with_usd:
                for det in results:
                    self.apply_usd_yaw_fallback(det, None)

        return results

    def _sample_depth(
        self, pixel_x, pixel_y, depth_frame, percentile: float = 50, radii=(10, 15, 25, 40)
    ):
        """Sample depth at pixel using progressive patch search.
        radii=(5,) -> single small patch (centroid median behavior).
        radii=(10,15,25,40) -> progressive search (default for moments)."""
        if depth_frame is None:
            return None
        H, W = depth_frame.shape[:2]
        for r in radii:
            y0, y1 = max(pixel_y - r, 0), min(pixel_y + r + 1, H)
            x0, x1 = max(pixel_x - r, 0), min(pixel_x + r + 1, W)
            patch = depth_frame[y0:y1, x0:x1]
            valid = patch[(patch > 0) & np.isfinite(patch)]
            if len(valid) > 0:
                return float(np.percentile(valid, percentile))
        return None

    def _normalize_yaw_90(self, world_yaw_deg: float) -> float:
        yaw = float(world_yaw_deg) % 180.0
        if yaw > 90.0:
            yaw -= 180.0
        return yaw

    def _usd_candidate_paths(self, color: str = None) -> list:
        configured = self.usd_cube_prim_paths
        if configured is None:
            return []

        paths = []
        if isinstance(configured, dict):
            values = []
            keys = []
            if color is not None:
                keys.append(color)
            keys.extend(["*", "all"])
            for key in keys:
                value = configured.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    values.append(value)
                else:
                    values.extend(value)
        else:
            values = configured

        if isinstance(values, str):
            values = [values]
        for path in values:
            if path and path not in paths:
                paths.append(path)
        return paths

    @staticmethod
    def _rgb_classify(r: float, g: float, b: float) -> "str | None":
        if r > 0.45 and r > g * 1.4 and r > b * 1.4:
            return "red"
        if b > 0.45 and b > r * 1.4 and b > g * 1.4:
            return "blue"
        if g > 0.45 and g > r * 1.4 and g > b * 1.4:
            return "green"
        return None

    def _classify_usd_color(self, prim) -> "str | None":
        path = str(prim.GetPath())
        cached = self._usd_color_cache.get(path)
        if cached is not None:
            return cached
        color = self._classify_usd_color_compute(prim)
        if color is not None:
            self._usd_color_cache[path] = color
        return color

    def _classify_usd_color_compute(self, prim) -> "str | None":
        # 1. Keyword in full prim path (e.g. "Cube_Blue_L1_2_001/Cube_021")
        full_path = str(prim.GetPath()).lower()
        for color in ("red", "blue", "green"):
            if color in full_path:
                return color

        # 2. displayColor on the prim itself and its Xform parent
        targets = [prim]
        parent = prim.GetParent()
        if parent and parent.IsValid():
            targets.append(parent)

        for target in targets:
            for attr_name in ("primvars:displayColor", "displayColor"):
                attr = target.GetAttribute(attr_name)
                if not attr.IsValid():
                    continue
                try:
                    value = attr.Get()
                    if value is None:
                        continue
                    if hasattr(value, "__len__") and len(value) > 0:
                        first = value[0]
                        rgb = first if hasattr(first, "__len__") else value
                    else:
                        rgb = value
                    c = self._rgb_classify(float(rgb[0]), float(rgb[1]), float(rgb[2]))
                    if c:
                        return c
                except Exception:
                    continue

        # 3. Bound material - check material path name and shader diffuse color
        try:
            for target in targets:
                material, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()
                if not material:
                    continue
                mat_path = str(material.GetPath()).lower()
                for color in ("red", "blue", "green"):
                    if color in mat_path:
                        return color
                # Walk shader nodes looking for diffuseColor / base_color
                for desc in material.GetPrim().GetAllDescendants():
                    shader = UsdShade.Shader(desc)
                    if not shader:
                        continue
                    for inp_name in ("diffuseColor", "base_color", "albedo"):
                        inp = shader.GetInput(inp_name)
                        if not inp or not inp.IsValid():
                            continue
                        val = inp.Get()
                        if val is None:
                            continue
                        try:
                            c = self._rgb_classify(float(val[0]), float(val[1]), float(val[2]))
                            if c:
                                return c
                        except Exception:
                            # Per-pixel classifier failure - silent on purpose
                            # to avoid log floods; the outer block below logs
                            # once per call if the whole walk fails.
                            pass
        except Exception as e:
            _dbg(f"[CV] USD-material color lookup failed: {e}")

        return None

    def _iter_usd_cube_prims(self, stage, color: str = None):
        # USD_CUBE_PRIM_PATHS acts as a fast prior: walks the explicit list
        # first, then auto-traverses the stage to pick up NEW siblings of those
        # prims (e.g. cubes added at runtime). The sibling-only filter avoids
        # mis-matching scene meshes named "Cube_*" elsewhere in the hierarchy
        # (containers, walls, etc.).
        used_paths = set()
        explicit_parents = set()
        for path in self._usd_candidate_paths(color):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                used_paths.add(str(prim.GetPath()))
                explicit_parents.add(str(prim.GetPath().GetParentPath()))
                yield prim

        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            name = prim.GetName().lower()
            if type_name != "Cube" and "cube" not in name:
                continue
            path = str(prim.GetPath())
            if path in used_paths:
                continue
            # With an explicit prior list - restrict to siblings only.
            # Without one - fall back to "any cube prim", needed for ad-hoc
            # scenes where USD_CUBE_PRIM_PATHS isn't configured yet.
            if explicit_parents:
                parent_path = str(prim.GetPath().GetParentPath())
                if parent_path not in explicit_parents:
                    continue
            used_paths.add(path)
            yield prim

    def _usd_pose_from_prim(self, prim, mpu: float) -> "dict | None":
        try:
            xf = UsdGeom.Xformable(prim)
            mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            center = np.array([mat[3][0], mat[3][1], mat[3][2]], dtype=float) * mpu
            x_world = np.array([mat[0][0], mat[0][1], mat[0][2]], dtype=float)

            n = float(np.linalg.norm(x_world))
            if n < 1e-8:
                return None
            x_world /= n

            yaw = float(np.rad2deg(np.arctan2(x_world[1], x_world[0])))
            yaw = self._normalize_yaw_90(yaw)
            top_xyz = center + np.array([0.0, 0.0, self.cube_size_m * 0.5])
            return {
                "path": str(prim.GetPath()),
                "color": self._classify_usd_color(prim),
                "center_xyz": center,
                "top_xyz": top_xyz,
                "yaw_deg": yaw,
                "src": "usd_xform",
            }
        except Exception as e:
            _dbg(f"[CV/USD] pose error for {prim.GetPath()}: {e}")
            return None

    def _find_usd_cube_pose(
        self, color: str = None, world_xyz: "np.ndarray | None" = None
    ) -> "dict | None":
        try:
            stage = self.depth_camera.prim.GetStage()
            mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0

            poses = []
            for prim in self._iter_usd_cube_prims(stage, color=color):
                pose = self._usd_pose_from_prim(prim, mpu)
                if pose is not None:
                    poses.append(pose)

            if not poses:
                return None

            color_matches = [p for p in poses if p["color"] == color]
            if color_matches:
                poses = color_matches
            elif color is not None:
                # Skip prims confirmed as a different color; unclassified prims
                # remain eligible as a last-resort fallback match.
                wrong = {"red", "blue", "green"} - {color}
                all_colors = sorted({p["color"] for p in poses})
                poses = [p for p in poses if p["color"] not in wrong]
                if not poses:
                    _dbg(f"[CV/USD] no {color} prims in stage - found only: {all_colors}")
                    return None

            if world_xyz is None:
                if not self.usd_quiet:
                    _dbg(f"[CV/USD] no world XYZ for color={color}; skip ambiguous USD match")
                return None

            world_xyz = np.array(world_xyz, dtype=float)
            for pose in poses:
                pose["match_dist_m"] = float(np.linalg.norm(pose["top_xyz"][:2] - world_xyz[:2]))

            pose = min(poses, key=lambda p: p["match_dist_m"])
            if pose["match_dist_m"] > self.usd_match_max_distance:
                if not self.usd_quiet:
                    _dbg(
                        f"[CV/USD] no match for color={color}: "
                        f"closest={pose['path']} "
                        f"usd_xy=({pose['top_xyz'][0]:+.3f},{pose['top_xyz'][1]:+.3f}) "
                        f"cv_xy=({world_xyz[0]:+.3f},{world_xyz[1]:+.3f}) "
                        f"dist={pose['match_dist_m'] * 100:.1f}cm > tol={self.usd_match_max_distance * 100:.0f}cm"
                    )
                return None
            return pose
        except Exception as e:
            _dbg(f"[CV/USD] _find_usd_cube_pose error: {e}")
            return None

    def debug_usd_cubes(self, limit: int = 30) -> None:
        try:
            stage = self.depth_camera.prim.GetStage()
            mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
            poses = []
            for prim in self._iter_usd_cube_prims(stage):
                pose = self._usd_pose_from_prim(prim, mpu)
                if pose is not None:
                    poses.append(pose)
            print(f"[Init] USD fallback cubes: {len(poses)}")
            for pose in poses[:limit]:
                _dbg(
                    f"[CV/USD]   {pose['path']} top={np.round(pose['top_xyz'], 3)} "
                    f"yaw={pose['yaw_deg']:.1f}° color={pose['color']} "
                    f"src={pose.get('src', '?')}"
                )
        except Exception as e:
            _dbg(f"[CV/USD] debug_usd_cubes error: {e}")

    def get_usd_cube_states(self, x_min: float = 0.20, x_max: float = 0.80) -> list:
        """Return USD cube prims whose world X is in [x_min, x_max].

        Each entry: ``{path, color, xyz, yaw_deg}``.
        """
        try:
            stage = self.depth_camera.prim.GetStage()
            mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
            result = []
            for prim in self._iter_usd_cube_prims(stage):
                pose = self._usd_pose_from_prim(prim, mpu)
                if pose is None:
                    continue
                xyz = pose["center_xyz"]
                x = float(xyz[0])
                if x < x_min or x > x_max:
                    continue
                result.append(
                    {
                        "path": pose["path"],
                        "color": pose["color"],
                        "xyz": [x, float(xyz[1]), float(xyz[2])],
                        "yaw_deg": float(pose["yaw_deg"]),
                    }
                )
            return result
        except Exception as e:
            _dbg(f"[CV/USD] get_usd_cube_states error: {e}")
            return []

    def get_usd_pose_for_detection(
        self, detection: dict, world_xyz: "np.ndarray | None" = None
    ) -> "dict | None":
        return self._find_usd_cube_pose(detection.get("color"), world_xyz)

    def _yaw_diff_folded(self, cv_yaw: float, usd_yaw: float) -> float:
        cv_norm = self._normalize_yaw_90(cv_yaw)
        cv_fold = self._normalize_yaw_45(cv_norm)
        usd_norm = self._normalize_yaw_90(usd_yaw)
        usd_fold = self._normalize_yaw_45(usd_norm)
        diff = abs(cv_fold - usd_fold)
        if diff > 45.0:
            diff = 90.0 - diff
        return float(diff)

    def _select_yaw_detail(self, cv_yaw: float, usd_yaw: "float | None") -> dict:
        cv_norm = self._normalize_yaw_90(cv_yaw)

        if usd_yaw is None:
            return {"yaw": cv_norm, "source": "cv", "diff": None}

        usd_norm = self._normalize_yaw_90(usd_yaw)
        diff = self._yaw_diff_folded(cv_norm, usd_norm)

        if diff <= self.usd_yaw_tolerance:
            return {"yaw": cv_norm, "source": "cv", "diff": diff}

        return {"yaw": usd_norm, "source": "usd", "diff": diff}

    def apply_usd_yaw_fallback(
        self, detection: dict, world_xyz: "np.ndarray | None" = None
    ) -> dict:
        pose = self.get_usd_pose_for_detection(detection, world_xyz)
        usd_yaw = None if pose is None else pose["yaw_deg"]
        selected = self._select_yaw_detail(detection.get("world_yaw", 0.0), usd_yaw)

        detection["angle_deg"] = selected["yaw"]
        detection["yaw_source"] = selected["source"]
        detection["yaw_diff_deg"] = selected["diff"]
        detection["usd_yaw"] = usd_yaw
        detection["usd_prim_path"] = None if pose is None else pose["path"]
        detection["usd_match_dist_m"] = None if pose is None else pose.get("match_dist_m")
        detection["usd_xyz"] = None if pose is None else [float(v) for v in pose["top_xyz"]]
        if world_xyz is not None:
            detection["world_xyz"] = [float(v) for v in world_xyz]
        return detection

    def _normalize_yaw_45(self, yaw_deg: float) -> float:
        yaw = float(yaw_deg) % 90.0
        if yaw > 45.0:
            yaw -= 90.0
        return yaw

    def _yaw_from_depth_pca(self, contour, depth_frame) -> float:
        if depth_frame is None or self.intrinsics is None:
            return 0.0
        mat_result = self._cam_to_world_matrix()
        if mat_result is None:
            return 0.0
        R, cam_pos, _ = mat_result

        H_img, W_img = depth_frame.shape[:2]

        mask = np.zeros((H_img, W_img), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return 0.0

        pts_world = []
        for px_i, py_i in zip(xs[::2], ys[::2], strict=False):
            z = float(depth_frame[py_i, px_i])
            if z <= 0 or not np.isfinite(z) or z > 10.0:
                continue
            wp = self._pixel_depth_to_world(int(px_i), int(py_i), z, R, cam_pos)
            if not np.all(np.isfinite(wp)):
                continue
            pts_world.append(wp)

        if len(pts_world) < 8:
            return 0.0

        pts_world = np.array(pts_world)

        z_vals = pts_world[:, 2]
        z_med = float(np.median(z_vals))
        z_std = float(np.std(z_vals))
        mask_z = np.abs(z_vals - z_med) < 2.0 * z_std + 0.01
        pts_world = pts_world[mask_z]

        if len(pts_world) < 6:
            return 0.0

        pts_xy = pts_world[:, :2]
        centered = pts_xy - pts_xy.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.ndim < 2:
            return 0.0

        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return 0.0

        major_axis = eigvecs[:, -1]
        yaw_deg = float(np.rad2deg(np.arctan2(major_axis[1], major_axis[0])))
        return self._normalize_yaw_90(yaw_deg)

    def _yaw_pca_with_pnp(
        self, contour, depth_frame, pixel_x: int, pixel_y: int, size_m: float
    ) -> float:
        if depth_frame is not None and self.intrinsics is not None:
            R_rot = self._cam_to_world_rotation()
            pnp_yaw = self.estimate_yaw_pnp(
                contour, depth_frame, pixel_x, pixel_y, size_m, R_rot=R_rot
            )
            if pnp_yaw is not None:
                if not self.usd_quiet:
                    _dbg(f"[Yaw] PnP={pnp_yaw:+.1f}° ✓")
                return pnp_yaw

        pca_yaw = self._yaw_from_depth_pca(contour, depth_frame)
        if not self.usd_quiet:
            _dbg(f"[Yaw] PCA={pca_yaw:+.1f}° (PnP failed - no 4 corners)")
        return pca_yaw

    def estimate_yaw_pnp(
        self,
        contour: np.ndarray,
        depth_frame: "np.ndarray | None",
        pixel_x: int,
        pixel_y: int,
        size_m: "float | None" = None,
        R_rot: "np.ndarray | None" = None,
    ) -> "float | None":
        if self.intrinsics is None:
            return None

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        if len(approx) != 4:
            return None

        pts_2d = approx.reshape(4, 2).astype(np.float32)
        pts_2d = _order_quad_points(pts_2d)

        z_center = self._sample_depth(pixel_x, pixel_y, depth_frame, percentile=10)
        if z_center is None or z_center <= 0:
            return None

        s = float(size_m) if size_m and size_m > 0.01 else self.cube_size_m
        half = s / 2.0
        obj_pts = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

        img_pts = pts_2d.astype(np.float64)
        camera_matrix = self.intrinsics.astype(np.float64)
        dist = self.dist_coeffs

        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                camera_matrix,
                dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    camera_matrix,
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except Exception as e:
                _dbg(f"[CV/PnP] solvePnP error: {e}")
                return None

        if not ok:
            return None

        R_obj, _ = cv2.Rodrigues(rvec)
        if tvec[2][0] < 0:
            rvec = -rvec
            R_obj, _ = cv2.Rodrigues(rvec)

        if R_rot is None:
            R_rot = self._cam_to_world_rotation()
        R_world = R_rot @ R_obj

        x_axis_world = R_world[:, 0]
        yaw_rad = np.arctan2(x_axis_world[1], x_axis_world[0])
        yaw_deg = self._normalize_yaw_90(float(np.rad2deg(yaw_rad)))

        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist)
        proj_pts = proj_pts.reshape(4, 2)
        reproj_err = float(np.mean(np.linalg.norm(proj_pts - img_pts, axis=1)))

        if reproj_err > 15.0:
            return None

        return yaw_deg

    def detect_stable(
        self,
        n_frames: int = 3,
        max_jitter: int = 20,
        color: "str | None" = None,
        target_pixel: "tuple[int, int] | None" = None,
        max_target_shift: float = 80.0,
    ) -> "dict | None":
        """Stable detection over n_frames consecutive jitter-free frames.

        color: if set, only look for cubes of this color; otherwise picks
               the first detection from active_colors.
        target_pixel: if set, prefer the detection closest to this pixel
                      in each frame instead of the largest by area.
        """
        results = []
        # Anchor updates after every frame so the next frame searches near
        # the same cube rather than the original target pixel. Without this,
        # two adjacent cubes can alternate between frames, making the
        # averaged pixel_x/y land between them and miss both.
        anchor_pixel = target_pixel
        for _ in range(n_frames):
            if color is not None:
                old = self.active_colors
                try:
                    self.active_colors = [color]
                    dets = self.detect_colored_objects(with_usd=False)
                finally:
                    self.active_colors = old
                if anchor_pixel is not None and dets:
                    target = np.array(anchor_pixel, dtype=float)
                    for det in dets:
                        det["_target_dist_px"] = float(
                            np.linalg.norm(
                                np.array([det["pixel_x"], det["pixel_y"]], dtype=float) - target
                            )
                        )
                    d = min(dets, key=lambda det: det["_target_dist_px"])
                    if d["_target_dist_px"] > max_target_shift:
                        _dbg(
                            f"[CV] nearest {color} cube far from target_pixel: "
                            f"{d['_target_dist_px']:.1f}px > {max_target_shift:.1f}px"
                        )
                        d = None
                else:
                    d = dets[0] if dets else None
            else:
                dets = self.detect_colored_objects(with_usd=False)
                d = dets[0] if dets else None
            if d is None:
                return None
            results.append(d)
            # Anchor next frame's search around the current detection.
            if target_pixel is not None:
                anchor_pixel = (d["pixel_x"], d["pixel_y"])

        # Stability check: all sampled frames must agree on the cube's color.
        colors_found = [r["color"] for r in results]
        if len(set(colors_found)) > 1:
            _dbg(f"[CV] unstable color: {colors_found}")
            return None

        xs = [r["pixel_x"] for r in results]
        ys = [r["pixel_y"] for r in results]
        if max(xs) - min(xs) > max_jitter or max(ys) - min(ys) > max_jitter:
            _dbg(f"[CV] unstable detection (Δx={max(xs) - min(xs)}, Δy={max(ys) - min(ys)})")
            return None

        best = results[-1].copy()
        best["pixel_x"] = int(np.mean(xs))
        best["pixel_y"] = int(np.mean(ys))
        return best

    # Coordinate transforms: pixel ↔ world XYZ via depth + camera intrinsics.

    def _cam_xform(self):
        """Return (mat, meters_per_unit) for the USD camera, or (None, 1.0) on error."""
        try:
            cam_prim = self.depth_camera.prim
            stage = cam_prim.GetStage()
            meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
            xf = UsdGeom.Xformable(cam_prim)
            mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return mat, meters_per_unit
        except Exception as e:
            _dbg(f"[CV] _cam_xform error: {e}")
            return None, 1.0

    def _cam_to_world_matrix(self):
        if self._cam_matrix_cache is not None:
            return self._cam_matrix_cache
        mat, meters_per_unit = self._cam_xform()
        if mat is None:
            return None
        M33 = np.array(
            [
                [mat[0][0], mat[0][1], mat[0][2]],
                [mat[1][0], mat[1][1], mat[1][2]],
                [mat[2][0], mat[2][1], mat[2][2]],
            ]
        )
        row_norms = np.linalg.norm(M33, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-8, row_norms, 1.0)
        R_proj = (M33 / row_norms).T
        cam_pos = np.array([mat[3][0], mat[3][1], mat[3][2]]) * meters_per_unit
        self._cam_matrix_cache = (R_proj, cam_pos, meters_per_unit)
        return self._cam_matrix_cache

    def _cam_to_world_rotation(self):
        if self._cam_rotation_cache is not None:
            return self._cam_rotation_cache
        mat, _ = self._cam_xform()
        if mat is None:
            return np.eye(3)
        cols = []
        for c in range(3):
            v = np.array([mat[0][c], mat[1][c], mat[2][c]])
            n = np.linalg.norm(v)
            cols.append(v / n if n > 1e-8 else v)
        R_usd_cam = np.column_stack(cols)
        flip = np.diag([1.0, -1.0, -1.0])
        R_cam_to_world = R_usd_cam @ flip
        self._cam_rotation_cache = R_cam_to_world
        return self._cam_rotation_cache

    def _pixel_depth_to_world(self, pixel_x, pixel_y, z, R, cam_pos) -> np.ndarray:
        x_cam = (pixel_x - self.cx) * z / self.fx
        y_cam = (pixel_y - self.cy) * z / self.fy
        point_cam = np.array([x_cam, -y_cam, -z])
        return R @ point_cam + cam_pos

    def world_to_pixel(
        self, world_xyz: np.ndarray, use_bias: bool = True
    ) -> "tuple[int, int] | None":
        """Project world XYZ to pixel coordinates. Returns (px, py) or None.

        use_bias=True for CV-detected coords (subtracts the systematic bias),
        use_bias=False for USD ground-truth coords (no correction needed).
        """
        if self.intrinsics is None:
            self._get_intrinsics()
        if self.intrinsics is None:
            return None
        mat_result = self._cam_to_world_matrix()
        if mat_result is None:
            return None
        R, cam_pos, _ = mat_result
        bias = self._world_bias if use_bias else np.zeros(3)
        p_cam = R.T @ (np.array(world_xyz) - bias - cam_pos)
        z = -p_cam[2]
        if z <= 0.01:
            return None
        px = int(p_cam[0] * self.fx / z + self.cx)
        py = int(-p_cam[1] * self.fy / z + self.cy)
        return px, py

    def pixel_to_world(
        self,
        pixel_x: int,
        pixel_y: int,
        depth_frame: "np.ndarray | None" = None,
    ) -> "np.ndarray | None":
        if depth_frame is None:
            depth_frame = self.get_depth_frame()
        if depth_frame is None:
            _dbg("[CV] pixel_to_world: depth_frame is None")
            return None

        H, W = depth_frame.shape[:2]
        if not (0 <= pixel_y < H and 0 <= pixel_x < W):
            return None

        if self.intrinsics is None:
            self._get_intrinsics()

        z = self._sample_depth(pixel_x, pixel_y, depth_frame, percentile=10)
        if z is None:
            _dbg("[CV] ✗ entire depth frame is zero")
            return None

        mat_result = self._cam_to_world_matrix()
        if mat_result is None:
            return None
        R, cam_pos, _ = mat_result

        world_pt = self._pixel_depth_to_world(pixel_x, pixel_y, z, R, cam_pos)
        world_pt = world_pt + self._world_bias

        return world_pt
