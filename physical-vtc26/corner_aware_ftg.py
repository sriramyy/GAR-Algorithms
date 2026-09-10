#!/usr/bin/env python3
import math
from enum import Enum, auto
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

# ------------------------------------------------------------
# FSM
# ------------------------------------------------------------
class OvertakeState(Enum):
    FOLLOW = auto()
    PREPARE = auto()
    OVERTAKE = auto()
    RETURN = auto()

# ------------------------------------------------------------
# Target representation
# ------------------------------------------------------------
class TargetVehicle:
    def __init__(self, pose: Pose, vel: float):
        self.pose = pose
        self.vel = vel

# ------------------------------------------------------------
# Corner analysis result
# ------------------------------------------------------------
class CornerInfo:
    """
    Encapsulates the geometry of the upcoming corner.

    sharpness : float in [0, 1]
        0.0 = perfectly straight / wide sweeper
        1.0 = 90-degree (or sharper) hairpin
    turn_dir  : float in [-1, 1]
        -1 = left turn,  +1 = right turn,  0 = straight
    interior_angle_deg : float
        The raw estimated interior wall angle in degrees.
    """
    def __init__(self, sharpness: float, turn_dir: float, interior_angle_deg: float):
        self.sharpness = sharpness
        self.turn_dir  = turn_dir
        self.interior_angle_deg = interior_angle_deg

# ------------------------------------------------------------
# Corner Analyzer
# ------------------------------------------------------------
class CornerAnalyzer:
    """
    Estimates corner sharpness and direction from a processed LiDAR array.

    Strategy
    --------
    1. Convert the 1-D range array into (x, y) Cartesian points in the
       car's local frame (x = forward, y = left).
    2. Fit a line to the LEFT wall band and another to the RIGHT wall band
       using a robust least-squares line fit (numpy polyfit degree-1).
    3. Compute the angle between the two fitted wall normals.
    4. Map that angle to a sharpness score and a turn direction.

    Wall bands are defined as the outermost angular slices of the scan
    (configurable).  Only points closer than `max_wall_dist` are used so
    that far-away open space doesn't pollute the fit.
    """

    def __init__(
        self,
        wall_band_deg: float = 40.0,   # angular width of each wall band
        max_wall_dist: float = 4.0,    # ignore points farther than this (m)
        min_points: int = 6,           # minimum points needed for a valid fit
        smooth_alpha: float = 0.25,    # EMA smoothing for sharpness/dir
        sharp_threshold_deg: float = 150.0,  # angles below this → "sharp"
    ):
        self.wall_band_deg      = wall_band_deg
        self.max_wall_dist      = max_wall_dist
        self.min_points         = min_points
        self.smooth_alpha       = smooth_alpha
        self.sharp_threshold_deg = sharp_threshold_deg

        # smoothed outputs (EMA state)
        self._smooth_sharpness: float = 0.0
        self._smooth_turn_dir:  float = 0.0

    # ------------------------------------------------------------------
    def update(
        self,
        proc: np.ndarray,
        radians_per_elem: float,
        scan_angle_min: float = -np.pi / 2,
    ) -> CornerInfo:
        """
        Call once per LiDAR scan with the preprocessed (trimmed + clipped)
        range array.  Returns a CornerInfo with smoothed estimates.

        Parameters
        ----------
        proc              : preprocessed range array (length N)
        radians_per_elem  : angular resolution (rad / index)
        scan_angle_min    : angle corresponding to index 0 (rad).
                            For the trimmed 270° → 540-element slice the
                            centre is 0 (forward); this default matches the
                            trimmed array produced by preprocess_lidar().
        """
        N = proc.size
        if N < 2 * self.min_points:
            return self._emit(0.0, 0.0, 180.0)

        # ---- build angle array for the trimmed scan ----
        angles = (np.arange(N) - N / 2.0) * radians_per_elem  # rad, centre=0

        # ---- Cartesian conversion (car frame: x-fwd, y-left) ----
        xs = proc * np.cos(angles)
        ys = proc * np.sin(angles)

        # ---- wall bands (angular, from the edges inward) ----
        band_rad = np.deg2rad(self.wall_band_deg)

        # RIGHT wall: most-negative angles (rightmost indices)
        right_mask = (angles < -np.pi / 2 + band_rad) & (proc < self.max_wall_dist) & (proc > 0.05)
        # LEFT wall:  most-positive angles (leftmost indices)
        left_mask  = (angles >  np.pi / 2 - band_rad) & (proc < self.max_wall_dist) & (proc > 0.05)

        angle_deg, turn_dir = self._wall_angle(
            xs, ys, left_mask, right_mask
        )

        # ---- sharpness mapping ----
        # interior_angle = 180° → straight road → sharpness 0
        # interior_angle = 90°  → hairpin       → sharpness 1
        # clamp so angles outside [90, 180] are handled gracefully
        clamped = float(np.clip(angle_deg, 90.0, self.sharp_threshold_deg))
        sharpness_raw = 1.0 - (clamped - 90.0) / (self.sharp_threshold_deg - 90.0)

        return self._emit(sharpness_raw, turn_dir, angle_deg)

    # ------------------------------------------------------------------
    def _wall_angle(
        self,
        xs: np.ndarray, ys: np.ndarray,
        left_mask: np.ndarray, right_mask: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit lines to left and right wall points; return (interior_angle_deg, turn_dir).
        turn_dir: +1 = right turn, -1 = left turn, 0 = straight.
        """
        left_ok  = int(np.sum(left_mask))
        right_ok = int(np.sum(right_mask))

        if left_ok < self.min_points or right_ok < self.min_points:
            return 180.0, 0.0

        # Fit y = m*x + b for each wall (x=forward, y=lateral)
        # Use np.polyfit with deg=1; swap axes if wall is more vertical.
        def fit_line_direction(px, py) -> np.ndarray:
            """Return a unit direction vector [dx, dy] for the fitted line."""
            # decide orientation: fit x = f(y) if points are nearly horizontal
            x_span = float(np.ptp(px))
            y_span = float(np.ptp(py))
            if x_span < 1e-6 and y_span < 1e-6:
                return np.array([1.0, 0.0])
            if y_span > x_span:
                # fit x as function of y
                coeffs = np.polyfit(py, px, 1)  # x = m*y + b
                # direction vector in (x,y): d = (m, 1) normalised
                d = np.array([coeffs[0], 1.0])
            else:
                coeffs = np.polyfit(px, py, 1)  # y = m*x + b
                d = np.array([1.0, coeffs[0]])
            norm = np.linalg.norm(d)
            if norm < 1e-9:
                return np.array([1.0, 0.0])
            return d / norm

        lx, ly = xs[left_mask],  ys[left_mask]
        rx, ry = xs[right_mask], ys[right_mask]

        d_left  = fit_line_direction(lx, ly)
        d_right = fit_line_direction(rx, ry)

        # interior angle between the two wall directions
        dot = float(np.clip(np.dot(d_left, d_right), -1.0, 1.0))
        interior_angle_deg = float(np.degrees(np.arccos(abs(dot))))

        # turn direction: compare mean lateral position of left vs right wall
        # If left wall mean y > right wall mean y (both positive/negative), corner turns right
        left_mean_y  = float(np.mean(ly))
        right_mean_y = float(np.mean(ry))

        # Heuristic: if walls converge to the right (right_mean_y > 0 asymmetrically), right turn
        # More robust: look at which side has smaller forward extent
        left_mean_x  = float(np.mean(lx))
        right_mean_x = float(np.mean(rx))

        if abs(left_mean_x - right_mean_x) < 0.3:
            turn_dir = 0.0  # essentially straight
        elif right_mean_x < left_mean_x:
            turn_dir = 1.0   # right wall closes in first → right turn
        else:
            turn_dir = -1.0  # left wall closes in first → left turn

        return interior_angle_deg, turn_dir

    # ------------------------------------------------------------------
    def _emit(self, sharpness: float, turn_dir: float, angle_deg: float) -> CornerInfo:
        """Apply EMA smoothing and return a CornerInfo."""
        a = self.smooth_alpha
        self._smooth_sharpness = a * sharpness + (1 - a) * self._smooth_sharpness
        self._smooth_turn_dir  = a * turn_dir  + (1 - a) * self._smooth_turn_dir
        return CornerInfo(
            sharpness=float(self._smooth_sharpness),
            turn_dir=float(self._smooth_turn_dir),
            interior_angle_deg=float(angle_deg),
        )


# ------------------------------------------------------------
# Node
# ------------------------------------------------------------
class OvertakeFollowGap(Node):
    def __init__(self):
        super().__init__("overtake_follow_gap")

        # ---------- topic names ----------
        self.scan_topic = "/scan"
        self.odom_topic = "/odom"
        self.opps_topic = "/opponents"
        self.drive_topic = "/drive"

        # ---------- ROS2 I/O ----------
        qos10 = QoSProfile(depth=10)
        qos20 = QoSProfile(depth=20)
        self.scan_sub  = self.create_subscription(LaserScan, self.scan_topic,  self.lidar_callback,     qos10)
        self.odom_sub  = self.create_subscription(Odometry,  self.odom_topic,  self.odom_callback,      qos20)
        self.opps_sub  = self.create_subscription(PoseArray, self.opps_topic,  self.opponents_callback, qos10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, qos10)

        # ---------- FOLLOW-gap knobs ----------
        self.BUBBLE_RADIUS       = 100
        self.PREPROCESS_CONV_SIZE = 3
        self.BEST_POINT_CONV_SIZE = 120
        self.MAX_LIDAR_DIST      = 7.0
        self.MAX_STEER_ABS       = np.deg2rad(40.0)

        # speeds
        self.STRAIGHT_SPEED = 3.0
        self.CORNER_SPEED   = 2.0
        self.SPEED_MAX      = 4.5

        # handling
        self.CENTER_BIAS_ALPHA    = 0.35
        self.EDGE_GUARD_DEG       = 12.0
        self.SIDE_REPULSION_GAIN  = 0.28
        self.TTC_HARD_BRAKE       = 0.55
        self.TTC_SOFT_BRAKE       = 0.9
        self.FWD_WEDGE_DEG        = 8.0
        self.STEER_SMOOTH_ALPHA   = 0.5
        self.STEER_RATE_LIMIT     = np.deg2rad(8.0)

        # ---------- OVERTAKING knobs ----------
        self.FOLLOW_TIME_GAP  = 0.8
        self.MIN_SPEED_ADV    = 0.3
        self.MIN_CLEAR_DIST   = 3.0
        self.PASS_SIDE        = "left"
        self.PASS_BIAS_DEG    = 18.0
        self.RETURN_LATENCY   = 0.8
        self.PREPARE_TIMEOUT  = 2.0

        # ---------- prediction knobs ----------
        self.PREDICTION_TIME     = 1.0
        self.OPEN_GAP_THRESHOLD  = 4.5
        self.PREDICTION_ENABLED  = True

        # ---------- overtaking boost ----------
        self.OVERTAKE_SPEED_BOOST = 0.4

        # ---------- safety features ----------
        self.MIN_RETURN_GAP     = 2.5
        self.MIN_LATERAL_CLEAR  = 1.0
        self.PASS_FUTURE_HORIZON = 0.7

        self.CAR_LENGTH     = 0.4
        self.FRONT_MARGIN   = 0.23
        self.PASS_TIME_EST  = 1.0

        # target persistence
        self.target_memory    = None
        self.target_memory_ts = 0.0
        self.TARGET_MEMORY_TIME = 0.8

        self.COLLISION_LONG_THRESH = 3.5
        self.COLLISION_LAT_THRESH  = 1.0

        self.EXTRA_OVERTAKE_BUBBLE = 80
        self.OPP_REPULSION_GAIN    = 0.35

        # ---------- CORNER-AWARE WIDE-LINE knobs ----------
        # Maximum additional index offset applied when sharpness == 1.0.
        # Pushes best_idx toward the outside of the corner before the apex,
        # creating a wider entry line.
        self.CORNER_WIDE_MAX_DEG  = 14.0   # degrees of extra steer bias at max sharpness
        # Speed reduction multiplier at max sharpness
        #   effective_corner_speed = CORNER_SPEED * (1 - CORNER_SPEED_REDUCTION * sharpness)
        self.CORNER_SPEED_REDUCTION = 0.30  # up to 30 % slower on a hairpin
        # Only apply wide-line logic when forward clearance is within this range (m).
        # Prevents the logic firing on distant geometry that isn't the immediate corner.
        self.CORNER_LOOKAHEAD_DIST  = 5.0
        # Minimum sharpness to trigger any wide-line adjustment (avoids noise on straights)
        self.CORNER_SHARP_DEADBAND  = 0.15

        # ---------- corner analyser ----------
        self.corner_analyzer = CornerAnalyzer(
            wall_band_deg=40.0,
            max_wall_dist=4.0,
            min_points=6,
            smooth_alpha=0.25,
            sharp_threshold_deg=150.0,
        )
        self.corner_info: Optional[CornerInfo] = None

        # ---------- runtime state ----------
        self.state        = OvertakeState.FOLLOW
        self.state_ts     = self.now_sec()
        self.ego_pose: Optional[Pose]       = None
        self.ego_speed: float               = 0.0
        self.opponents: List[TargetVehicle] = []
        self.radians_per_elem: Optional[float] = None
        self.proc_latest:      Optional[np.ndarray] = None
        self.prev_steer: float = 0.0

        self.get_logger().info("OvertakeFollowGap (corner-aware wide-line) started.")

    # ------------------------------------------------------------------
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def yaw_of(self, pose: Pose) -> float:
        q = pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.z * q.z + q.y * q.y)
        )

    def predict_front_gap(self, target: TargetVehicle, dt: float) -> float:
        if self.ego_pose is None:
            return 0.0
        s_now = self.longitudinal_gap(target.pose, self.ego_pose)
        rel_v = self.ego_speed - target.vel
        return s_now + rel_v * dt

    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        self.ego_pose = msg.pose.pose
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.ego_speed = float(np.hypot(vx, vy))

    def opponents_callback(self, msg: PoseArray):
        self.opponents = [TargetVehicle(p, 2.5) for p in msg.poses]

    # ------------------------------------------------------------------
    def lidar_callback(self, scan: LaserScan):
        if self.ego_pose is None:
            return

        ranges_full = np.array(scan.ranges, dtype=np.float32)
        proc = self.preprocess_lidar(ranges_full)
        self.proc_latest = proc.copy()

        # ---------- corner analysis (NEW) ----------
        if self.radians_per_elem is not None and proc.size > 0:
            self.corner_info = self.corner_analyzer.update(proc, self.radians_per_elem)
            self.get_logger().debug(
                f"Corner: sharpness={self.corner_info.sharpness:.2f}  "
                f"dir={self.corner_info.turn_dir:+.2f}  "
                f"angle={self.corner_info.interior_angle_deg:.1f}°"
            )

        # ---------- target selection ----------
        target = self.select_front_target_stable()

        # ---------- prediction ----------
        predicted = None
        if target and self.PREDICTION_ENABLED:
            predicted = self.predict_future(target)

        # ---------- FSM ----------
        self.step_fsm(target, predicted)

        # ---------- gap follower core ----------
        if proc.size == 0:
            return

        closest_idx = int(np.argmin(proc))

        radius = self.BUBBLE_RADIUS
        if self.state == OvertakeState.OVERTAKE:
            radius += self.EXTRA_OVERTAKE_BUBBLE

        proc = self.mask_bubble(proc, closest_idx, radius)
        gap_start, gap_end = self.find_max_gap(proc)
        best_idx = self.find_best_point(gap_start, gap_end, proc)

        # edge guard
        best_idx = self.apply_edge_guard(best_idx, proc.size)

        # center bias
        best_idx = self.apply_center_bias(best_idx, proc.size, self.CENTER_BIAS_ALPHA)

        # overtaking pass bias
        if self.state in (OvertakeState.PREPARE, OvertakeState.OVERTAKE):
            bias = self.index_bias(proc.size, np.deg2rad(self.PASS_BIAS_DEG))
            best_idx = int(np.clip(best_idx + bias, 0, proc.size - 1))

        # ---- corner-aware wide-line bias (NEW) ----
        best_idx = self.apply_corner_wide_bias(best_idx, proc.size)

        # opponent repulsion
        best_idx = self.apply_opponent_repulsion(best_idx, proc.size, target)

        # side repulsion from walls
        repel    = self.side_repulsion_shift(proc)
        best_idx = int(np.clip(best_idx + repel, 0, proc.size - 1))

        # ---------- steering ----------
        steer_raw = self.index_to_steer(best_idx, proc.size)
        steer_cmd = self.smooth_and_limit_steer(steer_raw)

        # ---------- speed ----------
        speed_cmd = self.speed_policy(steer_cmd)

        # ---------- publish ----------
        self.publish_drive(speed_cmd, steer_cmd)

    # ------------------------------------------------------------------
    # CORNER-AWARE WIDE-LINE BIAS  (NEW)
    # ------------------------------------------------------------------
    def apply_corner_wide_bias(self, idx: int, length: int) -> int:
        """
        Push best_idx toward the *outside* of the detected corner
        proportionally to sharpness, so the car takes a wider entry line
        through tight corners.

        Outside = opposite side from turn_dir:
          turn_dir > 0  (right turn)  → outside is LEFT  → decrease idx
          turn_dir < 0  (left turn)   → outside is RIGHT → increase idx
        """
        if (self.corner_info is None
                or self.radians_per_elem is None
                or self.state == OvertakeState.OVERTAKE):   # don't fight overtake bias
            return idx

        sharpness = self.corner_info.sharpness
        turn_dir  = self.corner_info.turn_dir

        # Only activate when we are close enough to the corner
        fwd = self.forward_clearance()
        if fwd > self.CORNER_LOOKAHEAD_DIST:
            return idx

        # Dead-band: ignore noise on straights
        if sharpness < self.CORNER_SHARP_DEADBAND:
            return idx

        # Magnitude of the index shift
        max_shift_rad = np.deg2rad(self.CORNER_WIDE_MAX_DEG)
        max_shift_idx = int(round(max_shift_rad / self.radians_per_elem))

        # Scale by sharpness (0 → no shift, 1 → full shift)
        # Direction: outside of the turn
        # turn_dir > 0 means right turn → outside is left → idx decreases
        shift = int(round(-np.sign(turn_dir) * sharpness * max_shift_idx))

        new_idx = int(np.clip(idx + shift, 0, length - 1))

        self.get_logger().debug(
            f"CornerWideBias: sharpness={sharpness:.2f} "
            f"turn_dir={turn_dir:+.2f} shift={shift:+d} "
            f"idx {idx}→{new_idx}"
        )
        return new_idx

    # ------------------------------------------------------------------
    # SPEED POLICY — augmented with corner sharpness (MODIFIED)
    # ------------------------------------------------------------------
    def speed_policy(self, steer: float) -> float:
        base = self.CORNER_SPEED if abs(steer) > np.deg2rad(10) else self.STRAIGHT_SPEED
        base = min(base, self.SPEED_MAX)

        # ----- corner-sharpness speed reduction (NEW) -----
        if (self.corner_info is not None
                and self.corner_info.sharpness >= self.CORNER_SHARP_DEADBAND):
            fwd = self.forward_clearance()
            if fwd <= self.CORNER_LOOKAHEAD_DIST:
                reduction = self.CORNER_SPEED_REDUCTION * self.corner_info.sharpness
                base = base * (1.0 - reduction)

        # forward TTC braking
        fwd = self.forward_clearance()
        v   = max(self.ego_speed, 0.05)
        ttc = fwd / v
        if ttc < self.TTC_HARD_BRAKE:
            base = 0.0
        elif ttc < self.TTC_SOFT_BRAKE:
            scale = (ttc - self.TTC_HARD_BRAKE) / (self.TTC_SOFT_BRAKE - self.TTC_HARD_BRAKE)
            base  = scale * base

        if abs(steer) < np.deg2rad(6.0) and fwd > 3.0:
            base = min(self.SPEED_MAX, base + 0.5)

        if self.state == OvertakeState.OVERTAKE:
            base = min(self.SPEED_MAX, base + self.OVERTAKE_SPEED_BOOST)

        return base

    # ------------------------------------------------------------------
    # Everything below is unchanged from the original
    # ------------------------------------------------------------------

    def index_bias(self, length: int, angle_rad: float) -> int:
        if self.radians_per_elem is None:
            return 0
        return int(round(angle_rad / self.radians_per_elem))

    def select_front_target_stable(self) -> Optional[TargetVehicle]:
        now = self.now_sec()
        raw = self.select_front_target()
        if raw is None:
            if self.target_memory and now - self.target_memory_ts < self.TARGET_MEMORY_TIME:
                return self.target_memory
            return None
        self.target_memory    = raw
        self.target_memory_ts = now
        return raw

    def select_front_target(self) -> Optional[TargetVehicle]:
        if self.ego_pose is None or not self.opponents:
            return None
        best   = None
        best_s = float("inf")
        yaw    = self.yaw_of(self.ego_pose)
        for o in self.opponents:
            dx = o.pose.position.x - self.ego_pose.position.x
            dy = o.pose.position.y - self.ego_pose.position.y
            s  =  math.cos(yaw)*dx + math.sin(yaw)*dy
            l  = -math.sin(yaw)*dx + math.cos(yaw)*dy
            if s > 0 and s < best_s and abs(l) < 3.0:
                best   = o
                best_s = s
        return best

    def longitudinal_gap(self, ego: Pose, other: Pose) -> float:
        yaw = self.yaw_of(ego)
        dx  = other.position.x - ego.position.x
        dy  = other.position.y - ego.position.y
        return math.cos(yaw)*dx + math.sin(yaw)*dy

    def lateral_gap(self, ego: Pose, other: Pose) -> float:
        yaw = self.yaw_of(ego)
        dx  = other.position.x - ego.position.x
        dy  = other.position.y - ego.position.y
        return -math.sin(yaw)*dx + math.cos(yaw)*dy

    def passed_target(self, target: TargetVehicle) -> bool:
        if self.ego_pose is None:
            return False
        s = self.longitudinal_gap(target.pose, self.ego_pose)
        return s > (self.CAR_LENGTH / 2.0 + self.FRONT_MARGIN)

    def predict_future(self, target: TargetVehicle) -> TargetVehicle:
        yaw_ego = self.yaw_of(self.ego_pose)
        dx = target.vel * self.PREDICTION_TIME * math.cos(yaw_ego)
        dy = target.vel * self.PREDICTION_TIME * math.sin(yaw_ego)
        fp = Pose()
        fp.position.x = target.pose.position.x + dx
        fp.position.y = target.pose.position.y + dy
        fp.orientation = target.pose.orientation
        return TargetVehicle(fp, target.vel)

    def step_fsm(self, target, predicted):
        now     = self.now_sec()
        t_state = now - self.state_ts

        if target is None:
            if self.state != OvertakeState.FOLLOW:
                self.set_state(OvertakeState.RETURN)
            return

        dist      = self.longitudinal_gap(self.ego_pose, target.pose)
        lat       = self.lateral_gap(self.ego_pose, target.pose)
        rel_v     = self.ego_speed - target.vel
        desired_gap = self.FOLLOW_TIME_GAP * max(self.ego_speed, 0.1)

        if self.state == OvertakeState.FOLLOW:
            if dist < desired_gap and rel_v < self.MIN_SPEED_ADV:
                self.set_state(OvertakeState.PREPARE)

        elif self.state == OvertakeState.PREPARE:
            if predicted:
                future_gap = self.longitudinal_gap(self.ego_pose, predicted.pose)
                if (future_gap >= self.OPEN_GAP_THRESHOLD and
                        self.clear_on_pass_side()):
                    self.set_state(OvertakeState.OVERTAKE)
            if t_state > self.PREPARE_TIMEOUT:
                self.set_state(OvertakeState.FOLLOW)

        elif self.state == OvertakeState.OVERTAKE:
            gap_future        = self.predict_front_gap(target, self.PASS_TIME_EST)
            enough_front_gap  = gap_future >= (self.CAR_LENGTH + self.FRONT_MARGIN)
            if (self.passed_target(target) and
                    enough_front_gap and
                    self.forward_clearance() >= self.MIN_CLEAR_DIST and
                    abs(lat) >= self.MIN_LATERAL_CLEAR):
                self.set_state(OvertakeState.RETURN)

        elif self.state == OvertakeState.RETURN:
            if t_state > self.RETURN_LATENCY:
                self.set_state(OvertakeState.FOLLOW)

    def set_state(self, s: OvertakeState):
        if self.state != s:
            self.state    = s
            self.state_ts = self.now_sec()
            self.get_logger().info(f"STATE -> {self.state.name}")

    def apply_opponent_repulsion(self, idx: int, length: int, target: Optional[TargetVehicle]) -> int:
        if target is None or self.radians_per_elem is None:
            return idx
        lat = self.lateral_gap(self.ego_pose, target.pose)
        if lat > 0:
            repel = -self.OPP_REPULSION_GAIN / (abs(lat) + 0.2)
        else:
            repel = +self.OPP_REPULSION_GAIN / (abs(lat) + 0.2)
        shift = int(round(repel / (self.radians_per_elem or 1e-6)))
        return int(np.clip(idx + shift, 0, length - 1))

    def preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        n = len(ranges)
        self.radians_per_elem = (2.0 * np.pi) / n if n > 0 else None

        if n > 270:
            proc = ranges[135:-135].copy()
        else:
            proc = ranges.copy()

        if self.PREPROCESS_CONV_SIZE > 1:
            k    = np.ones(self.PREPROCESS_CONV_SIZE, dtype=np.float32) / float(self.PREPROCESS_CONV_SIZE)
            proc = np.convolve(proc, k, mode="same")

        np.clip(proc, 0.0, self.MAX_LIDAR_DIST, out=proc)
        return proc

    def mask_bubble(self, arr: np.ndarray, center: int, radius: int) -> np.ndarray:
        a  = arr.copy()
        lo = max(0, center - radius)
        hi = min(a.size, center + radius + 1)
        a[lo:hi] = 0.0
        return a

    def find_max_gap(self, arr: np.ndarray):
        if arr.size == 0:
            return 0, 0
        masked = np.ma.masked_where(arr == 0.0, arr)
        spans  = np.ma.notmasked_contiguous(masked)
        if not spans:
            return 0, arr.size
        best = max(spans, key=lambda sl: (sl.stop - sl.start))
        return best.start, best.stop

    def find_best_point(self, start: int, stop: int, arr: np.ndarray) -> int:
        if arr.size == 0:
            return 0
        if stop <= start + 1:
            return start
        seg = arr[start:stop]
        if self.BEST_POINT_CONV_SIZE > 1:
            k   = np.ones(self.BEST_POINT_CONV_SIZE, dtype=np.float32) / float(self.BEST_POINT_CONV_SIZE)
            seg = np.convolve(seg, k, mode="same")
        return int(np.argmax(seg)) + start

    def apply_edge_guard(self, idx: int, length: int) -> int:
        if self.radians_per_elem is None or length == 0:
            return 0
        guard = int(round(np.deg2rad(self.EDGE_GUARD_DEG) / self.radians_per_elem))
        lo    = guard
        hi    = length - guard - 1
        return int(np.clip(idx, lo, hi))

    def apply_center_bias(self, idx: int, length: int, alpha: float) -> int:
        center = (length - 1) / 2.0
        biased = (1 - alpha) * idx + alpha * center
        return int(np.clip(round(biased), 0, length - 1))

    def side_repulsion_shift(self, proc: np.ndarray) -> int:
        if proc.size == 0 or self.radians_per_elem is None:
            return 0
        L         = proc.size
        band      = max(6, int(0.05 * L))
        left_avg  = float(np.mean(proc[:band]))
        right_avg = float(np.mean(proc[-band:]))
        diff      = right_avg - left_avg
        per       = self.radians_per_elem
        max_shift = int(round(self.SIDE_REPULSION_GAIN / per))
        shift     = int(np.clip(
            np.sign(diff) * min(abs(diff), 1.0) * max_shift,
            -max_shift, max_shift
        ))
        return shift

    def index_to_steer(self, idx: int, length: int) -> float:
        if self.radians_per_elem is None:
            return 0.0
        angle = (idx - (length / 2.0)) * self.radians_per_elem
        steer = angle / 2.0
        return float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))

    def smooth_and_limit_steer(self, steer: float) -> float:
        s     = (1.0 - self.STEER_SMOOTH_ALPHA)*steer + self.STEER_SMOOTH_ALPHA*self.prev_steer
        delta = np.clip(s - self.prev_steer, -self.STEER_RATE_LIMIT, self.STEER_RATE_LIMIT)
        s_lim = self.prev_steer + float(delta)
        self.prev_steer = s_lim
        return s_lim

    def forward_clearance(self) -> float:
        if self.proc_latest is None or self.radians_per_elem is None:
            return self.MAX_LIDAR_DIST
        a      = self.proc_latest
        L      = a.size
        center = L // 2
        half   = int(round(np.deg2rad(self.FWD_WEDGE_DEG) / self.radians_per_elem))
        lo     = max(0, center - half)
        hi     = min(L, center + half + 1)
        return float(np.min(a[lo:hi]))

    def clear_on_pass_side(self) -> bool:
        if self.proc_latest is None or self.radians_per_elem is None:
            return False
        a    = self.proc_latest
        L    = a.size
        center   = L // 2
        per      = self.radians_per_elem
        wedge_center = np.deg2rad(15.0)
        wedge_half   = np.deg2rad(8.0)
        off_idx  = int(round(wedge_center / per))
        half_idx = int(round(wedge_half   / per))
        c  = center - off_idx if self.PASS_SIDE == "left" else center + off_idx
        lo = max(0, c - half_idx)
        hi = min(L, c + half_idx + 1)
        if hi <= lo:
            return False
        return float(np.min(a[lo:hi])) >= self.MIN_CLEAR_DIST

    def publish_drive(self, speed: float, steer: float):
        msg = AckermannDriveStamped()
        msg.header.stamp              = self.get_clock().now().to_msg()
        msg.drive.steering_angle      = float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))
        msg.drive.speed               = float(max(0.0, speed))
        self.drive_pub.publish(msg)


# ------------------------------------------------------------
def main():
    rclpy.init()
    node = OvertakeFollowGap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()