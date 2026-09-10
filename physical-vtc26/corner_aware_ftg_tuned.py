#!/usr/bin/env python3
"""
Revised gap-follower for F1TENTH.

Main changes vs. the original:
  1.  Angular indexing now comes from scan.angle_increment (was hard-coded 2*pi/N,
      which is wrong by 1.33x for a 270-deg lidar and scaled every bias term).
  2.  CornerAnalyzer sharpness bug fixed (it used to return 1.0 permanently).
  3.  Target selection is now "widest gap -> centre of the deepest plateau", with
      hysteresis, instead of argmax on a saturated array.  This is the main
      anti-swerve fix.
  4.  Disparity extender replaces the single "bubble around closest point",
      which used to flip between left and right wall frame-to-frame.
  5.  Continuous speed law (was a binary 3.0 / 2.0 switch).
  6.  Sign fixes in clear_on_pass_side() and the corner wide-line bias.
  7.  Optional odometry-gated shortcut through the narrow corridor.
"""

import math
from enum import Enum, auto
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
from ackermann_msgs.msg import AckermannDriveStamped


class OvertakeState(Enum):
    FOLLOW = auto()
    PREPARE = auto()
    OVERTAKE = auto()
    RETURN = auto()


class TargetVehicle:
    def __init__(self, pose: Pose, vel: float):
        self.pose = pose
        self.vel = vel


# ------------------------------------------------------------------
# Corner analysis
# ------------------------------------------------------------------
class CornerInfo:
    def __init__(self, sharpness: float, turn_dir: float, interior_angle_deg: float):
        self.sharpness = sharpness
        self.turn_dir = turn_dir
        self.interior_angle_deg = interior_angle_deg


class CornerAnalyzer:
    """
    Estimates corner sharpness/direction by fitting a line to each wall band.

    FIXED: the original computed  arccos(|dot|)  which lies in [0, 90] deg and
    then clipped it into [90, 150], so the result was *always* exactly 90 deg
    and sharpness was *always* 1.0.  We now map to the interior angle in
    [90, 180] deg.

    turn_dir is continuous in [-1, 1] instead of a hard +-1 switch, so the
    wide-line bias no longer flips sign frame-to-frame.
    """

    def __init__(self, wall_band_deg=35.0, max_wall_dist=4.0, min_points=8,
                 smooth_alpha=0.12, straight_angle_deg=165.0):
        self.wall_band_deg = wall_band_deg
        self.max_wall_dist = max_wall_dist
        self.min_points = min_points
        self.smooth_alpha = smooth_alpha
        self.straight_angle_deg = straight_angle_deg
        self._s = 0.0
        self._d = 0.0

    def update(self, proc: np.ndarray, angles: np.ndarray) -> CornerInfo:
        if proc.size < 2 * self.min_points:
            return self._emit(0.0, 0.0, 180.0)

        xs = proc * np.cos(angles)
        ys = proc * np.sin(angles)

        band = np.deg2rad(self.wall_band_deg)
        edge = float(np.max(np.abs(angles)))          # actual FOV half-width
        valid = (proc > 0.05) & (proc < self.max_wall_dist)
        right_mask = (angles < -edge + band) & valid   # low index  = right
        left_mask = (angles > edge - band) & valid     # high index = left

        if int(left_mask.sum()) < self.min_points or int(right_mask.sum()) < self.min_points:
            return self._emit(0.0, 0.0, 180.0)

        d_l = self._fit_dir(xs[left_mask], ys[left_mask])
        d_r = self._fit_dir(xs[right_mask], ys[right_mask])
        dot = float(np.clip(abs(np.dot(d_l, d_r)), 0.0, 1.0))
        interior = 180.0 - float(np.degrees(np.arccos(dot)))     # [90, 180]

        lo = 90.0
        hi = self.straight_angle_deg
        sharp = 1.0 - (float(np.clip(interior, lo, hi)) - lo) / (hi - lo)

        # continuous turn direction from how much sooner one wall closes in
        dx = float(np.mean(xs[right_mask]) - np.mean(xs[left_mask]))
        turn = float(np.clip(-dx / 1.0, -1.0, 1.0))   # right wall nearer -> +1 (right turn)
        return self._emit(sharp, turn, interior)

    @staticmethod
    def _fit_dir(px, py) -> np.ndarray:
        if px.size < 2:
            return np.array([1.0, 0.0])
        x_span, y_span = float(np.ptp(px)), float(np.ptp(py))
        if max(x_span, y_span) < 1e-6:
            return np.array([1.0, 0.0])
        if y_span > x_span:
            m = np.polyfit(py, px, 1)[0]
            d = np.array([m, 1.0])
        else:
            m = np.polyfit(px, py, 1)[0]
            d = np.array([1.0, m])
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([1.0, 0.0])

    def _emit(self, s, t, ang) -> CornerInfo:
        a = self.smooth_alpha
        self._s = a * s + (1 - a) * self._s
        self._d = a * t + (1 - a) * self._d
        return CornerInfo(float(self._s), float(self._d), float(ang))


# ------------------------------------------------------------------
class OvertakeFollowGap(Node):
    def __init__(self):
        super().__init__("overtake_follow_gap")

        qos10 = QoSProfile(depth=10)
        qos20 = QoSProfile(depth=20)
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, qos10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, qos20)
        self.create_subscription(PoseArray, "/opponents", self.opponents_callback, qos10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos10)

        # ---------------- vehicle ----------------
        self.WHEELBASE = 0.33
        self.CAR_HALF_WIDTH = 0.16
        self.CAR_LENGTH = 0.40
        self.MAX_STEER_ABS = np.deg2rad(35.0)

        # ---------------- gap follower ----------------
        self.FOV_USED_DEG = 190.0      # trim the scan by ANGLE, not by index
        self.MAX_LIDAR_DIST = 10.0     # raised: 7.0 created large flat plateaus
        self.PREPROCESS_CONV_SIZE = 5
        self.SAFETY_INFLATE = 0.22     # disparity extender half-width (m)
        self.GAP_MIN_DEPTH = 1.2       # ranges below this are not drivable
        self.LOOKAHEAD_MIN = 1.0
        self.LOOKAHEAD_MAX = 3.0

        # target smoothing / hysteresis  (the anti-swerve knobs)
        self.TARGET_ALPHA = 0.22       # LPF on the *target angle*, not on steer
        self.TARGET_HYST_DEG = 6.0     # ignore target jumps smaller than this
        self.STEER_RATE_LIMIT = np.deg2rad(180.0)   # rad/s (was per-callback)

        # ---------------- speed ----------------
        self.V_MIN = 1.6
        self.V_MAX = 6.0
        self.V_CLEAR_GAIN = 1.15       # v <= gain * forward_clearance
        self.A_LAT_MAX = 6.0           # m/s^2 -> v <= sqrt(a/kappa)
        self.TTC_HARD = 0.45
        self.TTC_SOFT = 0.95
        self.FWD_WEDGE_DEG = 10.0

        # ---------------- corner wide line ----------------
        self.CORNER_WIDE_MAX_DEG = 6.0     # was 14 deg and permanently on
        self.CORNER_SPEED_REDUCTION = 0.15  # was 0.30 and permanently on
        self.CORNER_LOOKAHEAD_DIST = 4.5
        self.CORNER_SHARP_DEADBAND = 0.35   # was 0.15

        # ---------------- overtaking ----------------
        self.FOLLOW_TIME_GAP = 0.8
        self.MIN_SPEED_ADV = 0.3
        self.MIN_CLEAR_DIST = 3.0
        self.PASS_SIDE = "left"
        self.PASS_BIAS_DEG = 18.0
        self.RETURN_LATENCY = 0.8
        self.PREPARE_TIMEOUT = 2.0
        self.PREDICTION_TIME = 1.0
        self.OPEN_GAP_THRESHOLD = 4.5
        self.OVERTAKE_SPEED_BOOST = 0.6
        self.MIN_LATERAL_CLEAR = 1.0
        self.FRONT_MARGIN = 0.23
        self.PASS_TIME_EST = 1.0
        self.OPP_REPULSION_GAIN = 0.35
        self.TARGET_MEMORY_TIME = 0.8

        # ---------------- SHORTCUT (map-anchored) ----------------
        # Fill these three from the map's .yaml, then set SHORTCUT_ENABLED = True.
        self.SHORTCUT_ENABLED = False
        self.MAP_RESOLUTION = 0.05
        self.MAP_ORIGIN = (0.0, 0.0)     # yaml "origin" [x, y, yaw]
        self.MAP_HEIGHT_PX = 421         # vtc_new.pgm is 541 x 421

        # corridor centreline, in PGM pixel coords (col, row), entry -> exit
        self.SHORTCUT_PX = [(240, 224), (226, 224), (209, 223), (193, 222),
                            (178, 218), (162, 214), (147, 209), (131, 207),
                            (120, 199), (112, 188)]
        self.SHORTCUT_ARM_RADIUS = 1.20   # m: arm the gate this close to the entry
        self.SHORTCUT_WP_RADIUS = 0.45    # m: waypoint acceptance
        self.SHORTCUT_MAX_SPEED = 2.6
        self.SHORTCUT_MIN_CLEAR = 0.45    # abort if lidar says the corridor is blocked
        self.shortcut_wps = [self._px_to_world(p) for p in self.SHORTCUT_PX]
        self.in_shortcut = False
        self.shortcut_i = 0

        # ---------------- state ----------------
        self.corner = CornerAnalyzer()
        self.corner_info: Optional[CornerInfo] = None
        self.state = OvertakeState.FOLLOW
        self.state_ts = self.now_sec()
        self.ego_pose: Optional[Pose] = None
        self.ego_speed = 0.0
        self.opponents: List[TargetVehicle] = []
        self.opp_prev: List[Tuple[float, float]] = []
        self.opp_prev_ts = 0.0
        self.angles: Optional[np.ndarray] = None
        self.ainc: Optional[float] = None
        self.proc_latest: Optional[np.ndarray] = None
        self.target_angle = 0.0
        self.prev_steer = 0.0
        self.prev_ts = self.now_sec()
        self.target_memory = None
        self.target_memory_ts = 0.0

        self.get_logger().info("OvertakeFollowGap v2 started.")

    # ==============================================================
    # helpers
    # ==============================================================
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _px_to_world(self, px) -> Tuple[float, float]:
        col, row = px
        x = self.MAP_ORIGIN[0] + (col + 0.5) * self.MAP_RESOLUTION
        y = self.MAP_ORIGIN[1] + (self.MAP_HEIGHT_PX - 1 - row + 0.5) * self.MAP_RESOLUTION
        return (x, y)

    @staticmethod
    def yaw_of(pose: Pose) -> float:
        q = pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.z * q.z + q.y * q.y))

    def to_body(self, wx: float, wy: float) -> Tuple[float, float]:
        yaw = self.yaw_of(self.ego_pose)
        dx = wx - self.ego_pose.position.x
        dy = wy - self.ego_pose.position.y
        return (math.cos(yaw) * dx + math.sin(yaw) * dy,
                -math.sin(yaw) * dx + math.cos(yaw) * dy)

    # ==============================================================
    # callbacks
    # ==============================================================
    def odom_callback(self, msg: Odometry):
        self.ego_pose = msg.pose.pose
        t = msg.twist.twist.linear
        self.ego_speed = float(np.hypot(t.x, t.y))

    def opponents_callback(self, msg: PoseArray):
        # estimate opponent speed from successive messages instead of hard-coding 2.5
        now = self.now_sec()
        dt = max(now - self.opp_prev_ts, 1e-3)
        cur = [(p.position.x, p.position.y) for p in msg.poses]
        out = []
        for i, p in enumerate(msg.poses):
            v = 2.5
            if i < len(self.opp_prev) and dt < 0.5:
                v = float(np.hypot(cur[i][0] - self.opp_prev[i][0],
                                   cur[i][1] - self.opp_prev[i][1]) / dt)
                v = float(np.clip(v, 0.0, 8.0))
            out.append(TargetVehicle(p, v))
        self.opponents = out
        self.opp_prev = cur
        self.opp_prev_ts = now

    # ==============================================================
    def lidar_callback(self, scan: LaserScan):
        if self.ego_pose is None:
            return
        now = self.now_sec()
        dt = float(np.clip(now - self.prev_ts, 1e-3, 0.2))
        self.prev_ts = now

        proc = self.preprocess(scan)
        if proc.size == 0:
            return
        self.proc_latest = proc.copy()
        self.corner_info = self.corner.update(proc, self.angles)

        target = self.select_front_target_stable()
        predicted = self.predict_future(target) if target else None
        self.step_fsm(target, predicted)

        # ---------- shortcut override ----------
        sc_angle = self.shortcut_angle()
        if sc_angle is not None:
            desired = sc_angle
        else:
            desired = self.gap_angle(proc)
            desired += self.pass_bias()
            desired += self.corner_wide_bias()
            desired += self.opponent_repulsion(target)
            desired += self.centering_bias(proc)
            edge = float(np.max(np.abs(self.angles))) - np.deg2rad(12.0)
            desired = float(np.clip(desired, -edge, edge))

        # ---------- smoothing on the TARGET, then geometry ----------
        if abs(desired - self.target_angle) > np.deg2rad(self.TARGET_HYST_DEG):
            self.target_angle += self.TARGET_ALPHA * (desired - self.target_angle)
        else:
            self.target_angle += 0.35 * self.TARGET_ALPHA * (desired - self.target_angle)

        steer = self.pure_pursuit_steer(self.target_angle)
        steer = self.rate_limit(steer, dt)
        speed = self.speed_policy(steer)
        self.publish_drive(speed, steer)

    # ==============================================================
    # perception
    # ==============================================================
    def preprocess(self, scan: LaserScan) -> np.ndarray:
        r = np.asarray(scan.ranges, dtype=np.float32)
        r = np.nan_to_num(r, nan=0.0, posinf=self.MAX_LIDAR_DIST,
                          neginf=0.0)
        n = r.size
        self.ainc = float(scan.angle_increment)
        ang = scan.angle_min + np.arange(n) * self.ainc

        half = np.deg2rad(self.FOV_USED_DEG) / 2.0
        keep = np.abs(ang) <= half
        r = r[keep]
        self.angles = ang[keep]

        if self.PREPROCESS_CONV_SIZE > 1:
            k = np.ones(self.PREPROCESS_CONV_SIZE, np.float32) / self.PREPROCESS_CONV_SIZE
            r = np.convolve(r, k, mode="same")
        np.clip(r, 0.0, self.MAX_LIDAR_DIST, out=r)
        return r

    def disparity_extend(self, r: np.ndarray) -> np.ndarray:
        """Shrink every range whose neighbour is much closer, by the car half-width.
        Replaces the single 'bubble around the closest beam', which used to jump
        between the left and right wall when they were nearly equidistant."""
        out = r.copy()
        d = np.diff(r)
        idx = np.flatnonzero(np.abs(d) > 0.25)
        for i in idx:
            near = min(r[i], r[i + 1])
            if near < 0.05:
                continue
            span = int(math.ceil(math.atan2(self.SAFETY_INFLATE, near) / abs(self.ainc)))
            if r[i] < r[i + 1]:
                lo, hi = i, min(out.size, i + span + 1)
            else:
                lo, hi = max(0, i + 1 - span), i + 2
            out[lo:hi] = np.minimum(out[lo:hi], near)
        return out

    def gap_angle(self, proc: np.ndarray) -> float:
        """Widest drivable gap, aim at the CENTRE of its deepest plateau."""
        safe = self.disparity_extend(proc)
        thresh = max(self.GAP_MIN_DEPTH, 0.55 * float(np.max(safe)))
        ok = safe > thresh
        if not ok.any():
            return 0.0

        # longest run of True
        idx = np.flatnonzero(np.diff(np.r_[0, ok.view(np.int8), 0]))
        starts, ends = idx[0::2], idx[1::2]
        widths = ends - starts
        depths = np.array([safe[s:e].mean() for s, e in zip(starts, ends)])
        score = widths * depths          # prefer wide AND deep, not just wide
        b = int(np.argmax(score))
        s, e = int(starts[b]), int(ends[b])

        seg = safe[s:e]
        peak = float(seg.max())
        plateau = np.flatnonzero(seg >= peak - 0.15)   # <-- centre of plateau,
        best = s + int(round(float(plateau.mean())))   #     not np.argmax
        mid = (s + e - 1) / 2.0
        best = 0.6 * best + 0.4 * mid                  # bias to gap centre
        return float(self.angles[int(np.clip(round(best), 0, safe.size - 1))])

    def forward_clearance(self) -> float:
        if self.proc_latest is None or self.angles is None:
            return self.MAX_LIDAR_DIST
        m = np.abs(self.angles) <= np.deg2rad(self.FWD_WEDGE_DEG)
        return float(np.min(self.proc_latest[m])) if m.any() else self.MAX_LIDAR_DIST

    def range_at(self, angle: float) -> float:
        if self.proc_latest is None or self.angles is None:
            return 0.0
        i = int(np.argmin(np.abs(self.angles - angle)))
        lo = max(0, i - 3)
        hi = min(self.proc_latest.size, i + 4)
        return float(np.min(self.proc_latest[lo:hi]))

    # ==============================================================
    # bias terms  (all in RADIANS now, no index arithmetic)
    # ==============================================================
    def centering_bias(self, proc: np.ndarray) -> float:
        """Gentle centering using the +-70 deg beams (the original used the
        outermost 5% of a very wide FOV, which points almost sideways)."""
        if self.angles is None:
            return 0.0
        l = np.abs(self.angles - np.deg2rad(70.0)) < np.deg2rad(12.0)
        r = np.abs(self.angles + np.deg2rad(70.0)) < np.deg2rad(12.0)
        if not (l.any() and r.any()):
            return 0.0
        diff = float(np.mean(proc[l]) - np.mean(proc[r]))   # + => more room left
        if abs(diff) < 0.25:                                 # deadband kills the
            return 0.0                                       # left/right hunting
        return float(np.clip(diff * np.deg2rad(6.0), -np.deg2rad(10.0), np.deg2rad(10.0)))

    def corner_wide_bias(self) -> float:
        c = self.corner_info
        if c is None or self.state == OvertakeState.OVERTAKE:
            return 0.0
        if c.sharpness < self.CORNER_SHARP_DEADBAND:
            return 0.0
        if self.forward_clearance() > self.CORNER_LOOKAHEAD_DIST:
            return 0.0
        # outside of the corner: right turn (turn_dir > 0) -> outside is LEFT
        # -> POSITIVE angle.  The original used the opposite sign.
        return float(np.sign(c.turn_dir) * c.sharpness *
                     np.deg2rad(self.CORNER_WIDE_MAX_DEG))

    def pass_bias(self) -> float:
        if self.state not in (OvertakeState.PREPARE, OvertakeState.OVERTAKE):
            return 0.0
        s = 1.0 if self.PASS_SIDE == "left" else -1.0
        return s * np.deg2rad(self.PASS_BIAS_DEG)

    def opponent_repulsion(self, target: Optional[TargetVehicle]) -> float:
        if target is None:
            return 0.0
        lat = self.lateral_gap(self.ego_pose, target.pose)
        return -math.copysign(self.OPP_REPULSION_GAIN / (abs(lat) + 0.4), lat)

    # ==============================================================
    # shortcut
    # ==============================================================
    def shortcut_angle(self) -> Optional[float]:
        if not self.SHORTCUT_ENABLED or self.ego_pose is None:
            return None
        if self.state in (OvertakeState.PREPARE, OvertakeState.OVERTAKE):
            return None

        if not self.in_shortcut:
            ex, ey = self.shortcut_wps[0]
            bx, by = self.to_body(ex, ey)
            if bx > -0.2 and math.hypot(bx, by) < self.SHORTCUT_ARM_RADIUS:
                # only arm if the corridor really is open from here
                if self.range_at(math.atan2(*self.to_body(*self.shortcut_wps[3])[::-1])) > 1.5:
                    self.in_shortcut = True
                    self.shortcut_i = 1
                    self.get_logger().info("SHORTCUT armed")
            if not self.in_shortcut:
                return None

        # advance waypoints
        while self.shortcut_i < len(self.shortcut_wps):
            bx, by = self.to_body(*self.shortcut_wps[self.shortcut_i])
            if math.hypot(bx, by) < self.SHORTCUT_WP_RADIUS or bx < 0.0:
                self.shortcut_i += 1
            else:
                break
        if self.shortcut_i >= len(self.shortcut_wps):
            self.in_shortcut = False
            self.get_logger().info("SHORTCUT done")
            return None

        bx, by = self.to_body(*self.shortcut_wps[self.shortcut_i])
        ang = math.atan2(by, bx)
        if self.range_at(ang) < self.SHORTCUT_MIN_CLEAR:
            self.in_shortcut = False
            self.get_logger().warn("SHORTCUT aborted (blocked)")
            return None
        return ang

    # ==============================================================
    # control
    # ==============================================================
    def pure_pursuit_steer(self, angle: float) -> float:
        """Geometric steering instead of the arbitrary 'angle / 2'."""
        ld = float(np.clip(0.55 * max(self.ego_speed, 1.0),
                           self.LOOKAHEAD_MIN, self.LOOKAHEAD_MAX))
        steer = math.atan2(2.0 * self.WHEELBASE * math.sin(angle), ld)
        return float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))

    def rate_limit(self, steer: float, dt: float) -> float:
        m = self.STEER_RATE_LIMIT * dt
        s = self.prev_steer + float(np.clip(steer - self.prev_steer, -m, m))
        self.prev_steer = s
        return s

    def speed_policy(self, steer: float) -> float:
        fwd = self.forward_clearance()

        # 1. lateral-acceleration limit from the commanded curvature
        kappa = abs(math.tan(steer)) / self.WHEELBASE
        v = self.V_MAX if kappa < 1e-3 else math.sqrt(self.A_LAT_MAX / kappa)

        # 2. do not out-drive what we can see
        v = min(v, self.V_CLEAR_GAIN * fwd)

        # 3. corner sharpness trim
        c = self.corner_info
        if c is not None and c.sharpness >= self.CORNER_SHARP_DEADBAND and fwd <= self.CORNER_LOOKAHEAD_DIST:
            v *= (1.0 - self.CORNER_SPEED_REDUCTION * c.sharpness)

        # 4. TTC brake
        ttc = fwd / max(self.ego_speed, 0.05)
        if ttc < self.TTC_HARD:
            v = 0.0
        elif ttc < self.TTC_SOFT:
            v *= (ttc - self.TTC_HARD) / (self.TTC_SOFT - self.TTC_HARD)

        if self.state == OvertakeState.OVERTAKE:
            v += self.OVERTAKE_SPEED_BOOST
        if self.in_shortcut:
            v = min(v, self.SHORTCUT_MAX_SPEED)

        return float(np.clip(v, 0.0, self.V_MAX)) if v > 0.0 else 0.0

    def publish_drive(self, speed: float, steer: float):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.drive.steering_angle = float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))
        m.drive.speed = float(max(0.0, speed))
        self.drive_pub.publish(m)

    # ==============================================================
    # opponents / FSM  (logic unchanged, sign bugs fixed)
    # ==============================================================
    def longitudinal_gap(self, ego: Pose, other: Pose) -> float:
        yaw = self.yaw_of(ego)
        dx = other.position.x - ego.position.x
        dy = other.position.y - ego.position.y
        return math.cos(yaw) * dx + math.sin(yaw) * dy

    def lateral_gap(self, ego: Pose, other: Pose) -> float:
        yaw = self.yaw_of(ego)
        dx = other.position.x - ego.position.x
        dy = other.position.y - ego.position.y
        return -math.sin(yaw) * dx + math.cos(yaw) * dy

    def select_front_target(self) -> Optional[TargetVehicle]:
        if self.ego_pose is None or not self.opponents:
            return None
        best, best_s = None, float("inf")
        for o in self.opponents:
            s = self.longitudinal_gap(self.ego_pose, o.pose)
            l = self.lateral_gap(self.ego_pose, o.pose)
            if 0 < s < best_s and abs(l) < 3.0:
                best, best_s = o, s
        return best

    def select_front_target_stable(self) -> Optional[TargetVehicle]:
        now = self.now_sec()
        raw = self.select_front_target()
        if raw is None:
            if self.target_memory and now - self.target_memory_ts < self.TARGET_MEMORY_TIME:
                return self.target_memory
            return None
        self.target_memory, self.target_memory_ts = raw, now
        return raw

    def passed_target(self, t: TargetVehicle) -> bool:
        return self.longitudinal_gap(t.pose, self.ego_pose) > (self.CAR_LENGTH / 2 + self.FRONT_MARGIN)

    def predict_front_gap(self, t: TargetVehicle, dt: float) -> float:
        return self.longitudinal_gap(t.pose, self.ego_pose) + (self.ego_speed - t.vel) * dt

    def predict_future(self, t: TargetVehicle) -> TargetVehicle:
        yaw = self.yaw_of(self.ego_pose)
        p = Pose()
        p.position.x = t.pose.position.x + t.vel * self.PREDICTION_TIME * math.cos(yaw)
        p.position.y = t.pose.position.y + t.vel * self.PREDICTION_TIME * math.sin(yaw)
        p.orientation = t.pose.orientation
        return TargetVehicle(p, t.vel)

    def clear_on_pass_side(self) -> bool:
        """FIXED: positive angle is LEFT.  The original looked to the right when
        PASS_SIDE was 'left'."""
        if self.angles is None or self.proc_latest is None:
            return False
        s = 1.0 if self.PASS_SIDE == "left" else -1.0
        c = s * np.deg2rad(15.0)
        m = np.abs(self.angles - c) < np.deg2rad(8.0)
        return bool(m.any() and float(np.min(self.proc_latest[m])) >= self.MIN_CLEAR_DIST)

    def set_state(self, s: OvertakeState):
        if self.state != s:
            self.state, self.state_ts = s, self.now_sec()
            self.get_logger().info(f"STATE -> {s.name}")

    def step_fsm(self, target, predicted):
        t_state = self.now_sec() - self.state_ts
        if target is None:
            if self.state != OvertakeState.FOLLOW:
                self.set_state(OvertakeState.RETURN)
            return

        dist = self.longitudinal_gap(self.ego_pose, target.pose)
        lat = self.lateral_gap(self.ego_pose, target.pose)
        rel_v = self.ego_speed - target.vel
        desired_gap = self.FOLLOW_TIME_GAP * max(self.ego_speed, 0.1)

        if self.state == OvertakeState.FOLLOW:
            if dist < desired_gap and rel_v < self.MIN_SPEED_ADV:
                self.set_state(OvertakeState.PREPARE)
        elif self.state == OvertakeState.PREPARE:
            if predicted:
                fg = self.longitudinal_gap(self.ego_pose, predicted.pose)
                if fg >= self.OPEN_GAP_THRESHOLD and self.clear_on_pass_side():
                    self.set_state(OvertakeState.OVERTAKE)
            if t_state > self.PREPARE_TIMEOUT:
                self.set_state(OvertakeState.FOLLOW)
        elif self.state == OvertakeState.OVERTAKE:
            if (self.passed_target(target)
                    and self.predict_front_gap(target, self.PASS_TIME_EST) >= (self.CAR_LENGTH + self.FRONT_MARGIN)
                    and self.forward_clearance() >= self.MIN_CLEAR_DIST
                    and abs(lat) >= self.MIN_LATERAL_CLEAR):
                self.set_state(OvertakeState.RETURN)
        elif self.state == OvertakeState.RETURN:
            if t_state > self.RETURN_LATENCY:
                self.set_state(OvertakeState.FOLLOW)


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