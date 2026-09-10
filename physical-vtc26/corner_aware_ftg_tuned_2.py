#!/usr/bin/env python3
"""
F1TENTH reactive planner -- curvature-space ("arc scan") version.

Why this replaces the beam-argmax gap follower
----------------------------------------------
A gap follower picks ONE lidar beam as the aim point.  Approaching a corner
there are two near-equal candidates (down the straight, and around the corner),
so the aim point flips between them frame to frame -> right/left/right before
the car commits, and the speed law reacts to each flip.

Here we instead score a fan of ~61 constant-curvature arcs the car could
actually drive, and pick the best one.  Free arc length is a *continuous*
function of curvature, so there is no flip.  A quadratic penalty on
(kappa - kappa_previous) makes the choice sticky: once turn-in starts, it
holds.  Turn-in also happens ~4 m early, because the straight-ahead arc gets
blocked long before the corner arrives.

Corner speed comes from the length of the *chosen arc*, not from a
straight-ahead wedge.  The old `v <= gain * forward_clearance` was the main
reason the car crawled through corners: inside a corner the +-10 deg wedge
points at the outside wall 2 m away, so it capped speed at ~2.3 m/s no matter
how fast the corner actually was.

Localization: NOT required.  Driving uses /scan only.  /odom is used for ego
speed and for opponent relative geometry, both of which are frame-invariant.
The car can be started anywhere on the track, facing either way.
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


class ArcFollowGap(Node):
    def __init__(self):
        super().__init__("arc_follow_gap")

        qos10 = QoSProfile(depth=10)
        qos20 = QoSProfile(depth=20)
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, qos10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, qos20)
        self.create_subscription(PoseArray, "/opponents", self.opponents_callback, qos10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos10)

        # ================= vehicle =================
        self.WHEELBASE = 0.33
        self.CAR_HALF_WIDTH = 0.15
        self.CAR_LENGTH = 0.40
        self.MAX_STEER = np.deg2rad(35.0)

        # ================= arc scan =================
        self.N_ARCS = 61            # odd, so kappa = 0 is a candidate
        self.HORIZON = 8.0          # m of arc we look ahead. THE turn-in knob:
        #                             larger -> earlier, smoother turn-in.
        self.SWEEP_MAX = math.pi    # never evaluate more than 180 deg of arc
        self.INFLATE = 0.28         # half corridor the arc must keep clear (m)
        #                             = CAR_HALF_WIDTH + margin.  Raise if you
        #                             clip walls, lower to use more track.
        self.CLEAR_REF = 0.80       # clearance beyond which we stop caring (m)
        self.FOV_DEG = 190.0
        self.RANGE_MAX = 10.0
        self.BEAM_STRIDE = 3        # subsample lidar for the scan

        # --- cost weights (this is where the behaviour lives) ---
        self.W_PROGRESS = 1.00      # reward free arc length
        self.W_CLEAR = 0.30         # reward staying away from walls
        self.W_STRAIGHT = 0.05      # mild preference for going straight
        self.W_SMOOTH = 0.45        # <-- anti-jitter.  Raise to commit harder
        #                                to a turn, lower if turn-in feels lazy.
        self.W_PASS = 0.35          # pass-side preference while overtaking

        self.KAPPA_ALPHA = 0.45     # light LPF on the chosen curvature
        self.STEER_RATE = np.deg2rad(240.0)   # rad/s

        # ================= speed =================
        self.V_MIN = 1.5
        self.V_MAX = 6.5
        self.A_LAT = 7.0            # m/s^2 cornering limit -> v = sqrt(A/kappa)
        self.A_BRAKE = 10.0          # m/s^2 used for look-ahead braking
        self.STOP_MARGIN = 0.35     # m kept in hand at the end of the free arc
        self.V_TIGHT_CLEAR = 0.30   # below this clearance, start scaling speed
        self.OVERTAKE_BOOST = 0.6

        # ================= overtaking =================
        self.FOLLOW_TIME_GAP = 0.8
        self.MIN_SPEED_ADV = 0.3
        self.MIN_CLEAR_DIST = 3.0
        self.PASS_SIDE = "left"
        self.RETURN_LATENCY = 0.8
        self.PREPARE_TIMEOUT = 2.0
        self.PREDICTION_TIME = 1.0
        self.OPEN_GAP_THRESHOLD = 4.5
        self.MIN_LATERAL_CLEAR = 1.0
        self.FRONT_MARGIN = 0.23
        self.PASS_TIME_EST = 1.0
        self.TARGET_MEMORY_TIME = 0.8

        # ================= state =================
        km = math.tan(self.MAX_STEER) / self.WHEELBASE
        self.KAPPA_MAX = km
        k = np.linspace(-km, km, self.N_ARCS)
        k[np.abs(k) < 1e-3] = 1e-3          # avoid the 1/0 in R = 1/kappa
        self.kappas = k
        self.R = (1.0 / k).reshape(-1, 1)
        self.s_eval = np.minimum(self.HORIZON, self.SWEEP_MAX / np.abs(k))

        self.kappa_cmd = 0.0
        self.prev_steer = 0.0
        self.prev_ts = self.now_sec()
        self.ego_pose: Optional[Pose] = None
        self.ego_speed = 0.0
        self.opponents: List[TargetVehicle] = []
        self.opp_prev: List[Tuple[float, float]] = []
        self.opp_prev_ts = 0.0
        self.state = OvertakeState.FOLLOW
        self.state_ts = self.now_sec()
        self.target_memory = None
        self.target_memory_ts = 0.0
        self.angles: Optional[np.ndarray] = None
        self.ranges: Optional[np.ndarray] = None

        self.get_logger().info("ArcFollowGap started (curvature-space planner).")

    # ==============================================================
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def yaw_of(p: Pose) -> float:
        q = p.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.z * q.z + q.y * q.y))

    # ==============================================================
    def odom_callback(self, msg: Odometry):
        self.ego_pose = msg.pose.pose
        t = msg.twist.twist.linear
        self.ego_speed = float(np.hypot(t.x, t.y))

    def opponents_callback(self, msg: PoseArray):
        now = self.now_sec()
        dt = max(now - self.opp_prev_ts, 1e-3)
        cur = [(p.position.x, p.position.y) for p in msg.poses]
        out = []
        for i, p in enumerate(msg.poses):
            v = 2.5
            if i < len(self.opp_prev) and dt < 0.5:
                v = float(np.clip(np.hypot(cur[i][0] - self.opp_prev[i][0],
                                           cur[i][1] - self.opp_prev[i][1]) / dt, 0.0, 8.0))
            out.append(TargetVehicle(p, v))
        self.opponents, self.opp_prev, self.opp_prev_ts = out, cur, now

    # ==============================================================
    def lidar_callback(self, scan: LaserScan):
        now = self.now_sec()
        dt = float(np.clip(now - self.prev_ts, 1e-3, 0.2))
        self.prev_ts = now

        X, Y = self.scan_to_points(scan)
        if X.size == 0:
            return

        target = self.select_front_target_stable()
        self.step_fsm(target, self.predict_future(target) if target else None)

        kappa, s_free, clear = self.plan(X, Y)

        self.kappa_cmd += self.KAPPA_ALPHA * (kappa - self.kappa_cmd)
        steer = float(np.clip(math.atan(self.kappa_cmd * self.WHEELBASE),
                              -self.MAX_STEER, self.MAX_STEER))
        m = self.STEER_RATE * dt
        steer = self.prev_steer + float(np.clip(steer - self.prev_steer, -m, m))
        self.prev_steer = steer

        self.publish_drive(self.speed(self.kappa_cmd, s_free, clear), steer)

    # ==============================================================
    def scan_to_points(self, scan: LaserScan):
        r = np.asarray(scan.ranges, dtype=np.float32)
        r = np.nan_to_num(r, nan=0.0, posinf=self.RANGE_MAX, neginf=0.0)
        ang = scan.angle_min + np.arange(r.size) * scan.angle_increment

        keep = np.abs(ang) <= np.deg2rad(self.FOV_DEG) / 2.0
        r, ang = r[keep], ang[keep]
        self.ranges, self.angles = r, ang

        r = r[::self.BEAM_STRIDE]
        ang = ang[::self.BEAM_STRIDE]
        ok = (r > 0.08) & (r < self.RANGE_MAX)
        r, ang = r[ok], ang[ok]
        return r * np.cos(ang), r * np.sin(ang)

    # --------------------------------------------------------------
    def arc_scan(self, X: np.ndarray, Y: np.ndarray):
        """For every candidate curvature return (free arc length, min clearance).

        Path of a constant-curvature arc, body frame (x fwd, y left):
            p(s) = ( sin(k s)/k , (1 - cos(k s))/k )
        i.e. a circle of radius R = 1/k centred at (0, R).  A lidar point q
        blocks the arc if its distance to that centre is within INFLATE of |R|,
        and the arc length at which we reach it is s = R * atan2(qx/R, (R-qy)/R).
        """
        R = self.R                                     # (K,1)
        d = np.hypot(X[None, :], Y[None, :] - R)       # (K,M)
        e = np.abs(d - np.abs(R))                      # radial offset from arc
        psi = np.arctan2(X[None, :] / R, (R - Y[None, :]) / R)
        s = R * psi
        s = np.where(s < 0.0, s + 2.0 * np.pi * np.abs(R), s)   # wrap to ahead

        s_eval = self.s_eval.reshape(-1, 1)
        hit = (e < self.INFLATE) & (s <= s_eval)
        s_free = np.minimum(np.where(hit, s, np.inf).min(axis=1), self.s_eval)

        on_path = (s >= 0.0) & (s <= s_free.reshape(-1, 1))
        clear = np.where(on_path, e, np.inf).min(axis=1)
        clear = np.where(np.isfinite(clear), clear, self.CLEAR_REF)
        return s_free, clear

    def plan(self, X, Y):
        s_free, clear = self.arc_scan(X, Y)
        k = self.kappas
        kn = k / self.KAPPA_MAX

        progress = np.minimum(s_free, self.HORIZON) / self.HORIZON
        clear_n = np.clip(clear, 0.0, self.CLEAR_REF) / self.CLEAR_REF

        J = (-self.W_PROGRESS * progress
             - self.W_CLEAR * clear_n
             + self.W_STRAIGHT * kn ** 2
             + self.W_SMOOTH * ((k - self.kappa_cmd) / self.KAPPA_MAX) ** 2)

        # pass-side preference: reward arcs that end up on the chosen side
        if self.state in (OvertakeState.PREPARE, OvertakeState.OVERTAKE):
            side = 1.0 if self.PASS_SIDE == "left" else -1.0
            y_end = (1.0 - np.cos(k * s_free)) / k
            J -= self.W_PASS * side * np.clip(y_end / 1.5, -1.0, 1.0)

        b = int(np.argmin(J))
        return float(k[b]), float(s_free[b]), float(clear[b])

    # --------------------------------------------------------------
    def speed(self, kappa: float, s_free: float, clear: float) -> float:
        # 1. cornering limit
        v = math.sqrt(self.A_LAT / max(abs(kappa), 1e-3))
        # 2. be able to stop within the free part of the arc we chose
        v = min(v, math.sqrt(2.0 * self.A_BRAKE * max(s_free - self.STOP_MARGIN, 0.0)))
        # 3. squeeze through tight spots
        if clear < self.V_TIGHT_CLEAR:
            v *= max(0.35, clear / self.V_TIGHT_CLEAR)
        if self.state == OvertakeState.OVERTAKE:
            v += self.OVERTAKE_BOOST
        v = float(np.clip(v, 0.0, self.V_MAX))
        if s_free < 0.45:            # emergency
            return 0.0
        return max(v, self.V_MIN)

    def publish_drive(self, speed: float, steer: float):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.drive.steering_angle = float(steer)
        m.drive.speed = float(max(0.0, speed))
        self.drive_pub.publish(m)

    # ==============================================================
    # opponents / FSM
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
            if 0 < s < best_s and abs(self.lateral_gap(self.ego_pose, o.pose)) < 3.0:
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

    def predict_future(self, t: TargetVehicle) -> TargetVehicle:
        yaw = self.yaw_of(self.ego_pose)
        p = Pose()
        p.position.x = t.pose.position.x + t.vel * self.PREDICTION_TIME * math.cos(yaw)
        p.position.y = t.pose.position.y + t.vel * self.PREDICTION_TIME * math.sin(yaw)
        p.orientation = t.pose.orientation
        return TargetVehicle(p, t.vel)

    def forward_clearance(self) -> float:
        if self.ranges is None or self.angles is None:
            return self.RANGE_MAX
        m = np.abs(self.angles) <= np.deg2rad(10.0)
        return float(np.min(self.ranges[m])) if m.any() else self.RANGE_MAX

    def clear_on_pass_side(self) -> bool:
        if self.ranges is None:
            return False
        side = 1.0 if self.PASS_SIDE == "left" else -1.0   # +angle is LEFT
        m = np.abs(self.angles - side * np.deg2rad(15.0)) < np.deg2rad(8.0)
        return bool(m.any() and float(np.min(self.ranges[m])) >= self.MIN_CLEAR_DIST)

    def set_state(self, s: OvertakeState):
        if self.state != s:
            self.state, self.state_ts = s, self.now_sec()
            self.get_logger().info(f"STATE -> {s.name}")

    def step_fsm(self, target, predicted):
        t_state = self.now_sec() - self.state_ts
        if target is None or self.ego_pose is None:
            if self.state != OvertakeState.FOLLOW:
                self.set_state(OvertakeState.RETURN)
            return

        dist = self.longitudinal_gap(self.ego_pose, target.pose)
        lat = self.lateral_gap(self.ego_pose, target.pose)
        rel_v = self.ego_speed - target.vel

        if self.state == OvertakeState.FOLLOW:
            if dist < self.FOLLOW_TIME_GAP * max(self.ego_speed, 0.1) and rel_v < self.MIN_SPEED_ADV:
                self.set_state(OvertakeState.PREPARE)
        elif self.state == OvertakeState.PREPARE:
            if predicted and self.longitudinal_gap(self.ego_pose, predicted.pose) >= self.OPEN_GAP_THRESHOLD \
                    and self.clear_on_pass_side():
                self.set_state(OvertakeState.OVERTAKE)
            elif t_state > self.PREPARE_TIMEOUT:
                self.set_state(OvertakeState.FOLLOW)
        elif self.state == OvertakeState.OVERTAKE:
            passed = self.longitudinal_gap(target.pose, self.ego_pose) > (self.CAR_LENGTH / 2 + self.FRONT_MARGIN)
            gap_future = self.longitudinal_gap(target.pose, self.ego_pose) + rel_v * self.PASS_TIME_EST
            if passed and gap_future >= (self.CAR_LENGTH + self.FRONT_MARGIN) \
                    and self.forward_clearance() >= self.MIN_CLEAR_DIST \
                    and abs(lat) >= self.MIN_LATERAL_CLEAR:
                self.set_state(OvertakeState.RETURN)
        elif self.state == OvertakeState.RETURN:
            if t_state > self.RETURN_LATENCY:
                self.set_state(OvertakeState.FOLLOW)


def main():
    rclpy.init()
    node = ArcFollowGap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()