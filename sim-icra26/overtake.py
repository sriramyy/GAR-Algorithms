
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan, JointState, Imu
from std_msgs.msg import Float32
import numpy as np
from typing import Optional


class OvertakeFollowGap(Node):

    def __init__(self):
        super().__init__("overtake_follow_gap")

        # --- topic names
        self.scan_topic           = "/autodrive/roboracer_1/lidar"
        self.left_encoder_topic   = "/autodrive/roboracer_1/left_encoder"
        self.right_encoder_topic  = "/autodrive/roboracer_1/right_encoder"
        self.imu_topic            = "/autodrive/roboracer_1/imu"
        self.steer_feedback_topic = "/autodrive/roboracer_1/steering"
        self.steering_topic       = "/autodrive/roboracer_1/steering_command"
        self.throttle_topic       = "/autodrive/roboracer_1/throttle_command"

        qos10 = QoSProfile(depth=10)

        self.scan_sub     = self.create_subscription(LaserScan,  self.scan_topic,           self.lidar_callback,         qos10)
        self.l_enc_sub    = self.create_subscription(JointState, self.left_encoder_topic,   self.left_encoder_callback,  qos10)
        self.r_enc_sub    = self.create_subscription(JointState, self.right_encoder_topic,  self.right_encoder_callback, qos10)
        self.imu_sub      = self.create_subscription(Imu,        self.imu_topic,            self.imu_callback,           qos10)
        self.steer_fb_sub = self.create_subscription(Float32,    self.steer_feedback_topic, self.steer_fb_callback,      qos10)

        self.steering_pub = self.create_publisher(Float32, self.steering_topic,  qos10)
        self.throttle_pub = self.create_publisher(Float32, self.throttle_topic,  qos10)

        # --- PARAMETERS
        # -- LiDAR
        self.MAX_LIDAR_DIST       = 6.0
        self.PREPROCESS_CONV_SIZE = 3

        # -- obstacle bubble
        self.BUBBLE_RADIUS_BASE  = 90
        self.BUBBLE_RADIUS_MAX   = 180
        self.BUBBLE_WALL_THRESH  = 1.5

        # -- gap selection
        self.BEST_POINT_CONV_SIZE = 80

        # -- wall safety – IMMEDIATE (straight-ahead)
        self.EMERGENCY_DIST  = 0.40    # m – crawl trigger
        self.FWD_WEDGE_DEG   = 10.0   # half-angle for immediate e-stop check

        # -- wall safety – WIDE LOOKAHEAD 
        self.LOOKAHEAD_WEDGE_DEG  = 35.0  # half-angle for corner-approach detection
        self.LOOKAHEAD_BRAKE_DIST = 3.5   # m – start tapering speed on straight
        self.LOOKAHEAD_SLOW_DIST  = 2.0   # m – commit to full corner speed

        # -- hairpin detection – EXTRA-WIDE wedge
        # the large right-side of 180 deg hairpin wall sweeps in from the side
        # very wide wedge catches it before the narrow forward wedge does
        self.HAIRPIN_WEDGE_DEG   = 55.0   # half-angle
        self.HAIRPIN_BRAKE_DIST  = 2.2    # m – latch in_hairpin flag

        # -- lateral safety nudge
        self.LATERAL_SAFETY_DIST = 0.30   # m – min clearance at target heading
        self.LATERAL_CHECK_BAND  = 8      # ±indices to inspect around best_idx

        # -- side-repulsion sensing
        self.SIDE_BAND_FRAC      = 0.06

        # -- outside-wall positional bias on straights
        # nudge toward the outer wall to widen corner entry (racing line).
        self.OUTSIDE_BIAS_GAIN   = 0.30   # fraction of half-track shift
        self.OUTSIDE_BIAS_THRESH = 2.5    # m lookahead clearance required to apply bias

        # -- steering
        self.MAX_STEER_ABS              = np.deg2rad(40.0)
        self.STEER_DEADBAND_RAD         = np.deg2rad(1.8)
        self.EDGE_GUARD_DEG             = 14.0
        self.CENTER_BIAS_ALPHA          = 0.18
        self.SIDE_REPULSION_GAIN        = 0.50

        self.STEER_SMOOTH_ALPHA_STRAIGHT = 0.58
        self.STEER_SMOOTH_ALPHA_CORNER   = 0.22

        self.STEER_RATE_LIMIT_STRAIGHT  = np.deg2rad(4.5)
        self.STEER_RATE_LIMIT_CORNER    = np.deg2rad(20.0)
        self.STEER_RATE_LIMIT_HAIRPIN   = np.deg2rad(40.0)  # effectively no limit

        # -- speed
        self.SPEED_MAX       = 0.35 
        self.STRAIGHT_SPEED  = 0.28  
        self.CORNER_SPEED    = 0.14   
        self.HAIRPIN_SPEED   = 0.10   
        self.MIN_SPEED        = 0.05
        self.EMERGENCY_SPEED  = 0.04

        self.HAIRPIN_STEER_DEG       = 20.0   # |steer| above → hairpin mode
        self.CORNER_CLEARANCE_THRESH = 2.0    # fwd clearance below → corner mode

        # asymmetric speed filter: fast decel, slow accel
        self.SPEED_FILTER_ALPHA_DECEL = 0.85
        self.SPEED_FILTER_ALPHA_ACCEL = 0.28

        # speed feedback
        self.SPEED_FEEDBACK_GAIN = 0.65
        self.ENCODER_SPEED_ALPHA = 0.20

        # --- RUNTIME STATE
        self.left_wheel_speed:  Optional[float] = None
        self.right_wheel_speed: Optional[float] = None
        self.ego_speed:         float = 0.0
        self.yaw_rate:          float = 0.0
        self.radians_per_elem:  Optional[float] = None
        self.proc_latest:       Optional[np.ndarray] = None
        self.prev_steer:        float = 0.0
        self.prev_speed_cmd:    float = 0.0
        self.in_hairpin:        bool  = False

        self.get_logger().info("Follow-The-Gap controller v5 started.")

    # --- SENSOR CALLBACKS
    def imu_callback(self, msg: Imu):
        self.yaw_rate = msg.angular_velocity.z

    def steer_fb_callback(self, msg: Float32):
        if self.prev_steer == 0.0:
            self.prev_steer = float(msg.data)

    def _encoder_speed_sample(self, msg: JointState) -> Optional[float]:
        if not msg.velocity:
            return None
        return float(np.mean(np.abs(np.asarray(msg.velocity, dtype=np.float32))))

    def _update_speed_estimate(self):
        samples = [s for s in (self.left_wheel_speed, self.right_wheel_speed) if s is not None]
        if not samples:
            return
        measured = float(np.mean(samples))
        self.ego_speed = (1.0 - self.ENCODER_SPEED_ALPHA) * self.ego_speed + \
                          self.ENCODER_SPEED_ALPHA * measured

    def left_encoder_callback(self, msg: JointState):
        self.left_wheel_speed = self._encoder_speed_sample(msg)
        self._update_speed_estimate()

    def right_encoder_callback(self, msg: JointState):
        self.right_wheel_speed = self._encoder_speed_sample(msg)
        self._update_speed_estimate()

    #  --- lidar callback
    def lidar_callback(self, scan: LaserScan):
        ranges_full = np.array(scan.ranges, dtype=np.float32)
        proc = self.preprocess_lidar(ranges_full)
        self.proc_latest = proc.copy()

        if proc.size == 0:
            return

        # clearance
        fwd_clear       = self._wedge_clearance(proc, self.FWD_WEDGE_DEG)
        lookahead_clear = self._wedge_clearance(proc, self.LOOKAHEAD_WEDGE_DEG)
        hairpin_clear   = self._wedge_clearance(proc, self.HAIRPIN_WEDGE_DEG)

        # estop
        if fwd_clear < self.EMERGENCY_DIST:
            self._publish_drive(self.EMERGENCY_SPEED, self.prev_steer * 0.4)
            return

        # hairpin latch
        self.in_hairpin = (hairpin_clear < self.HAIRPIN_BRAKE_DIST)

        # dynamic bubble
        valid        = proc[proc > 0.0]
        closest_dist = float(np.min(valid)) if valid.size > 0 else self.MAX_LIDAR_DIST
        t            = float(np.clip(1.0 - (closest_dist / self.BUBBLE_WALL_THRESH), 0.0, 1.0))
        bubble_r     = int(self.BUBBLE_RADIUS_BASE + t * (self.BUBBLE_RADIUS_MAX - self.BUBBLE_RADIUS_BASE))

        closest_idx  = int(np.argmin(proc))
        proc_bubbled = self.mask_bubble(proc, closest_idx, bubble_r)

        # fiding gap
        gap_start, gap_end = self.find_max_gap(proc_bubbled)
        best_idx = self.find_best_point(gap_start, gap_end, proc_bubbled)

        # index refine
        best_idx = self.apply_edge_guard(best_idx, proc_bubbled.size)
        best_idx = self.apply_center_bias(best_idx, proc_bubbled.size, self.CENTER_BIAS_ALPHA)

        repel    = self.side_repulsion_shift(proc)
        best_idx = int(np.clip(best_idx + repel, 0, proc_bubbled.size - 1))

        # outside-wall bias only on straights
        if not self.in_hairpin and lookahead_clear > self.OUTSIDE_BIAS_THRESH:
            best_idx = self.apply_outside_bias(best_idx, proc, proc_bubbled.size)

        # lateral safety nudge
        best_idx = self.lateral_safety_nudge(best_idx, proc_bubbled)

        # steering
        steer_raw = self.index_to_steer(best_idx, proc_bubbled.size)
        steer_cmd = self.smooth_and_limit_steer(steer_raw, fwd_clear, lookahead_clear)

        # speed
        speed_cmd = self.speed_policy(steer_cmd, fwd_clear, lookahead_clear, hairpin_clear)

        self._publish_drive(speed_cmd, steer_cmd)

    # --- lidar preprocess
    def preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        n = len(ranges)
        self.radians_per_elem = (2.0 * np.pi) / n if n > 0 else None

        proc = ranges[135:-135].copy() if n > 270 else ranges.copy()
        proc = np.where(np.isfinite(proc), proc, self.MAX_LIDAR_DIST)
        np.clip(proc, 0.0, self.MAX_LIDAR_DIST, out=proc)

        if self.PREPROCESS_CONV_SIZE > 1:
            k    = np.ones(self.PREPROCESS_CONV_SIZE, dtype=np.float32) / float(self.PREPROCESS_CONV_SIZE)
            proc = np.convolve(proc, k, mode="same").astype(np.float32)
            np.clip(proc, 0.0, self.MAX_LIDAR_DIST, out=proc)

        return proc

    def mask_bubble(self, arr: np.ndarray, center: int, radius: int) -> np.ndarray:
        a  = arr.copy()
        lo = max(0, center - radius)
        hi = min(a.size, center + radius + 1)
        a[lo:hi] = 0.0
        return a

    # --- finding gap
    def find_max_gap(self, arr: np.ndarray):
        """Score gaps by width × mean_depth × min_depth_factor."""
        if arr.size == 0:
            return 0, 0
        masked = np.ma.masked_where(arr == 0.0, arr)
        spans  = np.ma.notmasked_contiguous(masked)
        if not spans:
            return 0, arr.size

        best_score, best_span = -1.0, spans[0]
        for sl in spans:
            seg    = arr[sl.start:sl.stop]
            width  = sl.stop - sl.start
            mean_d = float(np.mean(seg))
            min_d  = float(np.min(seg))
            score  = width * mean_d * (min_d / max(mean_d, 0.01))
            if score > best_score:
                best_score, best_span = score, sl
        return best_span.start, best_span.stop

    def find_best_point(self, start: int, stop: int, arr: np.ndarray) -> int:
        if arr.size == 0:
            return 0
        if stop <= start + 1:
            return start
        seg = arr[start:stop]
        if self.BEST_POINT_CONV_SIZE > 1 and seg.size > self.BEST_POINT_CONV_SIZE:
            k   = np.ones(self.BEST_POINT_CONV_SIZE, dtype=np.float32) / float(self.BEST_POINT_CONV_SIZE)
            seg = np.convolve(seg, k, mode="same")
        return int(np.argmax(seg)) + start

    # --- index refining
    def apply_edge_guard(self, idx: int, length: int) -> int:
        if self.radians_per_elem is None or length == 0:
            return 0
        guard = int(round(np.deg2rad(self.EDGE_GUARD_DEG) / self.radians_per_elem))
        return int(np.clip(idx, guard, length - guard - 1))

    def apply_center_bias(self, idx: int, length: int, alpha: float) -> int:
        center = (length - 1) / 2.0
        return int(np.clip(round((1.0 - alpha) * idx + alpha * center), 0, length - 1))

    def side_repulsion_shift(self, proc: np.ndarray) -> int:
        if proc.size == 0 or self.radians_per_elem is None:
            return 0
        band      = max(6, int(self.SIDE_BAND_FRAC * proc.size))
        left_avg  = float(np.mean(proc[:band]))
        right_avg = float(np.mean(proc[-band:]))
        diff      = right_avg - left_avg
        max_shift = int(round(self.SIDE_REPULSION_GAIN / self.radians_per_elem))
        return int(np.clip(
            np.sign(diff) * min(abs(diff), 1.0) * max_shift,
            -max_shift, max_shift
        ))

    def apply_outside_bias(self, idx: int, proc: np.ndarray, length: int) -> int:
        if self.radians_per_elem is None or length == 0:
            return idx
        band      = max(8, int(0.15 * length))
        left_far  = float(np.mean(proc[:band]))
        right_far = float(np.mean(proc[-band:]))
        # diff > 0 → left is further away → outer wall is left → nudge idx toward 0
        diff      = left_far - right_far
        total     = max(left_far + right_far, 0.1)
        max_shift = int(round(self.OUTSIDE_BIAS_GAIN / self.radians_per_elem))
        shift     = int(np.clip(
            -np.sign(diff) * (min(abs(diff), total) / total) * max_shift,
            -max_shift, max_shift
        ))
        return int(np.clip(idx + shift, 0, length - 1))

    def lateral_safety_nudge(self, idx: int, arr: np.ndarray) -> int:
        if arr.size == 0:
            return idx
        length  = arr.size
        lo      = max(0, idx - self.LATERAL_CHECK_BAND)
        hi      = min(length, idx + self.LATERAL_CHECK_BAND + 1)
        region  = arr[lo:hi]
        nz      = region[region > 0.0]
        if nz.size == 0 or float(np.min(nz)) >= self.LATERAL_SAFETY_DIST:
            return idx

        left_seg  = arr[lo: idx + 1]
        right_seg = arr[idx: hi]
        left_nz   = left_seg[left_seg > 0.0]
        right_nz  = right_seg[right_seg > 0.0]
        left_min  = float(np.min(left_nz))  if left_nz.size  > 0 else self.MAX_LIDAR_DIST
        right_min = float(np.min(right_nz)) if right_nz.size > 0 else self.MAX_LIDAR_DIST

        nudge = int(round(self.LATERAL_CHECK_BAND * 1.5))
        if left_min < right_min:
            return int(np.clip(idx + nudge, 0, length - 1))
        else:
            return int(np.clip(idx - nudge, 0, length - 1))

    # --- steering
    def index_to_steer(self, idx: int, length: int) -> float:
        if self.radians_per_elem is None:
            return 0.0
        angle = (idx - length / 2.0) * self.radians_per_elem
        return float(np.clip(angle / 2.0, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))

    def smooth_and_limit_steer(self, steer: float, fwd_clear: float, lookahead_clear: float) -> float:
        """
          hairpin  – minimal smoothing, near-unlimited rate -> snap to full lock
          corner   – light smoothing, generous rate limit
          straight – heavy smoothing, tight rate limit + deadband
        """
        abs_steer = abs(steer)
        if self.in_hairpin or abs_steer > np.deg2rad(self.HAIRPIN_STEER_DEG):
            alpha, rate_limit = 0.15, self.STEER_RATE_LIMIT_HAIRPIN
        elif (abs_steer > np.deg2rad(self.HAIRPIN_STEER_DEG * 0.55) or
              fwd_clear < self.CORNER_CLEARANCE_THRESH or
              lookahead_clear < self.LOOKAHEAD_SLOW_DIST):
            alpha, rate_limit = self.STEER_SMOOTH_ALPHA_CORNER, self.STEER_RATE_LIMIT_CORNER
        else:
            alpha, rate_limit = self.STEER_SMOOTH_ALPHA_STRAIGHT, self.STEER_RATE_LIMIT_STRAIGHT
            if abs_steer < self.STEER_DEADBAND_RAD:
                steer = 0.0

        s     = (1.0 - alpha) * steer + alpha * self.prev_steer
        delta = float(np.clip(s - self.prev_steer, -rate_limit, rate_limit))
        s_lim = self.prev_steer + delta
        self.prev_steer = s_lim
        return s_lim

    # --- clearance
    def _wedge_clearance(self, arr: np.ndarray, half_angle_deg: float) -> float:
        """Min non-zero range within ±half_angle_deg of straight ahead."""
        if self.radians_per_elem is None or arr.size == 0:
            return self.MAX_LIDAR_DIST
        length = arr.size
        center = length // 2
        half   = int(round(np.deg2rad(half_angle_deg) / self.radians_per_elem))
        lo     = max(0, center - half)
        hi     = min(length, center + half + 1)
        nz     = arr[lo:hi]
        nz     = nz[nz > 0.0]
        return float(np.min(nz)) if nz.size > 0 else self.MAX_LIDAR_DIST

    # --- speed
    def speed_policy(self, steer: float, fwd_clear: float,
                     lookahead_clear: float, hairpin_clear: float) -> float:
        """
          HAIRPIN   in_hairpin OR |steer| > HAIRPIN_STEER_DEG -> HAIRPIN_SPEED
          CORNER    lookahead_clear < LOOKAHEAD_SLOW_DIST -> CORNER_SPEED
          APPROACH  lookahead_clear < LOOKAHEAD_BRAKE_DIST  -> lerp down
          STRAIGHT  otherwise -> STRAIGHT_SPEED
        """
        abs_steer  = abs(steer)
        turn_ratio = min(abs_steer / self.MAX_STEER_ABS, 1.0) if self.MAX_STEER_ABS > 0 else 0.0
        clr_scale  = float(np.clip((fwd_clear - 0.50) / 3.00, 0.20, 1.0))

        if self.in_hairpin or abs_steer > np.deg2rad(self.HAIRPIN_STEER_DEG):
            target = float(np.clip(self.HAIRPIN_SPEED * clr_scale, self.MIN_SPEED, self.HAIRPIN_SPEED))

        elif (turn_ratio > 0.18 or
              fwd_clear < self.CORNER_CLEARANCE_THRESH or
              lookahead_clear < self.LOOKAHEAD_SLOW_DIST):
            turn_scale = 1.0 - 0.65 * turn_ratio
            speed  = self.CORNER_SPEED * turn_scale * clr_scale
            target = float(np.clip(speed, self.MIN_SPEED, self.CORNER_SPEED))

        elif lookahead_clear < self.LOOKAHEAD_BRAKE_DIST:
            t      = float(np.clip(
                (self.LOOKAHEAD_BRAKE_DIST - lookahead_clear) /
                (self.LOOKAHEAD_BRAKE_DIST - self.LOOKAHEAD_SLOW_DIST), 0.0, 1.0))
            speed  = ((1.0 - t) * self.STRAIGHT_SPEED + t * self.CORNER_SPEED) * clr_scale
            target = float(np.clip(speed, self.MIN_SPEED, self.STRAIGHT_SPEED))

        else:
            speed = self.STRAIGHT_SPEED * clr_scale
            if abs_steer < np.deg2rad(5.0) and fwd_clear > 3.5:
                speed = min(self.SPEED_MAX, speed + 0.03)
            target = float(np.clip(speed, self.MIN_SPEED, self.SPEED_MAX))

        # Asymmetric low-pass: near-instant decel, gradual accel
        alpha    = self.SPEED_FILTER_ALPHA_DECEL if target < self.prev_speed_cmd \
                   else self.SPEED_FILTER_ALPHA_ACCEL
        filtered = (1.0 - alpha) * self.prev_speed_cmd + alpha * target
        self.prev_speed_cmd = filtered
        return filtered

    # --- throttle
    def speed_to_throttle(self, target_speed: float) -> float:
        speed_error = target_speed - self.ego_speed
        throttle    = target_speed + self.SPEED_FEEDBACK_GAIN * speed_error
        return float(np.clip(throttle, 0.0, self.SPEED_MAX))

    def _publish_drive(self, target_speed: float, steer: float):
        steer_msg      = Float32()
        steer_msg.data = float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))
        thr_msg        = Float32()
        thr_msg.data   = self.speed_to_throttle(max(0.0, target_speed))
        self.steering_pub.publish(steer_msg)
        self.throttle_pub.publish(thr_msg)


# --- ENTRY POINT
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

