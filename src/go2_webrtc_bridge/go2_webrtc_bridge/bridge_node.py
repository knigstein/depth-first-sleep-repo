"""Experimental Unitree Go2 WebRTC -> ROS 2 bridge.

Targeted at unitree_webrtc_connect==2.1.2.

Design goals for the first field-data phase:
  * subscribe to only the navigation/SLAM/LiDAR topics that matter;
  * keep raw payloads for binary/poorly documented topics;
  * expose useful typed ROS messages where the WebRTC payload is understood;
  * expose commands needed to turn LiDAR/uSLAM/legacy SLAM on and off;
  * publish per-topic activity counters so inactive topics are obvious.

The bridge is deliberately conservative about motion: cmd_vel forwarding is
DISABLED by default and must be enabled with a parameter.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, PointCloud2, PointField
from std_msgs.msg import Header, String, UInt8MultiArray
from std_srvs.srv import SetBool

try:
    import sounddevice
except Exception:
    import sys as _sys
    import types as _types

    _sd = _types.ModuleType("sounddevice")
    _sd.query_devices = lambda *a, **k: []
    _sd.OutputStream = _sd.InputStream = _sd.RawOutputStream = object
    _sd.PortAudioError = Exception
    _sd.default = type("_SDDefault", (), {})()
    _sys.modules["sounddevice"] = _sd

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

DATA_TOPICS: dict[str, str] = {
    "LOW_STATE": RTC_TOPIC["LOW_STATE"],
    "LF_SPORT_MOD_STATE": RTC_TOPIC["LF_SPORT_MOD_STATE"],
    "SPORT_MOD_STATE": RTC_TOPIC["SPORT_MOD_STATE"],
    "ULIDAR_ARRAY": RTC_TOPIC["ULIDAR_ARRAY"],
    "ULIDAR_STATE": RTC_TOPIC["ULIDAR_STATE"],
    "ROBOTODOM": RTC_TOPIC["ROBOTODOM"],
    "SLAM_ODOMETRY": RTC_TOPIC["SLAM_ODOMETRY"],
    "SLAM_PC_TO_IMAGE_LOCAL": RTC_TOPIC["SLAM_PC_TO_IMAGE_LOCAL"],
    "SLAM_QT_NOTICE": RTC_TOPIC["SLAM_QT_NOTICE"],
    "LIDAR_MAPPING_CLOUD_POINT": RTC_TOPIC["LIDAR_MAPPING_CLOUD_POINT"],
    "LIDAR_MAPPING_ODOM": RTC_TOPIC["LIDAR_MAPPING_ODOM"],
    "LIDAR_MAPPING_SERVER_LOG": RTC_TOPIC["LIDAR_MAPPING_SERVER_LOG"],
    "LIDAR_LOCALIZATION_CLOUD_POINT": RTC_TOPIC["LIDAR_LOCALIZATION_CLOUD_POINT"],
    "LIDAR_LOCALIZATION_ODOM": RTC_TOPIC["LIDAR_LOCALIZATION_ODOM"],
    "GRID_MAP": RTC_TOPIC["GRID_MAP"],
}

CMD_TOPICS: dict[str, str] = {
    "ULIDAR_SWITCH": RTC_TOPIC["ULIDAR_SWITCH"],
    "LIDAR_MAPPING_CMD": RTC_TOPIC["LIDAR_MAPPING_CMD"],
    "SLAM_QT_COMMAND": RTC_TOPIC["SLAM_QT_COMMAND"],
    "SPORT_MOD": RTC_TOPIC["SPORT_MOD"],
}

# Recorded LOW_STATE frames always carry 20 motor_state entries: indices 0-11 are
# the twelve actuated leg joints (never zero, real temperatures), 12-19 are
# constant placeholders (q == 0.0, temperature == 0, reserve == [0, 0]). Publishing
# those placeholders would show up as eight dead joints in RViz, so only the
# first twelve are forwarded.
GO2_JOINT_NAMES = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
]

SPORT_MAP = {
    "sit": "Sit",
    "rise_sit": "RiseSit",
    "stand_up": "StandUp",
    "stand_down": "StandDown",
    "hello": "Hello",
    "stretch": "Stretch",
    "wiggle": "WiggleHips",
    "balance": "BalanceStand",
    "recovery": "RecoveryStand",
    "damp": "Damp",
    "stop": "StopMove",
}

# Resolve the api_ids once, at import. SPORT_CMD is the installed library's
# dict, and a name it does not define would otherwise raise KeyError at the
# moment somebody asks the robot to sit — i.e. in the middle of a rescue.
# Unknown aliases are dropped here so the misconfiguration is visible at
# startup instead of as a failed command later.
SPORT_API_IDS: dict[str, int] = {
    alias: SPORT_CMD[name] for alias, name in SPORT_MAP.items() if name in SPORT_CMD
}
_MISSING_SPORT = sorted({name for name in SPORT_MAP.values() if name not in SPORT_CMD})
if _MISSING_SPORT:
    print(
        f"[go2_webrtc_bridge] WARNING: unitree_webrtc_connect SPORT_CMD has no "
        f"entry for {_MISSING_SPORT}; those /go2/sport_cmd aliases will be rejected",
        file=sys.stderr,
    )

# Move and StopMove drive the whole motion-safety path (_cmd_vel_loop, the
# watchdog, shutdown). The library always ships them, but if one is genuinely
# absent we must not pretend the node can stop the robot. Not fatal at import:
# with enable_cmd_vel off, data collection is perfectly safe without them, so
# __init__ only refuses when motion is actually enabled.
_MISSING_MOTION_CMDS = [n for n in ("Move", "StopMove") if n not in SPORT_CMD]

RAW_BINARY_TOPICS = {
    DATA_TOPICS["SLAM_PC_TO_IMAGE_LOCAL"],
    DATA_TOPICS["LIDAR_MAPPING_CLOUD_POINT"],
    DATA_TOPICS["LIDAR_LOCALIZATION_CLOUD_POINT"],
    DATA_TOPICS["GRID_MAP"],
}

# Every payload key `_publish_odometry_generic` knows how to read. Used only as
# a recogniser: a payload matching none of them is refused rather than turned
# into an all-zero Odometry. `position` doubles as the orientation source when
# it carries 7 elements ([x,y,z,qx,qy,qz,qw]).
_ODOM_FIELD_PATHS: tuple[tuple[str, ...], ...] = (
    ("pose", "pose", "position"),
    ("pose", "position"),
    ("position",),
    ("pose",),
    ("pose", "pose", "orientation"),
    ("pose", "orientation"),
    ("orientation",),
    ("imu_state", "quaternion"),
    ("quaternion",),
    ("imu_state", "rpy"),
    ("twist", "twist", "linear"),
    ("twist", "linear"),
    ("velocity",),
    ("twist", "twist", "angular"),
    ("twist", "angular"),
    ("yaw_speed",),
    ("twist",),
)


@dataclass
class TopicStats:
    count: int = 0
    first_ns: int = 0
    last_ns: int = 0
    bytes_last: int = 0

    def update(self, size: int = 0) -> None:
        now = time.monotonic_ns()
        self.count += 1
        if self.first_ns == 0:
            self.first_ns = now
        self.last_ns = now
        self.bytes_last = size

    def hz(self) -> float:
        if self.count < 2 or self.first_ns == 0:
            return 0.0
        dt = (self.last_ns - self.first_ns) / 1e9
        return (self.count - 1) / dt if dt > 0 else 0.0

    def age(self) -> float | None:
        if self.last_ns == 0:
            return None
        return (time.monotonic_ns() - self.last_ns) / 1e9


def jsonable(value: Any, max_array_elements: int = 0) -> Any:
    """Convert common Unitree/numpy values into JSON-safe data.

    max_array_elements caps ndarray -> list expansion. <= 0 means no cap, which
    is what a recording run wants: the utlidar voxel map carries a (N, 3)
    float64 point array, and stubbing it out to a shape descriptor would throw
    away the entire LiDAR stream from the raw channel.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # json.dumps would emit bare NaN/Infinity, which is not valid JSON
        # (RFC 8259) and is rejected by strict parsers such as serde, JSON.parse
        # or a ROS bag tool. Null survives a round trip and is unambiguous.
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (np.floating,)):
        # NumPy floats are not Python floats, so they miss the branch above and
        # would end up as str() (or as bare NaN once inside an array's tolist()).
        return jsonable(float(value), max_array_elements)
    if isinstance(value, np.ndarray):
        # Guard against exploding huge arrays in the status/raw JSON channels.
        # max_array_elements <= 0 disables the guard entirely (the default), so
        # a recording run captures every point rather than a shape-only stub.
        if 0 < max_array_elements < value.size:
            return {
                "__ndarray__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "size": int(value.size),
                "__truncated__": True,
                "__limit__": int(max_array_elements),
            }
        # A numeric array is the common case by far (the LiDAR (N, 3) float64
        # cloud is ~10^5 elements). Check finiteness once, vectorised, instead of
        # recursing through every element — that recursion costs ~10^5 Python
        # calls per cloud on the WebRTC event-loop thread.
        if value.dtype.kind in "fiub" and np.isfinite(value).all():
            return value.tolist()
        return jsonable(value.tolist(), max_array_elements)
    if isinstance(value, (list, tuple)):
        return [jsonable(v, max_array_elements) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v, max_array_elements) for k, v in value.items()}
    return str(value)


def as_float_list(value: Any, n: int | None = None) -> list[float] | None:
    if value is None:
        return None
    try:
        out = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if n is not None and len(out) < n:
        return None
    return out


def pick(d: Any, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        cur = d
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def quaternion_msg(q: Any) -> tuple[float, float, float, float] | None:
    """Return ROS order x,y,z,w.

    Unitree IMU examples expose quaternion as [w,x,y,z].
    """
    vals = as_float_list(q, 4)
    if vals is None:
        return None
    w, x, y, z = vals[:4]
    return x, y, z, w


def rpy_to_quat(rpy) -> tuple[float, float, float, float] | None:
    vals = as_float_list(rpy, 3)
    if vals is None:
        return None
    roll, pitch, yaw = vals[:3]
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,
    )  # w


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    s = math.sin(yaw * 0.5)
    c = math.cos(yaw * 0.5)
    return 0.0, 0.0, s, c


class Go2WebRTCBridge(Node):
    def __init__(self) -> None:
        def _load_aes_key() -> str | None:
            v = (os.environ.get("UNITREE_AES_KEY") or "").strip()
            if v:
                return v
            try:
                with open(os.path.expanduser("~/.fleet/aes_key")) as f:
                    return f.read().strip() or None
            except OSError:
                return None

        super().__init__("go2_webrtc_bridge")

        # Connection parameters.
        self.declare_parameter(
            "robot_ip", os.environ.get("UNITREE_ROBOT_IP", "192.168.123.161")
        )
        self.declare_parameter("aes_128_key", _load_aes_key())

        # Sensor parameters.
        self.declare_parameter("lidar_auto_enable", False)
        self.declare_parameter("lidar_decoder", "native")
        self.declare_parameter("enable_slam_subscriptions", True)
        # The utlidar voxel_map is already expressed in "odom" by the robot. Keep
        # its own frame_id unless explicitly told otherwise.
        self.declare_parameter("lidar_use_payload_frame", True)

        # Raw-channel fidelity. These control /go2/raw/... only; the typed
        # topics are unaffected.
        #
        # publish_full_arrays=true expands every ndarray in the payload instead of
        # replacing arrays over `raw_array_limit` with a shape-only stub. The
        # default is full fidelity because the alternative silently deletes the
        # LiDAR point cloud (measured: 215 B recorded instead of 1.46 MB).
        #
        # exclude_raw_fields drops top-level payload keys from /go2/raw/... .
        # motor_state is excluded by default: it is 1105 of LOW_STATE's 1358
        # bytes and there is no velocity in it, so a recording run gets the same
        # information for ~0.4% less disk. Joint positions remain available on
        # the typed /go2/joint_states topic.
        self.declare_parameter("publish_full_arrays", True)
        self.declare_parameter("raw_array_limit", 4096)
        self.declare_parameter("exclude_raw_fields", ["motor_state"])

        # Motion is deliberately off by default.
        self.declare_parameter("enable_cmd_vel", False)
        self.declare_parameter("cmd_vel_timeout", 0.5)
        self.declare_parameter("max_cmd_vx", 1.0)
        self.declare_parameter("max_cmd_vy", 1.0)
        self.declare_parameter("max_cmd_wz", 2.0)

        if _MISSING_MOTION_CMDS and bool(self.get_parameter("enable_cmd_vel").value):
            raise RuntimeError(
                f"enable_cmd_vel is true but unitree_webrtc_connect SPORT_CMD is "
                f"missing {_MISSING_MOTION_CMDS}. Refusing to start: this node "
                "could not guarantee the robot can be stopped. Set "
                "enable_cmd_vel:=false, or install a library version that "
                "defines these api_ids."
            )

        # Frames.
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("imu_frame", "imu_link")
        self.declare_parameter("lidar_frame", "lidar_link")
        # "sport"  -> /go2/imu/data from rt/lf/sportmodestate (full IMU set)
        # "lowstate" -> only from rt/lf/lowstate (orientation from rpy)
        # "both"    -> both topics publish (duplicate orientation, mixed completeness)
        self.declare_parameter("imu_source", "sport")

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.imu_frame = str(self.get_parameter("imu_frame").value)
        self.lidar_frame = str(self.get_parameter("lidar_frame").value)
        # Read once, like the frames: these are read per message inside the
        # WebRTC callback, and get_parameter() allocates a Parameter each call.
        self.imu_source = str(self.get_parameter("imu_source").value)
        self.lidar_use_payload_frame = bool(
            self.get_parameter("lidar_use_payload_frame").value
        )
        self.publish_full_arrays = bool(self.get_parameter("publish_full_arrays").value)
        self._raw_array_limit = (
            0
            if self.publish_full_arrays
            else int(self.get_parameter("raw_array_limit").value)
        )
        self._excluded_raw_fields = frozenset(
            str(f)
            for f in (self.get_parameter("exclude_raw_fields").value or [])
            if str(f)
        )

        self._lock = threading.RLock()
        self._stats: dict[str, TopicStats] = {
            name: TopicStats() for name in DATA_TOPICS.values()
        }

        self._last_rx_time = 0.0
        self._latest_cmd_time = 0.0
        self._cmd_vel_latest = None
        self._lidar_wanted = False
        self._connection_ready = threading.Event()
        self._shutdown_requested = threading.Event()
        self._stopping = threading.Event()

        # Typed publishers.
        self.imu_pub = self.create_publisher(Imu, "/go2/imu/data", 10)
        self.joint_pub = self.create_publisher(JointState, "/go2/joint_states", 10)

        self.sport_lf_odom_pub = self.create_publisher(
            Odometry, "/go2/odom/sport_lf", 10
        )
        self.sport_odom_pub = self.create_publisher(Odometry, "/go2/odom/sport", 10)
        self.robot_pose_odom_pub = self.create_publisher(
            Odometry, "/go2/odom/robot_pose", 10
        )
        self.lio_odom_pub = self.create_publisher(Odometry, "/go2/odom/lio_sam", 10)
        self.uslam_mapping_odom_pub = self.create_publisher(
            Odometry, "/go2/odom/uslam_mapping", 10
        )
        self.uslam_localization_odom_pub = self.create_publisher(
            Odometry, "/go2/odom/uslam_localization", 10
        )

        self.lidar_raw_pub = self.create_publisher(PointCloud2, "/go2/lidar/points", 10)
        self.lidar_state_pub = self.create_publisher(String, "/go2/lidar/state", 10)

        # String/raw channels keep unknown payloads bag-recordable.
        self.raw_string_pubs: dict[str, Any] = {}
        self.raw_bytes_pubs: dict[str, Any] = {}
        self.raw_meta_pubs: dict[str, Any] = {}
        for topic in DATA_TOPICS.values():
            safe = self._safe_topic(topic)
            if topic in RAW_BINARY_TOPICS:
                self.raw_bytes_pubs[topic] = self.create_publisher(
                    UInt8MultiArray, f"/go2/raw/{safe}_bytes", 10
                )
                self.raw_meta_pubs[topic] = self.create_publisher(
                    String, f"/go2/raw/{safe}_meta", 10
                )
            else:
                self.raw_string_pubs[topic] = self.create_publisher(
                    String, f"/go2/raw/{safe}", 10
                )

        # State/status channels.
        self.topic_status_pub = self.create_publisher(
            String, "/go2/bridge/topic_status", 10
        )
        self.connection_status_pub = self.create_publisher(
            String, "/go2/bridge/connection", 10
        )

        # Commands.
        self.lidar_service = self.create_service(
            SetBool, "/go2/lidar/set_enabled", self._srv_lidar
        )
        self.uslam_command_sub = self.create_subscription(
            String, "/go2/uslam/command", self._on_uslam_command, 10
        )
        self.legacy_slam_command_sub = self.create_subscription(
            String, "/go2/legacy_slam/command", self._on_legacy_slam_command, 10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._on_cmd_vel_unstamped, 10
        )
        self.cmd_vel_stamped_sub = self.create_subscription(
            TwistStamped, "/cmd_vel_stamped", self._on_cmd_vel_stamped, 10
        )
        self.sport_cmd_sub = self.create_subscription(
            String, "/go2/sport_cmd", self._on_sport_cmd, 10
        )

        self.stop_timer = self.create_timer(0.1, self._cmd_vel_watchdog)
        self.status_timer = self.create_timer(1.0, self._publish_status)

        # Background asyncio / WebRTC thread.
        self._thread = threading.Thread(target=self._webrtc_thread_main, daemon=True)
        self._thread.start()

    @staticmethod
    def _safe_topic(topic: str) -> str:
        return topic.replace("/", "_").strip("_")

    def _ros_now(self):
        return self.get_clock().now().to_msg()

    def _stamp(self, msg) -> None:
        try:
            msg.header.stamp = self._ros_now()
        except AttributeError:
            pass

    def _publish_raw(self, topic: str, envelope: dict[str, Any]) -> None:
        payload = envelope.get("data") if isinstance(envelope, dict) else envelope

        if topic in self.raw_bytes_pubs:
            raw = payload.get("data") if isinstance(payload, dict) else payload
            if isinstance(raw, (bytes, bytearray)):
                m = UInt8MultiArray()
                m.data = list(bytes(raw))
                self.raw_bytes_pubs[topic].publish(m)

            # Keep metadata needed for later offline decoding: compression
            # parameters, origin, resolution, chunk info, etc.
            meta = dict(envelope) if isinstance(envelope, dict) else {"data": {}}
            if isinstance(meta.get("data"), dict):
                meta_data = dict(meta["data"])
                meta_data.pop("data", None)
                meta["data"] = meta_data
            if self._excluded_raw_fields:
                meta = {
                    k: v for k, v in meta.items() if k not in self._excluded_raw_fields
                }
            text = self._encode(meta)
            self.raw_meta_pubs[topic].publish(self._as_string(text))
            # For these topics the wire payload is the byte blob, so that is
            # what "bytes_last" should report; meta is the fallback when the
            # payload arrived already decoded.
            size = len(raw) if isinstance(raw, (bytes, bytearray)) else 0
            self._count(topic, size or len(text.encode("utf-8")))
            return

        # For normal topics publish the topic payload, not the WebRTC envelope.
        # exclude_raw_fields is applied here only — the typed converters still
        # see the untouched payload, so /go2/joint_states keeps working while
        # motor_state is kept out of the bulky JSON channel.
        if self._excluded_raw_fields and isinstance(payload, dict):
            dropped = self._excluded_raw_fields & payload.keys()
            if dropped:
                payload = {k: v for k, v in payload.items() if k not in dropped}
        text = self._encode(payload)
        self.raw_string_pubs[topic].publish(self._as_string(text))
        size = len(payload) if isinstance(payload, (bytes, bytearray)) else 0
        self._count(topic, size or len(text.encode("utf-8")))

    @staticmethod
    def _as_string(text: str):
        m = String()
        m.data = text
        return m

    def _encode(self, payload: Any) -> str:
        """JSON-encode a payload for a /go2/raw channel.

        Uses this node's raw_array_limit, which is 0 (expand everything) unless
        publish_full_arrays was turned off explicitly.
        """
        return json.dumps(
            jsonable(payload, self._raw_array_limit),
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _count(self, topic: str, size: int) -> None:
        with self._lock:
            self._stats[topic].update(size)
            self._last_rx_time = time.monotonic()

    def _publish_odometry_generic(
        self,
        payload: Any,
        pub,
        frame_id_default: str = "odom",
        child_frame_default: str = "base_link",
    ) -> bool:
        """Best-effort conversion for JSON nav_msgs/Odometry-like payloads.

        If the WebRTC payload is binary CDR, this returns False and raw bytes are
        kept on the corresponding /go2/raw topic for later decoding.
        """
        if not isinstance(payload, dict):
            return False
        # Binary CDR payloads surface as {"data": <bytes>, ...}. Typed conversion
        # is impossible and publishing would emit all-zero odometry on a live
        # topic — refuse and keep the raw bytes on /go2/raw instead.
        if isinstance(payload.get("data"), (bytes, bytearray)):
            return False
        # Same reasoning for a dict that simply does not look like odometry:
        # silently emitting an all-zero Odometry on /go2/odom/* is worse than
        # emitting nothing, because consumers cannot tell "at the origin" from
        # "we do not understand this payload". Refuse instead.
        if not any(pick(payload, path) is not None for path in _ODOM_FIELD_PATHS):
            return False

        msg = Odometry()
        self._stamp(msg)

        frame_id = (
            pick(payload, ("header", "frame_id"), ("frame_id",)) or frame_id_default
        )
        child = pick(payload, ("child_frame_id",)) or child_frame_default
        msg.header.frame_id = str(frame_id)
        # nav_msgs/Odometry requires the twist to be expressed in
        # child_frame_id, not in header.frame_id. SportModState.velocity is a
        # body-frame vector (it integrates to ROBOTODOM only after rotating by
        # the quaternion), so the child frame must be the body frame.
        msg.child_frame_id = str(child)

        # list: [x, y, z, qx, qy, qz, qw] — geometry_msgs/Pose ordering.
        p = pick(
            payload,
            ("pose", "pose", "position"),
            ("pose", "position"),
            ("position",),
            ("pose",),
        )
        if isinstance(p, dict):
            msg.pose.pose.position.x = float(p.get("x", 0.0))
            msg.pose.pose.position.y = float(p.get("y", 0.0))
            msg.pose.pose.position.z = float(p.get("z", 0.0))
        elif isinstance(p, (list, tuple)) and len(p) >= 3:
            msg.pose.pose.position.x = float(p[0])
            msg.pose.pose.position.y = float(p[1])
            msg.pose.pose.position.z = float(p[2])

        # Pose orientation.
        q = pick(
            payload,
            ("pose", "pose", "orientation"),
            ("pose", "orientation"),
            ("orientation",),
            ("imu_state", "quaternion"),
            ("quaternion",),
            ("imu_state", "rpy"),
        )
        if isinstance(q, dict):
            msg.pose.pose.orientation.x = float(q.get("x", 0.0))
            msg.pose.pose.orientation.y = float(q.get("y", 0.0))
            msg.pose.pose.orientation.z = float(q.get("z", 0.0))
            msg.pose.pose.orientation.w = float(q.get("w", 1.0))
        elif isinstance(q, (list, tuple)) and len(q) == 3:
            q2 = rpy_to_quat(q)
            if q2:
                (
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ) = q2
        elif isinstance(q, (list, tuple)) and len(q) >= 4:
            q2 = quaternion_msg(q)
            if q2:
                (
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ) = q2
        elif isinstance(p, (list, tuple)) and len(p) >= 7:
            # Flat ROBOTODOM pose list: [x, y, z, qx, qy, qz, qw].
            msg.pose.pose.orientation.x = float(p[3])
            msg.pose.pose.orientation.y = float(p[4])
            msg.pose.pose.orientation.z = float(p[5])
            msg.pose.pose.orientation.w = float(p[6])

        # Velocity.
        #
        # SportModState.velocity is the robot's velocity in its own body frame
        # (verified against the dumps: rotating d(position)/dt by the reported
        # quaternion into the body frame matches `velocity` at corr ~0.93,
        # while the unrotated odom-frame derivative anti-correlates in dump3).
        # yaw_speed is literally gyroscope[2] in every recorded frame, i.e. also
        # body frame. Since msg.child_frame_id is the body frame, both are
        # correct there without any rotation.
        v = pick(
            payload,
            ("twist", "twist", "linear"),
            ("twist", "linear"),
            ("velocity",),
        )
        if isinstance(v, dict):
            msg.twist.twist.linear.x = float(v.get("x", 0.0))
            msg.twist.twist.linear.y = float(v.get("y", 0.0))
            msg.twist.twist.linear.z = float(v.get("z", 0.0))
        elif isinstance(v, (list, tuple)) and len(v) >= 3:
            msg.twist.twist.linear.x = float(v[0])
            msg.twist.twist.linear.y = float(v[1])
            msg.twist.twist.linear.z = float(v[2])

        w = pick(
            payload,
            ("twist", "twist", "angular"),
            ("twist", "angular"),
        )
        if isinstance(w, dict):
            msg.twist.twist.angular.x = float(w.get("x", 0.0))
            msg.twist.twist.angular.y = float(w.get("y", 0.0))
            msg.twist.twist.angular.z = float(w.get("z", 0.0))
        elif isinstance(w, (list, tuple)) and len(w) >= 3:
            msg.twist.twist.angular.x = float(w[0])
            msg.twist.twist.angular.y = float(w[1])
            msg.twist.twist.angular.z = float(w[2])
        elif "yaw_speed" in payload:
            msg.twist.twist.angular.z = float(payload["yaw_speed"])

        # Flat ROBOTODOM twist list: [vx, vy, vz, wx, wy, wz].
        t = payload.get("twist")
        if isinstance(t, (list, tuple)) and len(t) >= 6:
            msg.twist.twist.linear.x = float(t[0])
            msg.twist.twist.linear.y = float(t[1])
            msg.twist.twist.linear.z = float(t[2])
            msg.twist.twist.angular.x = float(t[3])
            msg.twist.twist.angular.y = float(t[4])
            msg.twist.twist.angular.z = float(t[5])

        pub.publish(msg)
        return True

    def _dispatch_topic(self, topic: str, message: dict[str, Any]) -> None:
        """Entry point for every WebRTC data-channel callback.

        Everything is inside the guard: an exception escaping here lands in the
        WebRTC library's own callback and can tear the data channel down, which
        would cost us every topic, not just the one that misbehaved.
        """
        try:
            payload = message.get("data") if isinstance(message, dict) else message
            self._publish_raw(topic, message)

            if topic == DATA_TOPICS["LOW_STATE"]:
                self._handle_lowstate(payload)
            elif topic == DATA_TOPICS["LF_SPORT_MOD_STATE"]:
                self._handle_sport_state(payload, self.sport_lf_odom_pub)
            elif topic == DATA_TOPICS["SPORT_MOD_STATE"]:
                self._handle_sport_state(payload, self.sport_odom_pub)
            elif topic == DATA_TOPICS["ULIDAR_ARRAY"]:
                self._handle_lidar(payload, self.lidar_raw_pub)
            elif topic == DATA_TOPICS["ULIDAR_STATE"]:
                self._handle_lidar_state(payload)
            elif topic == DATA_TOPICS["ROBOTODOM"]:
                self._publish_odometry_generic(
                    payload, self.robot_pose_odom_pub, "odom", self.base_frame
                )
            elif topic == DATA_TOPICS["SLAM_ODOMETRY"]:
                self._publish_odometry_generic(
                    payload, self.lio_odom_pub, "map", self.base_frame
                )
            elif topic == DATA_TOPICS["LIDAR_MAPPING_ODOM"]:
                self._publish_odometry_generic(
                    payload, self.uslam_mapping_odom_pub, "map", self.base_frame
                )
            elif topic == DATA_TOPICS["LIDAR_LOCALIZATION_ODOM"]:
                self._publish_odometry_generic(
                    payload, self.uslam_localization_odom_pub, "map", self.base_frame
                )
        except Exception as exc:
            self.get_logger().warning(f"Failed typed conversion for {topic}: {exc}")

    def _publish_imu_from(self, imu_state: dict[str, Any]) -> None:
        """Publish sensor_msgs/Imu from a Unitree imu_state block.

        The two topics that carry an imu_state expose different subsets of it.
        Recorded payloads:

          rt/lf/lowstate        {'rpy': [3]}                       (10 Hz)
          rt/lf/sportmodestate  {'quaternion': [w,x,y,z], 'gyroscope': [3],
                                'accelerometer': [3], 'rpy': [3],
                                'temperature': int}               (10 Hz)

        so orientation is always derivable, but only the sport topic carries
        angular velocity / linear acceleration. Callers pass whichever block
        they hold; missing fields stay at zero rather than dropping the message.
        """
        imu = Imu()
        imu.header.stamp = self._ros_now()
        imu.header.frame_id = self.imu_frame

        q = quaternion_msg(imu_state.get("quaternion")) or rpy_to_quat(
            imu_state.get("rpy")
        )
        if q:
            (
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
                imu.orientation.w,
            ) = q

        gyro = as_float_list(imu_state.get("gyroscope"), 3)
        if gyro:
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (
                gyro[:3]
            )

        acc = as_float_list(imu_state.get("accelerometer"), 3)
        if acc:
            (
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ) = acc[:3]

        self.imu_pub.publish(imu)

    def _handle_lowstate(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        imu_state = payload.get("imu_state")
        if (
            self.imu_source in ("lowstate", "both")
            and isinstance(imu_state, dict)
            and imu_state
        ):
            self._publish_imu_from(imu_state)

        self._publish_joint_state(payload)

    def _publish_joint_state(self, payload: Any) -> None:
        """Publish sensor_msgs/JointState from LOW_STATE.motor_state.

        Recorded entries have exactly {q, temperature, lost, reserve} — there is
        no "dq" anywhere in any dump, so JointState.velocity is left unset rather
        than filled with zeros that would read as "joints are not moving".
        Temperatures are useful enough to keep on the raw channel only; JointState
        has no field for them.
        """
        motors = payload.get("motor_state")
        if not isinstance(motors, list) or not motors:
            return
        motors = [m for m in motors if isinstance(m, dict)][: len(GO2_JOINT_NAMES)]
        if not motors:
            return

        js = JointState()
        js.header.stamp = self._ros_now()
        js.header.frame_id = self.base_frame
        js.name = GO2_JOINT_NAMES[: len(motors)]
        js.position = [float(m.get("q", 0.0)) for m in motors]
        self.joint_pub.publish(js)

    def _handle_sport_state(self, payload: Any, pub) -> None:
        if not isinstance(payload, dict):
            return
        self._publish_odometry_generic(payload, pub, "odom", self.base_frame)

        # SportModState is the only recorded topic whose imu_state has the whole
        # sensor_msgs/Imu set, so it is the one worth publishing the full
        # message from. lowstate only carries rpy.
        imu_state = payload.get("imu_state")
        if (
            self.imu_source in ("sport", "both")
            and isinstance(imu_state, dict)
            and imu_state
        ):
            self._publish_imu_from(imu_state)

    def _handle_lidar(self, payload: Any, pub) -> None:
        if not isinstance(payload, dict):
            return
        data = payload.get("data")
        if not isinstance(data, dict):
            return
        points = data.get("points")
        if points is None:
            return

        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return
        arr = np.ascontiguousarray(arr[:, :3], dtype=np.float32)
        n_points = int(arr.shape[0])

        header = self._ros_now()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_header = Header()
        cloud_header.stamp = header
        # Recorded payloads always carry frame_id="odom" (the voxel map is a world
        # grid, and it tracks the robot through the map). Relabelling it as
        # lidar_link puts the cloud in a frame nothing else is expressed in.
        if self.lidar_use_payload_frame:
            cloud_header.frame_id = str(
                pick(payload, ("frame_id",)) or self.lidar_frame
            )
        else:
            cloud_header.frame_id = self.lidar_frame

        # Build the buffer with numpy instead of point_cloud2.create_cloud().
        # create_cloud() wants a Python list of points and struct.pack()s them
        # one at a time: measured ~1.8 ms for a 5k-point cloud and ~13 ms for
        # 35k points, versus ~0.03 ms for the numpy path below. That is pure
        # Python blocking the WebRTC event-loop thread, which is the same thread
        # that resends the motion command. The layout is identical (three
        # float32 fields, no padding), so the message is byte-for-byte the same.
        cloud = PointCloud2()
        cloud.header = cloud_header
        cloud.height = 1
        cloud.width = n_points
        cloud.fields = fields
        cloud.point_step = 12
        cloud.row_step = 12 * n_points
        cloud.data = arr.tobytes()
        cloud.is_dense = True
        pub.publish(cloud)

    def _handle_lidar_state(self, payload: Any) -> None:
        """Surface LiDAR health on /go2/lidar/state.

        Recorded rt/utlidar/lidar_state payloads are flat dicts:
        stamp(float sec), error_state, dirty_percentage, cloud_frequency,
        cloud_packet_loss_rate, cloud_size, cloud_scan_num, imu_frequency,
        imu_packet_loss_rate, imu_rpy(degrees!), serial_* and the version
        strings. Note imu_rpy here is degrees, unlike every other imu_rpy-style
        field in the other topics, which are radians — so it is passed through
        as-is under an explicit key name rather than being fed to any converter.
        """
        if not isinstance(payload, dict):
            return
        out = {
            "error_state": payload.get("error_state"),
            "dirty_percentage": payload.get("dirty_percentage"),
            "cloud_frequency_hz": payload.get("cloud_frequency"),
            "cloud_packet_loss_rate_percent": payload.get("cloud_packet_loss_rate"),
            "cloud_size_bytes": payload.get("cloud_size"),
            "cloud_scan_num": payload.get("cloud_scan_num"),
            "imu_frequency_hz": payload.get("imu_frequency"),
            "imu_packet_loss_rate": payload.get("imu_packet_loss_rate"),
            "imu_rpy_degrees": payload.get("imu_rpy"),
            "serial_recv_stamp": payload.get("serial_recv_stamp"),
            "serial_buffer_size": payload.get("serial_buffer_size"),
            "serial_buffer_read": payload.get("serial_buffer_read"),
            "firmware_version": payload.get("firmware_version"),
            "software_version": payload.get("software_version"),
            "sdk_version": payload.get("sdk_version"),
            "sys_rotation_speed": payload.get("sys_rotation_speed"),
            "com_rotation_speed": payload.get("com_rotation_speed"),
        }
        m = String()
        m.data = self._encode(out)
        self.lidar_state_pub.publish(m)

    def _schedule(self, coro) -> None:
        loop = getattr(self, "_async_loop", None)
        if loop is None or not loop.is_running():
            self.get_logger().warning("WebRTC event loop is not ready")
            coro.close()
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as exc:
            # The loop can still stop between is_running() and the hand-off
            # (loop.stop() is called from the WebRTC thread during teardown).
            # Close the coroutine so it does not leak as "never awaited".
            coro.close()
            self.get_logger().warning(f"WebRTC event loop unavailable: {exc}")
            return
        fut.add_done_callback(self._log_future_error)

    def _log_future_error(self, fut) -> None:
        try:
            fut.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.get_logger().error(f"WebRTC command failed: {exc}")

    def _on_uslam_command(self, msg: String) -> None:
        command = msg.data.strip()
        if not command:
            return
        self._schedule(self._send_plain(CMD_TOPICS["LIDAR_MAPPING_CMD"], command))

    def _on_legacy_slam_command(self, msg: String) -> None:
        command = msg.data.strip()
        if not command:
            return
        self._schedule(self._send_plain(CMD_TOPICS["SLAM_QT_COMMAND"], command))

    def _on_cmd_vel_stamped(self, msg: TwistStamped) -> None:
        self._handle_cmd_vel(msg.twist)

    def _on_cmd_vel_unstamped(self, msg: Twist) -> None:
        self._handle_cmd_vel(msg)

    def _handle_cmd_vel(self, twist: Twist) -> None:
        if not bool(self.get_parameter("enable_cmd_vel").value):
            return
        # np.clip propagates NaN, so a non-finite Twist would reach the robot as
        # a Move parameter. Treat that as "no command" instead.
        if not all(
            math.isfinite(v)
            for v in (
                twist.linear.x,
                twist.linear.y,
                twist.angular.z,
            )
        ):
            self.get_logger().warning("cmd_vel with non-finite value ignored")
            return
        vx = float(
            np.clip(
                twist.linear.x,
                -float(self.get_parameter("max_cmd_vx").value),
                float(self.get_parameter("max_cmd_vx").value),
            )
        )
        vy = float(
            np.clip(
                twist.linear.y,
                -float(self.get_parameter("max_cmd_vy").value),
                float(self.get_parameter("max_cmd_vy").value),
            )
        )
        wz = float(
            np.clip(
                twist.angular.z,
                -float(self.get_parameter("max_cmd_wz").value),
                float(self.get_parameter("max_cmd_wz").value),
            )
        )
        self._latest_cmd_time = time.monotonic()
        self._cmd_vel_latest = (vx, vy, wz)

    def _cmd_vel_watchdog(self) -> None:
        if not bool(self.get_parameter("enable_cmd_vel").value):
            return
        timeout = float(self.get_parameter("cmd_vel_timeout").value)
        if (
            self._latest_cmd_time
            and (time.monotonic() - self._latest_cmd_time) > timeout
        ):
            self._latest_cmd_time = 0.0
            self._cmd_vel_latest = None
            self._schedule(self._send_stop())

    def _on_sport_cmd(self, msg):
        alias = (msg.data or "").strip()
        if alias not in SPORT_MAP:
            self.get_logger().warning(f"unknown sport command: {msg.data!r}")
            return
        api_id = SPORT_API_IDS.get(alias)
        if api_id is None:
            self.get_logger().warning(
                f"sport command {alias!r} is not supported by the installed "
                f"unitree_webrtc_connect (no SPORT_CMD entry for "
                f"{SPORT_MAP[alias]!r})"
            )
            return
        self._schedule(self._send_sport_id(api_id))

    async def _send_sport_id(self, api_id: int) -> None:
        await self._connection.datachannel.pub_sub.publish_request_new(
            CMD_TOPICS["SPORT_MOD"], {"api_id": api_id}
        )

    async def _send_plain(self, topic: str, data: Any) -> None:
        self._connection.datachannel.pub_sub.publish_without_callback(topic, data)

    async def _send_sport_move(self, x: float, y: float, z: float) -> None:
        options = {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z},
        }
        await self._connection.datachannel.pub_sub.publish_request_new(
            CMD_TOPICS["SPORT_MOD"], options
        )

    async def _send_stop(self) -> None:
        options = {"api_id": SPORT_CMD["StopMove"]}
        await self._connection.datachannel.pub_sub.publish_request_new(
            CMD_TOPICS["SPORT_MOD"], options
        )

    def _srv_lidar(self, request: SetBool.Request, response: SetBool.Response):
        if not self._connection_ready.is_set():
            response.success = False
            response.message = "WebRTC not connected yet"
            return response
        self._lidar_wanted = bool(request.data)
        self._schedule(self._set_lidar(request.data))
        response.success = True
        response.message = (
            f"LiDAR {'enable' if request.data else 'disable'} command sent"
        )
        return response

    async def _set_lidar(self, enabled: bool) -> None:
        if enabled:
            await self._connection.datachannel.disableTrafficSaving(True)
            self._connection.datachannel.set_decoder(
                decoder_type=str(self.get_parameter("lidar_decoder").value)
            )
            self._connection.datachannel.pub_sub.publish_without_callback(
                CMD_TOPICS["ULIDAR_SWITCH"], "on"
            )
        else:
            self._connection.datachannel.pub_sub.publish_without_callback(
                CMD_TOPICS["ULIDAR_SWITCH"], "off"
            )

    def _webrtc_thread_main(self) -> None:
        """Run one connect/subscribe/reconnect session per loop iteration.

        The robot throttles rapid TCP probes to the signaling endpoint, so a
        single connect attempt fails transiently — retry inside the coroutine
        and, once a session is up, rebuild the whole session if the channel
        dies (another client grabbing the single WebRTC slot, WiFi drop,
        robot reboot). Previous bridge died permanently on the first failure.
        """
        while not self._shutdown_requested.is_set():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            try:
                self._async_loop.run_until_complete(self._connect_and_subscribe())
                self._async_loop.run_forever()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self.get_logger().error(f"WebRTC thread failed: {exc}")
            finally:
                # Tear the session down before dropping the loop. _watchdog_loop
                # normally disconnects, but if anything raised between a
                # successful connect() and the watchdog starting (decoder setup,
                # datachannel open, subscribe), nothing else would — and the
                # robot only serves one WebRTC client, so the orphaned session
                # would make every reconnect fail with RobotBusyError.
                async def _teardown() -> None:
                    conn = getattr(self, "_connection", None)
                    # Skip when the session is already down — shutdown()'s
                    # _close disconnects too, and this runs right behind it.
                    if conn is not None and self._connection_alive():
                        try:
                            # Bounded: this runs on the way out and must never
                            # be the reason the thread refuses to exit.
                            await asyncio.wait_for(conn.disconnect(), timeout=5.0)
                        except asyncio.TimeoutError:
                            self.get_logger().warning(
                                "_webrtc_thread_main, disconnect timed out"
                            )
                        except Exception as exc:
                            self.get_logger().error(
                                f"_webrtc_thread_main, disconnect: {exc}"
                            )
                    self._connection = None

                try:
                    pending = asyncio.all_tasks(self._async_loop)
                    for task in pending:
                        task.cancel()
                    self._async_loop.run_until_complete(asyncio.sleep(0))
                except Exception as exc:
                    self.get_logger().error(
                        f"_webrtc_thread_main, run_until_complete: {exc}"
                    )
                try:
                    self._async_loop.run_until_complete(_teardown())
                except Exception as exc:
                    self.get_logger().error(f"_webrtc_thread_main, teardown: {exc}")
                try:
                    self._async_loop.close()
                except Exception as exc:
                    self.get_logger().error(f"_webrtc_thread_main, close: {exc}")
                self._connection_ready.clear()
                self._cmd_vel_latest = None
                self._latest_cmd_time = 0.0
                self._stopping.set()
                if not self._shutdown_requested.is_set():
                    self._publish_connection("disconnected")
            if self._shutdown_requested.is_set():
                break
            for _ in range(50):  # ~5 s pause between sessions
                if self._shutdown_requested.is_set():
                    break
                time.sleep(0.1)

    async def _connect_and_subscribe(self) -> None:
        self._stopping.clear()
        self._latest_cmd_time = 0.0
        self._cmd_vel_latest = None
        last_exc: BaseException | None = None
        ip = self.get_parameter("robot_ip").value
        key = self.get_parameter("aes_128_key").value
        for attempt in range(1, 6):
            if key:
                conn = UnitreeWebRTCConnection(
                    WebRTCConnectionMethod.LocalSTA, ip=ip, aes_128_key=key
                )
            else:
                conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
            try:
                await conn.connect()
                self._connection = conn
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                self.get_logger().warning(
                    f"connect attempt {attempt}/5 failed: {type(exc).__name__}: {exc}"
                )
                try:
                    await conn.disconnect()
                except Exception as exc:
                    self.get_logger().error(f"_connect_and_subscribe: {exc}")
                if attempt < 5:
                    await asyncio.sleep(4.0)
        if last_exc is not None:
            raise last_exc

        await self._connection.datachannel.wait_datachannel_open()

        self._connection.datachannel.set_decoder("native")
        await self._connection.datachannel.disableTrafficSaving(True)

        # Subscribe before any stateful activation command. _ensure_normal_mode
        # below can take 20 s (mode switch + settle), and uSLAM only starts its
        # useful streams after a mapping/localization transition — so anything
        # published before we subscribe is data we simply never see.
        for name, topic in DATA_TOPICS.items():
            if not bool(
                self.get_parameter("enable_slam_subscriptions").value
            ) and name.startswith(("LIDAR_", "GRID_MAP", "SLAM_")):
                continue
            callback = lambda message, topic=topic: self._dispatch_topic(topic, message)
            self._connection.datachannel.pub_sub.subscribe(topic, callback)

        # Sport/pose commands behave predictably only in 'normal' mode
        # (backup go2.ensure_normal_mode); mcf/ai modes silently ignore them.
        # Only asked for when we actually intend to command motion: this query
        # can *switch* the robot's mode, and a data-only bridge must not
        # override whatever else is driving the dog.
        if bool(self.get_parameter("enable_cmd_vel").value):
            try:
                await self._ensure_normal_mode()
            except Exception as exc:
                self.get_logger().warning(f"ensure_normal_mode failed: {exc}")

        self._connection_ready.set()
        self._publish_connection("connected")

        if bool(self.get_parameter("lidar_auto_enable").value) or self._lidar_wanted:
            await self._set_lidar(True)

        self._last_rx_time = time.monotonic()
        self._watchdog_task = asyncio.ensure_future(self._watchdog_loop())
        self._cmd_vel_task = asyncio.ensure_future(self._cmd_vel_loop())

    async def _cmd_vel_loop(self):
        while True:
            # Stop re-sending Move as soon as we begin tearing the session down,
            # otherwise this loop can push one more Move onto the wire *after*
            # shutdown already sent StopMove and leave the robot walking.
            if self._stopping.is_set():
                return
            if not bool(self.get_parameter("enable_cmd_vel").value):
                self._latest_cmd_time = 0.0
                self._cmd_vel_latest = None
                await asyncio.sleep(0.1)
                continue

            cmd = self._cmd_vel_latest
            if cmd is not None:
                try:
                    await self._send_sport_move(*cmd)
                except Exception as exc:
                    self.get_logger().warning(f"cmd_vel send failed: {exc}")

            await asyncio.sleep(0.1)

    async def _ensure_normal_mode(self) -> str | None:
        """Query the motion switcher; if not 'normal', switch and wait.

        api_id 1001 = get current mode, 1002 = set mode (per go2.py backup).
        """
        resp = await asyncio.wait_for(
            self._connection.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
            ),
            timeout=15,
        )
        mode = None
        try:
            if resp["data"]["header"]["status"]["code"] == 0:
                mode = json.loads(resp["data"]["data"])["name"]
        except Exception as exc:
            self.get_logger().error(f"_ensure_normal_mode: {exc}")
        if mode and mode != "normal":
            self.get_logger().info(f"Motion mode is {mode!r}, switching to 'normal'")
            await asyncio.wait_for(
                self._connection.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["MOTION_SWITCHER"],
                    {"api_id": 1002, "parameter": {"name": "normal"}},
                ),
                timeout=15,
            )
            await asyncio.sleep(5)  # give it time to stand into normal mode
            mode = "normal"
        return mode

    def _connection_alive(self) -> bool:
        conn = getattr(self, "_connection", None)
        if conn is None:
            return False
        try:
            if not conn.datachannel.data_channel_opened:
                return False
            cs = getattr(conn.pc, "connectionState", None)
            return cs in (None, "new", "connecting", "connected")
        except Exception:
            return True  # cannot inspect - do not panic

    async def _watchdog_loop(self) -> None:
        while True:
            if self._shutdown_requested.is_set():
                return
            await asyncio.sleep(2.0)
            if self._shutdown_requested.is_set():
                return
            if not self._connection_alive():
                self.get_logger().warning("WebRTC channel lost - reconnecting")
                break
            if self._last_rx_time and (time.monotonic() - self._last_rx_time) > 5.0:
                self.get_logger().warning(
                    "No data received for 5 s - treating connection as dead"
                )
                break
        # Tear the session down; the thread loop builds a fresh one.
        try:
            # Gate the periodic Move resend first, so the StopMove below is
            # really the last motion command on the wire (same ordering as
            # shutdown(): _stopping, then StopMove).
            self._stopping.set()
            self._cmd_vel_latest = None
            self._latest_cmd_time = 0.0
            # Best effort only — the channel that carried the last Move is
            # exactly the thing that just died, so this may not get through. It
            # costs one round trip and it is the difference between the robot
            # stopping and the robot still walking when a link drops.
            if bool(self.get_parameter("enable_cmd_vel").value):
                try:
                    await asyncio.wait_for(self._send_stop(), timeout=2.0)
                except Exception as exc:
                    self.get_logger().warning(
                        f"_watchdog_loop, best-effort stop failed: {exc}"
                    )
            for task in asyncio.all_tasks(self._async_loop):
                if task is not asyncio.current_task():
                    task.cancel()
            conn = getattr(self, "_connection", None)
            if conn is not None:
                try:
                    await asyncio.wait_for(conn.disconnect(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.get_logger().warning("_watchdog_loop, disconnect timed out")
                except Exception as exc:
                    self.get_logger().error(f"_watchdog_loop, disconnect: {exc}")
        except Exception as exc:
            self.get_logger().error(f"_watchdog_loop, cancel: {exc}")
        try:
            self._async_loop.stop()
        except Exception as exc:
            self.get_logger().error(f"_watchdog_loop, stop: {exc}")

    def _publish_connection(self, state: str) -> None:
        msg = String()
        msg.data = state
        self.connection_status_pub.publish(msg)

    def _publish_status(self) -> None:
        snapshot: dict[str, Any] = {}
        with self._lock:
            for name, topic in DATA_TOPICS.items():
                st = self._stats[topic]
                snapshot[name] = {
                    "topic": topic,
                    "count": st.count,
                    "hz": round(st.hz(), 3),
                    "age_s": None if st.age() is None else round(st.age(), 3),
                    "last_bytes": st.bytes_last,
                    "active": st.count > 0
                    and (st.age() is not None and st.age() < 2.0),
                }
        msg = String()
        msg.data = self._encode(snapshot)
        self.topic_status_pub.publish(msg)

    def shutdown(self) -> None:
        self._shutdown_requested.set()
        # Gate the periodic Move resend before anything else, otherwise the
        # StopMove below can race with one more Move.
        self._stopping.set()
        self._cmd_vel_latest = None
        loop = getattr(self, "_async_loop", None)
        conn = getattr(self, "_connection", None)
        if loop and loop.is_running() and conn:

            async def _close():
                try:
                    # Never leave the robot walking on exit. Sent unconditionally
                    # (when cmd_vel is enabled at all) rather than only when a
                    # command is currently live: the last Move may already be on
                    # the robot even if our own timer has just expired.
                    if bool(self.get_parameter("enable_cmd_vel").value):
                        await self._send_stop()
                except Exception as exc:
                    self.get_logger().error(f"shutdown, _send_stop: {exc}")
                try:
                    if (
                        bool(self.get_parameter("lidar_auto_enable").value)
                        or self._lidar_wanted
                    ):
                        await self._set_lidar(False)
                except Exception as exc:
                    self.get_logger().error(f"shutdown, _set_lidar: {exc}")
                try:
                    # Bounded, and loop.stop() runs even if this hangs, so the
                    # WebRTC thread always gets to exit.
                    await asyncio.wait_for(conn.disconnect(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.get_logger().warning("shutdown, disconnect timed out")
                except Exception as exc:
                    self.get_logger().error(f"shutdown, disconnect: {exc}")
                try:
                    loop.stop()
                except Exception as exc:
                    self.get_logger().error(f"shutdown, stop: {exc}")

            asyncio.run_coroutine_threadsafe(_close(), loop)
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            # run_forever returns once loop.stop() lands; give the remaining
            # teardown a moment so it is not cut off by rclpy.shutdown().
            self._thread.join(timeout=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Go2WebRTCBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
