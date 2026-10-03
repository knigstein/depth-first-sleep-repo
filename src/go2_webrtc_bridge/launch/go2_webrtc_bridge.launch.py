import os

from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription

_UNSET = "__go2_webrtc_bridge_unset__"

TUNING_ARGS = (
    "lidar_decoder",
    "imu_source",
    "lidar_auto_enable",
    "enable_slam_subscriptions",
    "enable_cmd_vel",
    "lidar_use_payload_frame",
    "publish_full_arrays",
)

BOOL_ARGS = {
    "lidar_auto_enable",
    "enable_slam_subscriptions",
    "enable_cmd_vel",
    "lidar_use_payload_frame",
    "publish_full_arrays",
}


def generate_launch_description():
    def build(context, *args, **kwargs):
        lc = LaunchConfiguration

        def given(name):
            try:
                v = lc(name).perform(context)
            except Exception:
                return None
            return None if v == _UNSET else v

        params = {}

        for name in TUNING_ARGS:
            v = given(name)
            if v is None:
                continue
            if name in BOOL_ARGS:
                params[name] = ParameterValue(lc(name), value_type=bool)
            else:
                params[name] = v
        for name in ("robot_ip", "aes_128_key"):
            v = given(name)
            if v is not None:
                params[name] = v

        excl = given("exclude_raw_fields")
        if excl is not None:
            params["exclude_raw_fields"] = [
                f.strip() for f in excl.split(",") if f.strip()
            ]

        config = None
        try:
            from ament_index_python.packages import get_package_share_directory

            config = os.path.join(
                get_package_share_directory("go2_webrtc_bridge"),
                "config",
                "go2_webrtc_bridge.yaml",
            )
        except Exception:
            print("Config error.")
            config = None

        return [
            Node(
                package="go2_webrtc_bridge",
                executable="bridge_node",
                name="go2_webrtc_bridge",
                output="screen",
                parameters=([config] if config else []) + [params],
            )
        ]

    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=_UNSET) for name in TUNING_ARGS]
        + [
            DeclareLaunchArgument("exclude_raw_fields", default_value=_UNSET),
            OpaqueFunction(function=build),
        ]
    )
