"""Defines a visualization stub that broadcasts to ROS topics instead.

Typical usage:
  vis = RerunVis(intrinsics_3x3)
  vis.log_pose(pose_4x4, layer="cam0")
  vis.log_img(img, layer="rgb_img", pose_layer="cam0")
  vis.step()
"""
from functools import partial
import threading
from typing_extensions import override
from typing import Tuple
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

try:
  import rclpy
  from rclpy.executors import SingleThreadedExecutor
  from rclpy.node import Node
  from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
  from sensor_msgs.msg import Image, PointCloud2
  from visualization_msgs.msg import Marker, MarkerArray
  from geometry_msgs.msg import PoseStamped, Point
  from rayfronts import ros_utils
except ModuleNotFoundError:
  logger.warning("ROS2 modules not found !")

from rayfronts.visualizers.base import Mapping3DVisualizer
from rayfronts import geometry3d as g3d, feat_compressors


class Ros2Vis(Mapping3DVisualizer):
  """Broadcsasts to ROS2 topics instead of visualizing. 
  
  Topics can be visualized with RVIZ2

  Attributes:
    intrinsics_3x3: See base.
    img_size: See base.
    base_point_size: See base.
    global_heat_scale: See base.
    device: See base.
    feat_compressor: See base.
    time_step: See base.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               img_size: Tuple[int] = None,
               base_point_size: float = None,
               global_heat_scale: bool = False,
               feat_compressor: feat_compressors.FeatCompressor = None,
               topic_prefix: str = "rayfronts",
               reliability: str = "best_effort",
               **kwargs):
    """

    Args:
      intrinsics_3x3: See base.
      img_size: See base.
      base_point_size: See base.
      global_heat_scale: See base.
      feat_compressor: See base.
      topic_prefix: Prefix for the ROS2 topics.
      reliability: Reliability of the ROS2 topics. Can be "reliable" or
        "best_effort".
    """
    super().__init__(intrinsics_3x3, img_size, base_point_size,
                     global_heat_scale, feat_compressor)

    if not rclpy.ok():
      rclpy.init()
    self._rosnode = Node("rayfronts_vis")
    self.topic_prefix = topic_prefix
    self._height = None
    self._width = None

    self._rdf2flu_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform("rdf", "flu")).to(self.device)
    self._prev_poses_4x4 = dict()
    # We initialize the publishers on the fly since topic names rely on the
    # layer names only known at logging time.
    self._publishers = dict()

    if reliability == "reliable":
      self._reliability = ReliabilityPolicy.RELIABLE
    elif reliability == "best_effort":
      self._reliability = ReliabilityPolicy.BEST_EFFORT
    else:
      raise ValueError("Reliability must be 'reliable' or 'best_effort'.")

    self._qos_profile = QoSProfile(
      reliability=self._reliability,
      depth=5,
      history=HistoryPolicy.KEEP_LAST)

    self._shutdown_event = threading.Event()
    self._ros_executor = SingleThreadedExecutor()
    self._ros_executor.add_node(self._rosnode)
    self._spin_thread = threading.Thread(
      target=self._spin_ros, name="rayfronts_vis_spinner")
    self._spin_thread.daemon = True
    self._spin_thread.start()

    logger.info("ROS2 Visualizer initialized successfully.")

  def _spin_ros(self):
    try:
      self._ros_executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass

  def _get_publisher(self, key: str, msg_type):
    try:
      return self._publishers[key]
    except KeyError:
      pub =  self._rosnode.create_publisher(
        msg_type,
        f"{self.topic_prefix}/{key}",
        self._qos_profile)
      self._publishers[key] = pub
      logger.info("Publisher %s/%s initialized.", self.topic_prefix, key)
      return pub

  def _has_subscriber(self, pub) -> bool:
    """Return True if the publisher has at least one subscriber."""
    return pub.get_subscription_count() > 0

  @override
  def log_img(self,
              img: torch.FloatTensor,
              layer: str = "img",
              pose_layer: str = "pose") -> None:
    k = f"{pose_layer}/{layer}"
    pub = self._get_publisher(k, Image)
    if not self._has_subscriber(pub):
      return
    self._height, self._width, _ = img.shape
    ros_img = ros_utils.numpy_to_image(
      (img.cpu().numpy()*255).astype("uint8"), encoding="rgb8")
    pub.publish(ros_img)

  @override
  def log_pose(self,
               pose_4x4: torch.FloatTensor,
               layer: str = "pose") -> None:
    k = f"{layer}/pose"
    pub = self._get_publisher(k, PoseStamped)
    if not self._has_subscriber(pub):
      return
    ros_pose = PoseStamped()
    pose_4x4 = g3d.transform_pose_4x4(pose_4x4.cpu(),
                                      self._rdf2flu_transform.cpu())
    ros_pose.pose = ros_utils.numpy_to_pose(pose_4x4.numpy())
    ros_pose.header.frame_id = "map"
    pub.publish(ros_pose)

  @override
  def log_pc(self,
             pc_xyz: torch.FloatTensor,
             pc_rgb: torch.FloatTensor = None,
             pc_radii: torch.FloatTensor = None,
             layer: str = "pc"):
    if pc_xyz.shape[0] == 0:
      return
    pub = self._get_publisher(layer, PointCloud2)
    if not self._has_subscriber(pub):
      return
    pc_arr = np.recarray(
      (pc_xyz.shape[0],),
      dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32),
             ("r", np.uint8), ("g", np.uint8), ("b", np.uint8)])
    pc_xyz = g3d.transform_points(
      pc_xyz.to(self.device), self._rdf2flu_transform).cpu()
    pc_arr.x = pc_xyz[:, 0].numpy()
    pc_arr.y = pc_xyz[:, 1].numpy()
    pc_arr.z = pc_xyz[:, 2].numpy()
    if pc_rgb is not None:
      if pc_rgb.shape[1] == 4: # If alpha channel is present
        pc_rgb[:, :3] = pc_rgb[:, :3] * pc_rgb[:, 3:4]

      pc_rgb = (pc_rgb.cpu().numpy() * 255).astype("uint8")
      pc_arr.r = pc_rgb[:, 0]
      pc_arr.g = pc_rgb[:, 1]
      pc_arr.b = pc_rgb[:, 2]
    else:
      pc_arr.r = 255
      pc_arr.g = 255
      pc_arr.b = 255
    ros_pc = ros_utils.array_to_pointcloud2(
      ros_utils.merge_rgb_fields(pc_arr), frame_id="map")
    pub.publish(ros_pc)

  @override
  def log_arrows(self, arr_origins, arr_dirs, arr_rgb = None, layer="arrows"):
    N = arr_origins.shape[0]
    if N == 0:
      return
    pub = self._get_publisher(layer, MarkerArray)
    if not self._has_subscriber(pub):
      return
    arr_ends = arr_origins + arr_dirs
    arr_origins = g3d.transform_points(
      arr_origins.to(self.device),
      self._rdf2flu_transform).cpu().numpy().astype("float")
    arr_ends = g3d.transform_points(
      arr_ends.to(self.device),
      self._rdf2flu_transform).cpu().numpy().astype("float")

    if arr_rgb is not None:
      arr_rgb = arr_rgb.cpu().numpy().astype("float")
    marker_array = MarkerArray()
    for i in range(N):
      arrow = Marker()
      arrow.header.frame_id = 'map'
      arrow.ns = layer
      arrow.id = i
      arrow.type = Marker.ARROW
      arrow.action = Marker.ADD
      p0 = arr_origins[i]
      p1 = arr_ends[i]
      arrow.points=[Point(x=p0[0], y=p0[1], z=p0[2]),
                    Point(x=p1[0], y=p1[1], z=p1[2])]
      arrow.scale.x = self.base_point_size # shaft diameter
      arrow.scale.y = self.base_point_size*2 # head diameter
      arrow.scale.z = self.base_point_size*2 # head length
      if arr_rgb is not None:
        arrow.color.r = arr_rgb[i, 0]
        arrow.color.g = arr_rgb[i, 1]
        arrow.color.b = arr_rgb[i, 2]
        arrow.color.a = 1.0
      else:
        arrow.color.r = 1.0
        arrow.color.g = 0.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0
      marker_array.markers.append(arrow)
    pub.publish(marker_array)

  @override
  def log_box(self, box_mins, box_maxs, layer = ""):
    # TODO: Implement box logging
    pass

  @override
  def step(self):
    super().step()

  def shutdown(self):
    self._rosnode.context.try_shutdown()
    self._shutdown_event.set()
    
