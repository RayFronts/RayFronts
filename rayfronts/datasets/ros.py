"""Defines ROS related datasets

Typical usage example:
  dataset = Ros2Subscriber(
    rgb_topic="/robot/front_stereo/left/image_rect_color",
    pose_topic="/robot/front_stereo/pose",
    disparity_topic="/robot/front_stereo/disparity/disparity_image",
    intrinsics_topic="/robot/front_stereo/left/camera_info",
    src_coord_system="flu")

  dataloader = torch.utils.data.DataLoader(
    self.dataset, batch_size=4)

  for i, batch in enumerate(dataloader):
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
"""

import os
import threading
import queue
from typing_extensions import override, deprecated
from typing import Tuple, Union
from collections import OrderedDict
import logging
import json

logger = logging.getLogger(__name__)

import numpy as np
from scipy.spatial.transform import Rotation
import torch

try:
  import rclpy
  from rclpy.node import Node
  from rclpy.executors import SingleThreadedExecutor
  from rclpy.qos import QoSProfile, ReliabilityPolicy
  import message_filters
  from sensor_msgs.msg import Image, CameraInfo, PointCloud, PointCloud2
  from geometry_msgs.msg import PoseStamped
  from stereo_msgs.msg import DisparityImage
  from nav_msgs.msg import Odometry
  from rayfronts.ros_utils import (
    image_to_numpy,
    pose_to_numpy,
    pointcloud2_to_array,
  )
except ModuleNotFoundError:
  logger.warning("ROS2 modules not found !")

try:
  import cv2
except ModuleNotFoundError:
  cv2 = None

from rayfronts.datasets.base import PosedRgbdDataset
from rayfronts import geometry3d as g3d

class Ros2Subscriber(PosedRgbdDataset):
  """ROS2 subscriber node to subscribe to posed RGBD topics.
  
  Attributes:
    intrinsics_3x3:  See base.
    rgb_h: See base.
    rgb_w: See base.
    depth_h: See base.
    depth_w: See base.
    frame_skip: See base.
    interp_mode: See base.
  """
  def __init__(self,
               rgb_topic,
               pose_topic,
               rgb_resolution=None,
               depth_resolution=None,
               disparity_topic = None,
               depth_topic = None,
               confidence_topic = None,
               point_cloud_topic = None,
               intrinsics_topic = None,
               intrinsics_file = None,
               src_coord_system = "flu",
               frame_skip = 0,
               interp_mode="bilinear"):
    """

    There can be three sources of depth:
    1- Disparity topic
    2- Depth topic
    3- Point cloud topic (will be projected using pose and intrinsics)
       Using the point cloud through this rgbd loader is inefficient as points
       will be projected then likely unprojected again in the mapping system.

    Args:
      rgb_resolution: See base.
      depth_resolution: See base.
      rgb_topic: Topic containing RGB images of type sensor_msgs/msg/Image
      pose_topic: Topic containing poses of type geometry_msgs/msg/PoseStamped
      disparity_topic: Topic containing disparity images of type
        stereo_msgs/DisparityImage.
      depth_topic: Topic containing depth images of type sensor_msgs/msg/Image
        with 32FC1 encoding in metric scale.
      confidence_topic: (Optional) Topic containing confidence in depth values.
        Message type: sensor_msgs/msg/Image.
      point_cloud_topic: Topic containing point cloud of type
        sensor_msgs/msg/PointCloud.
      intrinsics_topic: Topic containing intrinsics information from messages
        of type sensor_msgs/msg/CameraInfo. Will be used at initialization only.
      intrinsics_file: Path to json file containing intrinsics with the
        following keys, fx, fy, cx, cy, w, h. This will be prioritized
        over the intrinsics topic.
      src_coord_system: A string of 3 letters describing the camera coordinate
        system in r/l u/d f/b in any order. (e.g, rdf, flu, rfu)
      frame_skip: See base.
      interp_mode: See base.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    if point_cloud_topic is not None and disparity_topic is not None:
      raise ValueError("You cannot set both the point cloud topic and "
                       "disparity topic as that will lead to an ambiguous "
                       "source of depth information.")

    if intrinsics_file is None and intrinsics_topic is None:
      raise ValueError("Must provide a source for the intrinsics")

    self._shutdown_event = threading.Event()

    self.f = 0
    self.intrinsics_3x3 = None
    if intrinsics_file is not None:
      intrinsics_topic = None
      with open(intrinsics_file, "r") as f:
        int_json = json.load(f)
        self.intrinsics_3x3 = torch.tensor([
          [int_json["fx"], 0, int_json["cx"]],
          [0, int_json["fy"], int_json["cy"]],
          [0, 0, 1]
        ])
    self._intrinsics_loaded_cond = threading.Condition()
    self.src2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform(src_coord_system, "rdf"))

    # Setup ros node
    msg_str_to_type = OrderedDict(
      rgb = Image,
      pose = PoseStamped,
      disp = DisparityImage,
      depth = Image,
      pc = PointCloud,
      conf = Image,
    )
    self._topics = [rgb_topic, pose_topic, disparity_topic, depth_topic,
    point_cloud_topic, confidence_topic]
    if not rclpy.ok():
      rclpy.init()
    self._rosnode = Node("rayfronts_input_streamer")

    if intrinsics_topic is not None:
      self.intrinsics_sub = self._rosnode.create_subscription(
        CameraInfo, intrinsics_topic, self._set_intrinsics_from_msg,
        QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1))

    self._subs = OrderedDict()
    for i, t in enumerate(self._topics):
      msg_str = list(msg_str_to_type.keys())[i]
      if t is not None:
        self._subs[msg_str] = message_filters.Subscriber(
          self._rosnode, msg_str_to_type[msg_str], t, qos_profile = 10)
    self._frame_msgs_queue = queue.Queue(maxsize=10)

    self._time_sync = message_filters.ApproximateTimeSynchronizer(
      list(self._subs.values()), queue_size = 10, slop = 0.01,
      allow_headerless = False)
    self._time_sync.registerCallback(self._buffer_frame_msgs)

    self._ros_executor = SingleThreadedExecutor()
    self._ros_executor.add_node(self._rosnode)
    self._spin_thread = threading.Thread(
      target=self._spin_ros, name="rayfronts_input_stream_spinner")
    self._spin_thread.daemon = True

    if intrinsics_topic is not None:
      self._intrinsics_loaded_cond.acquire()
      self._spin_thread.start()
      logger.info("Waiting for intrinsics to be published..")
      while True:
        try:
          r = self._intrinsics_loaded_cond.wait(2)
        except KeyboardInterrupt as e:
          self.shutdown()
          raise e
        if r:
          self._rosnode.destroy_subscription(self.intrinsics_sub)
          break
    else:
      self._spin_thread.start()

    logger.info("Ros2Subscriber initialized successfully.")

  def _spin_ros(self):
    try:
      self._ros_executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass

  def _set_intrinsics_from_msg(self, msg):
    self._intrinsics_loaded_cond.acquire()
    self.intrinsics_3x3 = torch.tensor(msg.k, dtype = torch.float).reshape(3,3)
    self.original_h = msg.height
    self.original_w = msg.width
    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    logger.info("Loaded intrinsics: \n%s", str(self.intrinsics_3x3))
    self._intrinsics_loaded_cond.notify()
    self._intrinsics_loaded_cond.release()

  def _buffer_frame_msgs(self, *msgs):
    if self.frame_skip <= 0 or self.f % (self.frame_skip+1) == 0:
      if self._frame_msgs_queue.full():
        self._frame_msgs_queue.get() # Discard and priortize newer.
      self._frame_msgs_queue.put(msgs)
    self.f += 1

  def __iter__(self):
    while True:
      msgs = None
      try:
        msgs = self._frame_msgs_queue.get(block=True, timeout=2)
      except queue.Empty:
        if not self._shutdown_event.is_set():
          continue

      if msgs is None:
        return

      msgs = dict(zip(self._subs.keys(), msgs))

      # Parse RGB
      bgra_img = image_to_numpy(msgs["rgb"]).astype("float") / 255
      bgr_img = bgra_img[..., :3]
      rgb_img = torch.tensor(bgr_img[..., (2,1,0)],
                             dtype=torch.float).permute(2, 0, 1)

      # Parse Pose
      src_pose_4x4 = torch.tensor(
        pose_to_numpy(msgs["pose"].pose), dtype=torch.float)
      rdf_pose_4x4 = g3d.transform_pose_4x4(
        src_pose_4x4, self.src2rdf_transform)

      if 'depth' in msgs.keys():
        depth_img = image_to_numpy(msgs["depth"])
        depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)
      elif "disp" in msgs.keys():
        # TODO: Why is disparity negative in ros2 zedx and why is max and min
        # flipped? Not sure if this is correct ros2 zedx behaviour but will
        # correct those here for now.
        disparity_img = -image_to_numpy(msgs["disp"].image)
        min_disp = msgs["disp"].max_disparity
        max_disp = msgs["disp"].min_disparity

        focal_length = msgs["disp"].f
        stereo_baseline = msgs["disp"].t
        depth_img = focal_length*stereo_baseline/disparity_img
        depth_img[disparity_img < min_disp] = np.inf
        depth_img[disparity_img > max_disp] = -np.inf
        depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)

      elif "pc" in msgs.keys():
        # TODO: This should be more efficient than a for loop
        pc_xyz = torch.tensor([[p.x, p.y, p.z] for p in msgs["pc"].points])
        if len(pc_xyz) == 0:
          continue
        pc_xyz_homo = g3d.pts_to_homogen(pc_xyz)
        pc_xyz_homo = g3d.transform_points_homo(pc_xyz_homo,
                                                self.src2rdf_transform)
        pc_xyz_homo_cam = pc_xyz_homo @ torch.linalg.inv(rdf_pose_4x4).T
        pc_xyz_homo_cam /= pc_xyz_homo_cam[:, -1].unsqueeze(-1)
        pc_depth = pc_xyz_homo_cam[:, 2]
        ch_n2i = {ch.name: i for i,ch in enumerate(msgs["pc"].channels)}
        if "kp_u" in ch_n2i and "kp_v" in ch_n2i:
          u = torch.tensor(msgs["pc"].channels[ch_n2i["kp_u"]].values,
                       dtype=torch.int32)
          v = torch.tensor(msgs["pc"].channels[ch_n2i["kp_v"]].values,
                          dtype=torch.int32)
        else:
          uv = pc_xyz_homo_cam[:, :3] @ self.intrinsics_3x3.T
          uv /= uv[:, -1]
          u = uv[:, 0]
          v = uv[:, 1]

        depth_img = torch.ones_like(rgb_img[0:1])*torch.nan
        mask = torch.logical_and(v < rgb_img.shape[1], u < rgb_img.shape[2])
        depth_img[:, v[mask], u[mask]] = pc_depth[mask]
        depth_img[depth_img < 0] = -torch.infs
      else:
        raise ValueError("Expected at least a depth or a disparity or point cloud topic")

      # Parse confidence map if it exists
      conf_img = None
      if "conf" in msgs:
        conf_img = image_to_numpy(msgs["conf"]).astype("float")
        conf_img = 1 - (torch.tensor(conf_img, dtype=torch.float) / 100)
        conf_img = conf_img.unsqueeze(0)

      if (self.rgb_h != rgb_img.shape[-2] or
          self.rgb_w != rgb_img.shape[-1]):
        rgb_img = torch.nn.functional.interpolate(rgb_img.unsqueeze(0),
          size=(self.rgb_h, self.rgb_w), mode=self.interp_mode,
          antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

      if (self.depth_h != depth_img.shape[-2] or
          self.depth_w != depth_img.shape[-1]):
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0),
          size=(self.depth_h, self.depth_w),
          mode="nearest-exact").squeeze(0)
        if conf_img is not None:
          conf_img = torch.nn.functional.interpolate(conf_img.unsqueeze(0),
            size=(self.depth_h, self.depth_w),
            mode="nearest-exact").squeeze(0)

      if torch.sum(~depth_img.isnan()) == 0:
        logger.warning("Ignoring received depth frame with no valid values")
        continue
      frame_data = dict(rgb_img = rgb_img, depth_img = depth_img,
                        pose_4x4 = rdf_pose_4x4)

      if conf_img is not None:
        frame_data["confidence_map"] = conf_img

      yield frame_data

  def shutdown(self):
    self._shutdown_event.set()
    self._rosnode.context.try_shutdown()
    logger.info("Ros2Subscriber shutdown.")


import re

def _strip_c_comments(text):
  """Remove C-style /* ... */ and /** ... **/ comments from text."""
  return re.sub(r'/\*[\s\S]*?\*/', '', text)


def _parse_extrinsics_conf(path):
  """Parse a ModalAI extrinsics.conf file.

  The file is JSON wrapped in C-style comments. Each entry in the
  ``extrinsics`` array has ``parent``, ``child``,
  ``T_child_wrt_parent`` (translation in metres) and
  ``RPY_parent_to_child`` (Tait-Bryan intrinsic XYZ in degrees).

  Returns:
    A list of dicts, each with keys: parent, child, R_4x4 (child-to-parent
    4x4 rigid transform).
  """
  with open(path, "r") as f:
    raw = f.read()
  cleaned = _strip_c_comments(raw).strip()
  data = json.loads(cleaned)
  entries = data.get("extrinsics", data if isinstance(data, list) else [])

  transforms = []
  for e in entries:
    rpy = e["RPY_parent_to_child"]  # [roll, pitch, yaw] degrees
    t = e["T_child_wrt_parent"]     # [tx, ty, tz] metres

    # Build child-to-parent rotation matrix.
    # ModalAI convention: intrinsic XYZ body-fixed rotation sequence.
    # R(roll, pitch, yaw) = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    # This R maps a vector in the child frame to the parent frame.
    # scipy intrinsic 'XYZ' gives exactly Rx(a) @ Ry(b) @ Rz(c):
    R = Rotation.from_euler(
      'XYZ', rpy, degrees=True).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    transforms.append(dict(
      parent=e["parent"],
      child=e["child"],
      T_child_to_parent=torch.tensor(T, dtype=torch.float),
    ))
  return transforms


def _build_extrinsic_chain(transforms, child_name, root_name="body"):
  """Chain extrinsic entries to get T_child_to_root (child-to-root 4x4).

  Walks the parent graph from *child_name* up to *root_name* by looking
  up matching ``(parent, child)`` entries, composing
  ``T_parent_to_root @ T_child_to_parent`` at each step.

  Args:
    transforms: List returned by ``_parse_extrinsics_conf``.
    child_name: The sensor frame name (e.g. ``"hires_front"``).
    root_name: Target root frame (default ``"body"``).

  Returns:
    4x4 float tensor mapping child frame to root frame.
  """
  lookup = {e["child"]: e for e in transforms}
  T = torch.eye(4, dtype=torch.float)
  current = child_name
  visited = set()
  while current != root_name:
    if current in visited:
      raise ValueError(f"Cycle detected in extrinsics chain at '{current}'")
    visited.add(current)
    if current not in lookup:
      raise ValueError(
        f"Cannot reach '{root_name}' from '{child_name}': "
        f"no entry with child='{current}'")
    entry = lookup[current]
    T = entry["T_child_to_parent"] @ T
    current = entry["parent"]
  return T


class StarlingMaxSubscriber(PosedRgbdDataset):
  """ROS2 subscriber for Starling Max (VOXL 2): RGB + ToF depth.

  Subscribes to RGB, body pose, and a depth source; registers depth to
  the RGB frame using body pose and extrinsics parsed from a ModalAI
  ``extrinsics.conf`` file. Outputs the standard PosedRgbdDataset
  contract: ``rgb_img``, ``depth_img``, ``pose_4x4``, ``intrinsics_3x3``.

  Two mutually-exclusive depth sources are supported:

  * **Depth image** -- set ``depth_topic`` and ``depth_intrinsics_file``.
    The depth image is unprojected to 3-D using the depth intrinsics and
    the ToF-to-world pose, then re-projected into the RGB camera.
  * **Point cloud** -- set ``point_cloud_topic`` (sensor_msgs/PointCloud2).
    The point cloud is assumed to be in the ``src_coord_system`` world
    frame (e.g. FLU). If it is instead in the ToF sensor frame, set
    ``point_cloud_frame="sensor"``. Uses ``ros_utils.pointcloud2_to_array``.

  Coordinate conventions
  ~~~~~~~~~~~~~~~~~~~~~~
  * The body pose from ROS is in ``src_coord_system`` (default FLU).
  * The extrinsics.conf defines sensor frames in the body's native
    convention, typically FRD for VOXL / PX4.  Set
    ``extrinsics_coord_system`` accordingly.
  * Internally, everything is converted to RDF (OpenCV) before being
    passed to the mapping pipeline.
  """

  def __init__(self,
               rgb_topic,
               pose_topic,
               extrinsics_conf_path,
               intrinsics_file,
               # Depth source A: depth image + intrinsics
               depth_topic=None,
               depth_intrinsics_file=None,
               depth_max_range=3.0,
               # Depth source B: point cloud (PointCloud2 only)
               point_cloud_topic=None,
               point_cloud_frame="world",
               rgb_frame_name="hires_front",
               depth_frame_name="tof",
               extrinsics_coord_system="frd",
               confidence_topic=None,
               src_coord_system="flu",
               rgb_resolution=None,
               depth_resolution=None,
               frame_skip=0,
               interp_mode="bilinear"):
    """
    Args:
      rgb_topic: Image topic for the RGB camera (sensor_msgs/Image).
      pose_topic: Body pose topic (geometry_msgs/PoseStamped).
      extrinsics_conf_path: Path to the ModalAI extrinsics.conf file.
        Used to derive T_body_rgb and T_body_depth automatically.
      intrinsics_file: JSON file with fx, fy, cx, cy, w, h for the RGB
        camera.
      depth_topic: Image topic for the ToF depth camera. Requires
        ``depth_intrinsics_file``. Mutually exclusive with
        ``point_cloud_topic``.
      depth_intrinsics_file: JSON file with fx, fy, cx, cy, w, h for
        the depth (ToF) sensor.  Required when ``depth_topic`` is set.
      depth_max_range: When the depth image is uint8, values are normalized
        as (v/255)*depth_max_range to get depth in metres. Default 3.0.
      point_cloud_topic: Point cloud topic (sensor_msgs/PointCloud2).
        Mutually exclusive with ``depth_topic``.
      point_cloud_frame: Coordinate frame of the point cloud data.
        ``"world"`` (default) means points are already in the world
        frame matching ``src_coord_system``.  ``"sensor"`` means points
        are in the depth sensor's local frame and will be transformed
        to world using the depth extrinsics and body pose.
      rgb_frame_name: Child frame name for the RGB camera in the extrinsics
        file (default ``"hires_front"``).
      depth_frame_name: Child frame name for the depth sensor in the
        extrinsics file (default ``"tof"``).
      extrinsics_coord_system: 3-letter body-frame convention used in the
        extrinsics file (default ``"frd"`` for VOXL / PX4).
      confidence_topic: Optional ToF confidence image topic.
      src_coord_system: Body-frame convention of the ROS pose topic.
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    # ---- Validate depth source ----
    has_depth_img = depth_topic is not None
    has_pc = point_cloud_topic is not None
    if not has_depth_img and not has_pc:
      raise ValueError("Provide depth_topic + depth_intrinsics_file, "
                       "or point_cloud_topic.")
    if has_depth_img and has_pc:
      raise ValueError("Cannot set both depth_topic and "
                       "point_cloud_topic.")
    if has_depth_img and depth_intrinsics_file is None:
      raise ValueError("depth_intrinsics_file is required when "
                       "depth_topic is set.")

    self._depth_max_range = float(depth_max_range)
    self._use_point_cloud = has_pc
    self._point_cloud_frame = point_cloud_frame
    self._shutdown_event = threading.Event()
    self.f = 0

    # ---- Extrinsics: parse conf and build camera-to-body chains ----
    ext_entries = _parse_extrinsics_conf(extrinsics_conf_path)
    T_rgb_to_body_ext = _build_extrinsic_chain(
      ext_entries, rgb_frame_name, "body")
    T_depth_to_body_ext = _build_extrinsic_chain(
      ext_entries, depth_frame_name, "body")

    # Convert body side from extrinsics convention to RDF.
    ext_to_rdf = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform(extrinsics_coord_system, "rdf"))
    self.T_body_rgb = ext_to_rdf @ T_rgb_to_body_ext
    self.T_body_depth = ext_to_rdf @ T_depth_to_body_ext

    # Transform to convert body pose from src_coord_system to RDF.
    self.src2rdf = g3d.mat_3x3_to_4x4(
        g3d.get_coord_system_transform(src_coord_system, "rdf"))

    logger.info("Extrinsics loaded from %s", extrinsics_conf_path)
    logger.info("T_body_rgb (ext→RDF):\n%s", self.T_body_rgb)
    logger.info("T_body_depth (ext→RDF):\n%s", self.T_body_depth)

    # ---- RGB intrinsics from file ----
    with open(intrinsics_file, "r") as f:
      int_json = json.load(f)

    # We have four types of intrinsics:
    # 1. Pre-rectification RGB intrinsics
    # 2. Post-rectification RGB intrinsics
    #    This is used for projecting world points into the RGB view to get the
    #    registered depth image.
    # 3. Depth img intrinsics in case we use depth image as depth source
    # 4. The intrinsics stored in self.intrinsics_3x3 which refers to the
    #    intrinsics of the depth image that is registered to the RGB image.

    # Pre-rectification RGB intrinsics
    fx = int_json["fx"]
    fy = int_json["fy"]
    cx = int_json["cx"]
    cy = int_json["cy"]
    self._rgb_K_orig = np.array(
      [[fx, 0.0, cx],
       [0.0, fy, cy],
       [0.0, 0.0, 1.0]],
      dtype=np.float32)

    # Optional distortion coefficients.
    dist = int_json.get("dist_coeffs", None)
    self._rgb_dist_coeffs = None
    if dist is not None:
      self._rgb_dist_coeffs = np.array(dist, dtype=np.float32).reshape(-1, 1)

    orig_h = int_json.get("h", int_json.get("height", -1))
    orig_w = int_json.get("w", int_json.get("width", -1))

    # Compute a rectified camera matrix that crops away black borders.
    self._rgb_newK = None # Post-rectification RGB intrinsics
    self._rgb_undist_roi = None  # (x, y, w, h)
    base_rgb_h, base_rgb_w = orig_h, orig_w
    if self._rgb_dist_coeffs is not None and orig_w > 0 and orig_h > 0:
      self._rgb_newK, self._rgb_undist_roi = cv2.getOptimalNewCameraMatrix(
        self._rgb_K_orig, self._rgb_dist_coeffs,
        (orig_w, orig_h),
        alpha=0,
        centerPrincipalPoint=True,
      )
      x, y, rw, rh = self._rgb_undist_roi
      base_rgb_w, base_rgb_h = rw, rh
      assert base_rgb_w > 0 and base_rgb_h > 0

    K_rect = self._rgb_newK if self._rgb_newK is not None else self._rgb_K_orig

    # Post-rectification RGB intrinsics
    self._rgb_intrinsics_3x3 = torch.tensor(K_rect, dtype=torch.float)

    self.rgb_h = base_rgb_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = base_rgb_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.rgb_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.rgb_w if self.depth_w <= 0 else self.depth_w

    # Depth img intrinsics in case we use depth image as depth source
    self._depth_intrinsics_3x3 = None
    if has_depth_img:
      with open(depth_intrinsics_file, "r") as f:
        d_json = json.load(f)
      self._depth_intrinsics_3x3 = torch.tensor([
          [d_json["fx"], 0, d_json["cx"]],
          [0, d_json["fy"], d_json["cy"]],
          [0, 0, 1]
      ], dtype=torch.float)
      depth_orig_h = d_json.get("h", d_json.get("height", base_rgb_h))
      depth_orig_w = d_json.get("w", d_json.get("width", base_rgb_w))
      if depth_orig_h <= 0:
        depth_orig_h = base_rgb_h
      if depth_orig_w <= 0:
        depth_orig_w = base_rgb_w


    if self.rgb_h != base_rgb_h or self.rgb_w != base_rgb_w:
      self._rgb_intrinsics_3x3[0, :] *= self.rgb_w / base_rgb_w
      self._rgb_intrinsics_3x3[1, :] *= self.rgb_h / base_rgb_h

    # The intrinsics stored in self.intrinsics_3x3 which refers to the
    # intrinsics of the depth image that is registered to the RGB image.
    self.intrinsics_3x3 = self._rgb_intrinsics_3x3.clone()

    if self.depth_h != self.rgb_h or self.depth_w != self.rgb_w:
      self.intrinsics_3x3[0, :] *= self.depth_w / self.rgb_w
      self.intrinsics_3x3[1, :] *= self.depth_h / self.rgb_h

    # ---- ROS setup ----
    if not rclpy.ok():
      rclpy.init()
    self._rosnode = Node("rayfronts_starlingmax_rgbd")

    # BEST_EFFORT to match typical VOXL publishers (avoids QoS mismatch)
    _qos = QoSProfile(
      reliability=ReliabilityPolicy.BEST_EFFORT,
      depth=10,
    )
    self._subs = OrderedDict(
        rgb=message_filters.Subscriber(
          self._rosnode, Image, rgb_topic, qos_profile=_qos),
        pose=message_filters.Subscriber(
          self._rosnode, PoseStamped, pose_topic, qos_profile=_qos),
    )
    if has_depth_img:
      self._subs["depth"] = message_filters.Subscriber(
          self._rosnode, Image, depth_topic, qos_profile=_qos)
    else:
      self._subs["pc"] = message_filters.Subscriber(
          self._rosnode, PointCloud2, point_cloud_topic, qos_profile=_qos)
    if confidence_topic is not None:
      self._subs["conf"] = message_filters.Subscriber(
          self._rosnode, Image, confidence_topic, qos_profile=_qos)

    self._frame_msgs_queue = queue.Queue(maxsize=10)
    self._time_sync = message_filters.ApproximateTimeSynchronizer(
        list(self._subs.values()), queue_size=10, slop=0.1,
        allow_headerless=False)
    self._time_sync.registerCallback(self._buffer_frame_msgs)

    self._ros_executor = SingleThreadedExecutor()
    self._ros_executor.add_node(self._rosnode)
    self._spin_thread = threading.Thread(
        target=self._spin_ros, name="rayfronts_starlingmax_spinner")
    self._spin_thread.daemon = True
    self._spin_thread.start()

    depth_src = point_cloud_topic if has_pc else depth_topic
    logger.info("StarlingMaxSubscriber initialized (depth source: %s).",
                depth_src)

  # ---------- ROS helpers ----------

  def _spin_ros(self):
    try:
      self._ros_executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass

  def _buffer_frame_msgs(self, *msgs):
    if self.frame_skip <= 0 or self.f % (self.frame_skip + 1) == 0:
      if self._frame_msgs_queue.full():
        self._frame_msgs_queue.get()
      self._frame_msgs_queue.put(msgs)
    self.f += 1

  # ---------- Iterator ----------

  def __iter__(self):
    while True:
      try:
        msgs = self._frame_msgs_queue.get(block=True, timeout=2)
      except queue.Empty:
        if self._shutdown_event.is_set():
          return
        continue
      msgs = dict(zip(self._subs.keys(), msgs))

      # ---- Body pose → RDF, then derive camera poses ----
      ros_pose = msgs["pose"].pose
      body_4x4 = torch.tensor(pose_to_numpy(ros_pose), dtype=torch.float)
      body_rdf = g3d.transform_pose_4x4(body_4x4, self.src2rdf)
      pose_rgb = body_rdf @ self.T_body_rgb
      pose_depth = body_rdf @ self.T_body_depth

      # ---- Parse RGB ----
      rgb_np = image_to_numpy(msgs["rgb"])
      if rgb_np.ndim == 2:
        rgb_np = np.stack([rgb_np] * 3, axis=-1)
      elif rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]

      # Rectify / undistort RGB using calibrated intrinsics, if available.
      if (getattr(self, "_rgb_dist_coeffs", None) is not None
          and getattr(self, "_rgb_newK", None) is not None
          and getattr(self, "_rgb_undist_roi", None) is not None):
        undist = cv2.undistort(
          rgb_np, self._rgb_K_orig, self._rgb_dist_coeffs,
          None, self._rgb_newK)
        x, y, rw, rh = self._rgb_undist_roi
        undist = undist[y:y+rh, x:x+rw]
        rgb_np = undist

      # Flip BGR → RGB (VOXL typically publishes bgra8 or bgr8)
      if rgb_np.shape[-1] == 3:
        rgb_np = rgb_np[..., ::-1].copy()
      rgb_img = torch.tensor(
        rgb_np.astype(np.float32) / 255.0,
        dtype=torch.float).permute(2, 0, 1)

      # ---- Get world points from depth source ----
      if self._use_point_cloud:
        # Point cloud path (PointCloud2): use ros_utils
        pc_arr = pointcloud2_to_array(msgs["pc"], squeeze=True)
        flat = pc_arr.reshape(-1)
        x = np.asarray(flat["x"], dtype=np.float64)
        y = np.asarray(flat["y"], dtype=np.float64)
        z = np.asarray(flat["z"], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(valid):
          continue
        pc_pts = torch.tensor(
          np.column_stack([x[valid], y[valid], z[valid]]),
          dtype=torch.float)
        pc_homo = g3d.pts_to_homogen(pc_pts)
        if self._point_cloud_frame == "sensor":
          # Points are in the depth sensor's local frame → world RDF
          xyz_world = g3d.transform_points(pc_homo, pose_depth)
        else:
          # Points are in body frame
          xyz_world = g3d.transform_points(pc_homo, body_rdf @ self.src2rdf)
        # transform_points returns Nx4 when given homogeneous; we need Nx3
        xyz_world = g3d.pts_to_nonhomo(xyz_world)
      else:
        # Depth image path: unproject to world
        depth_np = image_to_numpy(msgs["depth"])
        assert depth_np.dtype == np.uint8
        # Normalize 0–255 with max range (metres)
        depth_np = (depth_np.astype(np.float32) / 255.0) * self._depth_max_range
        depth_tensor = torch.tensor(
          depth_np, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        depth_tensor[
          ~torch.isfinite(depth_tensor) | (depth_tensor <= 0)] = float("nan")
        depth_tensor[depth_tensor > self._depth_max_range] = torch.inf
        xyz_world, _ = g3d.depth_to_pointcloud(
          depth_tensor, pose_depth.unsqueeze(0),
          self._depth_intrinsics_3x3, max_num_pts=-1)

      # ---- Project world points into RGB (use RGB intrinsics; depth stays in RGB view) ----
      reg_depth = g3d.world_points_to_depth_image(
          xyz_world, pose_rgb, self._rgb_intrinsics_3x3, (self.rgb_h, self.rgb_w))

      # ---- Optional resizing ----
      if self.rgb_h > 0 and (
          rgb_img.shape[-2] != self.rgb_h
          or rgb_img.shape[-1] != self.rgb_w):
        rgb_img = torch.nn.functional.interpolate(
            rgb_img.unsqueeze(0), size=(self.rgb_h, self.rgb_w),
            mode=self.interp_mode,
            antialias=self.interp_mode in ("bilinear", "bicubic")
        ).squeeze(0)

      if self.depth_h > 0 and (
          reg_depth.shape[-2] != self.depth_h
          or reg_depth.shape[-1] != self.depth_w):
        reg_depth = torch.nn.functional.interpolate(
            reg_depth.unsqueeze(0),
            size=(self.depth_h, self.depth_w),
            mode="nearest-exact").squeeze(0)

      frame_data = dict(
          rgb_img=rgb_img,
          depth_img=reg_depth,
          pose_4x4=pose_rgb,
      )

      # ---- Optional confidence map ----
      if "conf" in msgs:
        conf_np = image_to_numpy(msgs["conf"]).astype(np.float32)
        conf = torch.tensor(conf_np, dtype=torch.float).unsqueeze(0)
        if conf.shape[-2:] != frame_data["depth_img"].shape[-2:]:
          conf = torch.nn.functional.interpolate(
              conf.unsqueeze(0),
              size=frame_data["depth_img"].shape[-2:],
              mode="nearest-exact").squeeze(0)
        frame_data["confidence_map"] = 1.0 - (conf / 100.0)

      yield frame_data

  def shutdown(self):
    self._shutdown_event.set()
    self._rosnode.context.try_shutdown()
    logger.info("StarlingMaxSubscriber shutdown.")


@deprecated("Use Ros2Subscriber instead")
class RosnpyDataset(PosedRgbdDataset):
  """Processes datasets produced by the ros2npy utility from scripts dir.
  
  The ros2npy utility is located in the scripts directory and it converts ROS 
  bags to npz files to drop the ros dependency. The format can be quite slow 
  since it requires loading huge chunks of memory at a time.

  This will be removed in the future and replaced by a ROS1 bag reader or
  subscriber.

  Attributes:
    intrinsics_3x3:  See base.
    rgb_h: See base.
    rgb_w: See base.
    depth_h: See base.
    depth_w: See base.
    frame_skip: See base.
    interp_mode: See base.
  """

  def __init__(self,
               path: str,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear"):
    """
    Args:
      path: Path to directory. if path ends with .npz only a single file is 
        loaded. If the path is a directory then all .npz files within that
        directory will be loaded in lexsorted order assuming that order
        corresponds to the chronological order as well.
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    if os.path.isdir(path):
      self._data_files = [os.path.join(path, x)
                         for x in os.listdir(path)
                         if x.endswith(".npz")]

      self._data_files = sorted(self._data_files)
    else:
      self._data_files = [path]

    self._data_files_index = 0
    with np.load(self._data_files[self._data_files_index]) as npz_file:
      self._loaded_file = dict(npz_file.items())
    self._next_loaded_file = None
    self._loaded_file_frame = 0

    self._data_files_index += 1
    self._prefetch_thread = None
    if self._data_files_index < len(self._data_files):
      self._prefetch_thread = threading.Thread(target=self._prefetch_next_file)
      self._prefetch_thread.start()

    self.intrinsics_3x3 = self._loaded_file["intrinsics_3x3"][0].reshape(3,3)
    self.intrinsics_3x3 = torch.tensor(self.intrinsics_3x3, dtype=torch.float)

    self.original_h, self.original_w = self._loaded_file["rgb_img"][0].shape[:2]
    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    # ROS (X-Forward, Y-Left, Z-Up) to OpenCV (X-Right, Y-Down, Z-Forward):
    self._flu2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform("flu", "rdf"))

  def _prefetch_next_file(self):
    with np.load(self._data_files[self._data_files_index]) as npz_file:
      self._next_loaded_file = dict(npz_file.items())

  @override
  def __iter__(self):
    f = 0
    while True:
      seq_lens = [len(x) for x in self._loaded_file.values()]

      if self._loaded_file_frame >= min(seq_lens):
        if self._prefetch_thread is None:
          break

        # Make sure prefetch thread has terminated
        self._prefetch_thread.join()

        # Load next file and reset frame index to 0
        self._loaded_file = self._next_loaded_file
        self._loaded_file_frame = 0
        self._next_loaded_file = None

        # Start loading of next file in seperate thread
        self._data_files_index += 1
        if self._data_files_index < len(self._data_files):
          self._prefetch_thread = threading.Thread(
            target=self._prefetch_next_file)
          self._prefetch_thread.start()
        else:
          self._prefetch_thread = None

      i = self._loaded_file_frame
      frames_data = self._loaded_file
      self._loaded_file_frame += 1

      if self.frame_skip > 0 and f % (self.frame_skip+1) != 0:
          f += 1
          continue
      f += 1

      flu_pose_t = frames_data["pose_t"][i]
      flu_pose_q = frames_data["pose_q_wxyz"][i]
      flu_pose_R = Rotation.from_quat(flu_pose_q, scalar_first=True).as_matrix()
      # TODO: Verify
      flu_pose_Rt_3x4 = np.concatenate((flu_pose_R, flu_pose_t.reshape(3, 1)),
                                        axis=1)
      flu_pose_Rt_3x4 = torch.tensor(flu_pose_Rt_3x4, dtype=torch.float)
      flu_pose_4x4 = g3d.mat_3x4_to_4x4(flu_pose_Rt_3x4)

      rdf_pose_4x4 = g3d.transform_pose_4x4(flu_pose_4x4,
                                            self._flu2rdf_transform)

      disparity_img = frames_data["disparity_img"][i]
      min_disp = frames_data["min_disparity"][i]
      max_disp = frames_data["max_disparity"][i]

      focal_length = frames_data["focal_length"][i]
      stereo_baseline = frames_data["stereo_baseline"][i]

      depth_img = focal_length*stereo_baseline/disparity_img
      #TODO: Why does this have to be flipped ?
      depth_img[disparity_img < min_disp] = -np.inf
      depth_img[disparity_img > max_disp] = np.inf
      depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)

      rgb_img = frames_data["rgb_img"][i]
      rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)/255

      if (self.rgb_h != rgb_img.shape[-2] or
          self.rgb_w != rgb_img.shape[-1]):
        rgb_img = torch.nn.functional.interpolate(rgb_img.unsqueeze(0),
          size=(self.rgb_h, self.rgb_w), mode=self.interp_mode,
          antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

      if (self.depth_h != depth_img.shape[-2] or
          self.depth_w != depth_img.shape[-1]):
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0),
          size=(self.depth_h, self.depth_w),
          mode="nearest-exact").squeeze(0)

      frame_data = dict(rgb_img = rgb_img, depth_img = depth_img,
                        pose_4x4 = rdf_pose_4x4)
      yield frame_data

  def shutdown(self):
    if self._prefetch_thread is not None:
      self._prefetch_thread.join()
