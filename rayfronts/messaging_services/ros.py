"""ROS2 implementation of the messaging service.

Subscribes to a text-query topic and optionally publishes query results
(voxel_similarity, ray_similarity) as sensor_msgs/PointCloud2 on configurable
topics. Only computes and publishes when there is at least one subscriber.

The messaging service uses a separate prefix (query_results_topic_prefix, e.g.
/rayfronts/query_results) and reserved key segments (voxel_similarity/...,
ray_similarity/...) so its topics do not collide with visualizer topics (which
use keys like voxel_rgb, frontiers, layer/pose, etc. under the vis topic_prefix).
"""

import threading
import logging
from typing_extensions import override

import numpy as np
import torch
import std_msgs.msg
from rayfronts.messaging_services import MessagingService
from rayfronts import ros_utils

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
import std_msgs

logger = logging.getLogger(__name__)

# Reserved topic key segments for query results (avoid collision with visualizer
# layer names such as voxel_rgb, frontiers, layer/pose, etc.).
KEY_VOXEL_SIMILARITY = "voxels_sim"
KEY_RAY_SIMILARITY = "rays_sim"
KEY_ALL_QUERIES = "all"


class Ros2MessagingService(MessagingService):
  """ROS2 messaging service: text-query subscription and query-result publishing.

  Subscribes to a String topic for text queries and invokes a callback.
  When the mapping server runs queries, can publish filtered voxel and/or ray
  similarity as PointCloud2 (only when there is a subscriber).

  Attributes:
    text_query_topic: See __init__.
    text_query_callback: See __init__.
    query_publish_threshold: See __init__.
    query_results_topic_prefix: See __init__.
    frame_id: See __init__.
  """

  def __init__(self,
               text_query_topic,
               text_query_callback=None,
               query_publish_threshold: float = 0.0,
               query_results_topic_prefix: str = "/rayfronts/query_results",
               frame_id: str = "map"):
    """

    Args:
      text_query_topic: ROS2 topic name for incoming text queries (std_msgs/String).
      text_query_callback: Callable invoked with the string data of each query
        message. Can be None to ignore queries.
      query_publish_threshold: Only voxels/rays with similarity >= this value
        are included in published PointCloud2. Ignored when no subscriber.
      query_results_topic_prefix: Prefix for query-result topics (same
        convention as vis topic_prefix: full topic = prefix + "/" + key).
        Default /rayfronts/query_results matches vis root /rayfronts. Keys
        are voxel_similarity/..., ray_similarity/... so they do not collide
        with visualizer layer names.
      frame_id: Frame id set on published PointCloud2 headers.
    """
    super().__init__()
    self.text_query_topic = text_query_topic
    self.text_query_callback = text_query_callback
    self.query_publish_threshold = query_publish_threshold
    self.query_results_topic_prefix = query_results_topic_prefix
    self.frame_id = frame_id
    self._query_publishers = dict()

    if not rclpy.ok():
      rclpy.init()
    self._rosnode = Node("rayfronts_messaging_service")

    self.text_query_sub = self._rosnode.create_subscription(
      std_msgs.msg.String, text_query_topic, self.text_query_handler,
      QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=5))

    self._ros_executor = SingleThreadedExecutor()
    self._ros_executor.add_node(self._rosnode)
    self._spin_thread = threading.Thread(
      target=self._spin_ros, name="rayfronts_messaging_service_spinner")
    self._spin_thread.daemon = True
    self._spin_thread.start()

    logger.info("Messaging Service initialized successfully.")

  def _spin_ros(self):
    """Run the ROS executor; catches shutdown exceptions."""
    try:
      self._ros_executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass

  def _get_query_publisher(self, key: str):
    """Return the lazy-created PointCloud2 publisher for the given topic key."""
    try:
      return self._query_publishers[key]
    except KeyError:
      pub = self._rosnode.create_publisher(
          PointCloud2,
          f"{self.query_results_topic_prefix}/{key}",
          QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=5))
      self._query_publishers[key] = pub
      logger.info("Query results publisher %s/%s initialized.",
                  self.query_results_topic_prefix, key)
      return pub

  def _has_subscriber(self, pub) -> bool:
    """Return True if the publisher has at least one subscriber."""
    return pub.get_subscription_count() > 0

  def _sanitize_topic_name(self, s: str) -> str:
    """Make a string safe for ROS 2 topic names (alphanumeric and underscore).
    Replaces spaces and other invalid chars with underscore, collapses runs."""
    if not isinstance(s, str) or not s:
      return ""
    out = []
    for c in s:
      if c.isalnum() or c == "_":
        out.append(c)
      elif c.isspace() or not c.isalnum():
        out.append("_")
    name = "".join(out)
    while "__" in name:
      name = name.replace("__", "_")
    return name.strip("_") or ""

  def _query_topic_suffix(self, q: int, query_labels: list = None) -> str:
    """Return topic suffix for query index q: '{q}_{label}' or just '{q}'."""
    if query_labels is not None and q < len(query_labels):
      sanitized = self._sanitize_topic_name(str(query_labels[q]))
      if sanitized:
        return f"{q}_{sanitized}"
    return str(q)

  @override
  def publish_query_results(self, query_results: dict,
                            query_labels: list = None) -> None:
    """Publish voxel and/or ray similarity as PointCloud2 when subscribers exist.

    Broadcasts:
    - Per-query topics: voxel_similarity/{q}_{label} (e.g. 0_dog, 1_cat), one
      PointCloud2 per query with (x, y, z, sim).
    - All-queries topic: voxel_similarity/all, single PointCloud2 with
      (x, y, z, sim_0, sim_1, ...) so each point has one row and one sim per query.
    Same for ray_similarity. Uses max over queries to decide which points pass
    the threshold. Only processes and publishes for topics that have a subscriber.
    """
    if "vox_xyz" in query_results and "vox_sim" in query_results:
      vox_xyz = query_results["vox_xyz"]
      vox_sim = query_results["vox_sim"]
      num_queries = vox_sim.shape[0]
      # Only convert and publish if at least one voxel topic has a subscriber
      pub_all = self._get_query_publisher(
          f"{KEY_VOXEL_SIMILARITY}/{KEY_ALL_QUERIES}")
      need_vox = self._has_subscriber(pub_all)
      if not need_vox:
        for q in range(num_queries):
          suffix = self._query_topic_suffix(q, query_labels)
          if self._has_subscriber(self._get_query_publisher(
              f"{KEY_VOXEL_SIMILARITY}/{suffix}")):
            need_vox = True
            break
      if need_vox:
        # Max, mask, and indexing on GPU; convert only filtered to CPU
        score_max = vox_sim.max(dim=0)[0]
        mask = score_max >= self.query_publish_threshold
        n_filtered = int(mask.sum().item())
        if n_filtered > 0:
          vox_xyz = vox_xyz[mask].cpu().numpy().astype(np.float32)
          vox_sim = vox_sim[:, mask].cpu().numpy().astype(np.float32)
          if self._has_subscriber(pub_all):
            dtype_all = [("x", np.float32), ("y", np.float32), ("z", np.float32)]
            for q in range(num_queries):
              dtype_all.append((f"sim_{q}", np.float32))
            rec = np.recarray((n_filtered,), dtype=dtype_all)
            rec["x"] = vox_xyz[:, 0]
            rec["y"] = vox_xyz[:, 1]
            rec["z"] = vox_xyz[:, 2]
            for q in range(num_queries):
              rec[f"sim_{q}"] = vox_sim[q, :]
            cloud = ros_utils.array_to_pointcloud2(
                rec, frame_id=self.frame_id)
            pub_all.publish(cloud)
          for q in range(num_queries):
            suffix = self._query_topic_suffix(q, query_labels)
            pub = self._get_query_publisher(f"{KEY_VOXEL_SIMILARITY}/{suffix}")
            if self._has_subscriber(pub):
              rec = np.recarray(
                  (n_filtered,),
                  dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32),
                        ("sim", np.float32)])
              rec.x = vox_xyz[:, 0]
              rec.y = vox_xyz[:, 1]
              rec.z = vox_xyz[:, 2]
              rec.sim = vox_sim[q, :]
              cloud = ros_utils.array_to_pointcloud2(
                  rec, frame_id=self.frame_id)
              pub.publish(cloud)

    if "ray_orig_angles" in query_results and "ray_sim" in query_results:
      ray_orig_angles = query_results["ray_orig_angles"]
      ray_sim = query_results["ray_sim"]
      num_queries = ray_sim.shape[0]
      pub_all = self._get_query_publisher(
          f"{KEY_RAY_SIMILARITY}/{KEY_ALL_QUERIES}")
      need_ray = self._has_subscriber(pub_all)
      if not need_ray:
        for q in range(num_queries):
          suffix = self._query_topic_suffix(q, query_labels)
          if self._has_subscriber(self._get_query_publisher(
              f"{KEY_RAY_SIMILARITY}/{suffix}")):
            need_ray = True
            break
      if need_ray:
        # Max, mask, and indexing on GPU; convert only filtered to CPU
        score_max = ray_sim.max(dim=0)[0]
        mask = score_max >= self.query_publish_threshold
        m_filtered = int(mask.sum().item())
        if m_filtered > 0:
          ray_orig_angles = ray_orig_angles[mask].cpu().numpy().astype(np.float32)
          ray_sim = ray_sim[:, mask].cpu().numpy().astype(np.float32)
          if self._has_subscriber(pub_all):
            dtype_all = [
                ("x", np.float32), ("y", np.float32), ("z", np.float32),
                ("theta", np.float32), ("phi", np.float32)]
            for q in range(num_queries):
              dtype_all.append((f"sim_{q}", np.float32))
            rec = np.recarray((m_filtered,), dtype=dtype_all)
            rec["x"] = ray_orig_angles[:, 0]
            rec["y"] = ray_orig_angles[:, 1]
            rec["z"] = ray_orig_angles[:, 2]
            rec["theta"] = ray_orig_angles[:, 3]
            rec["phi"] = ray_orig_angles[:, 4]
            for q in range(num_queries):
              rec[f"sim_{q}"] = ray_sim[q, :]
            cloud = ros_utils.array_to_pointcloud2(
                rec, frame_id=self.frame_id)
            pub_all.publish(cloud)
          for q in range(num_queries):
            suffix = self._query_topic_suffix(q, query_labels)
            pub = self._get_query_publisher(f"{KEY_RAY_SIMILARITY}/{suffix}")
            if self._has_subscriber(pub):
              rec = np.recarray(
                  (m_filtered,),
                  dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32),
                        ("theta", np.float32), ("phi", np.float32),
                        ("sim", np.float32)])
              rec.x = ray_orig_angles[:, 0]
              rec.y = ray_orig_angles[:, 1]
              rec.z = ray_orig_angles[:, 2]
              rec.theta = ray_orig_angles[:, 3]
              rec.phi = ray_orig_angles[:, 4]
              rec.sim = ray_sim[q, :]
              cloud = ros_utils.array_to_pointcloud2(
                  rec, frame_id=self.frame_id)
              pub.publish(cloud)

  @override
  def text_query_handler(self, s):
    """Forward the incoming String message data to the configured callback."""
    if self.text_query_callback is not None:
      self.text_query_callback(s.data)

  @override
  def join(self, timeout=None):
    """Block until the ROS spin thread exits."""
    self._spin_thread.join(timeout)

  @override
  def shutdown(self):
    """Shut down the ROS node context."""
    self._rosnode.context.try_shutdown()
