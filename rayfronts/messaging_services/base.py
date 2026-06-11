"""Defines the abstract base class for messaging services.

Messaging services handle receiving queries (e.g. text) and optionally
publishing query results or point clouds for consumption by planners or
other nodes.
"""

import abc
from typing import Dict

import torch


class MessagingService(abc.ABC):
  """Base interface for messaging services used by the mapping server.

  Subclasses typically receive queries and invoke a callback, and may
  publish query results or point clouds for external consumption.
  """

  @abc.abstractmethod
  def text_query_handler(self, s):
    """Handle an incoming text query message.

    Args:
      s: The received message (type depends on implementation, e.g. ROS
        std_msgs/String).
    """
    pass

  @abc.abstractmethod
  def join(self, timeout=None):
    """Block until the service thread exits.

    Args:
      timeout: Optional timeout in seconds. If None, block indefinitely.
    """
    pass

  @abc.abstractmethod
  def shutdown(self):
    """Signal the service to shut down and release resources."""
    pass

  def publish_pc(self, pc_xyz: torch.FloatTensor,
                 features: Dict[str, torch.FloatTensor] = None,
                 layer: str = "pc") -> None:
    """Publish a point cloud with optional named scalar features.

    Default is no-op; override in implementations that support publishing.

    Args:
      pc_xyz: (Nx3) Float tensor of point positions.
      features: Optional dict mapping field names to (N,) float tensors.
      layer: Logical name for this point cloud (e.g. "frontiers").
    """
    pass

  def publish_query_results(self, query_results: dict,
                            query_labels: list = None) -> None:
    """Publish query results for programmatic consumption (e.g. by a planner).

    Expects a dict with optional keys: vox_xyz, vox_sim (voxel similarity),
    ray_orig_angles, ray_sim (ray similarity). Default is no-op; override in
    implementations that support publishing.

    Args:
      query_results: Dict returned from mapper.feature_query(), containing
        at least one of (vox_xyz, vox_sim) or (ray_orig_angles, ray_sim).
      query_labels: Optional list of string labels, one per query (e.g. the
        text queries). Implementations may use these for naming or routing.
    """
    pass
