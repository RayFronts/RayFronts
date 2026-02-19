"""Defines the abstract base class for messaging services.

Messaging services handle receiving queries (e.g. text) and optionally
publishing query results (e.g. voxel/ray similarity as PointCloud2) for
consumption by planners or other nodes.
"""

import abc
import threading


class MessagingService(abc.ABC):
  """Base interface for messaging services used by the mapping server.

  Subclasses typically subscribe to a query topic and invoke a callback,
  and may publish query results (vox_xyz/vox_sim, ray_orig_angles/ray_sim)
  as PointCloud2 when the mapping server runs queries.
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

  def publish_query_results(self, query_results: dict,
                            query_labels: list = None) -> None:
    """Publish query results for programmatic consumption (e.g. by a planner).

    Expects a dict with optional keys: vox_xyz, vox_sim (voxel similarity),
    ray_orig_angles, ray_sim (ray similarity). Default is no-op; override in
    implementations that support publishing (e.g. as PointCloud2).

    Args:
      query_results: Dict returned from mapper.feature_query(), containing
        at least one of (vox_xyz, vox_sim) or (ray_orig_angles, ray_sim).
      query_labels: Optional list of string labels, one per query (e.g. the
        text queries). Implementations may use these for topic naming instead
        of query index.
    """
    pass
