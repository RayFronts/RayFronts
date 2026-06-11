import abc
from typing_extensions import Tuple

import torch


class DepthEstimator(abc.ABC):
  """Interface for all depth estimators.

  A depth estimator consumes batched RGB images and optionally an initial depth
  estimate, camera pose and intrinsics, and produces refined metric depth
  estimates.
  """

  def __init__(self, device=None):
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

  @abc.abstractmethod
  def estimate_depth(
      self,
      rgb_image: torch.FloatTensor,
      depth_init: torch.FloatTensor | None = None,
      pose_4x4: torch.FloatTensor | None = None,
      intrinsics_3x3: torch.FloatTensor | None = None,
  ) -> torch.FloatTensor:
    """Estimate depth for a batch of RGB images.

    Implementations may optionally use an initial depth estimate and pose.

    Args:
      rgb_image: A float tensor of shape (B, 3, H, W) with values in [0, 1].
      depth_init: An optional float tensor of shape (B, H, W) representing an
        initial depth estimate in meters registered to `rgb_image`.
      pose_4x4: Optional camera/robot pose. A float tensor of shape (B, 4, 4) or
        (4, 4) in OpenCV RDF convention. A pose is the extrinsics
        transformation matrix that takes you from camera/robot coordinates to
        world coordinates. Last row should always be [0, 0, 0, 1].
      intrinsics_3x3: Optional camera intrinsics. A float tensor of shape
        (3, 3) representing the standard pinhole intrinsics matrix in pixel
        coordinates (fx, fy, cx, cy).

    Returns:
      A float tensor of shape (B, H, W) representing refined depth in meters.
    """

  @property
  def supports_points(self) -> bool:
    """Whether this estimator can directly produce point clouds."""
    return False

  def estimate_depth_and_points(
      self,
      rgb_image: torch.FloatTensor,
      depth_init: torch.FloatTensor | None = None,
      pose_4x4: torch.FloatTensor | None = None,
      intrinsics_3x3: torch.FloatTensor | None = None,
  ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Estimate depth and 3D points for a batch of RGB images.

    The default implementation computes depth via `estimate_depth` and expects
    subclasses that support points to override this method for efficiency.

    Args:
      rgb_image: A float tensor of shape (B, 3, H, W) with values in [0, 1].
      depth_init: An optional float tensor of shape (B, H, W) representing an
        initial depth estimate in meters registered to `rgb_image`.
      pose_4x4: Optional camera/robot pose. A float tensor of shape (B, 4, 4) or
        (4, 4) in OpenCV RDF convention. A pose is the extrinsics
        transformation matrix that takes you from camera/robot coordinates to
        world coordinates. Last row should always be [0, 0, 0, 1].
      intrinsics_3x3: Optional camera intrinsics. A float tensor of shape
        (3, 3) representing the standard pinhole intrinsics matrix in pixel
        coordinates (fx, fy, cx, cy).

    Returns:
      A tuple `(depth, points)` where:
        - `depth` is a float tensor of shape (B, H, W) in meters.
        - `points` is a float tensor of shape (B, H, W, 3) in camera space.

    Raises:
      NotImplementedError: If the estimator does not support point outputs.
    """
    raise NotImplementedError(
        "This depth estimator does not support point cloud outputs.")

