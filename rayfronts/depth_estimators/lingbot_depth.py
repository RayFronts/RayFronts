import logging
from typing_extensions import Tuple

import torch

from rayfronts.depth_estimators.base import DepthEstimator

logger = logging.getLogger(__name__)

_HAS_LINGBOT = True
try:
  # LingBot-Depth reference:
  # https://github.com/robbyant/lingbot-depth
  from mdm.model.v2 import MDMModel  # type: ignore[import]
except ModuleNotFoundError:
  _HAS_LINGBOT = False
  MDMModel = None  # type: ignore[assignment]


class LingbotDepthEstimator(DepthEstimator):
  """Depth estimator backed by the LingBot-Depth model.

  This wraps the `LingBot-Depth` masked depth modeling approach [1] and exposes
  a batched `estimate_depth` API compatible with the RayFronts ecosystem.

  References:
    [1] LingBot-Depth: Masked Depth Modeling for Spatial Perception
        (https://github.com/robbyant/lingbot-depth)
  """

  def __init__(
      self,
      model_name: str = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
      device=None,
      use_fp16: bool = True,
  ):
    """Initializes the LingBot-Depth estimator.

    Args:
      model_name: Pretrained model identifier passed to
        `MDMModel.from_pretrained`.
      device: Torch device or device string. Defaults to "cuda" if available,
        otherwise "cpu".
      use_fp16: Whether to enable mixed-precision inference when supported.

    Raises:
      ImportError: If the LingBot-Depth dependency is not installed.
    """
    super().__init__(device=device)

    if not _HAS_LINGBOT:
      raise ImportError(
          "LingBot-Depth is not installed. Install it via pip, for example:\n"
          "  pip install 'git+https://github.com/robbyant/lingbot-depth.git'")

    self._model_name = model_name
    self._use_fp16 = use_fp16
    self._model = MDMModel.from_pretrained(model_name).to(self.device)
    self._model.eval()
    logger.info("Initialized LingBot-Depth model '%s' on device %s",
                model_name, self.device)

  @property
  def supports_points(self) -> bool:
    return True

  @torch.inference_mode()
  def estimate_depth(
      self,
      rgb_image: torch.FloatTensor,
      depth_init: torch.FloatTensor | None = None,
      pose_4x4: torch.FloatTensor | None = None,
      intrinsics_3x3: torch.FloatTensor | None = None,
  ) -> torch.FloatTensor:
    """Estimate refined depth using LingBot-Depth.

    Args:
      rgb_image: A float tensor of shape (B, 3, H, W) with values in [0, 1].
      depth_init: Optional float tensor of shape (B, H, W) representing raw
        depth in meters registered to `rgb_image`.
      pose_4x4: Optional camera/robot pose. A float tensor of shape (B, 4, 4) or
        (4, 4) in OpenCV RDF convention. A pose is the extrinsics
        transformation matrix that takes you from camera/robot coordinates to
        world coordinates. This argument is currently unused by this
        estimator but included for API consistency.
      intrinsics_3x3: Optional camera intrinsics tensor of shape (3, 3)
        representing the standard pinhole intrinsics matrix in pixel
        coordinates (fx, fy, cx, cy). Internally converted to the normalized
        form expected by LingBot-Depth based on the input image resolution.

    Returns:
      A float tensor of shape (B, H, W) representing refined depth in meters.
    """
    depth, _ = self.estimate_depth_and_points(
        rgb_image=rgb_image,
        depth_init=depth_init,
        pose_4x4=pose_4x4,
        intrinsics_3x3=intrinsics_3x3)
    return depth

  @torch.inference_mode()
  def estimate_depth_and_points(
      self,
      rgb_image: torch.FloatTensor,
      depth_init: torch.FloatTensor | None = None,
      pose_4x4: torch.FloatTensor | None = None,
      intrinsics_3x3: torch.FloatTensor | None = None,
  ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Estimate depth and 3D points using LingBot-Depth.

    Args:
      rgb_image: A float tensor of shape (B, 3, H, W) with values in [0, 1].
      depth_init: Optional float tensor of shape (B, H, W) representing raw
        depth in meters registered to `rgb_image`.
      pose_4x4: Optional camera/robot pose. A float tensor of shape (B, 4, 4) or
        (4, 4) in OpenCV RDF convention. A pose is the extrinsics
        transformation matrix that takes you from camera/robot coordinates to
        world coordinates. This argument is currently unused by this
        estimator but included for API consistency.
      intrinsics_3x3: Optional camera intrinsics tensor of shape (3, 3)
        representing the standard pinhole intrinsics matrix in pixel
        coordinates (fx, fy, cx, cy). Internally converted to the normalized
        form expected by LingBot-Depth based on the input image resolution.

    Returns:
      A tuple `(depth, points)` where:
        - `depth` is a float tensor of shape (B, H, W) in meters.
        - `points` is a float tensor of shape (B, H, W, 3) in camera space.
    """
    if rgb_image.ndim != 4 or rgb_image.shape[1] != 3:
      raise ValueError(
          "Expected rgb_image of shape (B, 3, H, W), got "
          f"{tuple(rgb_image.shape)}")

    images = rgb_image.to(self.device)
    _, _, h, w = images.shape
    if depth_init is not None:
      if depth_init.ndim != 3:
        raise ValueError(
            "Expected depth_init of shape (B, H, W), got "
            f"{tuple(depth_init.shape)}")
      depth_in = depth_init.to(self.device)
    else:
      depth_in = None

    intrinsics_dev = None
    if intrinsics_3x3 is not None:
      if intrinsics_3x3.ndim != 2 or intrinsics_3x3.shape != (3, 3):
        raise ValueError(
            "Expected intrinsics_3x3 of shape (3, 3), got "
            f"{tuple(intrinsics_3x3.shape)}")
      intrinsics_dev = intrinsics_3x3.to(self.device).clone().unsqueeze(0)
      intrinsics_dev = intrinsics_dev.expand(images.shape[0], -1, -1)
      # Normalize intrinsics by image width/height, matching LingBot-Depth.
      intrinsics_dev[:, 0, :] = intrinsics_dev[:, 0, :] / float(w)
      intrinsics_dev[:, 1, :] = intrinsics_dev[:, 1, :] / float(h)

    output = self._model.infer(
        images,
        depth_in=depth_in,
        intrinsics=intrinsics_dev,
        use_fp16=self._use_fp16,
    )
    depth = output["depth"]
    points = output["points"]
    return depth, points

