"""Defines abstract base classes for all mappers."""

import abc
from typing_extensions import List, Dict, Tuple

import torch

from rayfronts.utils import compute_cos_sim
from rayfronts.visualizers import Mapping3DVisualizer
from rayfronts.image_encoders import ImageSpatialEncoder
from rayfronts.feat_compressors import FeatCompressor

class RGBDMapping(abc.ABC):
  """Base interface for all maps taking posed RGBD as input.
  
  Attributes:
    intrinsics_3x3: See __init__.
    device: See __init__.
    visualizer: See __init__.
    clip_bbox: See __init__.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None):
    """

    Args:
      intrinsics_3x3: A 3x3 float tensor including camera intrinsics. Can set
        to None if no rgbd processing will happen.
      device: Where to perform the computations
      visualizer: Which visualizer to submit the inputs + processed outputs to.
      clip_bbox: Limit mapping to this box [[a,b,c], [x,y,z]]
        where x > a, y > b, z > c. Set to None to not bound the map.
    """
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    self.intrinsics_3x3 = intrinsics_3x3
    if self.intrinsics_3x3 is not None:
      self.intrinsics_3x3 = self.intrinsics_3x3.to(self.device)
    self.visualizer = visualizer

    self.clip_bbox = clip_bbox
    if self.clip_bbox is not None:
      if isinstance(self.clip_bbox, torch.Tensor):
        self.clip_bbox = clip_bbox.to(self.device)
      else:
        self.clip_bbox = torch.tensor(self.clip_bbox,
                                      dtype=torch.float, device=self.device)
      assert self.clip_bbox.shape[0] == 2 and self.clip_bbox.shape[1] == 3

      if self.visualizer is not None:
        self.visualizer.log_box(self.clip_bbox[0],
                                self.clip_bbox[1],
                                layer = "clip_bbox")

  def _clip_pc(self, pc_xyz: torch.FloatTensor,
              *features) -> Tuple[torch.Tensor]:
    """Clip a point cloud and its associated features to be in self.clip_bbox.

    If self.clip_bbox is not set, then this is a pass through.
    
    Args:
      pc_xyz: (Nx3) Float tensor describing point locations.
      *features: arbitrary feature tensors of shape (NxC) that have a one to one
        correspondance with pc_xyz.
    Returns:
      A tuple with the clipped pc_xyz tensor and the passed feature tensors.
    """
    if self.clip_bbox is None:
      return [pc_xyz] + list(features)

    mask = (pc_xyz > self.clip_bbox[0]) & (pc_xyz < self.clip_bbox[1])
    mask = torch.all(mask, dim=-1)
    pc_xyz = pc_xyz[mask]
    r = [pc_xyz]
    for f in features:
      r.append(f[mask])
    return r

  @abc.abstractmethod
  def save(self, file_path) -> None:
    """Saves all map data to the given file path.

    Args:
      file_path: Path to which the map data should be saved.
    """
    pass

  @abc.abstractmethod
  def load(self, file_path: str) -> None:
    """Loads map data in-place from given file path.
    
    Args:
      file_path: Path to file in the format saved by the save method.
    """
    pass

  @abc.abstractmethod
  def is_empty(self) -> bool:
    """Checks if the map is empty"""
    pass

  @abc.abstractmethod
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None) -> dict:
    """Consumes a posed rgbd frame and updates the internal representation.

    Args:
      rgb_img: A Bx3xHxW RGB float tensor within the (0-1) range describing a 
        batch of images
      depth_img: A Bx1xH'xW' float tensor with values > 0 describing a batch of 
        depth images. May include NaN and Inf values which will be ignored.
      pose_4x4: A Bx4x4 tensor which includes a batch of poses in opencv RDF.
        a pose is the extrinsics transformation matrix that takes you from
        camera/robot coordinates to world coordinates.
      conf_map: (Optional) A Bx1xH'xW' float tensor with values in [0-1]
        with 1 being most confident in the depth value.
    Returns:
      Local update information which can be used to visualize further
      data about transitionary data structures not part of the input or the 
      final map output. This allows decoupling of transitionary data
      visualization code from this processing method.
    """
    pass

  @abc.abstractmethod
  def vis_map(self) -> None:
    """Visualizes the underlying map using the passed visualizer."""
    pass

  @abc.abstractmethod
  def vis_update(self, **kwargs) -> None:
    """Visualizes transitionary update information such as feature images.
    
    Args:
      kwargs: Child class dependent. Typically the dictionary returned from
        process_posed_rgbd.
    """
    pass

class SemanticRGBDMapping(RGBDMapping):
  """Abstract class for semantic maps that can be queried.
  
  Attributes:
    intrinsics_3x3: See base.
    device: See base.
    visualizer: See base.

    encoder: See __init__.
    stored_feat_dim: See __init__.
    interp_mode: See __init__.

    basis: PCA basis used to compress features. CxD torch tensor, where C
      represents the original feature dim and D is the compressed dim.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               encoder: ImageSpatialEncoder = None,
               feat_compressor: FeatCompressor = None,
               interp_mode: str = "bilinear"):
    """
    Args:
      intrinsics_3x3: See base.
      device: See base.
      visualizer: See base.
      clip_bbox: See base.
      encoder: ImageEncoder to use to generate a feature map from an image.
      stored_feat_dim: Length of the feature dimension to store. If smaller than
        encoder output then PCA is used to compress. Set to -1 to use full
        encoder output.
      feat_proj_basis_path: If you already have a precomputed PCA basis .pt file
        as CxD torch tensor, where C represents the original feature dim and D
        is the compressed dim, then you can pass the path here to avoid
        recomputing based on a single frame. This will be used for both
        visualization and mapping by default.
      interp_mode: Which pytorch interpolation mode for rgb and feature
        interpolation. See https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """
    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox)
    self.encoder = encoder
    self.interp_mode = interp_mode
    self.feat_compressor = feat_compressor

  @abc.abstractmethod
  def feature_query(self,
                    feat_query: torch.FloatTensor,
                    softmax: bool = False,
                    compressed: bool = False) -> dict:
    """Queries map using features in the same feature space as stored features.
    
    Args:
      feat_query: BxD float tensor where B is number of queries and D is the dim
        of the features. 
      softmax: If True, a softmax across queries is taken making a result of 
        feat_query[i] dependent on all other queries in the batch.
      compressed: Whether to query the map in compressed feature space.
        Only relevant if the map was using a feature compressor. If True,
        then feat_query must be in compressed feature space already.
    Returns:
      Query result as a dictionary mapping names to results which is child class
      dependent..
    """
    pass

  def text_query(self, text_query: List[str], query_type = "labels",
                 softmax: bool = False,
                 compressed: bool = False) -> dict:
    """Queries the representation using textual labels or prompts.
    
    Args:
      query: A list of string representing the labels or queries to query
      query_type: Choose from [labels, prompts]. Labels are encoded using 
        prompt templates whereas prompts are encoded directly.
      softmax: If True, a softmax across queries is taken making a result of 
        query[i] dependent on all other queries in the batch.
      compressed: Whether to query the map in compressed feature space.
        Only relevant if the map was using a feature compressor.
    Returns:
      Query result as a dictionary mapping names to results which is child class
      dependent.
    """
    if self.global_vox_xyz is None or len(self.global_vox_xyz) == 0:
      return

    if query_type == "labels":
      feat_query = self.encoder.encode_labels(text_query)
    elif query_type == "prompts":
      feat_query = self.encoder.encode_prompts(text_query)
    else:
      raise ValueError("Invalid query type")
    
    if self.feat_compressor is not None and compressed:
      feat_query = self.feat_compressor.compress(feat_query)

    return self.feature_query(feat_query, softmax, compressed)


  def image_query(self, img_query: torch.FloatTensor,
                  softmax: bool = False,
                  compressed: bool = False) -> dict:
    """Queries the representation using images.
    
    Args:
      img_query: A Bx3xHxW float tensor with values in [0, 1] representing
        B image queries and 
      query_type: Choose from [labels, prompts]. Labels are encoded using 
        prompt templates whereas prompts are encoded directly.
      softmax: If True, a softmax across queries is taken making a result of 
        query[i] dependent on all other queries in the batch.
      compressed: Whether to query the map in compressed feature space.
        Only relevant if the map was using a feature compressor.
    Returns:
      Query result as a dictionary mapping names to results which is child class
      dependent.
    """

    if self.is_empty():
      return

    feat_query = self.encoder.encode_image_to_vector(img_query)

    if self.feat_compressor is not None and compressed:
      feat_query = self.feat_compressor.compress(feat_query)

    return self.feature_query(feat_query, softmax, compressed)

  @abc.abstractmethod
  def vis_query_result(self,
                       query_results,
                       vis_labels: List[str] = None,
                       vis_colors: Dict[str, str] = None,
                       vis_thresh: float = 0) -> None:
    """Visualize results of a query.
    
    Args:
      query_results: Results dictionary from calling a querying method.
      vis_labels: Visualization labels for each query.
      vis_colors: Mapping of label to a tuple of 3 integers for RGB [0-255].
        Set to None to use default color visualization.
      vis_thresh: How confident should we be in the query map element similarity
        to visualize that map element for that query. Set to 0 to visualize all.
    """
    pass


  def _proj_resize_feat_map(self, feat_img, h, w):
    """Encodes the RGB image to a feature map, compress if needed, and resize.
    
    Args:
      rgb_img: (Bx3xHxW) Float tensor with values in [0-1].
      h: desired final height.
      w: desired final width.
    """
    B, FC, FH, FW = feat_img.shape

    if self.feat_compressor is not None:
      if not self.feat_compressor.is_fitted():
        self.feat_compressor.fit(feat_img.permute(0, 2, 3, 1))

      feat_img = self.feat_compressor.compress(feat_img.permute(0, 2, 3, 1))
      feat_img = feat_img.permute(0, 3, 1, 2)

    # TODO: This can be very large. Make this more memory efficient.
    # it is theoritically possible to only interpolate the indices of interest
    feat_img = torch.nn.functional.interpolate(
      feat_img,
      size =(h, w),
      mode=self.interp_mode,
      antialias=self.interp_mode in ["bilinear", "bicubic"])

    return feat_img
