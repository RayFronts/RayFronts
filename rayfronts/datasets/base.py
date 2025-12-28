"""Defines abstract base classes for all datasets/datasources."""

import abc
from typing import Tuple, Union, List, Hashable, Dict
import copy

import torch


class PosedRgbdDataset(torch.utils.data.IterableDataset, abc.ABC):
  """A base interface for loading from any posed RGBD source.
  
  Attributes:
    intrinsics_3x3:  A 3x3 float tensor including camera intrinsics. This
      must be set by the child class.
    rgb_h: RGB image height to resize output to. If -1, no resizing is done.
    rgb_w: RGB image width to resize output to. If -1, no resizing is done.
    depth_h: Depth image height to resize output to. If -1, no resizing is done.
    depth_w: Depth image width to resize output to. If -1, no resizing is done.
    frame_skip: See __init__.
    interp_mode: See __init__.
  """

  def __init__(self,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear"):
    """
    Args:
      rgb_resolution: Resolution of rgb frames. Set to None to keep the same
        size as original. Either a single integer or a tuple (height, width).
      depth_resolution: Resolution of depth frames. Set to None to keep the same
        size as original. Either a single integer or a tuple (height, width).
      frame_skip: Frame skipping when loading data. Ex. frame_skip: 2, means we
        consume a frame then drop 2 and so on.
      interp_mode: Which pytorch interpolation mode for rgb and feature
        interpolation (Depth and Segmentation always use nearest-exact).
    """

    self.intrinsics_3x3 = None
    if isinstance(rgb_resolution, int):
      self.rgb_h = rgb_resolution
      self.rgb_w = rgb_resolution
    elif hasattr(rgb_resolution, "__len__"):
      self.rgb_h, self.rgb_w = rgb_resolution
    else:
      self.rgb_h = -1
      self.rgb_w = -1

    if isinstance(depth_resolution, int):
      self.depth_h = depth_resolution
      self.depth_w = depth_resolution
    elif hasattr(depth_resolution, "__len__"):
      self.depth_h, self.depth_w = depth_resolution
    else:
      self.depth_h = -1
      self.depth_w = -1

    self.frame_skip = frame_skip
    self.interp_mode = interp_mode

  @abc.abstractmethod
  def __iter__(self):
    """Iterater returning posed RGBD frames in order

    Returns:
      A dict mapping keys {rgb_img, depth_img, pose_4x4} to tensors of shapes
      {3xHxW, 1xH'xW', 4x4} respectively. RGB images are in float32 and have
      (0-1) range. Depth images contain positive values with possible NaNs for
      non valid depth values, +Inf for too far, -Inf for too close. 
      Pose is a 4x4 float32 tensor in opencv RDF. a pose is the extrinsics
      transformation matrix that takes you from camera/robot coordinates to
      world coordinates. Last row should always be [0, 0, 0, 1].

      A confidence map with key {confidence_map} may or may not be included.
      The map is 1xH'xW' float32 tensors in range [0-1] where 0 is least
      confident and 1 is most confident.

      A time stamp key {ts} may or may not be returned with its corresponding
      float32 tensor of shape 1 containing seconds since the epoch.
    """
    pass

class SemSegDataset(PosedRgbdDataset, abc.ABC):
  """Base interface for datasets that provide semantic label images as well.
  
  **Implementing classes must call _init_semseg_mappings with cat_id_to_name 
  mapping to map original dataset ids to a contiguous space.**
  """

  def __init__(self,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear"):
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    self._cat_id_to_name: Dict = None

    self._cat_index_to_id: torch.LongTensor = None
    self._cat_id_to_index: torch.LongTensor = None

    self._cat_index_to_name: List = None
    self._cat_name_to_index: Dict = None

  @property
  def num_classes(self) -> int:
    return len(self._cat_name_to_index)

  @property
  def cat_index_to_name(self) -> List:
    """Returns a mapping from category index to category name.
    
    cat_id_to_name[cat_id] gives the name of that category index.
    """
    return self._cat_index_to_name

  @property
  def cat_name_to_index(self) -> Dict:
    """Returns a mapping from category name to category index.
    
    cat_name_to_index[cat_index] gives the name of that category index.
    """
    return self._cat_name_to_index

  @abc.abstractmethod
  def __iter__(self):
    """Iterater returning posed RGBD frames + semantic segmentaion in order.

    Returns:
      Returns the same items as PosedRgbdDataset.
      In addition semantic segmentation is returned with key {semseg_img} as
      a 1xHxW long tensor specifying class indices for each pixel
      for no semantic label. The user of the class should be able to get each
      class id correspondance to the class name through the cat_index_to_name 
      property
    """
    pass


  def _init_semseg_mappings(self,
                            cat_id_to_name: Dict[int, str],
                            white_list: List[str] = None,
                            black_list: List[str] = None):
    """Computes mapping from ids (original pixel labels) to contiguous indices.

    This function initializes the following mappings to the dataset:
    - _cat_id_to_name: Dict mapping from category id to category name (arg copy)
    - _cat_index_to_id: Tensor mapping category index to category id
    - _cat_id_to_index: Tensor mapping from category id to category index
    - _cat_index_to_name: List mapping category index to category name
    - _cat_name_to_index: Dict mapping category name to category index. 

    We differentiate between a category index and a category id. Ids need not be
    contiguous and must be provided by the original dataset. Indices are
    contiguous indices to be used to directly index a one hot encoded mask.
    Note that we always reserve index/id= 0 as the ignore index.

    Args:
      cat_id_to_name: Dictionary mapping an id to name.
      white_list: A list of category names to include. If None, all categories
        in cat_id_to_name are included. (Cannot be used with black_list)
      black_list: A list of category names to exclude. If None, all categories 
        in cat_id_to_name are included. (Cannot be used with white_list)
    """
    assert isinstance(cat_id_to_name, dict)
    self._cat_id_to_name = cat_id_to_name
    cin = copy.copy(cat_id_to_name)
    cin[0] = ""
    assert white_list is None or len(white_list) == 0 or \
          black_list is None or len(black_list) == 0, \
          "Cannot set both white_list and black_list at the same time"

    if white_list is not None and len(white_list) > 0:
      self._cat_index_to_id = torch.tensor(
        sorted([id for id, name in cin.items()
                if name in white_list or id==0]),
                dtype=torch.long, device="cuda")
    else:
      if black_list is None:
        black_list = []
      self._cat_index_to_id = torch.tensor(
        sorted([id for id, name in cin.items()
                if name not in black_list]),
                dtype=torch.long, device="cuda")

    self._cat_id_to_index = torch.zeros(
      max(cat_id_to_name.keys())+1,
      dtype=torch.long, device="cuda")

    self._cat_id_to_index[self._cat_index_to_id] = \
      torch.arange(len(self._cat_index_to_id),
                  dtype=torch.long, device="cuda")

    num_classes = len(self._cat_index_to_id)

    self._cat_index_to_name = \
      [cin[self._cat_index_to_id[i].item()]
      for i in range(num_classes)]

    self._cat_name_to_index = {n: i for i, n in
                               enumerate(self._cat_index_to_name)}
