import abc
from typing_extensions import Tuple, List, override
import torch

from rayfronts.image_encoders.prompt_templates import openai_imagenet_template

class ImageEncoder(abc.ABC):
  """Interface for all image encoders"""

  def __init__(self, device = None):
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

  @abc.abstractmethod
  def is_compatible_size(self, h: int, w: int) -> bool:
    """Checks if height and width of image is compatible with encoder"""
    pass

  @abc.abstractmethod
  def get_nearest_size(self, h, w) -> Tuple[int, int]:
    """Returns nearest compatible size as (h,w) tuple"""
    pass

class ImageGlobalEncoder(ImageEncoder):
  """Interface for all image level (global) encoders"""

  @abc.abstractmethod
  def encode_image_to_vector(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    """Encode rgb image into an image level feature vector

    Args:
      rgb_image: Bx3xHxW float tensor representing an image with values in [0-1]
    Returns:
      BxC float tensor representing global features.
    """

class ImageSpatialEncoder(ImageEncoder):
  """Interface for all pixel level / patch level encoders"""

  @abc.abstractmethod
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    """Encode rgb image into a feature map

    Args:
      rgb_image: Bx3xHxW float tensor representing an image with values in [0-1]
    Returns:
      BxCxH'xW' float tensor representing pixel / patch level features.
    """

class ImageSpatialGlobalEncoder(ImageSpatialEncoder, ImageGlobalEncoder):
  "Interface for encoders that implement both global and spatial encoding APIs"

  @abc.abstractmethod
  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) \
      -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Encode rgb image into a feature map and an image level feature vector
    
    Args:
      rgb_image: Bx3xHxW float tensor representing an image with values in [0-1]
    Returns:
      A tuple of 
      - BxCxH'xW' float tensor representing pixel / patch level features.
      - BxC float tensor representing global features.
    """

class LangImageEncoder(ImageEncoder):
  """Interface for all image encoders that produce language aligned features"""

  def __init__(self, device = None):
    super().__init__(device)
    self.prompt_templates = openai_imagenet_template

  def insert_labels_into_templates(self, labels: List[str]) -> List[List[str]]:
    """Inserts each labels into a set of stored templates.
    
    Args:
      labels: A list of length T of class names / labels. Ex. ['cat', 'dog']
    Returns:
      labeled_templates as a list of length T of a list of length P of strings.
      where T is the number of labels and P is the number of templates.
    """
    return [[pt(x) for pt in self.prompt_templates] for x in labels]

  @abc.abstractmethod
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    """Encodes a list of labels into feature vectors using template prompts
    
    Args:
      labels: A list of length T of class names / labels. Ex. ['cat', 'dog']
    Returns:
      A TxD float tensor where T represents the number of classes and D 
      represents the feature space dimension.
    """
    pass

  @abc.abstractmethod
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    """Encodes a list of prompts into feature vectors directly
    
    Args:
      prompts: A list of strings of arbitrary length of prompts to encode.
    Returns:
      A TxD float tensor where T represents the number of prompts and D 
      represents the feature space dimension.
    """
    pass

class LangSpatialImageEncoder(LangImageEncoder, ImageSpatialEncoder):

  @abc.abstractmethod
  def align_spatial_features_with_language(
    self, features: torch.FloatTensor) -> torch.FloatTensor:
    """Projects image features to the language space if it is not already there.

    Some encoders directly output the language aligned features in which case
    this function will simply be a pass through.

    Args:
      features: A (BxCxHxW) float tensor representing spatial features obtained
        from encode_image_to_feat_map.
    Returns:
      A (BxDxHxW) float tensor that is in the same feature space as the text 
      embeddings obtained from encode_labels or encode_prompts.
    """
    pass

class LangGlobalImageEncoder(LangImageEncoder, ImageGlobalEncoder):

  @abc.abstractmethod
  def align_global_features_with_language(
    self, features: torch.FloatTensor) -> torch.FloatTensor:
    """Projects image global features to the language space if needed

    Some encoders directly output the language aligned features in which case
    this function will simply be a pass through.

    Args:
      features: A (BxC) float tensor representing global features obtained
        from encode_image_to_vector.
    Returns:
      A (BxD) float tensor that is in the same feature space as the text 
      embeddings obtained from encode_labels or encode_prompts.
    """
    pass


class LangSpatialGlobalImageEncoder(LangGlobalImageEncoder,
                                    LangSpatialImageEncoder):
  pass


class ImageSemSegEncoder(LangSpatialImageEncoder):
  """Interface for encoders that are actually semantic segmentors.
  
  To avoid rewriting major portions of the codebase. We consider semantic
  segmentation as simply a onehot encoder but nevertheless still producing a 
  feature vector that can be aggregated and compared with using cos-sim...etc.
  Would be more efficient to deal with integers of course but we create this
  for convenience. This way these "encoders" can be used with existing  mappers.
  """

  def __init__(self, device = None):
    super().__init__(device)
    self.eps = 1e-10

  @property
  @abc.abstractmethod
  def num_classes(self) -> int:
    pass

  @property
  @abc.abstractmethod
  def cat_index_to_name(self):
    """Returns a mapping from category index to category name.
    
    cat_index_to_name[cat_index] gives the name of that category index.
    """
    pass

  @property
  @abc.abstractmethod
  def cat_name_to_index(self):
    """Returns a mapping from category index to category name.
    
    cat_name_to_index[cat_name] gives the one hot index of that category name.
    """
    pass

  @override
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    """Encodes a list of category names into onehot vectors using stored mapping
    
    Args:
      labels: A list of length T of category names / labels. Ex. ['cat', 'dog']
    Returns:
      A TxD float tensor where T represents the number of classes and D 
      represents the feature space dimension.
    """
    onehot = torch.full((len(labels), self.num_classes), self.eps,
                         dtype=torch.float, device=self.device)
    for i,c in enumerate(labels):
      onehot[i, self.cat_name_to_index[c]] = 1
    return onehot

  @override
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    """alias for encode_labels"""
    return self.encode_labels(prompts)

  @override
  def align_spatial_features_with_language(self, features):
    return features
