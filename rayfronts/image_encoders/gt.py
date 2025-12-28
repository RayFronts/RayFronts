from typing_extensions import override
from typing import List

import torch

from rayfronts.image_encoders import ImageSemSegEncoder

class GTEncoder(ImageSemSegEncoder):
  """Special encoder to send GT segmentations as the feature maps."""

  def __init__(self,
               device: str = None,
               classes: List[str] = None):
    super().__init__(device)
    self.prompts = classes
    self._cat_id_to_name = None
    if self.prompts is not None and len(self.prompts) > 0:
      self.prompts = list(self.prompts)
      if len(self.prompts[0]) == 0: # Remove ignore class so we don't prompt
        self.prompts.pop(0)
      self._cat_index_to_name = {0: ''}
      self._cat_index_to_name.update({i+1: v for i,v in enumerate(self.prompts)})
      self._cat_name_to_index = {
        v: k for k, v in self._cat_index_to_name.items()
      }

  @property
  @override
  def num_classes(self):
    if self.prompts is None:
      return 0
    else:
      return len(self.prompts) + 1 # + 1 for ignore label

  @property
  @override
  def cat_index_to_name(self):
    return self._cat_index_to_name

  @property
  @override
  def cat_name_to_index(self):
    return self._cat_name_to_index

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    raise NotImplementedError(
      "GT encoder acts as a placeholder to encode text only. "
      "Feature map is equal to the semseg ground truth onehot encoded.")

  @override
  def get_nearest_size(self, h, w):
    return h, w

  @override
  def is_compatible_size(self, h, w):
    return True
