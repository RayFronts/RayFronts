from typing_extensions import override
from typing import List

import torch
import numpy as np

from rayfronts.image_encoders import ImageSemSegEncoder, LangSpatialImageEncoder
from rayfronts import utils

class SemSegWrapEncoder(ImageSemSegEncoder):
  """Special encoder to output semseg labels based on an encoder features
  
  Instead of outputting the features this encoder queries and returns
  probabilities as features.
  """

  def __init__(self,
               encoder: LangSpatialImageEncoder,
               device: str = None,
               classes: List[str] = None,
               predict: bool = False,
               prompt_denoising_thresh: float = 0.5,
               prediction_thresh: float = 0.1,
               chunk_size: int = 10000,
               text_query_mode: str = "labels",
               compute_prob: bool = True,
               interp_mode: str = "bilinear"):

    super().__init__(device)
    self.encoder = encoder
    self.predict = predict
    self.prompt_denoising_thresh = prompt_denoising_thresh
    self.prediction_thresh = prediction_thresh
    self.chunk_size = chunk_size
    self.text_query_mode = text_query_mode
    self.compute_prob = compute_prob
    self.interp_mode = interp_mode
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

    if self.text_query_mode == "labels":
      self.text_embeds = self.encoder.encode_labels(self.prompts)
    elif self.text_query_mode == "prompts":
      self.text_embeds = self.encoder.encode_prompts(self.prompts)
    else:
      raise ValueError("Invalid query type")

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
    B, _, H, W = rgb_image.shape
    feat_img = self.encoder.encode_image_to_feat_map(rgb_image)
    # feat_img = torch.nn.functional.interpolate(
    #   feat_img,
    #   size=(H, W),
    #   mode=self.interp_mode,
    #   antialias=self.interp_mode in ["bilinear", "bicubic"])
    feat_img = self.encoder.align_spatial_features_with_language(feat_img)
    B, C, Hp, Wp = feat_img.shape
    flat_feat_img = feat_img.permute(0, 2, 3, 1).reshape(-1, C)
    num_chunks = int(np.ceil(flat_feat_img.shape[0] / self.chunk_size))
    results = list()
    for c in range(num_chunks):
      sim_vx = utils.compute_cos_sim(
        self.text_embeds,
        flat_feat_img[c*self.chunk_size: (c+1)*self.chunk_size],
        softmax=self.compute_prob)

      if not self.predict:
        results.append(sim_vx)
      else:
        # Prompt denoising
        max_sim = torch.max(sim_vx, dim=0).values
        low_conf_classes = torch.argwhere(
          max_sim < self.prompt_denoising_thresh)
        sim_vx[:, low_conf_classes] = -torch.inf

        sim_value, pred = torch.max(sim_vx, dim=-1)

        # 0 is the ignore id / no pred
        pred += 1
        pred[sim_value < self.prediction_thresh] = 0

        results.append(pred)

    results = torch.cat(results, dim=0).reshape(B, Hp, Wp, -1)
    if self.predict:
      results = torch.nn.functional.one_hot(
        results, self.num_classes).squeeze(-2)
    else:
      results = torch.cat(
        [torch.zeros_like(results[..., :1]), results], dim=-1)

    return results.permute(0, 3, 1, 2).float()


  @override
  def get_nearest_size(self, h, w):
    return h, w

  @override
  def is_compatible_size(self, h, w):
    return True
