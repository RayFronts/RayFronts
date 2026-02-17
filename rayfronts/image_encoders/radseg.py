"""Includes the RADSeg Encoder.

The module only relies on the base.py, prompt_templates.py, and sam_utils.py files in
image_encoders. Encoder can be copied with those files to your own project.

Typical Usage:

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0),size=(512, 512))

  labels = ["car", "person"]

  enc = RADSegEncoder(model_version="c-radio_v3-b", lang_model="siglip2")
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)
  lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

  text_features = enc.encode_labels(labels)

  from rayfronts.utils import compute_cos_sim
  r = compute_cos_sim(text_features, lang_aligned_feat_map, softmax=True)
"""

from typing_extensions import override, List, Tuple

import torch
import numpy as np

from rayfronts.image_encoders.base import ImageSemSegEncoder
from segment_anything import sam_model_registry, SamPredictor
from rayfronts.image_encoders.sam_utils import sam_refinement

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import use_fused_attn
import math

def compute_cos_sim(vec1: torch.FloatTensor,
                    vec2: torch.FloatTensor,
                    softmax: bool = False) -> torch.FloatTensor:
    """Compute cosine similarity between two batches of D dim vectors.

    Args:
    vec1: NxC float tensor representing batch of vectors
    vec2: MxC float tensor representing batch of vectors
    softmax: If False, cosine similarity is returned. If True, softmaxed
        probability is returned across the N dimension.
    Returns:
    result: MxN float tensor representing similarity/prob. where result[0,1]
        represents the similarity of vec1[0] with vec2[1]
    """
    N, C1 = vec1.shape
    M, C2 = vec2.shape
    if C1 != C2:
        raise ValueError(f"vec1 feature dimension '{C1}' does not match vec2"
                        f"feature dimension '{C2}'")
    C = C1

    vec1 = vec1 / vec1.norm(dim = -1, keepdim = True)
    vec1 = vec1.reshape(1, N, 1, C)

    vec2 = vec2 / vec2.norm(dim=-1, keepdim = True)
    vec2 = vec2.reshape(M, 1, C, 1)

    sim = (vec1 @ vec2).reshape(M, N)
    if softmax:
        return torch.softmax(100 * sim, dim = -1)
    else:
        return sim

class SelfCorrelatingRecursiveAttn(nn.Module):
  def __init__(
    self,
    orig_attn,
    device,
    dim: int,
    qk_norm: bool = False,
    scra_scaling: int = 10
  ) -> None:
    super().__init__()
    num_heads = orig_attn.num_heads
    assert dim % num_heads == 0, "dim should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.fused_attn = use_fused_attn()

    self.qkv = orig_attn.qkv
    self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
    self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
    self.attn_drop = orig_attn.attn_drop
    self.proj = orig_attn.proj
    self.proj_drop = orig_attn.proj_drop
    self.device = device
    self.scra_scaling = scra_scaling

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    B, N, C = x.shape
    x_out = self.custom_attn(x.permute(1, 0, 2))
    x_out = x_out.permute(1, 0, 2)
    return x_out

  
  def custom_attn(self, x):
    num_heads = self.num_heads
    num_tokens, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)

    # Self Correlating Recursive Attention 
    attn_output = attn_output.view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    sim_tokens =  F.normalize(attn_output,dim = -1)
    sim_matrix = torch.bmm(sim_tokens,sim_tokens.transpose(1,2))*self.scra_scaling
    sim_matrix[sim_matrix < 0] = -torch.inf
    sim_matrix = F.softmax(sim_matrix,dim=-1)
    attn_output = torch.bmm(sim_matrix, v)    
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)
    
    return attn_output

class RADSegEncoder(ImageSemSegEncoder):
  # TODO: Split into feature encoder and semseg encoder for clarity instead
  # of switching behaviors based on flags.

  def __init__(self,
               device: str = None,
               model_version: str = "radio_v3-b",
               lang_model: str ="siglip2",
               return_radio_features: bool = True,
               compile: bool = False,
               amp: bool = False,
               predict: bool = False,
               classes: List[str] = None,
               text_query_mode: str = "labels",
               prediction_thresh: float = 0.0,
               prompt_denoising_thresh: float = 0.5,
               scra_scaling: float = 10.0,
               scga_scaling: float = 10.0,
               slide_crop: int = 336,
               slide_stride: int = 224,
               # Sam refinement args
               sam_refinement: bool = False,
               sam_ckpt: str = 'sam_vit_h_4b8939.pth',
               coarse_thresh: float = 0.10,
               minimal_area: int = 225,
               sam_mask_coff: float = 0.005,
               sam_iou_thresh: float = 0.9):
    
    super().__init__(device)

    self.compile = compile
    self.amp = amp
    self.model_version = model_version
    self.return_radio_features = return_radio_features
    adaptor_names = [lang_model, "sam"]
    self.model = torch.hub.load("NVlabs/RADIO", "radio_model",
                                version=model_version, progress=True,
                                skip_validation=True,
                                adaptor_names=adaptor_names)
    self.model.eval()
    self.model = self.model.to(self.device)
    # Steal adaptors from RADIO so it does not auto compute adaptor output.
    # We want to control when that happens.
    self.lang_adaptor = self.model.adaptors[lang_model]
    self.sam_adaptor = self.model.adaptors["sam"]
    self.model.adaptors = None
    last_block = self.model.model.blocks[-1]
    last_block.attn = SelfCorrelatingRecursiveAttn(
      last_block.attn,
      dim=self.model.model.embed_dim,
      device=self.device,
      scra_scaling=scra_scaling)

    if self.compile:
      self.model.compile(fullgraph=True, options={"triton.cudagraphs":True})
      self.lang_adaptor.compile(fullgraph=True, options={"triton.cudagraphs":True})

    self.predict = predict
    self.prediction_thresh = prediction_thresh
    self.text_query_mode = text_query_mode
    self.prompt_denoising_thresh = prompt_denoising_thresh

    if self.predict:
      if classes is None or len(classes) < 1:
        raise Exception("Must pass list of classes when predict is True")
      self.prompts = list(classes)

      if len(self.prompts[0]) == 0: # Remove ignore class so we don't prompt
        self.prompts.pop(0)
      self._cat_index_to_name = {0: ''}
      self._cat_index_to_name.update({i+1: v for i,v in enumerate(self.prompts)})
      self._cat_name_to_index = {
        v: k for k, v in self._cat_index_to_name.items()
      }
      if self.text_query_mode == "labels":
        self.text_embeds = self.encode_labels(self.prompts, onehot=False)
      elif self.text_query_mode == "prompts":
        self.text_embeds = self.encode_prompts(self.prompts, onehot=False)
      else:
        raise ValueError("Invalid query type")
      

    self.slide_stride = slide_stride
    self.slide_crop = slide_crop
    self.scga_scaling = scga_scaling

    # Sam refinement args
    self.sam_refinement = sam_refinement
    if sam_refinement:
      self.sam_iou_thresh = sam_iou_thresh
      self.coarse_thresh = coarse_thresh
      self.minimal_area = minimal_area
      self.sam_mask_coff = sam_mask_coff
      self.sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device=self.device).eval()
      self.sam_predictor = SamPredictor(self.sam)
      del self.sam.image_encoder.blocks
      del self.sam.image_encoder.patch_embed
      torch.cuda.empty_cache()

  @property
  @override
  def num_classes(self) -> int:
    return len(self.prompts) + 1

  @property
  @override
  def cat_index_to_name(self):
    return self._cat_index_to_name

  @property
  @override
  def cat_name_to_index(self):
    return self._cat_name_to_index

  @override
  def encode_labels(self, labels: List[str], onehot: bool = True) -> torch.FloatTensor:
    if self.predict and onehot:
      return super().encode_labels(labels)
    prompts_per_label = self.insert_labels_into_templates(labels)
    all_text_features = list()
    for i in range(len(labels)):
      text_features = self.encode_prompts(prompts_per_label[i], onehot=False)
      text_features = text_features.mean(dim=0, keepdim=True)
      all_text_features.append(text_features)

    all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features

  @override
  def encode_prompts(self, prompts: List[str], onehot: bool = True) -> torch.FloatTensor:
    if self.predict and onehot:
      return super().encode_labels(prompts)
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      text = self.lang_adaptor.tokenizer(prompts).to(self.device)
      text_features = self.lang_adaptor.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor, orig_img_size: Tuple[int] = None,
    return_preds: bool = False, ignore_label: bool = True) -> torch.FloatTensor:
    if orig_img_size is None:
      orig_img_size = rgb_image.shape[-2:]
    
    use_scga = False
    if self.slide_crop > 0:
      feat_map, num_windows = self._sliding_inference(
        rgb_image,
        stride=self.slide_stride,
        crop_size=self.slide_crop,
      )
      # Only apply SCGA when sliding-window actually used multiple windows.
      # If sliding-window degenerates to a single window (e.g. input smaller
      # than crop size), SCGA is skipped.
      use_scga = num_windows > 1
    else:
      feat_map = self._single_inference(rgb_image)
    
    if use_scga:
      feat_map = self._self_correlating_global_aggregation(feat_map)
    if not self.predict:
      return feat_map

    # Predict
    seg_probs = self._get_seg_logits(feat_map)
    seg_probs = nn.functional.interpolate(
      seg_probs, size=orig_img_size, mode='bilinear', align_corners=False)

    B, C, H, W = seg_probs.shape
    num_cls = C

    # Prompt denoising
    seg_probs = seg_probs.permute(0, 2, 3, 1).reshape(-1, C)
    max_sim_per_class = torch.max(seg_probs, dim=0).values

    low_conf_classes = torch.argwhere(max_sim_per_class < self.prompt_denoising_thresh)
    seg_probs[:, low_conf_classes] = 0

    seg_probs = seg_probs.reshape(B, H, W, C).permute(0, 3, 1, 2) # BxCxHxW
    max_sim_per_pixel, seg_pred = torch.max(seg_probs, dim=1, keepdim=True) # Bx1xHxW

    if self.sam_refinement:
      seg_pred_ref = list()
      seg_probs_ref = list()
      for b in range(B):
        sam_image, new_h, new_w = self._preprocess_sam(rgb_image[b], target_size=1024)
        image_features = self._single_inference(sam_image)
        sam_features = self._get_sam_spatial_features(image_features).float()
        sam_features = self._interpolate_to_sam_dims(sam_features)
        sam_features = self.sam_predictor.model.image_encoder.neck(sam_features)
        self.sam_predictor.features = sam_features
        self.sam_predictor.is_image_set = True
        self.sam_predictor.original_size = orig_img_size
        self.sam_predictor.input_size = (new_h, new_w)
        
        refined_masks, scores, refined_logits, prompt_boxes = sam_refinement(
          orig_img_size, seg_pred[b], seg_probs[b], num_cls, self.sam_predictor,
          self.coarse_thresh, self.minimal_area,
          self.sam_mask_coff, self.sam_iou_thresh)
      
        seg_pred_ref.append(refined_masks)
        seg_probs_ref.append(refined_logits)
      seg_pred = torch.stack(seg_pred_ref, dim=0)
      seg_probs = torch.stack(seg_probs_ref, dim=0)

    # Set low confidence predictions to the ignore label
    if ignore_label:
      seg_pred += 1
      seg_pred[max_sim_per_pixel < self.prediction_thresh] = 0
      seg_probs = torch.cat(
        [torch.zeros_like(seg_probs[:, :1, :, :]), seg_probs], dim=1)
    if return_preds:
      return seg_probs, seg_pred
    else:
      return seg_probs

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor,
                                           onehot: bool = True):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features or (self.predict and onehot):
      return features
    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.lang_adaptor.head_mlp(features)
    return out.permute(0, 2, 1).reshape(B, -1, H, W)

  def _get_sam_spatial_features(self, features: torch.FloatTensor):
    if self.sam_adaptor is None:
      raise ValueError("Cannot align to sam without a sam model")
  
    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)

    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.sam_adaptor.feat_mlp(features)

    return out.permute(0, 2, 1).reshape(B, -1, H, W)
  
  @override
  def is_compatible_size(self, h: int, w: int):
    hh, ww = self.get_nearest_size(h, w)
    return hh == h and ww == w

  @override
  def get_nearest_size(self, h, w):
    return self.model.get_nearest_supported_resolution(h, w)

  def _interpolate_to_sam_dims(self, feature_map):
    feat_h = feature_map.shape[2]
    feat_w = feature_map.shape[3]
    feature_map = F.interpolate(
      feature_map, size=(feat_h, feat_w), mode='bilinear',
      align_corners=False, antialias=True)
    feature_map = F.pad(feature_map, (0, 64 - feat_w, 0, 64 - feat_h))
    return feature_map

  def _preprocess_sam(self, image: torch.Tensor, target_size: int = 1024) -> torch.Tensor:
    """
    Preprocess an image tensor for SAM.
    
    Args:
        image (torch.Tensor): Tensor of shape (C, H, W), values in [0, 255] or [0, 1].
        target_size (int): Size to pad/resize to (default=1024).
    
    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 3, target_size, target_size).
    """
    if image.dim() != 3 or image.shape[0] != 3:
      raise ValueError("Input must have shape (3, H, W)")

    C, H, W = image.shape

    if H >= W:
      new_h = target_size
      new_w = int(W * 64 /H) * 16
    else:
      new_w = target_size
      new_h = int(H * 64 / W) * 16

    image = image.unsqueeze(0)  # add batch dim
    image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)

    return image, new_h, new_w

  def _self_correlating_global_aggregation(self,feat_map):

    b,tokens,h,w = feat_map.shape
    feat_map = feat_map.flatten(2,3).transpose(1,2)
    sim_tokens = F.normalize(feat_map,dim = -1)
    sim_matrix = torch.bmm(sim_tokens,sim_tokens.transpose(1,2))
    sim_matrix = (sim_matrix - torch.mean(sim_matrix)) * self.scga_scaling
    sim_matrix[sim_matrix < 0] = -torch.inf
    sim_matrix = F.softmax(sim_matrix,dim=-1)
    attn_output = torch.bmm(sim_matrix, feat_map)
    attn_output = attn_output.transpose(1,2).view(b,tokens,h,w)    

    return attn_output

  def _get_seg_logits(self, feat_map):

    image_features = self.align_spatial_features_with_language(feat_map, onehot=False)
    feat_map = image_features.permute(0, 2, 3, 1)

    B,H,W,C = feat_map.shape
    feat_map = feat_map.reshape(-1, C)

    cos_sim = compute_cos_sim(
      self.text_embeds, feat_map,
      softmax=True)

    N = len(self.text_embeds)

    cos_sim = cos_sim.reshape(B,H,W,N)
    cos_sim = cos_sim.permute(0,3,1,2)
    
    return cos_sim

  def _preprocess_image(self, image, stride=16, slide_crop=336):
    longer_side = max(image.shape[2:])
    h, w = image.shape[2:]
    if h%stride == 0 and w%stride == 0:
      return image
    if longer_side % stride != 0:
      dst_longer = (longer_side // stride + 1) * stride
    else:
      dst_longer = longer_side
    new_h = int(h * dst_longer / longer_side)
    new_w = int(w * dst_longer / longer_side)
    if new_h % stride != 0: new_h = (new_h // stride + 1) * stride
    if new_w % stride != 0: new_w = (new_w // stride + 1) * stride
    new_h, new_w = max(new_h, slide_crop), max(new_w, slide_crop)
    image = torch.nn.functional.interpolate(image, (new_h, new_w), mode='bilinear', align_corners=False)

    return image

  def _get_windowed_imgs(self, img, patch_size=16):
    stride, crop_size = self.slide_stride, self.slide_crop
    if type(img) == list:
      img = img[0].unsqueeze(0)
    if type(stride) == int:
      stride = (stride, stride)
    if type(crop_size) == int:
      crop_size = (crop_size, crop_size)

    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    crop_imgs, patch_locs = [], []
    for h_idx in range(h_grids):
      for w_idx in range(w_grids):
        y1 = h_idx * h_stride
        x1 = w_idx * w_stride
        y2 = min(y1 + h_crop, h_img)
        x2 = min(x1 + w_crop, w_img)
        y1 = max(y2 - h_crop, 0)
        x1 = max(x2 - w_crop, 0)
        crop_img = img[:, :, y1:y2, x1:x2]
        assert y1 % patch_size == 0 and x1 % patch_size == 0
        assert y2 % patch_size == 0 and x2 % patch_size == 0
        patch_locs.append(torch.tensor([y1//patch_size, x1//patch_size, y2//patch_size, x2//patch_size]))
        # pad image when (image_size % patch_size != 0)
        crop_imgs.append(crop_img)

    batched_imgs = torch.cat(crop_imgs, dim=0) # [n_patches, 3, h, w]
    return batched_imgs, patch_locs, (h_grids,w_grids)

  def _single_inference(self, rgb_image):
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.model(rgb_image).features
      if not self.return_radio_features:
        out = self.lang_adaptor.head_mlp(out)
    return out.permute(0, 2, 1).reshape(B, -1, H_, W_)

  def _sliding_inference(self, img, stride=224, crop_size=336):
    """
    Inference by sliding-window with overlap. If h_crop > h_img or w_crop > w_img,
    the small patch will be used to decode without padding.
    """

    if type(img) == list:
      img = img[0].unsqueeze(0)
    if type(stride) == int:
      stride = (stride, stride)
    if type(crop_size) == int:
      crop_size = (crop_size, crop_size)
        
    img = self._preprocess_image(img, stride[0], crop_size[0])
    
    batched_imgs, patch_locs, (h_grids,w_grids) = self._get_windowed_imgs(img, patch_size=16)
    batch_size = img.shape[0]
    _, _, h_img, w_img = img.shape
    
    image_feats = self._single_inference(batched_imgs)
    feat_dim = image_feats.shape[1]
    dtype = image_feats.dtype
    device = image_feats.device
    h_feat = math.ceil(h_img / 16)
    w_feat = math.ceil(w_img / 16)
    feat_map = torch.zeros((batch_size, feat_dim, h_feat, w_feat),dtype = dtype, device=device)
    count_mat = torch.zeros((batch_size, 1, h_feat, w_feat),dtype = dtype, device=device)
    for h_idx in range(h_grids):
      for w_idx in range(w_grids):
        coord = patch_locs[h_idx*w_grids+w_idx]
        img_feat = image_feats[h_idx*w_grids+w_idx]
        feat_map[:, :,coord[0]:coord[2],coord[1]:coord[3]] += img_feat
        count_mat[:, :,coord[0]:coord[2],coord[1]:coord[3]] += 1
    feat_map = feat_map / count_mat  # 1, D, dst_h, dst_w

    num_windows = h_grids * w_grids
    return feat_map, num_windows
