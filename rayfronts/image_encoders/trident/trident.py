"""Includes the Trident encoder https://github.com/YuHengsss/Trident

Typical Usage:

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0),size=(512, 512))

  labels = ["car", "person"]

  enc = TridentEncoder(clip_type='openai', 
                       model_type='ViT-L/14', 
                       vfm_model='dino', 
                       sam_model_type='vit_l'
                       sam_ckpt='path/to/ckpt/sam.pth'
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)
  lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

  text_features = enc.encode_labels(labels)

  from rayfronts.utils import compute_cos_sim
  r = compute_cos_sim(text_features, lang_aligned_feat_map, softmax=True)
"""

from typing_extensions import override, List

import numpy as np
import cv2
import torch
from torchvision import transforms
import torch.nn as nn

import os
import sys
trident_path = os.path.dirname(os.path.abspath(__file__))+'/external'
sys.path.append(trident_path)

from rayfronts.image_encoders.base import LangSpatialImageEncoder
from rayfronts.image_encoders.trident.external.open_clip import create_model, tokenizer
from rayfronts.image_encoders.trident.external.segment_anything import sam_model_registry, SamPredictor
from rayfronts.image_encoders.trident.external.seg_utils.utils import preprocess_image


class TridentEncoder(LangSpatialImageEncoder):
  """Defines the Trident encoder."""

  def __init__(self, device: str = None, 
               clip_type: str = 'openai', 
               model_type: str = 'ViT-L/14', 
               vfm_model: str = 'dino', 
               sam_model_type: str = 'vit_l', 
               sam_ckpt: str = None):
    """

    Args:
      device: "cpu" or "cuda", set to None to use CUDA if available.
      clip_type: "openai" by default, indicates the pretrained weights for CLIP
        For more details check out the open_clip implementation in 
        https://github.com/mlfoundations/open_clip
      model_type: 'ViT-L/14', can be replaced with other models detailed in the above link
      vfm_model: 'dino', loads the DINO model from
        https://github.com/facebookresearch/dino
      sam_model_type: 'vit-x' where x can be b, l, or h
        For more details on models and ckpts,
        see https://github.com/facebookresearch/segment-anything
      sam_ckpt: Path to the SAM checkpoint which can be found in link above
    """
    
    super().__init__(device)

    self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
    self.clip.eval().to(self.device)
    self.tokenizer = tokenizer.tokenize
    self.clip_stride = int(model_type[-2:])

    self.vfm_model = vfm_model
    self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    self.vfm = self.vfm.half()
    for p in self.vfm.parameters():
      p.requires_grad = False
    self.vfm.eval().to(self.device)

    self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    self.slide_stride = 112
    self.slide_crop = 336
    self.beta = 1.2
    self.gamma = 3.0
    self.debug = False

    if sam_model_type != 'vit_h':
      self.sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(device=self.device).eval().half()
    self.sam.prompt_encoder = self.sam.prompt_encoder.float()
    self.sam.mask_decoder = self.sam.mask_decoder.float()

    self.cos_fac = 1.2
    self.refine_neg_cos = True
    self.sam_iou_thresh = 0.9
    self.sam_predictor = SamPredictor(self.sam)

    if sam_model_type == 'vit_b':
      self.sam_heads = 12
    elif sam_model_type == 'vit_l':
      self.sam_heads = 16
    elif sam_model_type == 'vit_h':
      self.sam_heads = 16

  @override
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    prompts_per_label = self.insert_labels_into_templates(labels)
    all_text_features = list()
    for i in range(len(labels)):
      text_features = self.encode_prompts(prompts_per_label[i])
      text_features = text_features.mean(dim=0, keepdim=True)
      all_text_features.append(text_features)

    all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features.float()

  @override
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    query = self.tokenizer(prompts).to(self.device)
    text_features = self.clip.encode_text(query)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.float()
  
  def get_sam_feat(self,tmp_img, stride):
    self.sam_predictor.set_image(tmp_img)
    sam_r = 1.0
    sam_valid_h, sam_valid_w = sam_r * self.sam_predictor.input_size[0] // stride, sam_r * \
                                self.sam_predictor.input_size[1] // stride  # get the feature shape in SAM Encoder
    sam_valid_h, sam_valid_w = int(sam_valid_h), int(sam_valid_w)
    sam_enc_feats = self.sam_predictor.features
    sam_enc_feats = sam_enc_feats[:, :, :sam_valid_h, :sam_valid_w]
    sam_hw = int(sam_r * 64)
    sam_attn = self.sam_predictor.model.image_encoder.last_attn
    sam_attn = sam_attn.view(1, self.sam_heads, sam_hw, sam_hw, sam_hw, sam_hw)[:, :, :sam_valid_h, :sam_valid_w,
                :sam_valid_h, :sam_valid_w]
    sam_attn = sam_attn.flatten(2, 3).flatten(3, 4)
    sam_v = self.sam_predictor.model.image_encoder.last_v
    sam_v = sam_v[:, :, :sam_valid_h, :sam_valid_w, :]

    return sam_enc_feats, sam_attn, sam_v, sam_valid_h, sam_valid_w
  
  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    stride = self.clip_stride

    tmp_img = self.tensor_to_cv2(rgb_image)
    tmp_h, tmp_w = tmp_img.shape[:2]
    if tmp_h % stride != 0: tmp_h = (tmp_h // stride + 1) * stride
    if tmp_w % stride != 0: tmp_w = (tmp_w // stride + 1) * stride
    tmp_img = cv2.resize(tmp_img, (tmp_w, tmp_h))
    sam_enc_feats, sam_attn, sam_v, sam_valid_h, sam_valid_w = self.get_sam_feat(tmp_img, 16)

    processed_img = preprocess_image(rgb_image, stride, self.slide_crop)
    clip_whole_h, clip_whole_w = processed_img.shape[-2:]
    clip_feat_h, clip_feat_w = clip_whole_h // stride, clip_whole_w // stride
    img_batch, paddings, patch_locs, win_sizes = self.get_windowed_imgs(processed_img, stride)

    imgs_norm = [self.norm(self.unnorm(img_batch[i])) for i in range(len(img_batch))]
    imgs_norm = torch.stack(imgs_norm, dim=0)
    imgs_norm = imgs_norm.half()
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    if self.vfm_model == 'dino':
        self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
    # Forward pass in the model
    patch_size = self.vfm.patch_embed.patch_size
    if type(patch_size) is tuple: patch_size = patch_size[0]
    feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
    nb_im = feat.shape[0]  # Batch size
    vfm_h = imgs_norm[0].shape[-2] // patch_size
    vfm_w = imgs_norm[0].shape[-1] // patch_size
    vfm_feats = feat[:, 1:, :].reshape(nb_im, vfm_h, vfm_w, -1)
    vfm_feats = vfm_feats.permute(0, 3, 1, 2) #batch, c, h, w

    image_features = self.clip.encode_image(img_batch.half(),
                                            external_feats=vfm_feats, 
                                            beta=self.beta, gamma=self.gamma,
                                            paddings=paddings,
                                            dst_coords=patch_locs,
                                            win_sizes=win_sizes,
                                            dst_vh=clip_feat_h, 
                                            dst_vw=clip_feat_w,
                                            sam_attn=sam_attn, sam_v=sam_v,
                                            cos_fac=self.cos_fac, 
                                            vfm_token_size = (vfm_h, vfm_w),
                                            refine_neg_cos=self.refine_neg_cos)

    B = rgb_image.shape[0]
    image_features = image_features.permute(0, 2, 1)
    image_features = image_features.reshape(B, -1, sam_valid_h, sam_valid_w)
    return image_features.float()

  @override
  def is_compatible_size(self, h: int, w: int):
    return h == 512 and w == 512

  @override
  def get_nearest_size(self, h, w):
    return (512, 512)

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor):
    return features
  
  def get_windowed_imgs(self, img, patch_size=14):
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
    crop_imgs, paddings, patch_locs = [], [], []
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
        patch_locs.append(
          torch.tensor([y1//patch_size, x1//patch_size, 
                        y2//patch_size, x2//patch_size]))
        # pad image when (image_size % patch_size != 0)
        H, W = crop_img.shape[2:]  # original image shape
        pad = self.compute_padsize(H, W, 56)
        if any(pad):
            crop_img = nn.functional.pad(crop_img, pad)  # zero padding
        crop_imgs.append(crop_img)
        paddings.append(pad)
    batched_imgs = torch.cat(crop_imgs, dim=0) # [n_patches, 3, h, w]
    return batched_imgs, paddings, patch_locs, (h_grids,w_grids)
  
  def compute_padsize(self, H: int, W: int, patch_size: int):
    l, r, t, b = 0, 0, 0, 0
    if W % patch_size:
      lr = patch_size - (W % patch_size)
      l = lr // 2
      r = lr - l

    if H % patch_size:
      tb = patch_size - (H % patch_size)
      t = tb // 2
      b = tb - t

    return l, r, t, b
  
  def tensor_to_cv2(self, tensor_images):
    mean = np.array([0.485, 0.456, 0.406])  # mean and std for ImageNet
    std = np.array([0.229, 0.224, 0.225])

    img = tensor_images.cpu().detach().numpy()[0]
    img = np.transpose(img, (1, 2, 0))

    for c in range(img.shape[2]):
      img[:, :, c] = img[:, :, c] * std[c] + mean[c]

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img

class UnNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, image):
    image2 = torch.clone(image)
    for t, m, s in zip(image2, self.mean, self.std):
      t.mul_(s).add_(m)
    return image2