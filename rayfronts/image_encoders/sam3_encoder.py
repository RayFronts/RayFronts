"""Includes the SAM 3 (Segment Anything with Concepts) Encoder.

Uses the official Meta SAM 3 model for open-vocabulary semantic segmentation
via text prompts. See https://github.com/facebookresearch/sam3.

Typical Usage:

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0), size=(512, 512))

  labels = ["car", "person"]

  enc = SAM3SemSegEncoder(classes=labels)

  feat_map = enc.encode_image_to_feat_map(rgb_img)
  text_features = enc.encode_labels(labels)
"""

from typing_extensions import override
from typing import List, Tuple

import torch

from rayfronts.image_encoders.base import ImageSemSegEncoder
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import BatchedInferenceMetadata, FindStage
from sam3.eval.postprocessors import PostProcessImage


def _wrap_single_stage(out, find_meta):
  """Wraps one forward_grounding output for PostProcessImage.process_results.

  The postprocessor expects find_stages to have .loss_stages and to be iterable;
  we pass a single stage so loss_stages is None and iteration yields (out,).
  """

  class _SingleStage:
    loss_stages = None

    def __len__(self):
      return 1

    def __iter__(self):
      return iter([out])

  return _SingleStage(), [find_meta]


class SAM3SemSegEncoder(ImageSemSegEncoder):
  """Semantic segmentation encoder using Meta SAM 3 with text prompts.

  SAM 3 supports open-vocabulary segmentation via text. This encoder runs
  SAM 3 with all class prompts in a single batched forward per image and
  aggregates instance masks into a per-class feature map compatible with
  RayFronts mappers.
  """

  # Official SAM 3 backbone and checkpoint expect this input size; do not change.
  RESOLUTION = 1008

  def __init__(
    self,
    device: str = None,
    classes: List[str] = None,
    confidence_threshold: float = 0.5,
    checkpoint_path: str = None,
    load_from_hf: bool = True,
    prompts_per_forward: int = 16,
  ):
    """Initializes the SAM 3 encoder.

    Args:
      device: Device to run the model on (default: cuda if available).
      classes: List of class names used as text prompts. Index 0 is reserved
        for ignore/unlabeled. If None, must be set later via dataset config.
      confidence_threshold: Minimum score to keep a detection (default: 0.5).
      checkpoint_path: Path to SAM 3 checkpoint. If None and load_from_hf True,
        downloads from Hugging Face (requires access and hf auth).
      load_from_hf: Whether to load checkpoint from Hugging Face if no path
        is given (default: True).
      prompts_per_forward: Max number of text prompts per model forward. Prompts
        are processed in chunks of this size to limit GPU memory. Lower values
        use less memory but more forwards (default: 16).
    """
    super().__init__(device)

    self._resolution = self.RESOLUTION
    self._confidence_threshold = confidence_threshold
    self._prompts_per_forward = max(1, prompts_per_forward)
    self._model = build_sam3_image_model(
      device=self.device,
      eval_mode=True,
      checkpoint_path=checkpoint_path,
      load_from_HF=load_from_hf,
      enable_segmentation=True,
      enable_inst_interactivity=False,
    )

    self._postprocessor = PostProcessImage(
      max_dets_per_img=-1,
      iou_type="segm",
      use_original_sizes_box=True,
      use_original_sizes_mask=True,
      convert_mask_to_rle=False,
      detection_threshold=confidence_threshold,
      to_cpu=False,
    ).to(self.device)

    self.prompts = classes
    self._cat_index_to_name = None
    self._cat_name_to_index = None
    self._text_backbone_out = None
    if self.prompts is not None and len(self.prompts) > 0:
      self.prompts = list(self.prompts)
      if len(self.prompts[0]) == 0:
        self.prompts.pop(0)
      self._cat_index_to_name = {0: ""}
      self._cat_index_to_name.update(
        {i + 1: v for i, v in enumerate(self.prompts)}
      )
      self._cat_name_to_index = {
        v: k for k, v in self._cat_index_to_name.items()
      }
      with torch.inference_mode():
        self._text_backbone_out = self._model.backbone.forward_text(
          self.prompts, device=self.device
        )

  def _ensure_text_encoded(self) -> None:
    """Encode all prompts once if not already cached (e.g. prompts set after init)."""
    if self._text_backbone_out is None and self.prompts is not None and len(self.prompts) > 0:
      with torch.inference_mode():
        self._text_backbone_out = self._model.backbone.forward_text(
          self.prompts, device=self.device
        )

  def _slice_text_backbone_out(self, start: int, end: int) -> dict:
    """Returns text backbone outputs for prompts[start:end] from precomputed.
    Shapes: language_features (seq, num_prompts, 256), language_mask (num_prompts, seq),
    language_embeds (seq, num_prompts, 1024).
    """
    self._ensure_text_encoded()
    t = self._text_backbone_out
    return {
      "language_features": t["language_features"][:, start:end].contiguous(),
      "language_mask": t["language_mask"][start:end].contiguous(),
      "language_embeds": t["language_embeds"][:, start:end].contiguous(),
    }

  @property
  @override
  def num_classes(self) -> int:
    if self.prompts is None:
      return 0
    return len(self.prompts) + 1

  @property
  @override
  def cat_index_to_name(self):
    return self._cat_index_to_name

  @property
  @override
  def cat_name_to_index(self):
    return self._cat_name_to_index

  def _tensor_to_image_batch(self, img: torch.Tensor):
    """Preprocesses RGB tensor on GPU and returns (img_batch, orig_h, orig_w).

    Input: img (3, H, W) or (1, 3, H, W) in [0, 1], any device.
    Output: img_batch (1, 3, resolution, resolution) normalized, on self.device.
    Same normalization as SAM3 transform: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].
    """
    if img.dim() == 4:
      img = img.squeeze(0)
    orig_h, orig_w = img.shape[-2], img.shape[-1]
    img = img.to(device=self.device, dtype=torch.float32)
    img = img.clamp(0.0, 1.0)
    # Resize to backbone resolution (same as RandomResizeAPI(sizes=resolution, square=True))
    img = torch.nn.functional.interpolate(
      img.unsqueeze(0),
      size=(self._resolution, self._resolution),
      mode="bilinear",
      align_corners=False,
    )
    # Normalize: (x - 0.5) / 0.5
    img = (img - 0.5) / 0.5
    return img, orig_h, orig_w

  def _build_find_stage_and_metadata(
    self, n: int, query_ids: List[int], orig_h: int, orig_w: int
  ):
    """Builds FindStage and BatchedInferenceMetadata for n text-only queries."""
    device = self.device
    find_stage = FindStage(
      img_ids=torch.zeros(n, dtype=torch.long, device=device),
      text_ids=torch.arange(n, dtype=torch.long, device=device),
      input_boxes=torch.zeros(0, n, 4, device=device),
      input_boxes_mask=torch.ones(0, n, device=device, dtype=torch.bool),
      input_boxes_label=torch.zeros(0, n, dtype=torch.long, device=device),
      input_points=torch.empty(n, 0, 257, device=device),
      input_points_mask=torch.zeros(n, 0, device=device),
      object_ids=None,
    )
    orig_size = torch.tensor([[orig_h, orig_w]] * n, dtype=torch.long, device=device)
    meta = BatchedInferenceMetadata(
      coco_image_id=torch.tensor(query_ids, dtype=torch.long, device=device),
      original_image_id=torch.tensor(query_ids, dtype=torch.long, device=device),
      original_category_id=torch.ones(n, dtype=torch.int, device=device),
      original_size=orig_size,
      object_id=torch.zeros(n, dtype=torch.long, device=device),
      frame_index=torch.zeros(n, dtype=torch.long, device=device),
      is_conditioning_only=[False] * n,
    )
    return find_stage, meta

  def _channels_from_processed(
    self, processed: dict, query_ids: List[int], orig_h: int, orig_w: int
  ) -> List[torch.Tensor]:
    """Builds one channel tensor per query id from postprocessor output."""
    channels = []
    for qid in query_ids:
      result = processed.get(qid)
      if result is None:
        channel = torch.zeros(
          (1, 1, orig_h, orig_w),
          dtype=torch.float32,
          device=self.device,
        )
      else:
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None or scores is None or masks.shape[0] == 0:
          channel = torch.zeros(
            (1, 1, orig_h, orig_w),
            dtype=torch.float32,
            device=self.device,
          )
        else:
          if scores.dim() == 1:
            scores = scores.unsqueeze(1)
          weighted = masks.float() * scores.view(-1, 1, 1, 1)
          channel, _ = weighted.max(dim=0, keepdim=True)
      channels.append(channel)
    return channels

  def _encode_single_image(
    self, rgb_image: torch.FloatTensor
  ) -> torch.FloatTensor:
    """Encodes a single image (1, 3, H, W) to (1, num_classes, H, W).

    Encodes the image once with the backbone, then runs prompt chunks through
    text encoder + grounding only (no repeated image encoding).
    """
    B, C, H, W = rgb_image.shape
    assert B == 1, "SAM3 encoder processes one image at a time"
    img = rgb_image[0]

    # Preprocess on GPU (resize + normalize), no PIL.
    with torch.inference_mode():
      img_batch, orig_h, orig_w = self._tensor_to_image_batch(img)
      image_backbone_out = {"img_batch_all_stages": img_batch}
      image_backbone_out.update(
        self._model.backbone.forward_image(img_batch)
      )

    chunk_size = self._prompts_per_forward
    all_channels = []
    base_qid = 1

    for start in range(0, len(self.prompts), chunk_size):
      end = min(start + chunk_size, len(self.prompts))
      prompt_slice = self.prompts[start:end]
      query_ids = list(range(base_qid, base_qid + len(prompt_slice)))
      base_qid += len(prompt_slice)

      with torch.inference_mode():
        text_outputs = self._slice_text_backbone_out(start, end)
        chunk_backbone_out = {**image_backbone_out, **text_outputs}
        find_input, find_meta = self._build_find_stage_and_metadata(
          len(prompt_slice), query_ids, orig_h, orig_w
        )
        geometric_prompt = self._model._get_dummy_prompt(num_prompts=len(prompt_slice))
        out = self._model.forward_grounding(
          backbone_out=chunk_backbone_out,
          find_input=find_input,
          find_target=None,
          geometric_prompt=geometric_prompt,
        )

      find_stages, find_metadatas = _wrap_single_stage(out, find_meta)
      processed = self._postprocessor.process_results(find_stages, find_metadatas)
      all_channels.extend(
        self._channels_from_processed(processed, query_ids, orig_h, orig_w)
      )

    feat = torch.cat(all_channels, dim=1)
    if orig_h != H or orig_w != W:
      feat = torch.nn.functional.interpolate(
        feat, size=(H, W), mode="bilinear", align_corners=False
      )
    ignore_ch = torch.full(
      (1, 1, H, W), self.eps, dtype=feat.dtype, device=self.device
    )
    feat = torch.cat([ignore_ch, feat], dim=1)
    return feat

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor
  ) -> torch.FloatTensor:
    B, _, H, W = rgb_image.shape
    feat_map = torch.full(
      (B, self.num_classes, H, W),
      self.eps,
      dtype=torch.float,
      device=self.device,
    )
    for b in range(B):
      feat_map[b : b + 1] = self._encode_single_image(rgb_image[b : b + 1])
    return feat_map

  @override
  def get_nearest_size(self, h: int, w: int) -> Tuple[int, int]:
    return h, w

  @override
  def is_compatible_size(self, h: int, w: int) -> bool:
    return True
