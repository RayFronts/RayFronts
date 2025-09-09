from typing_extensions import override, Optional, Union, Any

from dataclasses import dataclass
from typing import List, Tuple
import math

import torch
from transformers import (AutoModelForMaskGeneration,
                          AutoModelForZeroShotObjectDetection,
                          AutoProcessor)

from rayfronts.image_encoders import ImageSemSegEncoder

class GroundedSamSemSegEncoder(ImageSemSegEncoder):
  """Performs semantic segmentation using grounding dino + sam"""

  def __init__(self,
               device: str = None,
               detector_id: str = "IDEA-Research/grounding-dino-tiny",
               segmentor_id: str = "facebook/sam-vit-base",
               classes: List[str] = None,
               prompt_chunk_size: int = 64,
               det_box_threshold: float = 0.25,
               det_text_threshold: float = 0.25):
    super().__init__(device)
    self.det_box_threshold = det_box_threshold
    self.det_text_threshold = det_text_threshold
    self.prompt_chunk_size = prompt_chunk_size
    self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
      detector_id).to(self.device)
    self.det_processor = AutoProcessor.from_pretrained(
      detector_id, do_rescale=False, use_fast=True)
    self.segmentor = AutoModelForMaskGeneration.from_pretrained(
      segmentor_id).to(self.device)

    self.seg_processor = AutoProcessor.from_pretrained(
      segmentor_id, do_rescale=False, use_fast=True)
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

  
  def _gdino_uncombine(self,
                       masks: torch.BoolTensor,
                       text_labels: List[str]
                       ) -> Tuple[torch.BoolTensor, torch.LongTensor]:
    """Uncombines labels into distinct masks

    Args:
      masks: Nx3xHxW booleant tensor.
      text_labels: List[str] of length N.
    Returns:
      Tuple of (masks, class indices) uncombined.
    """
    masks_uncombined = list()
    indices_uncombined = list()
    mask_indices = list()
    for i, tl in enumerate(text_labels):
      for p in self.prompts:
        if p in tl.split(" "):
          cls_i = self.cat_name_to_index[p]
          indices_uncombined.append(cls_i)
          mask_indices.append(i)

    indices_uncombined = torch.tensor(indices_uncombined,
                                      dtype=torch.long, device=self.device)
    masks_uncombined = masks[torch.tensor(mask_indices,
                             dtype=torch.long, device=self.device)]
    return masks_uncombined, indices_uncombined

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    B, C, H, W = rgb_image.shape

    # GDino is limited in how many images it can prompt at once.
    num_prompt_chunks = math.ceil(len(self.prompts)/self.prompt_chunk_size)
    n = self.prompt_chunk_size
    all_dets = list()
    for c in range(num_prompt_chunks):
      prompts = ". ".join(self.prompts[c*n:(c+1)*n])
      inputs = self.det_processor(
        images=rgb_image, text=[prompts]*B,
        return_tensors="pt").to(self.device)

      detections = self.detector(**inputs)

      detections = self.det_processor.post_process_grounded_object_detection(
        detections, target_sizes=[(H, W)]*B,
        threshold=self.det_box_threshold,
        text_threshold=self.det_text_threshold,
      )
      all_dets.append(detections)

    # Merge detections
    detections = all_dets[0]
    for b in range(B):
      for i in range(1, num_prompt_chunks):
        detections[b]["scores"] = torch.cat(
          (detections[b]["scores"], all_dets[i][b]["scores"]), dim=0)
        detections[b]["boxes"] = torch.cat(
          (detections[b]["boxes"], all_dets[i][b]["boxes"]), dim=0)
        detections[b]["text_labels"].extend(all_dets[i][b]["text_labels"])

    input_boxes = [detections[i]["boxes"].cpu().tolist()
                   for i in range(len(detections))]
    inputs = self.seg_processor(
      images=rgb_image, input_boxes=input_boxes,
      return_tensors="pt").to(self.device)
    sam_outputs = self.segmentor(**inputs)
    masks = self.seg_processor.post_process_masks(
      masks=sam_outputs.pred_masks,
      original_sizes=inputs.original_sizes,
      reshaped_input_sizes=inputs.reshaped_input_sizes
    )
    feat_map = torch.full((B, self.num_classes, H, W), self.eps,
                          dtype=torch.float, device=self.device)
    for i in range(len(masks)):
      cur_masks, class_indices = self._gdino_uncombine(
        masks[i], detections[i]["text_labels"])

      feat_map[i, class_indices] += torch.sum(cur_masks.float(), dim=1)
      assert class_indices.min() > 0
    return feat_map


  @override
  def get_nearest_size(self, h, w):
    return h, w

  @override
  def is_compatible_size(self, h, w):
    return True

  @staticmethod
  def _vis_detections(rgb_image, detections):
    """Visualize output of grounding dino detections.
    
    If batched then only visualize the last.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    results = detections[-1]
    fig, ax = plt.subplots()
    ax.imshow(rgb_image[-1].permute(1,2,0).cpu())
    scores = results["scores"]
    text_labels = results["text_labels"]
    boxes = results["boxes"]

    for box, score, text_label in zip(boxes, scores, text_labels):
      xmin, ymin, xmax, ymax = box.cpu().tolist()
      width, height = xmax - xmin, ymax - ymin

      # Add rectangle
      rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1,
                                edgecolor='red', facecolor='none')
      ax.add_patch(rect)

      # Add text label
      ax.text(
        xmin, ymin,
        f"{text_label}: {round(score.item(),2)}",
        color="white", fontsize=8,
        bbox=dict(facecolor='red', alpha=0.5, edgecolor='red', pad=1)
      )

    plt.axis('off')
    plt.show()