"""Defines scannet datasets

Typical usage example:
  dataset = ScanNetDataset(path, "scene0011_00")
  dataloader = torch.utils.data.DataLoader(
    self.dataset, batch_size=4)
  for i, batch in enumerate(dataloader):
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
"""

import os
import pandas as pd
from typing_extensions import override
from typing import Union, Tuple
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional
import PIL

from rayfronts.datasets.base import SemSegDataset

class ScanNetDataset(SemSegDataset):
  """Loads from the ScanNet dataset.

  Dataset information and download:
  https://github.com/ScanNet/ScanNet/tree/master
  This loader only works with extracted RGB, depth, and pose information
  processed by
  https://github.com/ScanNet/ScanNet/tree/master/SensReader/python

  In addition, it outputs NYU40 labels for semantic segmentation maps.

  Attributes:
    intrinsics_3x3:  See base.
    rgb_h: See base.
    rgb_w: See base.
    depth_h: See base.
    depth_w: See base.
    frame_skip: See base.
    interp_mode: See base.
    path: See __init__.
    scene_name: See __init__.
    load_semseg: See __init__.
  """

  def __init__(self,
               path: str,
               scene_name: str,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear",
               load_semseg: bool = True):
    """
    Args:
      path: Path to the root processed scannet directory.
      scene_name: Name of the scene from scannet. E.g "scene0518_00".
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
      load_semseg: Whether to load semantic segmentation labels or not.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)
    self.path = path
    self.scene_name = scene_name
    self.original_h = 480
    self.original_w = 640
    self.load_semseg = load_semseg

    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    scene_dir = os.path.join(self.path, self.scene_name)
    self.rgb_dir = os.path.join(scene_dir, "color")
    self.depth_dir = os.path.join(scene_dir, "depth")
    self.pose_path = os.path.join(scene_dir, "pose")
    self.intrinsics_path = os.path.join(
      scene_dir, "intrinsic/intrinsic_depth.txt")

    if self.load_semseg:
      self.semseg_dir = os.path.join(
        scene_dir, f"{scene_name}_2d-label-filt", "label-filt")

    with open(self.intrinsics_path, "r", encoding="UTF-8") as f:
        intrinsics = np.loadtxt(f)
    self.intrinsics_3x3 = torch.tensor(intrinsics[:3, :3], dtype=torch.float32)

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    n = len(os.listdir(os.path.join(self.rgb_dir)))

    self.rgb_paths =[os.path.join(self.rgb_dir, f"{f}.jpg")
                     for f in range(n)]
    self.depth_paths = [os.path.join(self.depth_dir, f"{f}.png")
                        for f in range(n)]
    if self.load_semseg:
      self.semseg_paths = [os.path.join(self.semseg_dir, f"{f}.png")
                           for f in range(n)]

    self._poses_4x4 = []
    pose_files = [os.path.join(self.pose_path, f"{f}.txt") for f in range(n)]
    for pose_file in pose_files:
      pose = np.loadtxt(pose_file)
      self._poses_4x4.append(torch.tensor(pose, dtype=torch.float32))

    self._poses_4x4 = torch.stack(self._poses_4x4, dim=0)

    if self.load_semseg:
      self.semseg_label_map_path = os.path.join(
        self.path, "scannetv2-labels.combined.tsv")
      self.semseg_label_map = pd.read_csv(self.semseg_label_map_path, sep="\t")
      self.scannet_to_nyu40 = {row["id"]: row["nyu40id"] 
                               for _, row in self.semseg_label_map.iterrows()}
      self.scannet_to_nyu40[0] = 0

      self._init_semseg_mappings(
        {row["nyu40id"]: row["nyu40class"]
         for _, row in self.semseg_label_map.iterrows()})
  
  @override
  def __iter__(self):
    for f in range(len(self._poses_4x4)):
      if self.frame_skip > 0 and f % (self.frame_skip + 1) != 0:
        continue

      rgb_img = torchvision.io.read_image(self.rgb_paths[f])
      rgb_img = rgb_img.type(torch.float32) / 255.0

      depth_img = PIL.Image.open(self.depth_paths[f])
      depth_img = torchvision.transforms.functional.pil_to_tensor(depth_img)
      depth_img = depth_img.float() / 1e3

      if (self.rgb_h != rgb_img.shape[-2] or
          self.rgb_w != rgb_img.shape[-1]):
        rgb_img = torch.nn.functional.interpolate(rgb_img.unsqueeze(0),
          size=(self.rgb_h, self.rgb_w), mode=self.interp_mode,
          antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

      if (self.depth_h != depth_img.shape[-2] or
          self.depth_w != depth_img.shape[-1]):
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0),
          size=(self.depth_h, self.depth_w),
          mode="nearest-exact").squeeze(0)

      pose_4x4 = self._poses_4x4[f]
      frame_data = dict(rgb_img=rgb_img, depth_img=depth_img, pose_4x4=pose_4x4)

      if self.load_semseg:
        img = np.array(PIL.Image.open(self.semseg_paths[f]))
        semseg_img = np.vectorize(self.scannet_to_nyu40.get)(img)
        semseg_img = torch.from_numpy(semseg_img).long().unsqueeze(0)
        # Translate from ids to indices
        semseg_img = self._cat_id_to_index[semseg_img]

        if (self.rgb_h != semseg_img.shape[-2] or
          self.rgb_w != semseg_img.shape[-1]):
          semseg_img = torch.nn.functional.interpolate(
            semseg_img.unsqueeze(0).float(),
            size=(self.rgb_h, self.rgb_w),
            mode="nearest-exact").squeeze(0).long()
        frame_data["semseg_img"] = semseg_img

      yield frame_data
