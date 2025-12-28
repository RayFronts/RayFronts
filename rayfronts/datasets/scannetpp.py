"""Defines scannet++ dataset

Typical usage example:
  dataset = ScanNetPPDataset(path, "bcd2436daf")
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
import json
import torch
import torchvision
import torchvision.transforms.functional
import PIL
from scipy.spatial.transform import Rotation as R
from rayfronts.datasets.base import SemSegDataset

class ScanNetPPDataset(SemSegDataset):
  """Loads from the ScanNet++ dataset.

  Dataset information and download:
  https://github.com/scannetpp/scannetpp
  This loader works with DSLR RGB images, rendered depth maps, and poses
  from COLMAP reconstruction. It loads undistorted DSLR images and their
  corresponding depth maps rendered from the 3D reconstruction.

  In addition, it outputs semantic segmentation labels using the top-100
  classes from the ScanNet++ semantic benchmark.

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
      path: Path to the root processed scannet++ directory.
      scene_name: Name of the scene from scannet++. E.g "bcd2436daf".
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
    self.original_h = 1168
    self.original_w = 1752
    self.load_semseg = load_semseg

    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    scene_dir = os.path.join(self.path, self.scene_name)
    self.rgb_dir = os.path.join(scene_dir, "dslr", "undistorted_images")
    self.depth_dir = os.path.join(self.path, "depth", self.scene_name, "dslr", "render_depth")
    self.intrinsics_path = os.path.join(scene_dir, "dslr", "nerfstudio", "transforms_undistorted.json")

    self.image_txt_pth = os.path.join(scene_dir, "dslr", "colmap", "images.txt")
    image_pose_dict = self.read_images_text(self.image_txt_pth)

    name_to_id = {img_name: v['image_id'] for img_name, v in image_pose_dict.items()}
    from pdb import set_trace as bp
    n = len(os.listdir(os.path.join(self.rgb_dir)))
    self.rgb_paths = sorted([os.path.join(self.rgb_dir, f) for f in os.listdir(self.rgb_dir) if f in name_to_id],key=lambda f: name_to_id[os.path.basename(f)])
    self.depth_paths = sorted([os.path.join(self.depth_dir, f) for f in os.listdir(self.depth_dir) if f.endswith('.png')],key=lambda f: name_to_id[os.path.basename(f).replace('.png', '.JPG')])
    sorted_items = sorted(image_pose_dict.items(), key=lambda kv: kv[1]['image_id'])
    self._poses_4x4 = []
    for image_name, info in sorted_items:
      tx,ty,tz = info['tvec']
      qw,qx,qy,qz = info['qvec']
      T = np.eye(4)
      T[:3, :3] = R.from_quat((qx, qy, qz, qw)).as_matrix()  # rotation
      T[:3, 3] = (tx, ty, tz)                                  # translation
      T_inv = np.linalg.inv(T)                                  # convert to camera-to-world
      self._poses_4x4.append(torch.tensor(T_inv, dtype=torch.float32))
    
    self._poses_4x4 = torch.stack(self._poses_4x4, dim=0)

    with open(self.intrinsics_path, "r", encoding="UTF-8") as f:
        intrinsics = json.load(f)
        fx,fy,cx,cy = intrinsics['fl_x'], intrinsics['fl_y'], intrinsics['cx'], intrinsics['cy']
        pose_frames = intrinsics['frames']
    self.intrinsics_3x3 = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy],[0.0, 0.0, 1.0]], dtype=torch.float32)

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    if self.load_semseg:
      top100file = '/path/to/scannet++/metadata/semantic_benchmark/top100.txt'
      semf = open(top100file,'r')
      aaa = semf.readlines()
      semdict = {}
      for i, aa in enumerate(aaa):
        aa = aa.strip()
        semdict[i+1] = aa
      self._cat_id_to_name = semdict

  @property
  @override
  def num_classes(self):
    return len(self._cat_id_to_name)

  @property
  @override
  def cat_id_to_name(self):
    return self._cat_id_to_name

  @property
  @override
  def cat_id_to_index(self):
    return self._cat_id_to_index
  
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

      yield frame_data

  def read_images_text(self, path):
    image_pose_dict = {}
    with open(path,'r') as f:
      raw = f.read().splitlines()
      raw = [line for line in raw if not line.startswith("#")]
    
    raw = raw[::2]
    for row in raw:
      elems = row.split()
      image_id = int(elems[0])
      qvec = np.array(tuple(map(float, elems[1:5])))
      tvec = np.array(tuple(map(float,elems[5:8])))
      image_name = elems[9]
      image_pose_dict[image_name] = {
        'image_id': image_id,
        'qvec': qvec,
        'tvec': tvec}
    return image_pose_dict