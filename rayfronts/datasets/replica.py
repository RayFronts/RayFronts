"""Defines replica datasets from the niceslam and semanticnerf trajectories.

Typical usage example:
  dataset = NiceReplicaDataset(path, "office0")
  dataloader = torch.utils.data.DataLoader(
    self.dataset, batch_size=4)
  for i, batch in enumerate(dataloader):
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
"""

import os
import json
from typing import Union, Tuple
from typing_extensions import override

import torch
import torchvision
import torchvision.transforms.functional
import PIL

from rayfronts.datasets.base import PosedRgbdDataset, SemSegDataset

class NiceReplicaDataset(PosedRgbdDataset):
  """Loads from the Replica dataset version processed by Nice-Slam.
  
  Dataset can be found at:
  https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh

  File structure:
    root
    --cam_params.json
    --office0
    ----traj.txt
    ----results
    ------depth000000.png
    ------frame000000.jpg
    --semantic_info (Optional. See note below)
    ----office_0
    ------info_semantic.json

  Note: semantic_info can be obtained by copying it from the SemanticNerfReplica
  dataset or more laboriously from the original Replica dataset. This dataloader
  will attempt to load it if its available to provide the list of categories in
  each scene. The list allows us to perform open-vocabulary semantic
  segmentation. Ground truth would need to be provided externally however.

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
  """

  def __init__(self,
               path: str,
               scene_name: str,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear"):
    """
    Args:
      path: Path to the root nice slam directory.
      scene_name: Name of the scene from replica. E.g "office0".
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)
    self.path = path
    self.scene_name = scene_name
    self.original_h = 680
    self.original_w = 1200

    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    with open(os.path.join(self.path, "cam_params.json"),
              "r", encoding="UTF-8") as f:
      cam_params = json.load(f)["camera"]

    self.intrinsics_3x3 = torch.tensor([[cam_params["fx"], 0, cam_params["cx"]],
                                        [0, cam_params["fy"], cam_params["cy"]],
                                        [0, 0, 1]])

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    self._depth_scale = cam_params["scale"]
    self._traj_path = os.path.join(path, scene_name, "traj.txt")
    self._poses_4x4 = list()

    with open(self._traj_path, "r", encoding="UTF-8") as traj_file:
      for traj_line in traj_file:
        traj_tsr = torch.tensor([float(x) for x in traj_line.strip().split()])
        self._poses_4x4.append(traj_tsr.reshape(4,4))

    self._poses_4x4 = torch.stack(self._poses_4x4, dim=0)

    N = len(self._poses_4x4)

    imgs_dir = os.path.join(self.path, self.scene_name, "results")
    self._depth_paths = [os.path.join(imgs_dir, f"depth{f:06d}.png")
                         for f in range(N)]

    self._rgb_paths = [os.path.join(imgs_dir, f"frame{f:06d}.jpg")
                       for f in range(N)]

    # In case we can load category lists
    semseg_info_f = os.path.join(
      self.path, "semantic_info",
      self.scene_name[:-1] + "_" + self.scene_name[-1], "info_semantic.json")

    if os.path.exists(semseg_info_f):
      with open(semseg_info_f, "r", encoding="UTF-8") as f:
        semseg_info = json.load(f)
      self._cat_id_to_name = \
        {item["id"]: item["name"] for item in semseg_info["classes"]}
      self.cat_id_to_name = self._cat_id_to_name
      self.num_classes = len(self._cat_id_to_name)

  @override
  def __iter__(self):
    for f in range(len(self._poses_4x4)):
      if self.frame_skip > 0 and f % (self.frame_skip+1) != 0:
        continue

      rgb_img = torchvision.io.read_image(self._rgb_paths[f])
      rgb_img = rgb_img.type(torch.float) / 255
      # Cannot use torchvision because depth PNG is not 8 bit.
      depth_img = PIL.Image.open(self._depth_paths[f])
      depth_img = torchvision.transforms.functional.pil_to_tensor(depth_img)
      depth_img = depth_img / self._depth_scale
      depth_img[depth_img==0] = torch.nan

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


class SemanticNerfReplicaDataset(SemSegDataset):
  """Loads from the Replica dataset version processed by Semantic_nerf.
  
  Dataset can be found at:
  https://github.com/Harry-Zhi/semantic_nerf

  File structure:
    root
    --office_0
    ----Sequence_1
    ------traj_w_c.txt
    ------depth
    --------depth_0.png
    ------rgb
    --------rgb_0.png
    ------semantic_class
    --------semantic_class_0.png
    --semantic_info
    ----office_0
    ------info_semantic.json

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
      path: Path to the root semantic nerf replica directory.
      scene_name: Name of the scene from replica. E.g "office_0/Sequence_1".
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

    self.intrinsics_3x3 = torch.tensor(
      [[320, 0, self.original_w/2],
       [0, 320, self.original_h/2],
       [0, 0, 1]])

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    self._depth_scale = 1000
    if len(scene_name.split("/")) == 1:
      seqs = sorted(os.listdir(os.path.join(path, scene_name)))
      scene_names = [os.path.join(scene_name, x) for x in seqs]
    else:
      scene_names = [scene_name]

    self._poses_4x4 = list()
    self._depth_paths = list()
    self._rgb_paths = list()
    if load_semseg:
      self._semseg_paths = list()

    for scene_name in scene_names:
      self._traj_path = os.path.join(path, scene_name, "traj_w_c.txt")

      n = 0
      with open(self._traj_path, "r", encoding="UTF-8") as traj_file:
        for traj_line in traj_file:
          traj_tsr = torch.tensor([float(x) for x in traj_line.strip().split()])
          self._poses_4x4.append(traj_tsr.reshape(4,4))
          n += 1

      depth_dir = os.path.join(self.path, scene_name, "depth")
      self._depth_paths.extend([os.path.join(depth_dir, f"depth_{f}.png")
                                for f in range(n)])

      rgb_dir = os.path.join(self.path, scene_name, "rgb")
      self._rgb_paths.extend([os.path.join(rgb_dir, f"rgb_{f}.png")
                              for f in range(n)])
      if load_semseg:
        semseg_dir = os.path.join(self.path, scene_name, "semantic_class")
        self._semseg_paths.extend(
          [os.path.join(semseg_dir, f"semantic_class_{f}.png")
           for f in range(n)])

    self._poses_4x4 = torch.stack(self._poses_4x4, dim=0)

    N = len(self._poses_4x4)

    if self.load_semseg:
      semseg_info_f = os.path.join(self.path, "semantic_info",
                                   self.scene_name.split("/")[0],
                                   "info_semantic.json")

      with open(semseg_info_f, "r", encoding="UTF-8") as f:
        semseg_info = json.load(f)

      self._init_semseg_mappings(
        {item["id"]: item["name"] for item in semseg_info["classes"]})

  @override
  def __iter__(self):
    for f in range(len(self._poses_4x4)):
      if self.frame_skip > 0 and f % (self.frame_skip+1) != 0:
        continue

      rgb_img = torchvision.io.read_image(self._rgb_paths[f])
      rgb_img = rgb_img.type(torch.float) / 255
      # Cannot use torchvision because depth PNG is not 8 bit.
      depth_img = PIL.Image.open(self._depth_paths[f])
      depth_img = torchvision.transforms.functional.pil_to_tensor(depth_img)
      depth_img = depth_img / self._depth_scale
      depth_img[depth_img==0] = torch.nan

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
        semseg_img = torchvision.io.read_image(self._semseg_paths[f]).long()
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
