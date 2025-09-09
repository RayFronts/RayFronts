"""Defines TartanAir+TartanGround datasets.

Typical usage example:
  dataset = TartanAirDataset(path, "AbandonedCableDay")
  dataloader = torch.utils.data.DataLoader(
    self.dataset, batch_size=4)
  for i, batch in enumerate(dataloader):
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
"""

import os
import json
from typing_extensions import override
from typing import Union, Tuple, List
import numpy as np

import torch
import torchvision
from scipy.spatial.transform import Rotation as R
from rayfronts.datasets.base import SemSegDataset

from rayfronts import geometry3d as g3d
import cv2

class TartanAirDataset(SemSegDataset):
  """Loads the front sensors from the TartanAir dataset.

  This dataloader may be volatile as the TartanAir dataset is actively being
  developed.

  Data can be found here
  https://tartanair.org/

  File structure:
    root
    --<scene_name>
    ----seg_label.json
    ----seg_label_clean.json (Obtained from RayFronts repo)
    ----<data_partition>
    ------<sequence>
    --------depth_lcam_front
    ----------000000_lcam_front_depth.png
    --------image_lcam_front
    ----------000000_lcam_front.png
    --------seg_lcam_front
    ----------000000_lcam_front_seg.png
  
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
    depth_cutoff: See __init__.
    label_file: See __init__.
    data_partition: See __init__.
    sequence: See __init__.
  """
  def __init__(self,
               path: str,
               scene_name: str,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear",
               load_semseg: bool = True,
               label_file: str = "seg_label_clean.json",
               depth_cutoff: float = 100,
               data_partition: str = "Data_ground",
               sequence: Union[List[str], str] = "P0000"):
    """
    Args:
      path: Path to the root tartanair dataset.
      scene_name: Name of the scene from replica. E.g "AbandonedFactory".
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
      load_semseg: Whether to load semantic segmentation labels or not.
      label_file: which label_file to load from.
      depth_cutoff: Limit depth to this value. TartanAir has sky/background
        as large encompassing spheres and as such the depth cutoff must be set
        to avoid mapping that.
      data_partition: Choose from ["Data_ground", "Data_easy", "Data_hard"]
      sequence: which sequence/trajectory to load. Can pass a list of sequences
        to play them all.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)
    self.path = path
    self.scene_name = scene_name
    self.original_h = 640
    self.original_w = 640
    self.load_semseg = load_semseg
    self.depth_cutoff = depth_cutoff
    self.data_partition = data_partition
    self.sequence = sequence
    self.label_file = label_file

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

    self._poses_4x4 = list()
    self._depth_paths = list()
    self._rgb_paths = list()
    if load_semseg:
      self._semseg_paths = list()

    self._depth_scale = 1

    self._poses_4x4 = list()
    self._depth_paths = list()
    self._rgb_paths = list()
    if load_semseg:
      self._semseg_paths = list()

    self._frd2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform("frd", "rdf"))

    sequences_dir = os.path.join(self.path, scene_name, self.data_partition)

    if self.sequence is None:
      sequences = sorted(os.listdir(sequences_dir))
    else:
      sequences = [self.sequence]

    for sequence in sequences: #P000, P001, P002, ...
      seq_dir = os.path.join(sequences_dir, sequence)
      pose_file = os.path.join(seq_dir, 'pose_lcam_front.txt')
      n = 0
      with open(pose_file, 'r', encoding="UTF-8") as pose_file:
        for pose_line in pose_file:
          tq = [float(x) for x in pose_line.strip().split()]
          frd_pose = self._pose_to_matrix(tq)
          rdf_pose = g3d.transform_pose_4x4(frd_pose, self._frd2rdf_transform)
          self._poses_4x4.append(rdf_pose)
          n += 1

      depth_dir = os.path.join(seq_dir, 'depth_lcam_front')
      self._depth_paths.extend(
        [os.path.join(depth_dir, f"{f:06d}_lcam_front_depth.png")
         for f in range(n)])

      rgb_dir = os.path.join(seq_dir, 'image_lcam_front')
      self._rgb_paths.extend(
        [os.path.join(rgb_dir, f"{f:06d}_lcam_front.png")
         for f in range(n)])

      if load_semseg:
        semseg_dir = os.path.join(seq_dir, 'seg_lcam_front')
        self._semseg_paths.extend(
          [os.path.join(semseg_dir, f"{f:06d}_lcam_front_seg.png")
           for f in range(n)])

    if self.load_semseg:
      semseg_info_f = os.path.join(self.path, scene_name, self.label_file)
      semrgb_text_f = os.path.join(self.path, 'seg_rgbs.txt')
      semrgb_text_np = np.loadtxt(semrgb_text_f, delimiter=',', dtype=int)
      sem_color_dict = {k:semrgb_text_np[k,2]
                        for k in range(1,len(semrgb_text_np)-1)}

      with open(semseg_info_f, "r") as f:
          semseg_info = json.load(f)
      
      obj_class_dict = semseg_info['name_map']

      self._init_semseg_mappings(
        {i+1: v for i, v in enumerate(sorted(obj_class_dict.keys()))})
      self._name_to_cat_id = {v: k for k,v in self._cat_id_to_name.items()}

      self._cat_imgid_to_cat_id = torch.zeros(max(sem_color_dict.values()),
                                              dtype=torch.long)
      for k, imgids in obj_class_dict.items():
        if isinstance(imgids, int):
          imgids = [imgids]
        for imgid in imgids:
          self._cat_imgid_to_cat_id[sem_color_dict[imgid]] = \
            self._name_to_cat_id[k]

    self._poses_4x4 = torch.stack(self._poses_4x4, dim=0)

  def _pose_to_matrix(self, pose):
    tx,ty,tz,qx,qy,qz,qw = pose
    rotation = R.from_quat([qx,qy,qz,qw]).as_matrix()
    transformation = torch.eye(4, dtype=torch.float32)
    transformation[:3,:3] = torch.tensor(rotation, dtype=torch.float32)
    transformation[:3,3] = torch.tensor([tx,ty,tz], dtype=torch.float32)
    return transformation

  @override
  def __iter__(self):
    for f in range(len(self._poses_4x4)):
      if self.frame_skip > 0 and f % (self.frame_skip+1) != 0:
        continue

      rgb_img = torchvision.io.read_image(self._rgb_paths[f])
      rgb_img = rgb_img.type(torch.float) / 255

      depth_rgba = cv2.imread(self._depth_paths[f], cv2.IMREAD_UNCHANGED)

      #640x640x4
      depth = depth_rgba.view("<f4")[:,:,0] #640 x 640 x1

      depth_img = torch.from_numpy(depth).unsqueeze(0)
      depth_img = depth_img / self._depth_scale
      depth_img[depth_img > self.depth_cutoff] = torch.inf
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
        semseg_img = self._cat_id_to_index[self._cat_imgid_to_cat_id[semseg_img]]

        if (self.rgb_h != semseg_img.shape[-2] or
          self.rgb_w != semseg_img.shape[-1]):
          semseg_img = torch.nn.functional.interpolate(
            semseg_img.unsqueeze(0).float(),
            size=(self.rgb_h, self.rgb_w),
            mode="nearest-exact").squeeze(0).long()
        frame_data["semseg_img"] = semseg_img

      yield frame_data
