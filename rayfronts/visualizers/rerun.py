"""Defines a ReRun.io implementation of a 3D mapping visualizer.

Typical usage:
  vis = RerunVis(intrinsics_3x3)
  vis.log_pose(pose_4x4, layer="cam0")
  vis.log_img(img, layer="rgb_img", pose_layer="cam0")
  vis.step()
"""
from typing_extensions import override
from typing import Tuple

import numpy as np
import rerun as rr
import torch

from rayfronts.visualizers.base import Mapping3DVisualizer
from rayfronts import feat_compressors

class RerunVis(Mapping3DVisualizer):
  """Semantic RGBD visualizer using ReRun.io
  
  Attributes:
    intrinsics_3x3: See base.
    img_size: See base.
    base_point_size: See base.
    global_heat_scale: See base.
    device: See base.
    feat_compressor: See base.
    time_step: See base.
    split_label_vis: See __init__.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               img_size: Tuple[int] = None,
               base_point_size: float = None,
               global_heat_scale: bool = False,
               feat_compressor: feat_compressors.FeatCompressor = None,
               split_label_vis: bool = False,
               **kwargs):
    """

    Args:
      intrinsics_3x3: See base.
      img_size: See base.
      base_point_size: See base.
      global_heat_scale: See base.
      feat_compressor: See base.
      split_label_vis: Whether to log labeled points and arrows to different
        layers in rerun or not.
    """
    super().__init__(intrinsics_3x3, img_size, base_point_size,
                     global_heat_scale, feat_compressor)

    rr.init("semantic_mapping_vis", spawn=True)
    rr.set_time_seconds("stable_time", 0)
    self._base_name = "world"
    rr.log(self._base_name, rr.ViewCoordinates.RDF, static=True)
    rr.log(self._base_name,
      rr.Arrows3D(
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
      ))
    self._height = None
    self._width = None
    self._prev_poses_4x4 = dict()
    self.split_label_vis = split_label_vis

  @override
  def log_img(self,
              img: torch.FloatTensor,
              layer: str = "img",
              pose_layer: str = "pose") -> None:
    self._height, self._width, _ = img.shape
    # Convert to uint8 for Rerun compatibility
    img_np = img.cpu().numpy()
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    img_uint8 = np.ascontiguousarray(img_uint8)
    rr.log(f"{self._base_name}/{pose_layer}/{layer}", rr.Image(img_uint8))

  @override
  def log_depth_img(self,
                    depth_img: torch.FloatTensor,
                    layer: str = "img_depth",
                    pose_layer: str = "pose") -> None:
    self._height, self._width = depth_img.shape
    # Convert to contiguous numpy array for Rerun compatibility
    depth_np = depth_img.cpu().numpy()
    if not depth_np.flags['C_CONTIGUOUS']:
      depth_np = np.ascontiguousarray(depth_np)
    rr.log(f"{self._base_name}/{pose_layer}/{layer}",
           rr.DepthImage(depth_np))

  @override
  def log_pose(self,
               pose_4x4: torch.FloatTensor,
               layer: str = "pose") -> None:
    rr.log(f"{self._base_name}/{layer}",
           rr.Pinhole(image_from_camera=self.intrinsics_3x3,
                      height=self._height, width=self._width,
                      camera_xyz=rr.ViewCoordinates.RDF))
    rr_transform = rr.Transform3D(translation=pose_4x4[:3, 3].cpu(),
                                  mat3x3=pose_4x4[:3, :3].cpu(),
                                  from_parent=False)
    rr.log(f"{self._base_name}/{layer}", rr_transform)

    if layer in self._prev_poses_4x4.keys():
      origin = self._prev_poses_4x4[layer][:3, -1]
      direction = pose_4x4[:3, -1] - origin
      rr_traj_arrows = rr.Arrows3D(origins=[origin.cpu()],
                                   vectors=[direction.cpu()],
                                   colors=[[255,0,0]])

      rr.log(f"{self._base_name}/{layer}_trajectory", rr_traj_arrows)
    self._prev_poses_4x4[layer] = pose_4x4

  @override
  def log_pc(self,
             pc_xyz: torch.FloatTensor,
             pc_rgb: torch.FloatTensor = None,
             pc_radii: torch.FloatTensor = None,
             layer: str = "pc"):
    super().log_pc(pc_xyz, pc_rgb, layer)
    radii = self.base_point_size if pc_radii is None else pc_radii
    try:
      radii = radii.cpu()
    except AttributeError:
      pass
    pc_rgb = pc_rgb.cpu() if pc_rgb is not None else pc_rgb
    rr.log(f"{self._base_name}/{layer}",
           rr.Points3D(positions=pc_xyz.cpu(), colors=pc_rgb, radii=radii))

  @override
  def log_arrows(self, arr_origins, arr_dirs, arr_rgb = None, layer="arrows"):
    rr.log(f"{self._base_name}/{layer}",
           rr.Arrows3D(vectors=(arr_dirs*self.base_point_size*5).cpu(),
                       origins=arr_origins.cpu(),
                       colors=arr_rgb.cpu(),
                       radii=self.base_point_size/3))

  @override
  def log_label_img(self, img_label, layer="img_label", pose_layer="pose"):
    rr.log(f"{self._base_name}/{pose_layer}/{layer}",
           rr.SegmentationImage(img_label.cpu()))

  @override
  def log_label_pc(self,
                   pc_xyz: torch.FloatTensor,
                   pc_labels: torch.FloatTensor = None,
                   layer: str = "pc_label"):
    if (self.split_label_vis and
        len(pc_labels.shape)==1 and
        pc_labels.shape[0] > 0):
      unique = torch.unique(pc_labels)
      pc_labels_onehot = torch.nn.functional.one_hot(pc_labels)
      for i in unique:
        rr.log(f"{self._base_name}/{layer}/{i}",
               rr.Points3D(positions=pc_xyz[pc_labels_onehot[..., i]==1].cpu(),
                           class_ids=i.item(),
                           radii=self.base_point_size))
    else:
      rr.log(f"{self._base_name}/{layer}",
            rr.Points3D(positions=pc_xyz.cpu(), class_ids=pc_labels.cpu(),
                        radii=self.base_point_size))

  @override
  def log_label_arrows(self, arr_origins, arr_dirs, arr_labels,
                       layer="arr_labels"):
    if (self.split_label_vis and len(arr_labels.shape)==1 and
        arr_labels.shape[0] > 0):
      unique = torch.unique(arr_labels)
      labels_onehot = torch.nn.functional.one_hot(arr_labels)
      for i in unique:
        m = labels_onehot[..., i]==1
        rr.log(f"{self._base_name}/{layer}/{i}",
              rr.Arrows3D(vectors=(arr_dirs*self.base_point_size*5)[m].cpu(),
                          origins=arr_origins[m].cpu(),
                          class_ids=i.item(), radii=self.base_point_size/3))
    else:
      rr.log(f"{self._base_name}/{layer}",
            rr.Arrows3D(vectors=(arr_dirs*self.base_point_size*5).cpu(),
                        origins=arr_origins.cpu(), class_ids=arr_labels.cpu(),
                        radii=self.base_point_size/3))
  @override
  def log_box(self, box_mins, box_maxs, layer = ""):
    box_centers = (box_maxs + box_mins) / 2
    box_half_sizes = (box_maxs - box_mins) / 2
    rr.log(f"{self._base_name}/{layer}_boxes",
           rr.Boxes3D(centers=box_centers.cpu(),
                      half_sizes=box_half_sizes.cpu()))

  def _log_axes(self, pose_4x4, layer="axes"):
    for i in range(pose_4x4.shape[0]):
      rr_transform = rr.Transform3D(
        translation=pose_4x4[i, :3, 3].cpu(), mat3x3=pose_4x4[i, :3, :3].cpu(),
        from_parent=False, axis_length=self.base_point_size*4,
        scale=2)
      rr.log(f"{self._base_name}/{layer}/{i}", rr_transform)

  @override
  def step(self):
    super().step()
    rr.set_time_seconds("stable_time", self.time_step*0.1)
