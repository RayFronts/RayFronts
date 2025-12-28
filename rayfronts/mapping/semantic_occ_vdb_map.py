"""Module defining a semantic voxel + openvdb occupancy map class.

Typical usage example:

  map = SemanticOccVDBMap(intrinsics_3x3, None, visualizer, encoder)
  for batch in dataloader:
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
    map.process_posed_rgbd(rgb_img, depth_img, pose_4x4)
  map.vis_map()

  r = map.text_query(["man wearing a blue shirt"])
  map.vis_query_result(r)

  map.save("test.pt")
"""

from typing_extensions import override, List, Tuple, Dict
import sys
import os

import torch
import openvdb

from rayfronts.mapping.base import SemanticRGBDMapping
from rayfronts import (geometry3d as g3d, visualizers, image_encoders,
                       feat_compressors)
from rayfronts.utils import compute_cos_sim

sys.path.insert(
  0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../csrc/build/"))
)
import rayfronts_cpp

class SemanticOccVDBMap(SemanticRGBDMapping):
  """Semantic voxel + openvdb occupancy map.

  Attributes:
    intrinsics_3x3: See base.
    device: See base.
    visualizer: See base.
    clip_bbox: See base
    encoder: See base.
    feat_compressor: See base.
    interp_mode: See base.

    max_pts_per_frame: See __init__.
    vox_size: See __init__.
    max_empty_pts_per_frame: See __init__.
    max_depth_sensing: See __init__.
    max_empty_cnt: See __init__.
    max_occ_cnt: See __init__.
    occ_observ_weight: See __init__.
    occ_thickness: See __init__.
    vox_accum_period: See __init__.
    occ_pruning_tolerance: See __init__.
    occ_pruning_period: See __init__.
    sem_pruning_thresh: See __init__.
    sem_pruning_period: See __init__.

    occ_map_vdb: An OpenVDB Int8 Grid storing log-odds occupancy.
      more information about available functions on the grid can be found below:
      https://www.openvdb.org/documentation/doxygen/classopenvdb_1_1v12__0_1_1Grid.html
    global_vox_xyz: Nx3 Float tensor describing voxel centroids in world
      coordinate frame.
    global_vox_rgb_feat_cnt: (Nx(3+C+1)) Float tensor; 3 for rgb, C for
      features, 1 for observation confidence. This tensor is aligned with
      global_vox_xyz. Currently observation confidence is just hit count.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: visualizers.Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               encoder: image_encoders.ImageEncoder = None,
               feat_compressor: feat_compressors.FeatCompressor = None,
               interp_mode: str = "bilinear",
               max_pts_per_frame: int = 1000,
               vox_size: int = 1,
               vox_accum_period: int = 1,
               max_empty_pts_per_frame: int = 1000,
               max_depth_sensing: float = -1,
               max_empty_cnt: int = 3,
               max_occ_cnt: int = 5,
               occ_observ_weight: int = 5,
               occ_thickness: int = 2,
               occ_pruning_tolerance: int = 2,
               occ_pruning_period: int = 1,
               sem_pruning_thresh: int = 0,
               sem_pruning_period: int = 1):
    """
    Args:
      intrinsics_3x3: See base.
      device: See base.
      visualizer: See base.
      clip_bbox: See base.
      encoder: See base.
      feat_compressor: See base.
      interp_mode: See base.

      max_pts_per_frame: How many points to project per frame. Set to -1 to 
        project all valid depth points.
      vox_size: Length of a side of a voxel in meters.
      vox_accum_period: How often do we aggregate voxels into the global 
        representation. Setting to 10, will accumulate point clouds from 10
        frames before voxelization. Should be tuned to balance memory,
        throughput, and min latency.
      max_empty_pts_per_frame: How many empty points to project per frame.
        Set to -1 to project all valid depth points.
      max_depth_sensing: Depending on the max sensing range, we project empty
        voxels up to that range if that pixel had +inf depth or out of range.
        Set to -1 to use the max depth in that frame as the max sensor range.
      max_empty_cnt: The maximum log odds value for empty voxels. 3 means the
        cell will be capped at -3 which corresponds to a
        probability of e^-3 / ( e^-3 + 1 ) ~= 0.05 Lower values help compression
        and responsivness to dynamic objects whereas higher values help
        stability and retention of more evidence.
      max_occ_cnt: The maximum log odds value for occupied voxels. Same
        discussion of max_empty_cnt applies here.
      occ_observ_weight: How much weight does an occupied observation hold
        over an empty observation.
      occ_thickness: When projecting occupied points, how many points do we
        project as occupied? e.g. Set to 3 to project 3 points centered around
        the original depth value with vox_size/2 spacing between them. This
        helps reduce holes in surfaces.
      occ_pruning_tolerance: Tolerance when merging voxels into bigger nodes.
      occ_pruning_period: How often do we prune occupancy into bigger voxels.
        Set to -1 to disable.
      sem_pruning_thresh: Voxels with log-odds occupancy below this value are
        removed.
      sem_pruning_period: How often do we prune semantic voxels to reflect
        occupancy (That is erase semantic voxels that are no longer occupied).
        Set to -1 to disable.
    """
    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox, encoder,
                     feat_compressor, interp_mode)

    self.max_pts_per_frame = max_pts_per_frame
    self.interp_mode = interp_mode
    self.occ_pruning_period = occ_pruning_period
    self.occ_pruning_tolerance = occ_pruning_tolerance
    self._occ_pruning_cnt = 0
    self.sem_pruning_period = sem_pruning_period
    self.sem_pruning_thresh = sem_pruning_thresh
    self._sem_pruning_cnt = 0

    self.vox_size = vox_size
    self.encoder = encoder

    self.max_empty_pts_per_frame = max_empty_pts_per_frame
    self.max_depth_sensing = max_depth_sensing
    self.max_empty_cnt = max_empty_cnt
    self.max_occ_cnt = max_occ_cnt
    self.occ_observ_weight = occ_observ_weight
    self.occ_thickness = occ_thickness

    v = self.vox_size
    self.occ_map_vdb = openvdb.Int8Grid()
    # FIXME: This following line causes "nanobind: leaked 1 instances!" upon
    # exiting.
    self.occ_map_vdb.transform = openvdb.createLinearTransform(voxelSize=v)

    # (Nx3)
    self.global_vox_xyz = None
    # (Nx(3+C+1)) 3 for rgb, C for features, 1 for observation count
    # We keep this in the same tensor to avoid having to constantly continue
    # concatenating and slicing when performing voxelization.
    # TODO: cap count such that it can be updated with dynamic env.
    self.global_vox_rgb_feat_cnt = None

    # Temporary separate voxel grid accumulation before aggregation
    self.vox_accum_period = vox_accum_period
    self._vox_accum_cnt = 0

    self._tmp_vox_xyz = list()
    self._tmp_vox_occ = list()
    self._tmp_vox_xyz_since_prune = list()

    self._tmp_pc_xyz = list()
    self._tmp_pc_rgb_feat_cnt = list()

  @property
  def global_vox_rgb(self) -> torch.FloatTensor:
    if self.global_vox_rgb_feat_cnt is not None:
      return self.global_vox_rgb_feat_cnt[:, :3]
    else:
      return None

  @property
  def global_vox_feat(self) -> torch.FloatTensor:
    if self.global_vox_rgb_feat_cnt is not None:
      return self.global_vox_rgb_feat_cnt[:, 3:-1]
    else:
      return None

  @property
  def global_vox_conf(self) -> torch.FloatTensor:
    if self.global_vox_rgb_feat_cnt is not None:
      return self.global_vox_rgb_feat_cnt[:, -1:]
    else:
      return None

  @override
  def save(self, file_path) -> None:
    raise NotImplementedError()

  @override
  def load(self, file_path) -> None:
    raise NotImplementedError()

  @override
  def is_empty(self) -> bool:
    return (self.occ_map_vdb.empty() and (self.global_vox_xyz is None or
                                          self.global_vox_xyz.shape[0] == 0))

  @override
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None,
                         feat_img: torch.FloatTensor = None) -> dict:
    update_info = dict()

    vox_xyz, vox_occ, pc_xyz, selected_indices = \
      g3d.depth_to_sparse_occupancy_voxels(
      depth_img, pose_4x4, self.intrinsics_3x3, self.vox_size, conf_map,
      max_num_pts = self.max_pts_per_frame,
      max_num_empty_pts = self.max_empty_pts_per_frame,
      max_depth_sensing = self.max_depth_sensing,
      occ_thickness=self.occ_thickness,
      return_pc=True
    )
    vox_xyz, vox_occ = self._clip_pc(vox_xyz, vox_occ)
    pc_xyz, selected_indices = self._clip_pc(
      pc_xyz, selected_indices.unsqueeze(-1))
    selected_indices = selected_indices.squeeze(-1)

    B, _, rH, rW = rgb_img.shape
    B, _, dH, dW = depth_img.shape

    if rH != dH or rW != dW:
      pts_rgb = torch.nn.functional.interpolate(
        rgb_img,
        size=(dH, dW),
        mode=self.interp_mode,
        antialias=self.interp_mode in ["bilinear", "bicubic"])
    else:
      pts_rgb = rgb_img

    if self.encoder is not None:
      if feat_img is None:
        feat_img = self.encoder.encode_image_to_feat_map(rgb_img)
      feat_img = self._proj_resize_feat_map(feat_img, dH, dW)
      update_info["feat_img"] = feat_img

    if pc_xyz.shape[0] > 0:
      pts_rgb = pts_rgb.permute(0, 2, 3, 1).reshape(-1, 3)[selected_indices]
      N = pts_rgb.shape[0]

      if self.encoder is not None:
        pts_feat = feat_img.permute(0, 2, 3, 1).reshape(-1, feat_img.shape[1])
        pts_feat = pts_feat[selected_indices]
        pts_rgb_feat_cnt = torch.cat(
          (pts_rgb, pts_feat, torch.ones((N, 1), device=self.device)), dim=-1)
      else:
        pts_rgb_feat_cnt = torch.cat(
          (pts_rgb, torch.ones((N, 1), device=self.device)), dim=-1)

      self._tmp_pc_xyz.append(pc_xyz)
      self._tmp_pc_rgb_feat_cnt.append(pts_rgb_feat_cnt)

    if vox_xyz.shape[0] > 0:
      # [0, 1] to [-1, occ_observ_weight]
      vox_occ = vox_occ*self.occ_observ_weight - 1
      self._tmp_vox_occ.append(vox_occ)
      self._tmp_vox_xyz.append(vox_xyz)

    self._vox_accum_cnt += B
    self._occ_pruning_cnt += B
    self._sem_pruning_cnt += B

    if self._vox_accum_cnt >= self.vox_accum_period:
      self._vox_accum_cnt = 0

      self.accum_occ_voxels()

      if (self.occ_pruning_period > -1 and
          self._occ_pruning_cnt >= self.occ_pruning_period):
        self._occ_pruning_cnt = 0
        self.occ_map_vdb.prune(self.occ_pruning_tolerance)

      self.accum_semantic_voxels()
      if (self.sem_pruning_period > -1 and
          self._sem_pruning_cnt >= self.sem_pruning_period):
        self._sem_pruning_cnt = 0
        self.prune_semantic_voxels(
          torch.cat(self._tmp_vox_xyz_since_prune, dim=0))
        self._tmp_vox_xyz_since_prune.clear()

    return update_info

  def accum_occ_voxels(self) -> None:
    """Accumulate the temporarilly gathered occupancy voxels."""
    if len(self._tmp_vox_xyz) == 0:
      return
    vox_xyz = torch.cat(self._tmp_vox_xyz, dim = 0)
    self._tmp_vox_xyz.clear()
    vox_occ = torch.cat(self._tmp_vox_occ, dim = 0)
    self._tmp_vox_occ.clear()
    if self.sem_pruning_period > 0:
      self._tmp_vox_xyz_since_prune.append(vox_xyz)

    rayfronts_cpp.occ_pc2vdb(
      self.occ_map_vdb, vox_xyz.cpu(), vox_occ.cpu().squeeze(-1),
      self.max_empty_cnt, self.max_occ_cnt)

  def accum_semantic_voxels(self) -> None:
    """Accumulate the temporarilly gathered semantic points into voxels."""
    if len(self._tmp_pc_xyz) == 0:
      return
    pc_xyz = torch.cat(self._tmp_pc_xyz, dim = 0)
    self._tmp_pc_xyz.clear()
    pc_rgb_feat_cnt = torch.cat(self._tmp_pc_rgb_feat_cnt, dim = 0)
    self._tmp_pc_rgb_feat_cnt.clear()

    if self.global_vox_xyz is None:
      vox_xyz, vox_rgb_feat_cnt, vox_cnt = g3d.pointcloud_to_sparse_voxels(
        pc_xyz, feat_pc=pc_rgb_feat_cnt, vox_size=self.vox_size,
        return_counts=True)

      vox_rgb_feat_cnt[:, -1] = vox_cnt.squeeze()
      self.global_vox_xyz = vox_xyz
      self.global_vox_rgb_feat_cnt = vox_rgb_feat_cnt
    else:
      self.global_vox_xyz, self.global_vox_rgb_feat_cnt = \
        g3d.add_weighted_sparse_voxels(
          self.global_vox_xyz,
          self.global_vox_rgb_feat_cnt,
          pc_xyz,
          pc_rgb_feat_cnt,
          vox_size=self.vox_size
        )

  def prune_semantic_voxels(self, updated_pts_xyz) -> None:
    """Remove semantic voxels that are no longer occupied.
    
    Args:
      updated_pts_xyz: (Nx3) Float tensor describing the voxels/points that
        have been updated. Only these points will be considered for removal.
    """
    if self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0:
      return

    updated_vox_xyz = g3d.pointcloud_to_sparse_voxels(
      updated_pts_xyz, vox_size=self.vox_size)
    updated_vox_occ = rayfronts_cpp.query_occ(
      self.occ_map_vdb, updated_vox_xyz.cpu()).to(self.device)

    vox_xyz_to_remove = updated_vox_xyz[updated_vox_occ.squeeze(-1) <=
                                        self.sem_pruning_thresh]

    self.global_vox_xyz, flag = g3d.intersect_voxels(
      self.global_vox_xyz, vox_xyz_to_remove, self.vox_size)

    self.global_vox_xyz = self.global_vox_xyz[flag == 1]

    # Strong assumption here that the original global_vox_xyz is sorted !
    # and that the produced global_vox_xyz is also sorted.
    # If both the first input and the output are sorted then the filtered flag
    # will be aligned with the first input.
    # TODO: Double check and have stronger guarantees / fail-safes
    self.global_vox_rgb_feat_cnt = \
      self.global_vox_rgb_feat_cnt[flag[flag >= 0] == 1]

  @override
  def vis_map(self) -> None:
    if self.visualizer is None or self.is_empty():
      return
    if self.global_vox_xyz is not None and self.global_vox_xyz.shape[0] > 0:
      self.visualizer.log_pc(self.global_vox_xyz, self.global_vox_rgb,
                            layer="voxel_rgb")
      if self.encoder is not None:
        self.visualizer.log_feature_pc(
          self.global_vox_xyz, self.global_vox_feat, layer="voxel_feature")

      log_hit_count = torch.log2(self.global_vox_conf.squeeze())
      self.visualizer.log_heat_pc(self.global_vox_xyz, log_hit_count,
                                  layer="voxel_log_hit_count")

    if not self.occ_map_vdb.empty():
      pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.occ_map_vdb)

      self.visualizer.log_occ_pc(
        pc_xyz_occ_size[:, :3],
        torch.clamp(pc_xyz_occ_size[:, -2:-1], min=-1, max=1),
        layer="voxel_occ"
      )

      tiles = pc_xyz_occ_size[pc_xyz_occ_size[:, -1] > self.vox_size, :]
      if tiles.shape[0] > 0:
        self.visualizer.log_occ_pc(
          tiles[:, :3],
          torch.clamp(tiles[:, -2:-1], min=-1, max=1),
          tiles[:, -1:],
          layer="voxel_occ_tiles"
        )

  @override
  def vis_update(self, **kwargs) -> None:
    if "feat_img" in kwargs:
      self.visualizer.log_feature_img(kwargs["feat_img"][-1].permute(1, 2, 0))

  @override
  def feature_query(self,
                    feat_query: torch.FloatTensor,
                    softmax: bool = False,
                    compressed: bool = True)-> dict:
    if self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0:
      return
    vox_feat = self.global_vox_feat

    if self.feat_compressor is not None and not compressed:
      vox_feat = self.feat_compressor.decompress(vox_feat)

    vox_feat = self.encoder.align_spatial_features_with_language(
      vox_feat.unsqueeze(-1).unsqueeze(-1)
    ).squeeze()

    r = compute_cos_sim(feat_query, vox_feat, softmax=softmax).T
    return dict(vox_xyz=self.global_vox_xyz, vox_sim=r)

  @override
  def vis_query_result(self,
                       query_results: dict,
                       vis_labels: List[str] = None,
                       vis_colors: Dict[str, str] = None,
                       vis_thresh: float = 0) -> None:
    vox_xyz = query_results["vox_xyz"]
    vox_sim = query_results["vox_sim"]
    for q in range(vox_sim.shape[0]):
      kwargs = dict()
      label = vis_labels[q]
      if vis_colors is not None and label in vis_colors.keys():
        kwargs["high_color"] = vis_colors[label]
        kwargs["low_color"] = (0, 0, 0)
      self.visualizer.log_heat_pc(
        vox_xyz, vox_sim[q, :],
        layer=f"queries/{label.replace(' ', '_').replace('/', '_')}",
        vis_thresh=vis_thresh,
        **kwargs)
