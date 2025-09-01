"""Module defining a minimal semantic voxel map class.

Typical usage example:

  map = SemanticVoxelMap(intrinsics_3x3, None, visualizer, encoder)
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
from typing_extensions import override, List, Dict, Tuple

import torch

from rayfronts.mapping.base import SemanticRGBDMapping
from rayfronts import (geometry3d as g3d, image_encoders, visualizers,
                       feat_compressors)
from rayfronts.utils import compute_cos_sim

class SemanticVoxelMap(SemanticRGBDMapping):
  """A minimalist semantic voxel map. 
  
  Given Posed RGBD input, an encoder is used to extract vision based semantics
  and the semantics are unprojected into the world and voxelized. Voxels can 
  then be queried with images or with text (If encoder produces language aligned
  features).

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
    windowing: See __init__.

    global_vox_xyz: Nx3 Float tensor describing voxel centroids in world
      coordinate frame.
    global_vox_rgb_feat_conf: (Nx(3+C+1)) Float tensor; 3 for rgb, C for
      features, 1 for observation confidence. This tensor is aligned with
      global_vox_xyz. Currently observation confidence is just hit count.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: visualizers.Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               encoder: image_encoders.ImageSpatialEncoder = None,
               feat_compressor: feat_compressors.FeatCompressor = None,
               interp_mode: str = "bilinear",
               max_pts_per_frame: int = 1000,
               vox_size: int = 1,
               vox_accum_period: int = 1,
               windowing: bool = False):
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
      windowing: (Experimental) whether to filter out voxels within the FoV
        window first before aggregation for efficiency.
    """

    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox, encoder,
                     feat_compressor, interp_mode)

    self.max_pts_per_frame = max_pts_per_frame
    self.interp_mode = interp_mode

    self.vox_size = vox_size
    self.windowing = windowing

    # (Nx3)
    self.global_vox_xyz = None
    # (Nx(3+C+1)) 3 for rgb, C for features, 1 for observation confidence
    # We keep this in the same tensor to avoid having to constantly continue
    # concatenating and slicing when performing voxelization.
    # TODO: Keep conf vs hit count distinctions consistent and clear.
    #   Also cap confidence such that it can be updated with dynamic env.
    self.global_vox_rgb_feat_conf = None

    # Temporary point cloud accumulation before voxelization
    self.vox_accum_period = vox_accum_period
    self._vox_accum_cnt = 0
    self._tmp_pc_xyz = list()
    self._tmp_pc_rgb_feat_conf = list()

  @property
  def global_vox_rgb(self) -> torch.FloatTensor:
    if self.global_vox_rgb_feat_conf is not None:
      return self.global_vox_rgb_feat_conf[:, :3]
    else:
      return None

  @property
  def global_vox_feat(self) -> torch.FloatTensor:
    if self.global_vox_rgb_feat_conf is not None:
      return self.global_vox_rgb_feat_conf[:, 3:-1]
    else:
      return None

  @property
  def global_vox_conf(self) -> torch.FloatTensor:
    if self.global_vox_rgb_feat_conf is not None:
      return self.global_vox_rgb_feat_conf[:, -1:]
    else:
      return None

  @override
  def save(self, file_path) -> None:
    torch.save(dict(global_vox_xyz=self.global_vox_xyz,
                    global_vox_rgb_feat_conf=self.global_vox_rgb_feat_conf),
                    file_path)

  @override
  def load(self, file_path) -> None:
    d = torch.load(file_path, weights_only=True)
    self.global_vox_xyz = d["global_vox_xyz"].to(self.device)
    self.global_vox_rgb_feat_conf = \
      d["global_vox_rgb_feat_conf"].to(self.device)

  @override
  def is_empty(self) -> bool:
    return self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0

  @override
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None,
                         feat_img: torch.FloatTensor = None) -> dict:
    update_info = dict()

    pts_xyz, selected_indices = g3d.depth_to_pointcloud(
      depth_img, pose_4x4, self.intrinsics_3x3,
      conf_map=conf_map,
      max_num_pts=self.max_pts_per_frame
    )

    pts_xyz, selected_indices = self._clip_pc(pts_xyz,
                                              selected_indices.unsqueeze(-1))
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

    pts_rgb = pts_rgb.permute(0, 2, 3, 1).reshape(-1, 3)[selected_indices]

    # Encode if needed

    N = pts_rgb.shape[0]
    if self.encoder is not None:
      if feat_img is None:
        feat_img = self.encoder.encode_image_to_feat_map(rgb_img)
      feat_img = self._proj_resize_feat_map(feat_img, dH, dW)
      update_info["feat_img"] = feat_img

      pts_feat = feat_img.permute(0, 2, 3, 1).reshape(-1, feat_img.shape[1])
      pts_feat = pts_feat[selected_indices]

      pts_rgb_feat_conf = torch.cat(
        (pts_rgb, pts_feat, torch.ones((N, 1), device=self.device)), dim=-1)
    else:
      pts_rgb_feat_conf = torch.cat(
        (pts_rgb, torch.ones((N, 1), device=self.device)), dim=-1)

    self._tmp_pc_rgb_feat_conf.append(pts_rgb_feat_conf)
    self._tmp_pc_xyz.append(pts_xyz)
    self._vox_accum_cnt += B

    if self._vox_accum_cnt >= self.vox_accum_period:
      self._vox_accum_cnt = 0
      self.accum_semantic_voxels()

    return update_info

  def accum_semantic_voxels(self) -> None:
    """Accumulate the temporarilly gathered semantic points into voxels."""

    if len(self._tmp_pc_xyz) == 0:
      return
    pts_xyz = torch.cat(self._tmp_pc_xyz)
    pts_rgb_feat_conf = torch.cat(self._tmp_pc_rgb_feat_conf, dim=0)
    self._tmp_pc_xyz.clear()
    self._tmp_pc_rgb_feat_conf.clear()

    if self.global_vox_xyz is None:
      vox_xyz, vox_rgb_feat_conf, vox_cnts = \
        g3d.pointcloud_to_sparse_voxels(
          pts_xyz, feat_pc=pts_rgb_feat_conf,
          vox_size=self.vox_size, return_counts=True)

      vox_rgb_feat_conf[:, -1] = vox_cnts.squeeze()
      self.global_vox_xyz = vox_xyz
      self.global_vox_rgb_feat_conf = vox_rgb_feat_conf
      return

    if not self.windowing:
      self.global_vox_xyz, self.global_vox_rgb_feat_conf = \
        g3d.add_weighted_sparse_voxels(
          self.global_vox_xyz,
          self.global_vox_rgb_feat_conf,
          pts_xyz,
          pts_rgb_feat_conf,
          vox_size=self.vox_size
      )
      return

    # Compute active window/bbox.
    # TODO: Test if its faster to project boundary points and pose centers
    # instead of doing min max over all tmp voxels.
    active_bbox_min = torch.min(pts_xyz, dim=0).values
    active_bbox_max = torch.max(pts_xyz, dim=0).values

    active_bbox_mask = torch.logical_and(
      torch.all(self.global_vox_xyz >= active_bbox_min, dim=-1),
      torch.all(self.global_vox_xyz <= active_bbox_max, dim=-1))

    global_vox_xyz_update, global_vox_rgb_feat_conf_update = \
      g3d.add_weighted_sparse_voxels(
        self.global_vox_xyz[active_bbox_mask],
        self.global_vox_rgb_feat_conf[active_bbox_mask],
        pts_xyz,
        pts_rgb_feat_conf,
        vox_size=self.vox_size
      )

    self.global_vox_xyz = torch.cat(
      [global_vox_xyz_update,
      self.global_vox_xyz[~active_bbox_mask]], dim=0)
    self.global_vox_rgb_feat_conf = torch.cat(
      [global_vox_rgb_feat_conf_update,
      self.global_vox_rgb_feat_conf[~active_bbox_mask]], dim=0)

  @override
  def vis_map(self) -> None:
    if self.visualizer is None or self.is_empty():
      return
    self.visualizer.log_pc(self.global_vox_xyz, self.global_vox_rgb,
                           layer="voxel_rgb")
    if self.encoder is not None:
      self.visualizer.log_feature_pc(self.global_vox_xyz, self.global_vox_feat,
                                     layer="voxel_feature")
    log_hit_count = torch.log2(self.global_vox_conf.squeeze())
    self.visualizer.log_heat_pc(self.global_vox_xyz, log_hit_count,
                                layer="voxel_log_hit_count")

  @override
  def vis_update(self, **kwargs) -> None:
    if "feat_img" in kwargs:
      self.visualizer.log_feature_img(kwargs["feat_img"][-1].permute(1, 2, 0))

  @override
  def feature_query(self,
                    feat_query: torch.FloatTensor,
                    softmax: bool = False,
                    compressed: bool = True)-> dict:
    if self.is_empty():
      return
    vox_feat = self.global_vox_feat

    if self.feat_compressor is not None and not compressed:
      vox_feat = self.feat_compressor.decompress(vox_feat)

    vox_feat = self.encoder.align_spatial_features_with_language(
      vox_feat.unsqueeze(-1).unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)

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
