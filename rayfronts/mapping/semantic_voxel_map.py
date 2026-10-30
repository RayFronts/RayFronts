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
    feat_dtype: See __init__.

    global_vox_xyz: Nx3 Float tensor describing voxel centroids in world
      coordinate frame.
    global_vox_rgb_conf: (Nx(3+1)) Float tensor; 3 for rgb, 1 for observation
      confidence. This tensor is aligned with global_vox_xyz. Currently
      observation confidence is just hit count.
    global_vox_feat: (NxC) Float tensor of voxel features aligned with
      global_vox_xyz. They are stored in feat_dtype, and read as float32.
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
               windowing: bool = False,
               feat_dtype: str = "float32"):
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
      feat_dtype: Name of the torch dtype in which voxel features are stored
        while mapping and when saved, e.g. "float16" or "bfloat16" to halve
        their memory. Features are running means whose update never forms
        their sum, so 16-bit features don't overflow. Colors and hit counts
        stay float32.
    """

    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox, encoder,
                     feat_compressor, interp_mode)

    self.max_pts_per_frame = max_pts_per_frame
    self.interp_mode = interp_mode

    self.vox_size = vox_size
    self.windowing = windowing
    self.feat_dtype = getattr(torch, feat_dtype)

    # (Nx3)
    self.global_vox_xyz = None
    # (Nx(3+1)) 3 for rgb, 1 for observation confidence. Kept in float32 so
    # that hit counts stay exact whatever feat_dtype is.
    # TODO: Keep conf vs hit count distinctions consistent and clear.
    #   Also cap confidence such that it can be updated with dynamic env.
    self.global_vox_rgb_conf = None
    # (NxC) in feat_dtype.
    self._global_vox_feat = None

    # Temporary point cloud accumulation before voxelization
    self.vox_accum_period = vox_accum_period
    self._vox_accum_cnt = 0
    self._tmp_pc_xyz = list()
    self._tmp_pc_rgb_conf = list()
    self._tmp_pc_feat = list()

  @property
  def global_vox_rgb(self) -> torch.FloatTensor:
    if self.global_vox_rgb_conf is not None:
      return self.global_vox_rgb_conf[:, :3]
    else:
      return None

  @property
  def global_vox_feat(self) -> torch.FloatTensor:
    if self._global_vox_feat is not None:
      return self._global_vox_feat.float()
    else:
      return None

  @property
  def global_vox_conf(self) -> torch.FloatTensor:
    if self.global_vox_rgb_conf is not None:
      return self.global_vox_rgb_conf[:, -1:]
    else:
      return None

  @override
  def save(self, file_path) -> None:
    torch.save(dict(global_vox_xyz=self.global_vox_xyz,
                    global_vox_rgb_conf=self.global_vox_rgb_conf,
                    global_vox_feat=self._global_vox_feat),
                    file_path)

  @override
  def load(self, file_path) -> None:
    d = torch.load(file_path, weights_only=True)
    self.global_vox_xyz = d["global_vox_xyz"].to(self.device)
    if "global_vox_rgb_feat_conf" in d:
      # Maps saved with rgb, features, and confidence in a single tensor.
      rgb_feat_conf = d["global_vox_rgb_feat_conf"]
      d["global_vox_rgb_conf"] = torch.cat(
        (rgb_feat_conf[:, :3], rgb_feat_conf[:, -1:]), dim=-1)
      d["global_vox_feat"] = rgb_feat_conf[:, 3:-1]
      if d["global_vox_feat"].shape[-1] == 0:
        d["global_vox_feat"] = None
    self.global_vox_rgb_conf = d["global_vox_rgb_conf"].to(self.device)
    self._global_vox_feat = d["global_vox_feat"]
    if self._global_vox_feat is not None:
      self._global_vox_feat = self._global_vox_feat.to(self.device,
                                                       self.feat_dtype)

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
      self._tmp_pc_feat.append(pts_feat[selected_indices])

    self._tmp_pc_rgb_conf.append(torch.cat(
      (pts_rgb, torch.ones((N, 1), device=self.device)), dim=-1))
    self._tmp_pc_xyz.append(pts_xyz)
    self._vox_accum_cnt += B

    if self._vox_accum_cnt >= self.vox_accum_period:
      self._vox_accum_cnt = 0
      self.accum_semantic_voxels()

    return update_info

  def _merge(self, vox_xyz, vox_rgb_conf, vox_feat,
             pts_xyz, pts_rgb_conf, pts_feat):
    """Merge points into voxels by updating the voxels' running means.

    The colors and features of a voxel are means over its observations,
    weighted by the hit count in the last column of vox_rgb_conf, and each
    point is an observation weighted by the last column of pts_rgb_conf. A
    voxel with mean f over a weight n that receives points with mean x and
    total weight m is updated to f + m / (n + m) * (x - f). Unlike
    (n * f + m * x) / (n + m), this never forms the sum of the voxel's
    features, so no value exceeds the magnitude of a single feature, and the
    features can be stored in 16 bits without overflowing. The update runs in
    float32 on the voxels that receive points, and is stored in feat_dtype.

    Args:
      vox_xyz: Nx3 float tensor of voxel centers, or None if there are no
        voxels yet.
      vox_rgb_conf: Nx(3+1) float tensor of voxel colors and hit counts.
      vox_feat: NxC tensor of voxel features. None if there are no features.
      pts_xyz: Mx3 float tensor of point positions.
      pts_rgb_conf: Mx(3+1) float tensor of point colors and weights.
      pts_feat: MxC tensor of point features. None if there are no features.

    Returns:
      A tuple of (xyz, rgb_conf, feat) of the merged voxels.
    """
    if vox_xyz is None:
      all_xyz = pts_xyz
    else:
      all_xyz = torch.cat((vox_xyz, pts_xyz), dim=0)
    num_vox = all_xyz.shape[0] - pts_xyz.shape[0]

    # TODO Flooring is faster
    xyz_vx, reduce_ind = torch.unique(
      torch.round(all_xyz/self.vox_size).type(torch.int64),
      return_inverse=True, dim=0)
    xyz_vx = xyz_vx.type(torch.float)*self.vox_size
    vox_ind = reduce_ind[:num_vox]

    if num_vox > 0:
      # Windowing can leave several voxels with the same center. Merge each
      # group into its first voxel, with the others as observations weighted
      # by their counts, before merging the points.
      i = torch.arange(num_vox, device=vox_ind.device)
      first = torch.full((xyz_vx.shape[0],), num_vox, device=vox_ind.device)
      first.scatter_reduce_(0, vox_ind, i, "amin")
      dup = first[vox_ind] != i
      if dup.any():
        return self._merge(
          *self._merge(vox_xyz[~dup], vox_rgb_conf[~dup],
                       None if vox_feat is None else vox_feat[~dup],
                       vox_xyz[dup], vox_rgb_conf[dup],
                       None if vox_feat is None else vox_feat[dup]),
          pts_xyz, pts_rgb_conf, pts_feat)

    # Voxels that receive points, and which of them each point falls in.
    upd_ind, pts_upd_ind = torch.unique(reduce_ind[num_vox:],
                                        return_inverse=True)

    d = pts_xyz.device
    pts_w = pts_rgb_conf[:, -1:].float()
    m = torch.zeros((upd_ind.shape[0], 1), device=d, dtype=torch.float32)
    m.index_add_(0, pts_upd_ind, pts_w)
    cnt = torch.zeros((xyz_vx.shape[0], 1), device=d, dtype=torch.float32)
    if num_vox > 0:
      cnt[vox_ind] = vox_rgb_conf[:, -1:].float()
    n = cnt[upd_ind]
    cnt[upd_ind] = n + m
    frac = m / (n + m)

    def update_mean(vox_val, pts_val, dtype):
      mean = torch.zeros((xyz_vx.shape[0], pts_val.shape[-1]), device=d,
                         dtype=dtype)
      if num_vox > 0:
        mean[vox_ind] = vox_val.to(dtype)
      x = torch.zeros((upd_ind.shape[0], pts_val.shape[-1]), device=d,
                      dtype=torch.float32)
      x.index_add_(0, pts_upd_ind, pts_val.float() * pts_w)
      x /= m
      f = mean[upd_ind].float()
      mean[upd_ind] = (f + frac * (x - f)).to(dtype)
      return mean

    rgb = update_mean(None if vox_rgb_conf is None else vox_rgb_conf[:, :3],
                      pts_rgb_conf[:, :3], torch.float32)
    feat = None
    if pts_feat is not None:
      feat = update_mean(vox_feat, pts_feat, self.feat_dtype)

    return xyz_vx, torch.cat((rgb, cnt), dim=-1), feat

  def accum_semantic_voxels(self) -> None:
    """Accumulate the temporarilly gathered semantic points into voxels."""

    if len(self._tmp_pc_xyz) == 0:
      return
    pts_xyz = torch.cat(self._tmp_pc_xyz)
    pts_rgb_conf = torch.cat(self._tmp_pc_rgb_conf, dim=0)
    pts_feat = None
    if len(self._tmp_pc_feat) > 0:
      pts_feat = torch.cat(self._tmp_pc_feat, dim=0)
    self._tmp_pc_xyz.clear()
    self._tmp_pc_rgb_conf.clear()
    self._tmp_pc_feat.clear()

    if self.global_vox_xyz is None or not self.windowing:
      self.global_vox_xyz, self.global_vox_rgb_conf, self._global_vox_feat = \
        self._merge(self.global_vox_xyz, self.global_vox_rgb_conf,
                    self._global_vox_feat, pts_xyz, pts_rgb_conf, pts_feat)
      return

    # Compute active window/bbox.
    # TODO: Test if its faster to project boundary points and pose centers
    # instead of doing min max over all tmp voxels.
    active_bbox_min = torch.min(pts_xyz, dim=0).values
    active_bbox_max = torch.max(pts_xyz, dim=0).values

    active_bbox_mask = torch.logical_and(
      torch.all(self.global_vox_xyz >= active_bbox_min, dim=-1),
      torch.all(self.global_vox_xyz <= active_bbox_max, dim=-1))

    active_feat = None
    if self._global_vox_feat is not None:
      active_feat = self._global_vox_feat[active_bbox_mask]

    xyz_update, rgb_conf_update, feat_update = self._merge(
      self.global_vox_xyz[active_bbox_mask],
      self.global_vox_rgb_conf[active_bbox_mask],
      active_feat,
      pts_xyz,
      pts_rgb_conf,
      pts_feat
    )

    self.global_vox_xyz = torch.cat(
      [xyz_update,
      self.global_vox_xyz[~active_bbox_mask]], dim=0)
    self.global_vox_rgb_conf = torch.cat(
      [rgb_conf_update,
      self.global_vox_rgb_conf[~active_bbox_mask]], dim=0)
    if feat_update is not None:
      self._global_vox_feat = torch.cat(
        [feat_update,
        self._global_vox_feat[~active_bbox_mask]], dim=0)

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
