"""Module defining RayFronts ! The semantic ray frontier map.

Typical usage example:

  map = SemanticRayFrontiersMap(intrinsics_3x3, None, visualizer, encoder)
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
import math
import logging

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

logger = logging.getLogger(__name__)

class SemanticRayFrontiersMap(SemanticRGBDMapping):
  """RayFronts: Semantic Rays + Frontiers + Semantic Voxels + Occupancy map.

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
    max_rays_per_frame: See __init__.

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

    fronti_neighborhood_r: See __init__.
    fronti_min_unobserved: See __init__.
    fronti_min_empty: See __init__.
    fronti_min_occupied: See __init__.
    fronti_subsampling: See __init__.
    fronti_subsampling_min_fronti: See __init__.

    ray_accum_period: See __init__.
    ray_accum_phase: See __init__.
    angle_bin_size: See __init__.
    ray_erosion: See __init__.
    zero_depth_mode: See __init__.
    ray_tracing: See __init__.
    global_encoding: See __init__.

    occ_map_vdb: An OpenVDB Int8 Grid storing log-odds occupancy.
      more information about available functions on the grid can be found below:
      https://www.openvdb.org/documentation/doxygen/classopenvdb_1_1v12__0_1_1Grid.html
    global_vox_xyz: Nx3 Float tensor describing voxel centroids in world
      coordinate frame.
    global_vox_rgb_feat_cnt: (Nx(3+C+1)) Float tensor; 3 for rgb, C for
      features, 1 for hit count. This tensor is aligned with
      global_vox_xyz.
    frontiers: (Nx3) Float tensor describing locations of map frontiers.
    global_rays_orig_angles: Mx(3+2) 3 for origin xyz and 2 for spherical angles
      theta (azimuthal angle in the xy-plane from the x-axis with -pi<=theta<pi)
      and phi (polar/zenith angle from the positive z-axis with 0<=phi<=pi)
    global_rays_feats_cnt: Mx(C+1) C for features, 1 for confidence weight.
  """
  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: visualizers.Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               encoder: image_encoders.ImageEncoder = None,
               feat_compressor: feat_compressors.FeatCompressor = None,
               interp_mode: str = "bilinear",

               max_pts_per_frame: int = -1,
               vox_size: int = 1,
               vox_accum_period: int = 1,
               max_empty_pts_per_frame: int = -1,
               max_rays_per_frame: int = -1,

               max_depth_sensing: float = -1,
               max_empty_cnt: int = 3,
               max_occ_cnt: int = 5,
               occ_observ_weight: int = 5,
               occ_thickness: int = 2,
               occ_pruning_tolerance: int = 2,
               occ_pruning_period: int = 1,
               sem_pruning_thresh: int = 0,
               sem_pruning_period: int = 1,

               fronti_neighborhood_r: int = 1,
               fronti_min_unobserved: int = 4,
               fronti_min_empty: int = 2,
               fronti_min_occupied: int = 0,
               fronti_subsampling: int = 4,
               fronti_subsampling_min_fronti: int = 10,

               ray_accum_period: int = 2,
               ray_accum_phase: int = 1,
               angle_bin_size: float = 30,
               ray_erosion: int = 1,
               ray_tracing: bool = False,
               global_encoding: bool = False,
               zero_depth_mode: bool = False,
               infer_direction: bool = False):
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

      fronti_neighborhood_r: 3D neighborhood radius to compute if a voxel is a
        frontier or not.
      fronti_min_unobserved: Minimum number of unobserved cells in the
        neighborhood of a cell for it to be considered a frontier.
      fronti_min_empty: Minimum number of empty/free cells in the
        neighborhood of a cell for it to be considered a frontier.
      fronti_min_occupied: Minimum number of occupied cells in the
        neighborhood of a cell for it to be considered a frontier.
      fronti_subsampling: After computing frontiers, we subsample using the
        below factor.
      fronti_subsampling_min_fronti: When subsampling (Clustering frontiers into
        bigger cells), how many frontiers should lie in the big cell to consider
        it as a frontier. This is heavilly tied to the subsampling factor. Ex. A
        subsampling factor of 4 means 4^3 cells will cluster into one cell.

      ray_accum_period: How often do we accumulate/bin the rays.
      ray_accum_phase: A phase term to offset the ray accumulation such that it
        does not happen with voxel accumulation.
      angle_bin_size: Bin size when discretizing the angles of rays in degrees.
      ray_erosion: Should we erode the out of range depth mask before shooting
        the rays ? Set to 0 to disable erosion. 1 means an erosion kernel that 
        is 3x3.
      ray_tracing: Enables ray tracing when projecting rays to frontiers. Slows
        things down but gives more accurate ray to frontier placement.
      global_encoding: Instead of using the spatial/dense features for rays, 
        encode the whole image into one feature vector.
      zero_depth_mode: Pose graph mode when depth is not available/unreliable, 
        or there is no desire for dense voxel mapping. All rays attach to the
        current location.
      infer_direction: Whether to infer frontier directions based on occupancy.
    """
    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox, encoder,
                     feat_compressor, interp_mode)

    self.max_pts_per_frame = max_pts_per_frame
    self.max_dirs_per_frame = max_rays_per_frame

    self.occ_pruning_period = occ_pruning_period
    self.occ_pruning_tolerance = occ_pruning_tolerance
    self._occ_pruning_cnt = 0

    self.sem_pruning_period = sem_pruning_period
    self.sem_pruning_thresh = sem_pruning_thresh
    self._sem_pruning_cnt = 0

    self.fronti_neighborhood_r = fronti_neighborhood_r
    self.fronti_min_unobserved = fronti_min_unobserved
    self.fronti_min_empty = fronti_min_empty
    self.fronti_min_occupied = fronti_min_occupied
    self.fronti_subsampling = fronti_subsampling
    self.fronti_subsampling_min_fronti = fronti_subsampling_min_fronti
    self.angle_bin_size = angle_bin_size

    self.vox_size = vox_size

    self.max_empty_pts_per_frame = max_empty_pts_per_frame
    self.max_depth_sensing = max_depth_sensing
    self.max_empty_cnt = max_empty_cnt
    self.max_occ_cnt = max_occ_cnt
    self.occ_observ_weight = occ_observ_weight
    self.occ_thickness = occ_thickness

    self.ray_erosion = ray_erosion
    self.ray_tracing = ray_tracing
    self.global_encoding = global_encoding
    self.zero_depth_mode = zero_depth_mode
    if infer_direction and angle_bin_size < 360:
      logger.warning("Setting infer direction to true while not in semantic "
                     "frontier mode may not make sense !")
    self.infer_direction = infer_direction

    v = self.vox_size

    ### Core data structures

    self.occ_map_vdb = openvdb.Int8Grid()
    # FIXME: This following line causes "nanobind: leaked 1 instances!" upon
    # exiting.
    self.occ_map_vdb.transform = openvdb.createLinearTransform(voxelSize=v)

    # (Nx3)
    self.global_vox_xyz = None
    # (Nx(3+C+1)) 3 for rgb, C for features, 1 for observation count
    # We keep this in the same tensor to avoid having to constantly continue
    # concatenating and slicing when performing voxelization.
    # TODO: cap count such that it can be updated if with dynamic env.
    self.global_vox_rgb_feat_cnt = None

    # Fx3
    self.frontiers = None
    # Mx(3+2) 3 for origin and 2 for angle
    self.global_rays_orig_angles = None
    # Mx(C+1) C for features, 1 for count.
    self.global_rays_feats_cnt = None

    ### Temporary accumulation variables

    # Point cloud and voxel accumulation before global update
    self.vox_accum_period = vox_accum_period
    self._vox_accum_cnt = 0

    self._tmp_vox_xyz = list()
    self._tmp_vox_occ = list()
    self._tmp_vox_xyz_since_prune = list()

    self._tmp_pc_xyz = list()
    self._tmp_pc_rgb_feat_cnt = list()

    # Semantic ray accumulation before global update
    self.ray_accum_period = ray_accum_period
    self.ray_accum_phase = ray_accum_phase
    self._ray_accum_delay = self.ray_accum_phase
    self._ray_accum_cnt = 0
    self._tmp_ray_orig = list()
    self._tmp_ray_dir = list()
    self._tmp_ray_feat = list()

  @property
  def global_vox_rgb(self):
    if self.global_vox_rgb_feat_cnt is None:
      return None
    else:
      return self.global_vox_rgb_feat_cnt[:, :3]

  @property
  def global_vox_feat(self):
    if self.global_vox_rgb_feat_cnt is None:
      return None
    else:
      return self.global_vox_rgb_feat_cnt[:, 3:-1]

  @property
  def global_vox_cnt(self):
    if self.global_vox_rgb_feat_cnt is None:
      return None
    else:
      return self.global_vox_rgb_feat_cnt[:, -1:]

  @property
  def global_rays_feat(self):
    if self.global_rays_feats_cnt is None:
      return None
    else:
      return self.global_rays_feats_cnt[:, :-1]

  @property
  def global_rays_cnt(self):
    if self.global_rays_feats_cnt is None:
      return None
    else:
      return self.global_rays_feats_cnt[:, -1:]

  @override
  def save(self, file_path):
    raise NotImplementedError()

  @override
  def load(self, file_path):
    raise NotImplementedError()

  @override
  def is_empty(self) -> bool:
    return (self.occ_map_vdb.empty() and
            (self.global_vox_xyz is None or
             self.global_vox_xyz.shape[0] == 0) and
            (self.global_rays_orig_angles is None or
             self.global_rays_orig_angles.shapes[0]))

  def summarize_frontier_feats(self):
    """Summarize the frontier rays into frontiers by averaging features.
    
    Returns:
      xyz: Nx3 float tensor describing the locations of frontiers that have
        features.
      feats: NxC float tensor describing frontier features.
    """
    if self.global_rays_orig_angles is None:
      return None, None
    fronti_xyz = torch.clone(self.global_rays_orig_angles[:, :3])
    fronti_feats_cnt = torch.clone(self.global_rays_feats_cnt)
    fronti_xyz, fronti_feats_cnt = g3d.add_weighted_sparse_voxels(
      fronti_xyz, fronti_feats_cnt,
      torch.zeros_like(fronti_xyz[1:1]),
      torch.zeros_like(fronti_feats_cnt[1:1]),
      self.vox_size)

    return fronti_xyz, fronti_feats_cnt[:, :-1]

  @override
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None,
                         feat_img: torch.FloatTensor = None) -> dict:
    update_info = dict()

    # TODO: Decouple ray resolution from depth resolution. Would be beneficial
    # to project a smaller number of rays at lower resolution to reduce
    # object semantics leaking at the boundaries.

    r = g3d.depth_to_sparse_occupancy_voxels(
      depth_img, pose_4x4, self.intrinsics_3x3, self.vox_size, conf_map,
      max_num_pts = self.max_pts_per_frame,
      max_num_empty_pts = self.max_empty_pts_per_frame,
      max_num_dirs = self.max_dirs_per_frame,
      max_depth_sensing = self.max_depth_sensing,
      occ_thickness=self.occ_thickness,
      return_pc=True, return_dirs= not self.global_encoding,
      dirs_erosion=self.ray_erosion,
    )
    if not self.global_encoding:
      vox_xyz, vox_occ, pc_xyz, selected_pc_ind, \
        origs, dirs, selected_dir_ind = r
    else:
      vox_xyz, vox_occ, pc_xyz, selected_pc_ind = r
      origs = torch.zeros(pose_4x4.shape[0], 1, 3,
                          dtype=torch.float, device=self.device)
      dirs = torch.zeros(pose_4x4.shape[0], 1, 3,
                         dtype=torch.float, device=self.device)
      dirs[..., -1] = 1
      origs = g3d.transform_points(origs, pose_4x4).reshape(-1, 3)
      dirs = g3d.transform_points(dirs, pose_4x4).reshape(-1, 3) - origs

    vox_xyz, vox_occ = self._clip_pc(vox_xyz, vox_occ)
    pc_xyz, selected_pc_ind = self._clip_pc(
      pc_xyz, selected_pc_ind.unsqueeze(-1))
    selected_pc_ind = selected_pc_ind.squeeze(-1)

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

    pts_rgb = pts_rgb.permute(0, 2, 3, 1).reshape(-1, 3)[selected_pc_ind]

    if not self.global_encoding:
      if feat_img is None:
        feat_img = self.encoder.encode_image_to_feat_map(rgb_img)
      feat_img = self._proj_resize_feat_map(feat_img, dH, dW)
      update_info["feat_img"] = feat_img

      feat_img_flat = feat_img.permute(0, 2, 3, 1).reshape(-1,
                                                           feat_img.shape[1])

      dirs_feat = feat_img_flat[selected_dir_ind]
      pts_feat = feat_img_flat[selected_pc_ind]
      del feat_img_flat

    else:
      feat_vec = self.encoder.encode_image_to_vector(rgb_img)
      dirs_feat = feat_vec
      pts_feat = feat_vec[selected_pc_ind // (dH*dW)]

    N = pts_rgb.shape[0]
    pts_rgb_feat_cnt = torch.cat(
      (pts_rgb, pts_feat, torch.ones((N, 1), device=self.device)), dim=-1)
    # [0, 1] to [-1, occ_observ_weight]
    vox_occ = vox_occ*self.occ_observ_weight-1

    self._vox_accum_cnt += B
    self._occ_pruning_cnt += B
    self._sem_pruning_cnt += B

    if self._ray_accum_delay > 0:
      self._ray_accum_delay -= B
    else:
      self._ray_accum_cnt += B

    if vox_xyz.shape[0] > 0:
      self._tmp_vox_occ.append(vox_occ)
      self._tmp_vox_xyz.append(vox_xyz)
    if pc_xyz.shape[0] > 0:
      self._tmp_pc_xyz.append(pc_xyz)
      self._tmp_pc_rgb_feat_cnt.append(pts_rgb_feat_cnt)

    if origs.shape[0] > 0:
      self._tmp_ray_feat.append(dirs_feat)
      self._tmp_ray_orig.append(origs)
      self._tmp_ray_dir.append(dirs)

    if self._vox_accum_cnt >= self.vox_accum_period:
      self._vox_accum_cnt = 0

      ## 1. Accumulate occupancy voxels
      updated_vox_xyz = self.accum_occ_voxels()

      ## 2. Accumulate semantic voxels
      self.accum_semantic_voxels()

      ## 3. Prune occupancy map
      if (self.occ_pruning_period > -1 and
          self._occ_pruning_cnt >= self.occ_pruning_period):

        self._occ_pruning_cnt = 0
        self.occ_map_vdb.prune(self.occ_pruning_tolerance)

      ## 4. Prune semantic voxels
      if (self.sem_pruning_period > -1 and
          self.global_vox_xyz is not None and
          self.global_vox_xyz.shape[0] > 0 and
          self._sem_pruning_cnt >= self.sem_pruning_period):

        self._sem_pruning_cnt = 0
        self.prune_semantic_voxels(
          torch.cat(self._tmp_vox_xyz_since_prune, dim=0))
        self._tmp_vox_xyz_since_prune.clear()

      ## 5. Update Frontiers

      # Compute active window/bbox.
      # TODO: Test if its faster to project boundary points and pose centers
      # instead of doing min max over all tmp voxels. Or maybe let occ_pc2vdb
      # return the bounding box since it will iterate over all voxels already.
      if updated_vox_xyz.shape[0] > 0:
        active_bbox_min = torch.min(updated_vox_xyz, dim = 0).values
        active_bbox_max = torch.max(updated_vox_xyz, dim = 0).values
        self.update_frontiers(active_bbox_min, active_bbox_max)

    if self._ray_accum_cnt >= self.ray_accum_period:
      self._ray_accum_cnt = 0

      ## 6. Prune semantic rays
      self.prune_semantic_rays()

      ## 7. Update semantic rays
      self.cast_semantic_rays()

      if self.infer_direction:
        self.compute_inferred_directions()

    return update_info

  def cast_semantic_rays(self) -> None:
    """Cast semantic rays accumulated in the temporary buffers onto frontiers.
    """
    for l in self._tmp_ray_dir:
      if len(l) > 0:
        break
    else:
      # No rays to cast
      return
    ray_dir = torch.cat(self._tmp_ray_dir, dim = 0) # Nx3
    self._tmp_ray_dir.clear()
    ray_orig = torch.cat(self._tmp_ray_orig, dim = 0) # Nx3
    self._tmp_ray_orig.clear()
    ray_feat = torch.cat(self._tmp_ray_feat, dim = 0) # Nx3
    self._tmp_ray_feat.clear()
    N = ray_orig.shape[0]
    M = self.frontiers.shape[0] if self.frontiers is not None else 0
    if N > 0 and M > 0:
      frontier_vox_size = self.vox_size*self.fronti_subsampling
      frontier_dir = self.frontiers.reshape(M,1,3) - ray_orig.reshape(1,N,3)
      dot_prod = frontier_dir.reshape(M, N, 1, 3) @ \
        ray_dir.reshape(1, N, 3, 1)
      dot_prod = dot_prod.squeeze(-1).squeeze(-1)
      closest_pts = dot_prod.reshape(M, N, 1) * ray_dir.reshape(1, N, 3) + \
        ray_orig.reshape(1, N, 3)

      # MxN distance matrix where [i,j] represents the shortest distance
      # between frontier i and ray-origin j.
      dist = torch.norm(frontier_dir, dim=-1)

      if not self.zero_depth_mode:
        # MxN distance matrix where [i,j] represents the shortest distance
        # between frontier i and ray j.
        ortho_dist = torch.norm(
          closest_pts - self.frontiers.reshape(M, 1, 3), dim=-1)

        # In [0-1]
        cost_matrix = (ortho_dist/ortho_dist.max() + dist/dist.max()) / 2

        # Only consider distances where frontier is in front of ray,
        # frontier is at a minimum distance from ray origin, and orthogonal
        # distance is at a maximum distance.
        cost_matrix[(dot_prod <= 0) |
                    (dist < frontier_vox_size*2) |
                    (ortho_dist > frontier_vox_size)] = torch.inf

        if self.max_depth_sensing > 0:
          cost_matrix[dist > self.max_depth_sensing*3] = torch.inf

      else:
        # If global_vox_xyz is None then we are in 0 depth mode
        cost_matrix = dist

      # Match rays with frontiers
      # TODO: Add option for interpolation instead of assigning each ray to a
      # single frontier.
      min_cost, min_cost_ind = torch.min(cost_matrix, dim=0)
      mask = min_cost.isfinite()
      min_cost = min_cost[mask]
      min_cost_ind = min_cost_ind[mask]
      ray_orig = ray_orig[mask, :]
      ray_dir = ray_dir[mask, :]
      ray_feat = ray_feat[mask, :]

      if self.ray_tracing and not self.zero_depth_mode and \
        ray_orig.shape[0] > 0:
        # Now we have assigned a ray to the best frontier candidate, let us
        # make sure that ray is not occluded by observed occupied voxels or
        # potentially occupied unobserved voxels.

        # We select the end point for each ray which is the closest point to
        # the selected frontier on that ray.
        closest_pts = closest_pts[:, mask, :]
        closest_pts = closest_pts[min_cost_ind, torch.arange(
          min_cost_ind.shape[0], device=self.device)]

        if self.max_depth_sensing > 0:
          max_dist = self.max_depth_sensing * 3
        else:
          max_dist = torch.norm(closest_pts - ray_orig, dim=-1).max().item()

        max_steps = int(math.ceil(max_dist/self.vox_size))
        marching_rays = torch.clone(ray_orig)
        # Marching mask
        m = torch.ones_like(ray_orig[:, 0], dtype=torch.bool)

        for s in range(max_steps):
          marching_rays[m] = marching_rays[m] + self.vox_size*ray_dir[m]

          occ = rayfronts_cpp.query_occ(
            self.occ_map_vdb, marching_rays[m].cpu()).to(self.device)
          m[m.nonzero()[occ >= 0]] = False
          m &= (torch.norm(marching_rays - closest_pts, dim=-1) >
                self.vox_size)

        # Valid ray mask
        vrm = torch.norm(marching_rays - closest_pts, dim=-1) <= self.vox_size
        min_cost = min_cost[vrm]
        min_cost_ind = min_cost_ind[vrm]
        ray_orig = ray_orig[vrm, :]
        ray_dir = ray_dir[vrm, :]
        ray_feat = ray_feat[vrm, :]

        ## Below is more parallel but extremely expensive for memory
        # coefs = torch.linspace(0, 1, max_steps, device=self.device)
        # coefs = coefs.reshape(1, -1, 1)
        # ray_trace_xyz = coefs * ray_orig.unsqueeze(1) + \
        #   (1-coefs) * closest_pts.unsqueeze(1)

      # Change the ray's origin to its matching frontier.
      ray_orig = self.frontiers[min_cost_ind]

      # TODO: Partition angle space in a better way
      _, theta, phi = g3d.cartesian_to_spherical(
        ray_dir[:, 0], ray_dir[:, 1], ray_dir[:, 2])

      ray_orig_angle = torch.cat(
        [ray_orig, torch.rad2deg(theta).unsqueeze(-1),
        torch.rad2deg(phi).unsqueeze(-1)], dim=-1)

      ray_weights = (1-min_cost).unsqueeze(-1)
      ray_weights /= torch.sum(ray_weights)
      if self.global_rays_orig_angles is None:
        ray_orig_angle, ray_feat_cnt = g3d.bin_rays(
          ray_orig_angle, self.vox_size, self.angle_bin_size,
          torch.cat((ray_feat, ray_weights), dim=-1),
          aggregation="weighted_mean")

        self.global_rays_orig_angles = ray_orig_angle
        self.global_rays_feats_cnt = ray_feat_cnt
      else:
        self.global_rays_orig_angles, self.global_rays_feats_cnt = \
          g3d.add_weighted_binned_rays(
            self.global_rays_orig_angles,
            self.global_rays_feats_cnt,
            ray_orig_angle,
            torch.cat((ray_feat, ray_weights), dim=-1),
            vox_size=self.vox_size,
            bin_size=self.angle_bin_size,
          )

  def prune_semantic_rays(self) -> None:
    """Remove semantic rays that are no longer on a frontier.
    
    If ray_tracing is enabled, then the removed rays are added back to the
    accumulation buffer to be recast in the next iteration.
    """
    if self.global_rays_orig_angles is None:
      return
    # if a ray origin does not lie on a frontier then that ray will
    # be removed.
    M = self.frontiers.shape[0]
    N = self.global_rays_orig_angles.shape[0]
    dist = torch.norm(
      self.frontiers.reshape(M, 1, 3) -
      self.global_rays_orig_angles[:, :3].reshape(1, N, 3), p=1, dim=-1)
    mask = torch.any(dist < 1e-6, dim=0)
    if self.ray_tracing:
      # If ray tracing is enabled then we push these rays instead of
      # destroying them
      re_shoot_orig_angles =  self.global_rays_orig_angles[~mask]
      if re_shoot_orig_angles.shape[0] > 0:
        re_shoot_feats_cnt =  self.global_rays_feats_cnt[~mask]
        ray_orig = re_shoot_orig_angles[:, :3]
        angles = torch.deg2rad(re_shoot_orig_angles[:, 3:])
        ray_dir = torch.stack(
          g3d.spherical_to_cartesian(1, angles[:, 0], angles[:, 1]), dim=-1)
        self._tmp_ray_dir.append(ray_dir)
        self._tmp_ray_orig.append(ray_orig)
        self._tmp_ray_feat.append(re_shoot_feats_cnt[:, :-1])


    self.global_rays_orig_angles = self.global_rays_orig_angles[mask]
    self.global_rays_feats_cnt = self.global_rays_feats_cnt[mask]

  def accum_occ_voxels(self) -> torch.FloatTensor:
    """Accumulate the temporarilly gathered occupancy voxels.
    
    Returns:
      Voxels that were updated (With repetitions)
    """
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
    return vox_xyz

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

  def update_frontiers(self, active_bbox_min, active_bbox_max) -> None:
    frontiers_update = rayfronts_cpp.parallel_filter_cells_in_bbox(
      self.occ_map_vdb,
      cell_type_to_iterate = rayfronts_cpp.CellType.Empty,
      world_bbox_min = rayfronts_cpp.Vec3d(*active_bbox_min),
      world_bbox_max = rayfronts_cpp.Vec3d(*active_bbox_max),
      neighborhood_r = self.fronti_neighborhood_r,
      min_unobserved = self.fronti_min_unobserved,
      min_empty = self.fronti_min_empty,
      min_occupied = self.fronti_min_occupied,
    ).to(self.device)

    # Subsample frontiers using voxel grid
    if self.fronti_subsampling > 1:
      frontiers_update, cnt = g3d.pointcloud_to_sparse_voxels(
        frontiers_update, vox_size=self.vox_size*self.fronti_subsampling,
        feat_pc=torch.ones_like(frontiers_update[:, 0:1]),
        aggregation="sum")
      frontiers_update = \
        frontiers_update[cnt.squeeze(-1) > self.fronti_subsampling_min_fronti]

    # Update global frontiers
    if self.frontiers is None:
      self.frontiers = frontiers_update
    else:
      # Replace old frontiers in active window with updated frontiers
      self.frontiers = self.frontiers[
        torch.logical_or(torch.any(self.frontiers < active_bbox_min, dim=-1),
                          torch.any(self.frontiers > active_bbox_max, dim=-1))]
      self.frontiers = torch.cat([self.frontiers, frontiers_update], dim=0)

  def compute_inferred_directions(self):
    """Infer frontier directions based on occupancy map."""
    frontiers = self.global_rays_orig_angles[:, :3]
    a = torch.arange(-1, 2, 1, device=self.device, dtype=torch.float)
    a *= self.vox_size
    window = torch.stack(torch.meshgrid(a, a, a, indexing="xy"),
                          dim=-1).reshape(-1, 3)
    window = window[torch.arange(0, 27) != 13] # Remove center
    query_pts = (frontiers.unsqueeze(1) + window.unsqueeze(0))
    query_pts = query_pts.reshape(-1, 3)
    occ = rayfronts_cpp.query_occ(self.occ_map_vdb, query_pts.cpu())
    occ = occ.to(self.device).reshape(-1, 26).float()
    weight = -torch.ones_like(occ)
    weight[occ==0] = 1
    dirs = weight @ window
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    _, theta, phi = g3d.cartesian_to_spherical(
      dirs[:, 0], dirs[:, 1], dirs[:, 2])
    rays_orig_angles = torch.cat(
      [frontiers, torch.rad2deg(theta).unsqueeze(-1),
       torch.rad2deg(phi).unsqueeze(-1)], dim=-1)

    self.global_rays_orig_angles = rays_orig_angles

  @override
  def feature_query(self,
                    feat_query: torch.FloatTensor,
                    softmax: bool = False,
                    compressed: bool = True)-> dict:
    if self.is_empty():
      return

    r = dict()
    # Query semantic voxels
    if self.global_vox_xyz is not None:
      vox_feat = self.global_vox_feat
      if self.feat_compressor is not None and not compressed:
        vox_feat = self.feat_compressor.decompress(vox_feat)
      vox_feat = self.encoder.align_spatial_features_with_language(
        vox_feat.unsqueeze(-1).unsqueeze(-1)
      ).squeeze(-1).squeeze(-1)
      r["vox_xyz"] = self.global_vox_xyz
      r["vox_sim"] = compute_cos_sim(feat_query, vox_feat, softmax=softmax).T

    # Query semantic ray frontiers
    if self.global_rays_orig_angles is not None and \
       self.global_rays_orig_angles.shape[0] > 0:

      rays_feat = self.global_rays_feat
      if self.feat_compressor is not None and not compressed:
        rays_feat = self.feat_compressor.decompress(rays_feat)
      rays_feat = self.encoder.align_spatial_features_with_language(
        rays_feat.unsqueeze(-1).unsqueeze(-1)
      ).squeeze(-1).squeeze(-1)
      r["ray_orig_angles"] = self.global_rays_orig_angles
      r["ray_sim"] = compute_cos_sim(feat_query, rays_feat, softmax=softmax).T

    return r

  @override
  def vis_map(self) -> None:
    if self.visualizer is None or self.is_empty():
      return

    # Vis semantic voxels
    if self.global_vox_xyz is not None and self.global_vox_xyz.shape[0] > 0:
      self.visualizer.log_pc(self.global_vox_xyz, self.global_vox_rgb,
                            layer="voxel_rgb")
      if self.encoder is not None:
        self.visualizer.log_feature_pc(
          self.global_vox_xyz, self.global_vox_feat, layer="voxel_feature")

      log_hit_count = torch.log2(self.global_vox_cnt.squeeze())
      self.visualizer.log_heat_pc(self.global_vox_xyz, log_hit_count,
                                  layer="voxel_log_hit_count")

    # Vis occupancy voxels
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

    # Vis frontiers
    if self.frontiers is not None and self.frontiers.shape[0] > 0:
      self.visualizer.log_pc(self.frontiers, layer="frontiers")

    # Vis rays
    if (self.global_rays_orig_angles is not None and
        self.global_rays_orig_angles.shape[0] > 0):

      if self.angle_bin_size >= 360 and not self.infer_direction:
        self.visualizer.log_feature_pc(self.global_rays_orig_angles[:, :3],
                                       self.global_rays_feat,
                                       layer="semantic_frontiers")
      else:
        ray_orig = self.global_rays_orig_angles[:, :3]
        angles = torch.deg2rad(self.global_rays_orig_angles[:, 3:])
        ray_dir = torch.stack(
          g3d.spherical_to_cartesian(1, angles[:, 0], angles[:, 1]), dim=-1)
        self.visualizer.log_feature_arr(ray_orig, ray_dir,
                                        self.global_rays_feat,
                                        layer="semantic_ray_frontiers")

  @override
  def vis_update(self, **kwargs) -> None:
    if "feat_img" in kwargs:
      self.visualizer.log_feature_img(kwargs["feat_img"][-1].permute(1, 2, 0))

  @override
  def vis_query_result(self,
                       query_results: dict,
                       vis_labels: List[str] = None,
                       vis_colors: Dict[str, str] = None,
                       vis_thresh: float = 0) -> None:
    if query_results is None:
      return

    # Vis voxel results
    if "vox_sim" in query_results:
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
          layer=f"queries/{label.replace(' ', '_').replace('/', '_')}/voxels",
          vis_thresh=vis_thresh,
          **kwargs)

    # Vis rays
    if "ray_sim" in query_results:
      ray_sim = query_results["ray_sim"]
      ray_orig = query_results["ray_orig_angles"][:, :3]
      angles = torch.deg2rad(query_results["ray_orig_angles"][:, 3:])
      ray_dir = torch.stack(
        g3d.spherical_to_cartesian(1, angles[:, 0], angles[:, 1]), dim=-1)

      for q in range(ray_sim.shape[0]):
        kwargs = dict()
        label = vis_labels[q]
        if vis_colors is not None and label in vis_colors and \
            vis_colors[label] is not None:

          kwargs["high_color"] = vis_colors[label]
          kwargs["low_color"] = vis_colors[label]

        if self.angle_bin_size >= 360 and not self.infer_direction:
          self.visualizer.log_heat_pc(
            ray_orig, ray_sim[q, :],
            layer=f"queries/{label.replace(' ', '_').replace('/', '_')}/frontiers",
            vis_thresh=vis_thresh,
            **kwargs)
        else:
          self.visualizer.log_heat_arrows(
            ray_orig, ray_dir, ray_sim[q, :],
            layer=f"queries/{label.replace(' ', '_').replace('/', '_')}/rays",
            vis_thresh=vis_thresh,
            **kwargs)
