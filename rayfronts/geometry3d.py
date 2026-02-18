"""Functions related to geometrical manipulation in pytorch"""

from typing import Tuple

import torch
import numpy as np
import torch_scatter

def pts_to_homogen(pts):
  """Works on last dimension"""
  if pts.shape[-1] != 3:
    raise ValueError(f"Invalid points tensor shape {pts.shape}. "
                      "Last dim should have length 3")
  return torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)

def pts_to_nonhomo(pts):
  """Works on last dimension"""
  if pts.shape[-1] != 4:
    raise ValueError(f"Invalid points tensor shape {pts.shape}. "
                      "Last dim should have length 4")
  return pts[..., :3]

def mat_3x4_to_4x4(mat):
  """Works on last two dimensions"""
  row = torch.tensor([[0,0,0,1]], device=mat.device)
  mat = torch.cat((mat, row.repeat(*mat.shape[:-2], 1, 1)), axis=-2)
  return mat

def mat_3x3_to_4x4(mat):
  """Works on last two dimensions"""
  zeros = torch.zeros(size=(*mat.shape[:-2], 3, 1), device=mat.device)
  mat = torch.cat((mat, zeros), dim=-1)
  return mat_3x4_to_4x4(mat)

def transform_points_homo(homo_points, transform_mat_4x4):
  return homo_points @ torch.transpose(transform_mat_4x4, -2, -1)

def transform_points(points, transform_mat):
  """Batched application of a 4x4 transformation matrix to 3d points
  
  points: a BxNx3 or Nx3 or BxNx4 or Nx4 float tensor representing points in
    xyz. If last dimension has length 4 the points are assumed to be in 
    homogeneus coordinates.
  transform_mat: a Bx4x4 or 4x4 or Bx3x4 or 3x4 or Bx3x3 or 3x3 float tensor. 
    If points are batched, then transform_mat must be batched as well.
  """
  # Input checks and conditioning
  if transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 3:
    transform_mat = mat_3x3_to_4x4(transform_mat)
  elif transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 4:
    transform_mat = mat_3x4_to_4x4(transform_mat)
  elif transform_mat.shape[-2] != 4 or transform_mat.shape[-1] != 4:
    raise ValueError(f"Invalid transform matrix shape {transform_mat.shape}")

  is_input_nonhomo = points.shape[-1] == 3
  if is_input_nonhomo:
    points = pts_to_homogen(points)
  if points.shape[-1] != 4:
    raise ValueError(f"Invalid points shape {points.shape}")

  transformed_points = transform_points_homo(points, transform_mat)

  if is_input_nonhomo:
    return pts_to_nonhomo(transformed_points)
  else:
    return transformed_points

def transform_pose_4x4(pose_4x4, transform_mat_4x4):
  """

  If pose_4x4 is in basis A, then transform_mat_4x4 should be a transform that
  transforms a point from basis A to basis B. The resulting pose will be in
  basis B.
  """
  return transform_mat_4x4 @ pose_4x4 @ torch.inverse(transform_mat_4x4)

def transform_pose(pose, transform_mat):
  # Input checks and conditioning
  if transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 3:
    transform_mat = mat_3x3_to_4x4(transform_mat)
  elif transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 4:
    transform_mat = mat_3x4_to_4x4(transform_mat)
  elif transform_mat.shape[-2] != 4 or transform_mat.shape[-1] != 4:
    raise ValueError(f"Invalid transform matrix shape {transform_mat.shape}")

  if pose.shape[-2] == 3 and pose.shape[-1] == 3:
    pose = mat_3x3_to_4x4(pose)
    return transform_pose_4x4(pose, transform_mat)[..., :3, :3]
  elif pose.shape[-2] == 3 and pose.shape[-1] == 4:
    pose = mat_3x4_to_4x4(pose)
    return transform_pose_4x4(pose, transform_mat)[..., :3, :4]
  elif pose.shape[-2] != 4 or pose.shape[-1] != 4:
    raise ValueError(f"Invalid pose matrix shape {pose.shape}")

  return transform_pose_4x4(pose, transform_mat)


def get_coord_system_transform(src, tgt):
  """Get the transformation matrix from source to target coordinate systems.
  
  Args:
    src: A length 3 string defining the source coordinate system convention. 
      Ex. "RDF"
    tgt: Same format as source defining the target coordinate system.
  """
  axes=dict(r=0,l=0,u=1,d=1,f=2,b=2)
  T = torch.zeros((3,3), dtype=torch.float)
  for i, tgt_dir in enumerate(tgt.lower()):
    a =axes[tgt_dir]
    for j, src_dir in enumerate(src.lower()):
      b = axes[src_dir]
      if a == b:
        if src_dir == tgt_dir:
          sign = 1
        else:
          sign = -1
        T[i, j] = sign
        break
  return T

def disparity_to_depth(disparity_img: torch.FloatTensor,
                       focal_length: float,
                       stereo_baseline: float,
                       depth_scale: float = 1.0):
  """Converts disparity per pixel to depth per pixel.
  
  Args:
    disparity_img: A Bx1xHxW float tensor describing a batch of disparity images
    focal_length: Describes the focal length of the cameras 
      from which the disparity was calculated.
    stereo_baseline: Describes the horizontal distance between the left/right 
      cameras in the stereo setup.
    depth_scale: Resulting depth will be divided by this scaling factor
  """
  depth_img = focal_length*stereo_baseline/disparity_img
  depth_img = depth_img/depth_scale
  return depth_img

def depth_to_pointcloud(depth_img: torch.FloatTensor,
                        pose_4x4: torch.FloatTensor,
                        intrinsics_3x3: torch.FloatTensor,
                        conf_map: torch.FloatTensor = None,
                        max_num_pts: int = -1):
  """Unprojects valid depth img pixels to the world.
  
  Args:
    depth_img: A Bx1xHxW float tensor with values > 0 describing a batch of 
      depth images. May include NaN for missing values, +Inf for too far,
      -Inf for too close.
    pose_4x4: A Bx4x4 tensor which includes a batch of poses in opencv RDF.
      a pose is the extrinsics transformation matrix that takes you from
      camera/robot coordinates to world coordinates.
    intrinsics_3x3: A 3x3 float tensor including camera intrinsics.
    conf_map: A Bx1xHxW float tensor with values > 0 describing a batch of
      confidence maps in [0-1] range with 1 being most confident.
      This is optional and will only be used if max_num_pts > 0.
    max_num_pts: Maximum number of points to project per image.
      If confidence map is given, it will be used to choose the most confident
      points. If no confidence map then will uniformly sample from the whole
      batch. Set to -1 to project all valid depth points.
  Returns:
    A tuple of (xyz_pc, selected_indices)
    xyz_pc: Nx3 float tensor including the xyz positions of each point.
    selected_indices: A flattenned int tensor of size (B*H*W) signifying
      the points that have been projected from the depth image. this can be used
      to index rgb, feature, or label images to match the returned xyz tensor.
      Ex. `rgb_img.permute(0, 2, 3, 1).reshape(-1, 3)[selected_indices]`

  TODO: (Low priority) Add option to return list of Nx3 corresponding to each
  batch.

  TODO: Filter invalid depth before performing computations for higher
  efficiency. Problem is filtering before hand makes keeping images separate
  hard without resorting to for loops since each image will have a different
  number of points.
  """
  # Unproject
  B, _, H, W = depth_img.shape
  # Note that original depth_img above is indexed using [y,x] since it is HxW
  img_xi, img_yi = torch.meshgrid(torch.arange(W, device=depth_img.device),
                                  torch.arange(H, device=depth_img.device),
                                  indexing="xy")
  img_xi = img_xi.tile((B, 1, 1))
  img_yi = img_yi.tile((B, 1, 1))

  img_plane_pts = torch.stack([
    img_xi.flatten(-2),
    img_yi.flatten(-2),
    torch.ones(B, H*W, device=depth_img.device),
  ], axis=-1)
  img_plane_pts = depth_img.reshape(B, H*W, 1) * img_plane_pts

  unproj_mat = pose_4x4 @ mat_3x3_to_4x4(torch.inverse(intrinsics_3x3))
  world_pts_xyz = transform_points(img_plane_pts, unproj_mat)

  # Select valid points

  valid_depth_mask = torch.logical_and(torch.isfinite(depth_img), depth_img > 0)
  valid_depth_indices = torch.argwhere(valid_depth_mask.flatten()).squeeze(-1)
  max_num_pts *= B
  if max_num_pts > 0 and max_num_pts < len(valid_depth_indices):
    if conf_map is None:
      indices_indices = torch.randperm(len(valid_depth_indices))
    else:
      indices_indices = torch.argsort(conf_map[valid_depth_mask],
                                      descending=True)

    selected_indices = valid_depth_indices[indices_indices[:max_num_pts]]
  else:
    selected_indices = valid_depth_indices

  world_pts_xyz = world_pts_xyz.reshape(-1, 3)[selected_indices]
  return world_pts_xyz, selected_indices


def world_points_to_depth_image(
    world_pts_xyz: torch.FloatTensor,
    pose_4x4: torch.FloatTensor,
    intrinsics_3x3: torch.FloatTensor,
    resolution: Tuple[int, int]) -> torch.FloatTensor:
  """Projects world points into the camera and rasterizes to a depth image.

  Pose is camera-to-world (extrinsics). Points in front of the camera (z > 0)
  are projected; on pixel overlap the closest depth is kept.

  Args:
    world_pts_xyz: Nx3 float tensor of points in world coordinates.
    pose_4x4: 4x4 camera-to-world pose (RDF).
    intrinsics_3x3: 3x3 RGB camera intrinsics.
    resolution: (H, W) of the output depth image.

  Returns:
    A 1xHxW float tensor with depth per pixel; NaN where no point projects.
  """
  H, W = resolution
  device = world_pts_xyz.device
  cam_pts = transform_points(world_pts_xyz, torch.linalg.inv(pose_4x4))
  z = cam_pts[:, 2]
  valid = (z > 0) & torch.isfinite(z)
  if not valid.any():
    return torch.full((1, H, W), float("nan"), device=device, dtype=torch.float)
  cam_pts = cam_pts[valid]
  z = cam_pts[:, 2]
  # Project: uv = K @ cam_pts; (u, v) = (x/z, y/z) in pixel coords
  uv_h = (intrinsics_3x3 @ cam_pts.T).T  # Nx3
  u = (uv_h[:, 0] / uv_h[:, 2]).long()
  v = (uv_h[:, 1] / uv_h[:, 2]).long()
  in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
  if not in_bounds.any():
    return torch.full((1, H, W), float("nan"), device=device, dtype=torch.float)
  u = u[in_bounds]
  v = v[in_bounds]
  z = z[in_bounds]
  # Rasterize: keep closest (minimum) depth per pixel
  linear_idx = (v * W + u).long()
  depth_flat = torch.full(
      (H * W,), float("inf"), device=device, dtype=torch.float)
  depth_flat.scatter_reduce_(0, linear_idx, z, reduce="amin", include_self=False)
  depth_flat[depth_flat == float("inf")] = float("nan")
  return depth_flat.reshape(1, H, W)


def npy_pointcloud_to_sparse_voxels(xyz_pc, vox_size, feat_pc=None,
                                aggregation="mean", return_counts=False):
  """Numpy version of pointcloud_to_sparse_voxels"""
  d = xyz_pc.device
  xyz_pc = xyz_pc.cpu().numpy()
  xyz_vx = np.round(xyz_pc/vox_size).astype("int32")

  if feat_pc is None:
    xyz_vx = xyz_vx.astype("float")*vox_size

    xyz_vx, _pos, count_vx = np.unique(xyz_vx, return_index=True,
                                       return_counts=True, axis=0)
    if return_counts:
      return (torch.tensor(xyz_vx, dtype=torch.float32).to(d),
              torch.tensor(count_vx, dtype=torch.float32).unsqueeze(-1).to(d))
    else:
      return torch.tensor(xyz_vx, dtype=torch.float32).to(d)

  # Refer to:
  # https://stackoverflow.com/questions/50950231/group-by-with-numpy-mean
  sort_ind = np.lexsort((xyz_vx[:, 2],
                         xyz_vx[:, 1],
                         xyz_vx[:, 0]))
  xyz_vx = xyz_vx[sort_ind]

  xyz_vx, _pos, counts_vx = np.unique(xyz_vx, return_index=True,
                                      return_counts=True, axis=0)
  feat_pc = feat_pc.cpu().numpy()
  g_sum = np.add.reduceat(feat_pc[sort_ind], _pos, axis=0)
  if aggregation == "mean" or aggregation == "avg":
    feat_vx = g_sum / counts_vx[:,None]
  elif aggregation == "sum":
    feat_vx = g_sum
  else:
    raise ValueError(f"Unrecognized aggregation method '{aggregation}'.")

  xyz_vx = torch.tensor(xyz_vx.astype("float")*vox_size,
                        dtype=torch.float32).to(d)
  feat_vx = torch.tensor(feat_vx, dtype=torch.float32).to(d)
  counts_vx = torch.tensor(counts_vx, dtype=torch.float32).to(d).unsqueeze(-1)

  if return_counts:
    return xyz_vx, feat_vx, counts_vx
  else:
    return xyz_vx, feat_vx

def pointcloud_to_sparse_voxels(xyz_pc, vox_size, feat_pc=None,
                                aggregation="mean", return_counts=False):
  """Convert a point cloud to sparse voxels possibly aggregating point features.

  Args:
    xyz_pc: Nx3 float tensor including the xyz positions of each point.
    vox_size: Size of a voxel in world units.
    feat_pc: NxC float tensor including any features that will be aggregated
      through voxelization.
    aggregation: ['mean', 'sum']
    return_counts: Whether to return the number of points aggregated within each
    voxel or not.

  Returns:
    xyz_vx: Nx3 float tensor representing xyz centers of the voxels.
    feat_vx: (Returned if feat_pc is not None) NxC float tensor representing the
      aggregated features in each voxel
    count_vx: (Returned if return_counts is True) Nx1 float tensor representing
      how many points were aggregated in each voxel.

  TODO: Add option for weighted aggregation
  """
  d = xyz_pc.device

  # TODO Flooring is faster
  xyz_vx = torch.round(xyz_pc/vox_size).type(torch.int64)

  if feat_pc is None:
    xyz_vx, count_vx = torch.unique(xyz_vx, return_counts=True, dim=0)
    xyz_vx = xyz_vx.type(torch.float)*vox_size
    count_vx = count_vx.type(torch.float).unsqueeze(-1)
    if return_counts:
      return xyz_vx, count_vx
    else:
      return xyz_vx

  # TODO: torch.unique will be much faster if we convert xyz to a single number
  # that perserves the order.
  xyz_vx, reduce_ind, counts_vx = torch.unique(xyz_vx, return_inverse=True,
                                               return_counts=True,dim=0)
  feat_vx = torch.zeros((xyz_vx.shape[0], feat_pc.shape[-1]), device=d,
                         dtype=feat_pc.dtype)

  torch_scatter.scatter(
    src=feat_pc, index=reduce_ind, out=feat_vx, reduce=aggregation, dim=0)

  xyz_vx = xyz_vx.type(torch.float)*vox_size
  counts_vx = counts_vx.type(torch.float).unsqueeze(-1)

  if return_counts:
    return xyz_vx, feat_vx, counts_vx
  else:
    return xyz_vx, feat_vx

def add_weighted_sparse_voxels(xyz_vx1, feat_cnt_vx1, xyz_vx2, feat_cnt_vx2,
                               vox_size):
  """Aggregate two sparse voxel representations with voxel weights.
  
  Feature aggregation is done through multiplying features with their weights,
  summing then dividing by total weights.

  Does some calculations in-place !

  Can accept point clouds instead of voxels. Makes no assumption that the passed
  xyz locations are discretized or unique.

  Args:
    xyz_vx1: Nx3 float tensor including the xyz centers of each voxel.
    feat_cnt_vx1:  Nx(C+1) float tensor including any features that will be
      aggregated through voxelization + a weight/count column.

    xyz_vx2:
    feat_cnt_vx2:
  """
  # Multiply features by their respective weights
  feat_cnt_vx1[:, :-1] = feat_cnt_vx1[:, :-1] * feat_cnt_vx1[:, -1:]
  feat_cnt_vx2[:, :-1] = feat_cnt_vx2[:, :-1] * feat_cnt_vx2[:, -1:]

  xyz_vx, feat_cnt_vx = pointcloud_to_sparse_voxels(
      torch.cat((xyz_vx1, xyz_vx2), dim=0),
      vox_size=vox_size,
      feat_pc=torch.cat((feat_cnt_vx1, feat_cnt_vx2), dim=0),
      aggregation="sum", return_counts=False
  )
  # Divide features by total weight
  feat_cnt_vx[:, :-1] = feat_cnt_vx[:, :-1] / feat_cnt_vx[:, -1:]
  return xyz_vx, feat_cnt_vx

# TODO: Break this huge function down to components
def depth_to_sparse_occupancy_voxels(depth_img: torch.FloatTensor,
                                     pose_4x4: torch.FloatTensor,
                                     intrinsics_3x3: torch.FloatTensor,
                                     vox_size: float,
                                     conf_map: torch.FloatTensor = None,
                                     max_num_pts: int = -1,
                                     max_num_empty_pts: int = -1,
                                     max_num_dirs: int = -1,
                                     max_depth_sensing: float = -1,
                                     occ_thickness: int = 1,
                                     algorithm: str = "frustum_culling",
                                     return_pc: bool = False,
                                     return_dirs: bool = False,
                                     dirs_erosion: int = 0):
  """Raycasts a depth map into an occupancy sparse grid of voxels in world coord

  Args:
    depth_img: A Bx1xHxW float tensor with values > 0 describing a batch of 
      depth images. May include NaN for missing values, +Inf for too far,
      -Inf for too close.
    pose_4x4: A Bx4x4 tensor which includes a batch of poses in opencv RDF.
      a pose is the extrinsics transformation matrix that takes you from
      camera/robot coordinates to world coordinates.
    intrinsics_3x3: A 3x3 float tensor including camera intrinsics.
    vox_size: The voxel size of the voxel grid.
    conf_map: A Bx1xHxW float tensor with values > 0 describing a batch of
      confidence maps in [0-1] range with 1 being most confident.
      This is optional and will only be used if max_num_pts > 0.
    max_num_pts: Maximum number of points to project per image. 
      Will uniformly sample from the whole batch. Set to -1 to project all valid
      depth points.
    max_num_empty_pts: Maximum number of empty points to project per image.
      Will uniformly sample from the whole batch. Set to -1 to project all valid
      empty points.
    max_num_dirs: Maximum number of dirs to return. Set to -1 return all.
    max_depth_sensing: Depending on the max sensing range, we project 
      empty voxels up to that range if depth was not provided for that pixel.
      Set to -1 to use the max depth in batch as the max sensor range.
    algorithm: The algorithm used for tracing voxels. Choose from:
      'ray_sampling': The simplest, and least accurate method where we simply
        sample points along the ray at vox_size/2 intervals to get voxels along
        the ray.
      'bresenham': TODO More common/accurate way of rasterizing lines.
      'frustum_culling': Get all voxels in an axis aligned bbox in front of the 
        pose, project them onto your image, remove voxels that lie outside the 
        viewing frustum / image plane
    return_pc: If true, then the projected point cloud will be returned
      with the selected indices as well.
    return_dirs: If true, then oor pixels will be returned as directions.
    dirs_erosion: Should we erode the out of range mask for more
      conservative shooting of rays.
  Returns:
    xyz_vx: Nx3 float tensor representing xyz centers of the voxels.
    occ_vx: Nx1 float tensor representing the occupancy value. 
      Currently discretized as 0 For empty, 1 for occupied.

    If return_pc was True, then will also return these intermediate results:

    xyz_pc: LxKx3 float tensor including the xyz positions of each point.
      K represents the thickness dimension.
    selected_pc_indices: A flattenned int tensor of size L and with indices in 
      [0, B*H*W] the points that have been projected from the depth image. Can
      be used to index rgb, feature, or label images to match the returned xyz.
      Ex. `rgb_img.permute(0, 2, 3, 1).reshape(-1, 3)[selected_indices]`

    If return_dirs was True, then these will also be returned:
    origins: Mx3 float tensor 
    dirs: Mx3 float tensor including the unit direction from origins
      in world coordinates. i.e each direction represents the ray/line t*d+o
      where t is in [0, +Inf], d is direction, and o is the origin.
    selected_dir_indices: A flattenned int tensor of size M and with indices in 
      [0, B*H*W].
  """
  B, _, H, W = depth_img.shape
  device = depth_img.device
  valid_depth_mask = torch.logical_and(torch.isfinite(depth_img), depth_img > 0)

  min_depth = vox_size/2
  if max_depth_sensing > 0:
    max_depth = max_depth_sensing
  else:
    try:
      max_depth = depth_img[valid_depth_mask].max()
    except RuntimeError:
      max_depth = -1

  # img_plane_pts is fixed if B, H, W are fixed so no need to recreate with
  # with every function call so we use a function static variable.
  f = depth_to_sparse_occupancy_voxels
  if not hasattr(f, "img_plane_pts") or f.img_plane_pts[0] != (B, H, W):
    # Note that original depth_img above is indexed using [y,x] since it is HxW
    img_xi, img_yi = torch.meshgrid(torch.arange(W, device=device),
                                    torch.arange(H, device=device),
                                    indexing="xy")
    img_xi = img_xi.tile((B, 1, 1))
    img_yi = img_yi.tile((B, 1, 1))

    img_plane_pts = torch.stack([
      img_xi.flatten(-2),
      img_yi.flatten(-2),
      torch.ones(B, H*W, device=device),
    ], axis=-1)

    f.img_plane_pts = (B, H, W), img_plane_pts
  else:
    img_plane_pts = f.img_plane_pts[1]

  unproj_mat = pose_4x4 @ mat_3x3_to_4x4(torch.inverse(intrinsics_3x3))

  if return_dirs:
    # Select any particular depth. We will normalize direction later
    world_dirs = transform_points(img_plane_pts, unproj_mat)
    origins = pose_4x4[:, :3, -1].unsqueeze(1).tile(1, world_dirs.shape[1], 1)
    world_dirs = world_dirs - origins

  if max_depth <= 0:
    # No finite depth. We only return a single empty voxel for each pose
    # representing the robot.
    world_empty_pts_xyz = pose_4x4[:, :3, -1]
    xyz_vx = pointcloud_to_sparse_voxels(world_empty_pts_xyz, vox_size=vox_size)
    occupancy_vx = torch.zeros(1, 1, device=device)
    if return_pc:
      world_occ_pts_xyz = torch.empty(0, 3, device=device)
      occ_pts_img_indices = torch.empty(0, dtype=torch.long, device=device)
  else:
    if algorithm == "ray_sampling":
      sampled_depths = torch.arange(min_depth, max_depth, vox_size/2,
                                    device=device) # D
      # Unproject
      img_plane_empty_pts = (sampled_depths.reshape(1, 1, -1, 1) *
                             img_plane_pts.reshape(B, H*W, 1, 3))
      assert occ_thickness >= 1

      depth_step = vox_size/2
      depth_modifiers = torch.linspace(
        torch.tensor(-depth_step*(occ_thickness-1)/2),
        torch.tensor(+depth_step*(occ_thickness-1)/2),
        occ_thickness, device=device, dtype=torch.float)

      L = len(depth_modifiers)
      tmp = depth_img.reshape(B, H*W, 1) + depth_modifiers.reshape(1, 1, -1)
      img_plane_occ_pts = tmp.unsqueeze(-1) * img_plane_pts.unsqueeze(-2)

      # After doing pose transformations we can mix all points in the batch
      world_occ_pts_xyz = transform_points(
        img_plane_occ_pts.reshape(B, -1, 3),
        unproj_mat).reshape(-1, 3)

      occ_pts_img_indices = torch.arange(0, B*W*H, device=device).unsqueeze(-1)
      occ_pts_img_indices = occ_pts_img_indices.tile(1, L).flatten()

      mask = valid_depth_mask.reshape(-1, 1).tile(1, L).flatten()
      world_occ_pts_xyz = world_occ_pts_xyz[mask]
      occ_pts_img_indices = occ_pts_img_indices[mask]

      world_empty_pts_xyz = transform_points(
        img_plane_empty_pts.reshape(B, -1, 3),
        unproj_mat).reshape(B, H*W, -1, 3)

      mask = \
        (img_plane_empty_pts.flatten(end_dim=1)[..., -1]
        < depth_img.reshape(B*H*W, 1))

      world_observed_pts_xyz = world_empty_pts_xyz.flatten(end_dim=1)
      world_observed_pts_xyz = world_observed_pts_xyz[mask]

    elif algorithm == "frustum_culling":
      bbox_mn, bbox_mx = get_update_bbox(
        pose_4x4, intrinsics_3x3, resolution=(H,W),
        near=0, far=max_depth)

      xx = torch.arange(bbox_mn[0]-vox_size, bbox_mx[0]+vox_size*occ_thickness,
                        vox_size, device=device)
      yy = torch.arange(bbox_mn[1]-vox_size, bbox_mx[1]+vox_size*occ_thickness,
                        vox_size, device=device)
      zz = torch.arange(bbox_mn[2]-vox_size, bbox_mx[2]+vox_size*occ_thickness,
                        vox_size, device=device)

      world_bbox_pts = torch.stack(torch.meshgrid(xx, yy, zz, indexing="xy"),
                                   dim=-1).reshape(-1, 3)
      world_bbox_pts = world_bbox_pts.unsqueeze(0).tile(B, 1, 1)
      cam_bbox_pts = transform_points(world_bbox_pts, torch.inverse(pose_4x4))

      img_plane_bbox_pts = transform_points(
        cam_bbox_pts, intrinsics_3x3.unsqueeze(0).tile(B, 1, 1))
      img_plane_bbox_pts = img_plane_bbox_pts[..., :2] / \
        img_plane_bbox_pts[..., -1:]
      img_plane_bbox_pts = torch.round(img_plane_bbox_pts).long()

      mask = ((img_plane_bbox_pts[..., 0] > 0) &
             (img_plane_bbox_pts[..., 0] < W) &
             (img_plane_bbox_pts[..., 1] > 0) &
             (img_plane_bbox_pts[..., 1] < H))

      # Now we convert from h,w indexing per image to a flat index for the whole
      # batched image.
      img_bbox_indices = W*img_plane_bbox_pts[..., 1] + \
        img_plane_bbox_pts[..., 0]
      img_bbox_indices = img_bbox_indices + \
        (torch.arange(0, B, device=device) * H * W).unsqueeze(-1)

      in_frustum_indices = img_bbox_indices[mask]
      world_bbox_pts = world_bbox_pts[mask]
      world_bbox_pts_assoc_depth = depth_img.flatten()[in_frustum_indices]
      world_bbox_pts_actual_depth = cam_bbox_pts[..., -1][mask]

      # Now we have world points and their corresponding flat indices to use
      # to associate with depth, rgb, and feature images.
      # Now we can choose the empty points and the occupied points and filter
      # out of bound / invalid points.

      occ_mask = torch.abs(
        world_bbox_pts_actual_depth-world_bbox_pts_assoc_depth) \
          < vox_size/2 * occ_thickness

      world_occ_pts_xyz = world_bbox_pts[occ_mask]
      occ_pts_img_indices = in_frustum_indices[occ_mask]

      world_observed_pts_xyz = world_bbox_pts[
        (world_bbox_pts_actual_depth < world_bbox_pts_assoc_depth-vox_size) &
        (world_bbox_pts_actual_depth < max_depth-vox_size) &
        (world_bbox_pts_actual_depth > min_depth)]
    else:
      raise ValueError(f"Algorithm '{algorithm}' is not implemented.")

    ## Select occupied points
    max_num_pts *= B
    if max_num_pts > 0 and max_num_pts < len(occ_pts_img_indices):
      if conf_map is None:
        indices_indices = torch.randperm(len(occ_pts_img_indices),
                                         device=device)
      else:
        indices_indices = torch.argsort(conf_map[occ_pts_img_indices],
                                        descending=True)

      occ_pts_img_indices = occ_pts_img_indices[indices_indices[:max_num_pts]]
      world_occ_pts_xyz = world_occ_pts_xyz[indices_indices[:max_num_pts], :]

    max_num_empty_pts *= B
    ## Select empty points
    if (max_num_empty_pts > 0 and
        max_num_empty_pts < world_observed_pts_xyz.shape[0]):

      all_indices = torch.arange(0, world_observed_pts_xyz.shape[0],
                                 device=device)
      if conf_map is None:
        indices_indices = torch.randperm(len(all_indices),
                                         device=device)
      else:
        indices_indices = torch.argsort(conf_map[valid_depth_mask],
                                        descending=True)

      selected_indices = all_indices[indices_indices[:max_num_empty_pts]]
      world_observed_pts_xyz = \
        world_observed_pts_xyz.reshape(-1, 3)[selected_indices]

    # At this point we selected empty points and occupied points.
    # Now we voxelize.

    flat_world_occ_pts_xyz = world_occ_pts_xyz.reshape(-1, 3)
    occupancy_pts = torch.vstack(
      [torch.ones_like(flat_world_occ_pts_xyz[..., -1:]),
      torch.zeros_like(world_observed_pts_xyz[..., -1:])])

    # We voxelize such that if a voxel has at least one point its occupancy
    # is set to 1. We do not overemphasize voxels that have many occupied pts.
    xyz_pts = torch.vstack([flat_world_occ_pts_xyz, world_observed_pts_xyz])
    xyz_vx, occupancy_vx = pointcloud_to_sparse_voxels(
      xyz_pts, vox_size, feat_pc = occupancy_pts,
      aggregation = "sum")

    occupancy_vx = torch.clamp(occupancy_vx, max=1)

  ## Select directions
  if return_dirs:
    oor_depth_mask = torch.isposinf(depth_img)
    if dirs_erosion > 0:
      r = dirs_erosion*2+1
      erosion_kernel = torch.ones(size =(1,1,r,r),
                                  dtype=torch.float, device=device)
      oor_depth_mask = torch.nn.functional.conv2d(oor_depth_mask.float(),
                                                  erosion_kernel,
                                                  padding="same")
      # Only if the whole neighborhood was positive then we cast.
      oor_depth_mask = oor_depth_mask.reshape(B*H*W, 1) >= (r**2)

  
    oor_depth_indices = torch.argwhere(oor_depth_mask.flatten()).squeeze(-1)
    max_num_dirs *= B
    if max_num_dirs > 0 and max_num_dirs < len(oor_depth_indices):
      indices_indices = torch.randperm(len(oor_depth_indices), device=device)
      dir_img_indices = oor_depth_indices[indices_indices[:max_num_dirs]]
    else:
      dir_img_indices = oor_depth_indices

    world_dirs = world_dirs.reshape(-1, 3)[dir_img_indices]
    origins = origins.reshape(-1, 3)[dir_img_indices]
    if world_dirs.shape[0] > 0:
      world_dirs = world_dirs / torch.norm(world_dirs, dim=-1, keepdim=True)

  return_vals = [xyz_vx, occupancy_vx]
  if return_pc:
    return_vals.extend([world_occ_pts_xyz, occ_pts_img_indices])
  if return_dirs:
    return_vals.extend([origins, world_dirs, dir_img_indices])


  return tuple(return_vals)

def pts_to_plane(pts: torch.FloatTensor):
  """Computes the plane that the 3 points lie on.

  Args:
    pts: *x3x3 Float Tensor
  
  Returns:
    *x4 Float tensor representing the 4 plane coefficients ax+by+cz+d=0
  """
  M = torch.cat([pts, torch.ones_like(pts[..., :3, :1])], dim=-1)
  U, S, Vt = torch.linalg.svd(M)
  return Vt[..., -1, :]

def get_update_bbox(pose_4x4: torch.FloatTensor,
                    intrinsics_3x3: torch.FloatTensor,
                    resolution: Tuple[float],
                    far: float,
                    near: float = 0):
  H, W = resolution
  B = pose_4x4.shape[0]

  plane_pts = lambda d: torch.tensor(
    [[0.0, 0.0, 1],
     [0.0, H, 1],
     [W, 0.0, 1],
     [W, H, 1]],
     dtype=torch.float, device=pose_4x4.device)*d

  near_plane_pts = plane_pts(near)
  far_plane_pts = plane_pts(far)

  all_plane_pts = torch.stack([near_plane_pts, far_plane_pts], dim=0)

  unproj_mat = pose_4x4 @ mat_3x3_to_4x4(torch.inverse(intrinsics_3x3))
  all_plane_pts = transform_points(
    all_plane_pts.reshape(1, -1, 3).tile(B, 1, 1),
    unproj_mat).reshape(B, 8, 3)
  
  mn = all_plane_pts.reshape(-1, 3).min(dim=0).values
  mx = all_plane_pts.reshape(-1, 3).max(dim=0).values
  return mn, mx

def get_frustum_planes(pose_4x4: torch.FloatTensor,
                       intrinsics_3x3: torch.FloatTensor,
                       resolution: Tuple[float],
                       far: float,
                       near: float = 0,
                       return_bbox: bool = False):

  """Compute the 6 planes comprising the frustum of the camera

  Normals are corrected to point inwards.

  Args:
    pose_4x4: A Bx4x4 tensor which includes a batch of poses in opencv RDF.
      a pose is the extrinsics transformation matrix that takes you from
      camera/robot coordinates to world coordinates.
    intrinsics_3x3: A 3x3 float tensor including camera intrinsics.
    resolution: A tuple of (H, W) describing the img size.
    far: Far plane depth in world units.
    near: Near plane depth in world units.

  Returns:
    Bx6x4 Float tensor where 6 corresponds to near,far,top,bottom,left,right
    respectively and 4 represents the plane coefficients abcd in ax+by+cz+d=0
    respectively.
  """
  # Unproject corner points
  H, W = resolution
  B = pose_4x4.shape[0]

  plane_pts = lambda d: torch.tensor(
    [[0.0, 0.0, 1],
     [0.0, H, 1],
     [W, 0.0, 1],
     [W, H, 1]],
     dtype=torch.float, device=pose_4x4.device)*d

  near_plane_pts = plane_pts(near)
  far_plane_pts = plane_pts(far)

  # 6x3x3
  all_plane_pts = torch.stack([
    near_plane_pts[:3], # Near
    far_plane_pts[:3], # Far
    torch.vstack([near_plane_pts[0], near_plane_pts[2], far_plane_pts[0]]), # Top
    torch.vstack([near_plane_pts[1], near_plane_pts[3], far_plane_pts[1]]), # Bottom
    torch.vstack([near_plane_pts[0], near_plane_pts[1], far_plane_pts[0]]), # Left
    torch.vstack([near_plane_pts[2], near_plane_pts[3], far_plane_pts[2]]), # Right
  ], dim=0)

  unproj_mat = pose_4x4 @ mat_3x3_to_4x4(torch.inverse(intrinsics_3x3))

  all_plane_pts = transform_points(
    all_plane_pts.reshape(1, -1, 3).tile(B, 1, 1),
    unproj_mat).reshape(B, 6, 3, 3)

  all_planes = pts_to_plane(all_plane_pts)

  # Correct normals by ensuring a center point is in front of all planes
  center_pt = torch.tensor(
    [[W/2, H/2, 1]], dtype=torch.float, device=pose_4x4.device)*(near+far)/2

  center_pt = transform_points(center_pt.reshape(1,1,3).tile(B, 1, 1),
                               unproj_mat)

  v = torch.linalg.vecdot(center_pt, all_planes[..., :3]) + all_planes[..., -1]
  all_planes[v<0] *= -1
  if return_bbox:
    mn = all_plane_pts.reshape(-1, 3).min(dim=0).values
    mx = all_plane_pts.reshape(-1, 3).max(dim=0).values
    return all_planes, (mn, mx)
  else:
    return all_planes

def rays_to_pose_4x4(rays):
  """Convert rays into a 4x4 pose that aligns the forward z axis with the ray.

  The angle of rotation around the ray itself (Roll) is arbitrary since that
  information is missing from the ray.

  Args:
    rays: Nx5 float tensor tensor including the xyz origin of each ray and its
      theta & phi angles in degrees.
  
  Returns: 
    Nx4x4 float tensor representing the corresponding poses.
  """
  d = rays.device
  N = rays.shape[0]
  xyz = rays[:, :3]
  theta = torch.deg2rad(rays[:, 3])
  phi = torch.deg2rad(rays[:, 4])
  dir_xyz = torch.stack(spherical_to_cartesian(1, theta, phi), dim=-1)
  initial_axis = torch.tensor([[0, 0, 1]], device=d, dtype=torch.float)
  rot_axes = torch.cross(initial_axis, dir_xyz, dim=-1)
  rot_angles = torch.arccos(torch.linalg.vecdot(initial_axis, dir_xyz))

  # https://en.wikipedia.org/wiki?curid=856005
  ca, sa = torch.cos(rot_angles), torch.sin(rot_angles)
  ux, uy, uz = rot_axes[:, 0], rot_axes[:, 1], rot_axes[:, 2]
  pose_4x4 = torch.zeros(N, 4, 4, device=d, dtype=torch.float)

  pose_4x4[:, 0, 0] = ux**2 * (1-ca) + ca
  pose_4x4[:, 1, 0] = ux*uy * (1-ca) + uz*sa
  pose_4x4[:, 2, 0] = ux*uz * (1-ca) - uy*sa

  pose_4x4[:, 0, 1] = ux*uy * (1-ca) - uz*sa
  pose_4x4[:, 1, 1] = uy**2 * (1-ca) + ca
  pose_4x4[:, 2, 1] = uy*uz * (1-ca) + ux*sa

  pose_4x4[:, 0, 2] = ux*uz * (1-ca) + uy*sa
  pose_4x4[:, 1, 2] = uy*uz * (1-ca) - ux*sa
  pose_4x4[:, 2, 2] = uz**2 * (1-ca) + ca

  pose_4x4[:, :3, -1] = xyz
  pose_4x4[:, -1, -1] = 1

  return pose_4x4

def intrinsics_3x3_to_fov(intrinsics_3x3: torch.FloatTensor, resolution: tuple):
  """
  
  Args:
    intrinsics_3x3: A 3x3 float tensor including camera intrinsics.
    resolution: H, W
  Returns:
    fovx,fovy representing the field of view in radians.
  """
  H, W = resolution
  fx = intrinsics_3x3[0, 0]
  fovx = 2 * torch.arctan(W / (2*fx))
  fy = intrinsics_3x3[1, 1]
  fovy = 2 * torch.arctan(H / (2*fy))
  return fovx, fovy


def get_cone_planes(pose_4x4: torch.FloatTensor,
                    far: float,
                    apex_angle: float,
                    near: float = 0,
                    start_radius: float = 0,
                    num_segs: int = 8):
  """Compute the planes that define cones emenating from a batch of poses

  Normals are corrected to point inwards.

  This can be used as opposed to frustums when the roll angle across the 
  forward axis (z-axis) is unknown/arbitrary in the poses.

  Args:
    pose_4x4: A Bx4x4 tensor which includes a batch of poses in opencv RDF.
      a pose is the extrinsics transformation matrix that takes you from
      camera/robot coordinates to world coordinates.
    far: Far plane depth in world units.
    apex_angle: Cone angle in radians.
    near: Near plane depth in world units.
    start_radius: The radius of the near cap circle.
    num_segs: How many segments to use for the start and end circles

  Returns:
    Bx(2+num_segs)x4 Float tensor where (2+num_segs) corresponds to near, far, 
    and the circumeference cone planes respectively. The last dimension (4)
    represents the plane coefficients abcd in ax+by+cz+d=0 respectively.
  """
  B = pose_4x4.shape[0]
  d = pose_4x4.device
  angle_delta = 2*torch.pi / num_segs
  theta = torch.arange(0, 2*torch.pi, angle_delta, device=d, dtype=torch.float)
  end_radius = far * torch.tan(apex_angle)

  near_cap = torch.stack([
    start_radius * torch.cos(theta),
    start_radius * torch.sin(theta),
    near * torch.ones_like(theta)
  ], dim=-1).unsqueeze(0).tile(B, 1, 1)
  far_cap = torch.stack([
    end_radius * torch.cos(theta),
    end_radius * torch.sin(theta),
    far * torch.ones_like(theta)
  ], dim=-1).unsqueeze(0).tile(B, 1, 1)

  near_cap = transform_points(near_cap, pose_4x4)
  far_cap = transform_points(far_cap, pose_4x4)

  plane_pts = [near_cap[:, :3, :], far_cap[:, :3, :]]
  for i in range(num_segs):
    p1 = near_cap[:, i, :]
    p2 = near_cap[:, (i+1) % num_segs, :]
    p3 = far_cap[:, i, :]
    plane_pts.append(
      torch.stack([p1, p2, p3], dim=1)
    )
  plane_pts = torch.stack(plane_pts, dim=1)
  planes = pts_to_plane(plane_pts)

  # Correct normals by ensuring a center point is in front of all planes
  center_pt = torch.tensor(
    [[0, 0, (near+far)/2]], dtype=torch.float, device=d)

  center_pt = transform_points(center_pt.reshape(1,1,3).tile(B, 1, 1),
                               pose_4x4)

  v = torch.linalg.vecdot(center_pt, planes[..., :3]) + planes[..., -1]
  planes[v<0] *= -1

  return planes



def cartesian_to_spherical(x, y, z):
  """Convert cartesian coordinates to spherical coordinates
  
  Args:
    x: Float tensor of length N
    y: ...
    z: ...

  Returns:
    r: magnitude
    theta: the azimuthal angle in the xy-plane from the x-axis with 
      -pi<=theta<pi
    phi: the polar/zenith angle from the positive z-axis with 0<=phi<=pi
  """
  r = torch.sqrt(x**2 + y**2 + z**2)
  theta = torch.atan2(y, x)
  phi = torch.acos(z/r)
  return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
  """Convert spherical coordinates to cartesian coordinates

  See `cartesian_to_spherical` doc for more information.
  """
  sin_phi = torch.sin(phi)
  x = r * torch.cos(theta) * sin_phi
  y = r * torch.sin(theta) * sin_phi
  z = r * torch.cos(phi)
  return x, y, z


def bin_rays(rays, vox_size, bin_size=30, feat=None,
             aggregation="mean", return_counts=False):
  """Aggregate rays into angle bins possibly aggregating features.
  
  Args:
    rays: Nx5 Float tensor including the xyz origin of each ray and its 2 angles
      in degrees.
    vox_size: Size of a voxel.
    bin_size: Size of the angle bin.
    feat: NxC float tensor including any features that will be aggregated
      through binning. If aggregation is `weighted_mean` then the last column
      of the features is taken to be the weights.
    aggregation: ['mean', 'sum', 'weighted_mean']
    return_counts: Whether to return the number of points aggregated within each
      bin or not.
  Returns:
    rays: Nx5 float tensor tensor including the xyz origin of each ray and its 2
      angles in degrees.
    feat: (Returned if feat is not None) NxC float tensor representing the
      aggregated features in each ray bin
    count: (Returned if return_counts is True) Nx1 float tensor representing
      how many rays were aggregated in each bin.
  """
  d = rays.device

  binned_rays = torch.clone(rays)
  binned_rays[:, :3] /= vox_size
  binned_rays[:, 3:] /= bin_size

  # TODO Flooring is faster
  binned_rays = torch.round(binned_rays).type(torch.int64)

  if feat is None:
    binned_rays, bin_counts = torch.unique(
      binned_rays, return_counts=True, dim=0)

    binned_rays = binned_rays.type(torch.float)
    binned_rays[:, :3] *= vox_size
    binned_rays[:, 3:] *= bin_size
    bin_counts = bin_counts.type(torch.float).unsqueeze(-1)

    if return_counts:
      return binned_rays, bin_counts
    else:
      return binned_rays

  binned_rays, reduce_ind, bin_counts = torch.unique(
    binned_rays, return_inverse=True, return_counts=True, dim=0)

  binned_feat = torch.zeros((binned_rays.shape[0], feat.shape[-1]), device=d)
  reduce = aggregation
  if aggregation == "weighted_mean":
    # Multiply features by their respective weights
    feat[:, :-1] = feat[:, :-1] * feat[:, -1:]
    feat[:, :-1] = feat[:, :-1] * feat[:, -1:]
    reduce = "sum"

  torch_scatter.scatter(
    src=feat, index=reduce_ind, out=binned_feat, reduce=reduce, dim=0)

  if aggregation == "weighted_mean":
    # Divide features by total weight
    feat[:, :-1] = feat[:, :-1] / feat[:, -1:]

  binned_rays = binned_rays.type(torch.float)
  binned_rays[:, :3] *= vox_size
  binned_rays[:, 3:] *= bin_size
  bin_counts = bin_counts.type(torch.float).unsqueeze(-1)

  if return_counts:
    return binned_rays, binned_feat, bin_counts
  else:
    return binned_rays, binned_feat


def add_weighted_binned_rays(rays1, feat_weight1, rays2, feat_weight2, vox_size,
                             bin_size):
  """Aggregate two binned ray representations with weights.
  
  Feature aggregation is done through multiplying features with their weights,
  summing then dividing by total weights.

  Does some calculations in-place !

  Can accept non-binned rays instead of binned rays. Makes no assumption that
  the passed rays are discretized or unique.

  Args:
    rays1: Nx5 float tensor tensor including the xyz origin of each ray and its
      theta & phi angles in degrees.
    feat_weight1:  Nx(C+1) float tensor including any features that will be
      aggregated through binning + a weight/count column.

    rays2: ...
    feat_weight2: ...

  Returns:
    rays: Mx5 float tensor.
    feat_weight: Mx(C+1) float tensor.
  """
  # Multiply features by their respective weights
  feat_weight1[:, :-1] = feat_weight1[:, :-1] * feat_weight1[:, -1:]
  feat_weight2[:, :-1] = feat_weight2[:, :-1] * feat_weight2[:, -1:]

  rays, feat_weight = bin_rays(
      torch.cat((rays1, rays2), dim=0),
      vox_size=vox_size,
      bin_size=bin_size,
      feat=torch.cat((feat_weight1, feat_weight2), dim=0),
      aggregation="sum", return_counts=False
  )
  # Divide features by total weight
  feat_weight[:, :-1] = feat_weight[:, :-1] / feat_weight[:, -1:]
  return rays, feat_weight

def intersect_voxels(xyz_vx1, xyz_vx2, vox_size):
  """Compute the set intersection between voxels

  Strong assumption: xyz_vx1 and xyz_vx2 are already voxelized with the same
  resolution passed in vox_size

  Args:
    xyz_vx1: Nx3 float tensor including the xyz centers of each voxel.
    xyz_vx2: Mx3 float tensor including the xyz centers of each voxel.
    vox_size: float representing the voxel grid size that both of these sets
      were voxelized with.
  
  Returns:
    The union xyz_vx between the two sets of voxels as Kx3 float tensor.
    A flag long tensor of size K which can be used to compute subtraction and
    intersection results with the following interpretation of its value:
     1: A - B
     0: A & B
    -1: B - A
  """
  union_vx, flag = pointcloud_to_sparse_voxels(
    torch.cat([xyz_vx1, xyz_vx2], dim = 0),
    feat_pc=torch.cat([torch.ones_like(xyz_vx1[:, 0:1]),
                       torch.ones_like(xyz_vx2[:, 0:1])*-1],
                       dim = 0),
    vox_size=vox_size, aggregation="sum"
  )
  flag = torch.round(flag).int().squeeze(-1)

  return union_vx, flag

def get_voxels_infront_planes(
    vx_xyz: torch.FloatTensor, planes: torch.FloatTensor) -> torch.FloatTensor:
  """Returns the filtered voxels that are in front of all given planes
  
  Args:
    vx_xyz: Nx3 float tensor representing xyz centers of the voxels.
    planes: Px4 Float tensor where P represents the number of planes and 4 
      represents the plane coefficients abcd in ax+by+cz+d=0 respectively.
  
  Returns:
    Mx3 float tensor representing the voxels from vx_xyz that lie in front of
    all planes.
  """
  N = vx_xyz.shape[0]
  P = planes.shape[0]
  vx_xyz = vx_xyz.reshape(N, 1, 3)
  planes = planes.reshape(1, P, 4)
  v = torch.linalg.vecdot(vx_xyz, planes[..., :3]) + planes[..., -1]
  mask = torch.all(v > 0, dim=-1)
  return vx_xyz[mask]

def get_voxels_infront_planes_mask(
    vx_xyz: torch.FloatTensor, planes: torch.FloatTensor) -> torch.FloatTensor:
  """Returns the filtered voxels that are in front of all given planes
  Args:
    vx_xyz: Nx3 float tensor representing xyz centers of the voxels.
    planes: BxPx4 Float tensor where P represents the number of planes and 4 
      represents the plane coefficients abcd in ax+by+cz+d=0 respectively.
  
  Returns:
    BxN boolean tensor
  """
  N = vx_xyz.shape[0]
  B, P, _ = planes.shape
  vx_xyz = vx_xyz.reshape(1, N, 1, 3)
  planes = planes.reshape(B, 1, P, 4)
  v = torch.linalg.vecdot(vx_xyz, planes[..., :3]) + planes[..., -1]
  mask = torch.all(v > 0, dim=-1)
  return mask
