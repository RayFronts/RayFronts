import os
import sys
from typing import List
import copy
import random

from sklearn.neighbors import BallTree
import torch
import torch_scatter
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from rayfronts import mapping, utils, geometry3d as g3d, datasets

def reset_seed(seed: int):
  """Reset the random seed for reproducibility."""
  if seed < 0:
    return
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def compute_semseg_preds(map_feats: torch.FloatTensor,
                         text_embeds: torch.FloatTensor,
                         prompt_denoising_thresh: float = 0.5,
                         prediction_thresh: float = 0.1,
                         chunk_size: int = 100000):
  """Compute semantic segmentation predictions from features and text embeds.

  Args:
    map_feats: NxC Float Tensor
    text_embeds: MxC Float Tensor
    prompt_denoising_thresh: Prompt denoising introduced in MaskClip+, removes
      the prompt with target class if its class confidence at all spatial
      locations is less than a threshold.
    prediction_thresh: Threshold after which a prediction is made. Otherwise
      will be set to ignore.
    chunk_size: How many points to process at once. This is used to avoid
      running out of memory.

  Returns:
    Nx1 Long Tensor of predictions. The predictions are the indices of the
    classes in the text_embeds + 1 to reserve 0 for the ignore class. 
    The predictions are in the same order as map_feats.
  """

  num_chunks = int(np.ceil(map_feats.shape[0] / chunk_size))
  preds = list()
  for c in range(num_chunks):
    sim_vx = utils.compute_cos_sim(
      text_embeds, map_feats[c*chunk_size: (c+1)*chunk_size], softmax=True)

    # Prompt denoising
    max_sim = torch.max(sim_vx, dim=0).values
    low_conf_classes = torch.argwhere(
      max_sim < prompt_denoising_thresh)
    sim_vx[:, low_conf_classes] = -torch.inf

    sim_value, pred = torch.max(sim_vx, dim=-1)

    # 0 is the ignore id / no pred
    pred += 1
    pred[sim_value < prediction_thresh] = 0

    preds.append(pred)

  preds = torch.cat(preds, dim=0)
  return preds

def eval_gt_pred(pred_ids, gt_ids, num_classes):
  """Compute TP, FP, and FN between predictions and GT.

  Args:
    pred_mask: N integer like tensor representing voxel-wise class indices.
    gt_mask: N integer like tensor representing voxel-wise GT class indices.
    num_classes: Total number of classes including the ignore class.

  Returns:
    A (tp, fp, fn, tn) tuple where `tp` is a Long tensor of length `num_classes`
    representing true positives. `fp` and `fn` correspond to false positives and
    false negatives resepectively and have the same format as `tp`.
  """

  pred_bin_mask = torch.nn.functional.one_hot(
    pred_ids, num_classes=num_classes)
  pred_bin_mask = pred_bin_mask.bool()

  gt_bin_mask = torch.nn.functional.one_hot(
    gt_ids, num_classes=num_classes)
  gt_bin_mask = gt_bin_mask.bool()

  valid_mask = (gt_ids != 0).unsqueeze(-1)
  tp = torch.sum(gt_bin_mask & pred_bin_mask & valid_mask, dim=0)
  fp = torch.sum(~gt_bin_mask & pred_bin_mask & valid_mask, dim=0)
  fn = torch.sum(gt_bin_mask & ~pred_bin_mask & valid_mask, dim=0)
  tn = torch.sum(~gt_bin_mask & ~pred_bin_mask & valid_mask, dim=0)

  return tp[1:], fp[1:], fn[1:], tn[1:]

def align_labels_with_vox_grid(xyz1, labels1, xyz2, labels2, vox_size):
  """Aligns labels1 and labels2 to the same voxel grid and the same order.

  To compare the labels of two sparse voxel maps, we need to align them such
  that labels1[0] and labels2[0] are in the same voxel. 

  Args: 
    xyz1: Nx3 float tensor
    labels1: N long tensor with class ids
    xyz2: Mx3 float tensor
    labels2: M long tensor with class ids
    vox_size: Voxel size of the grid used to align the labels.
  Returns:
    aligned_labels1: L long tensor
    aligned_labels2: L long tensor
    where L <= N+M
  """
  labels1_labels2 = torch.zeros(labels1.shape[0] + labels2.shape[0], 2,
                                device=labels1.device, dtype=labels1.dtype)
  labels1_labels2[:labels1.shape[0], 0] = labels1
  labels1_labels2[labels1.shape[0]:, 1] = labels2

  union_xyz, labels1_labels2 = g3d.pointcloud_to_sparse_voxels(
    torch.cat([xyz1, xyz2], dim = 0), feat_pc=labels1_labels2,
    vox_size=vox_size, aggregation="sum")
  aligned_labels1 = labels1_labels2[:, 0]
  aligned_labels2 = labels1_labels2[:, 1]

  return aligned_labels1, aligned_labels2


def align_labels_with_knn(xyz1, labels1, xyz2, labels2, k=1):
  """Align labels1 and labels2 using K-Nearest Neighbor. 
  
  For every point in xyz1 labels1, we generate the corresponding value from
  xyz2 labels2 through KNN and return aligned_labels1, aligned_labels2
  Both aligned_labels1 and aligned_labels2 are aligned to xyz1.

  Typically xyz1,labels1 would be the ground truth and xyz2,labels2 would be the
  predictions.

  Args:
    xyz1: Nx3 float tensor
    labels1: N long tensor with class ids
    xyz2: Mx3 float tensor
    labels2: M long tensor with class ids
    k: int number of neighbors to consider
  
  Returns:
    aligned_labels1: N long tensor
    aligned_labels2: N long tensor
  """
  # TODO: Replace with FAISS for GPU computation
  ball_tree = BallTree(xyz2.cpu())

  matched_indices = torch.from_numpy(ball_tree.query(xyz1.cpu(), k=k)[1]).cuda()
  aligned_labels2_k = labels2[matched_indices]
  aligned_labels2 = torch.mode(aligned_labels2_k, dim=-1).values

  return labels1, aligned_labels2


def rays_to_searchvol(rays_orig_angles: torch.FloatTensor,
                      rays_preds: torch.LongTensor,
                      vol_xyz: torch.FloatTensor,
                      vox_size: int,
                      cone_angle: float = 30,
                      start_radius: float = None,
                      searchvol_thresh: float = 0.05,
                      chunk_size: int = 10000):
  """Rasterizes rays to class id volumes bounded by vol_xyz.
  
  Args:
    rays_orig_angles: (Nx5) float tensor where every row is (x,y,z,theta,phi)
    rays_preds: N Long tensor represnting the class of each ray.
    vol_xyz: (Mx3) float tensor representing the voxel centers of the volume
      bounding the search.
    vox_size: Voxel side length.
    cone_angle: Apex angle in degrees of the cone shooting from each ray.
    start_radius: Starting radius of the cones.
    searchvol_thresh: At what threshold should the search volume be defined.
      Each semantic ray will cast a +1 in the voxels in front of its cone. Then
      the rasterized volume will be normalized by the max such that values are
      in [0-1] then it is thresholded to determine the search volume.
    chunk_size: How many voxels to process at once. Reduce if getting OOM.

  Returns:
    MxB boolean mask describing the locations in vol_xyz that apply to this
      search volume.
    B Long tensor describing the corresponding labels to the mask.
  """

  d = rays_orig_angles.device
  if start_radius is None:
    start_radius = vox_size

  # We compute the cones planes defined by each ray and use that to filter
  # the unobserved volume. That will be our "Search volume"
  ray_poses = g3d.rays_to_pose_4x4(rays_orig_angles)

  cone_angle = torch.deg2rad(cone_angle)
  cone_planes = g3d.get_cone_planes(ray_poses, far=vox_size*5, near=vox_size,
                                    apex_angle=cone_angle,
                                    start_radius=start_radius,
                                    num_segs=6)

  plane_mask = torch.ones(cone_planes.shape[1], dtype=torch.bool, device=d)
  plane_mask[1] = False # Ignore far plane
  cone_planes = cone_planes[:, plane_mask, :]

  num_chunks = int(np.ceil(vol_xyz.shape[0] / chunk_size))
  cone_unobserved_masks = list()
  for c in range(num_chunks):  
    cu = g3d.get_voxels_infront_planes_mask(
      vol_xyz[c*chunk_size: (c+1)*chunk_size], cone_planes)
    cone_unobserved_masks.append(cu)
  cone_unobserved_masks = torch.cat(cone_unobserved_masks, dim=1)

  # Now we need to combine search volumes from different rays that have
  # have the same class to get the final search volume defined for class X
  combined_cones_preds, reduce_ind = rays_preds.unique(return_inverse=True)

  combined_cones_cnt = list()
  num_chunks = int(np.ceil(cone_unobserved_masks.shape[1] / chunk_size))
  for c in range(num_chunks):
    cu = cone_unobserved_masks[:, c*chunk_size: (c+1)*chunk_size]
    ccc = torch.zeros(combined_cones_preds.shape[0],
                      cu.shape[1],
                      dtype=torch.long, device=d)
    torch_scatter.scatter(cu.long(), reduce_ind, dim=0, out=ccc)
    combined_cones_cnt.append(ccc)

  combined_cones_cnt = torch.cat(combined_cones_cnt, dim=1)

  assert combined_cones_cnt.shape[0] == combined_cones_preds.shape[0]
  assert combined_cones_cnt.shape[1] == vol_xyz.shape[0]

  norm_factor = torch.max(combined_cones_cnt, dim=-1).values.unsqueeze(-1)
  combined_cones_cnt = combined_cones_cnt / norm_factor

  combined_cones_masks = combined_cones_cnt > searchvol_thresh

  return combined_cones_masks, combined_cones_preds



def frontiers_to_searchvol(fronti_orig, fronti_preds, vol_xyz,
                           srchvol_thresh, chunk_size, def_r):
  d = fronti_orig.device

  fronti_dist = torch.norm(fronti_orig.unsqueeze(0) - fronti_orig.unsqueeze(1),
                           dim=-1)
  diag_mask = torch.eye(
    fronti_dist.shape[0], device=d, dtype=torch.bool)
  fronti_dist[diag_mask] = torch.inf
  min_dists = torch.min(fronti_dist, dim=0).values
  r = min_dists

  r[r.isinf()] = def_r

  num_chunks = int(np.ceil(vol_xyz.shape[0] / chunk_size))
  sphere_unobserved_masks = list()
  for c in range(num_chunks):
    cu = torch.norm(
      vol_xyz[c*chunk_size: (c+1)*chunk_size].unsqueeze(0) -
      fronti_orig.unsqueeze(1), dim=-1) < r.unsqueeze(-1)

    sphere_unobserved_masks.append(cu)
  sphere_unobserved_masks = torch.cat(sphere_unobserved_masks, dim=1)

  # Now we need to combine search volumes from different frontiers that have
  # have the same class to get the final search volume defined for class X
  combined_sphere_preds, reduce_ind = fronti_preds.unique(return_inverse=True)

  combined_sphere_cnt = list()
  num_chunks = int(np.ceil(sphere_unobserved_masks.shape[1] / chunk_size))
  for c in range(num_chunks):
    cu = sphere_unobserved_masks[:, c*chunk_size: (c+1)*chunk_size]
    ccc = torch.zeros(combined_sphere_preds.shape[0],
                      cu.shape[1],
                      dtype=torch.long, device=d)
    torch_scatter.scatter(cu.long(), reduce_ind, dim=0, out=ccc)
    combined_sphere_cnt.append(ccc)

  combined_sphere_cnt = torch.cat(combined_sphere_cnt, dim=1)

  assert combined_sphere_cnt.shape[0] == combined_sphere_preds.shape[0]
  assert combined_sphere_cnt.shape[1] == vol_xyz.shape[0]

  norm_factor = torch.max(combined_sphere_cnt, dim=-1).values.unsqueeze(-1)
  combined_sphere_cnt = combined_sphere_cnt / norm_factor

  combined_sphere_masks = combined_sphere_cnt > srchvol_thresh

  return combined_sphere_masks, combined_sphere_preds

