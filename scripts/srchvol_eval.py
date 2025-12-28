"""Perform online search volume + semantic mapping evaluation.

Typical Usage:
  python scripts/srchvol_eval.py \
    --config-dir experiments/srchvol_configs/ --config-name rayfronts_0 --multirun
"""
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing_extensions import override
from typing import Tuple, Union
import logging
from dataclasses import dataclass
import os
import sys
from collections import OrderedDict
import shutil

import torch
import torch_scatter
import hydra
from hydra.core.config_store import ConfigStore

from semseg_eval import SemSegEval, SemSegEvalConfig
import eval_utils

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from rayfronts import mapping, utils, geometry3d as g3d, datasets
import rayfronts_cpp

logger = logging.getLogger(__name__)

@dataclass
class SrchVolEvalConfig(SemSegEvalConfig):
  # The default search volume prediction if no evidence is available.
  # True means the default assumption is that a voxel can have the class unless
  # further evidence is provided to the contrary.
  default_prediction: bool = True

  # At what threshold should the search volume be defined. Each semantic ray
  # will cast a +1 in the voxels in front of its cone. Then the rasterized
  # volume will be normalized by the max such that values are in [0-1] then
  # it is thresholded to determine the search volume.
  srchvol_thresh: float = 0.05

  # See SemSegEvalConfig for the rest of the options

cs = ConfigStore.instance()
cs.store(name="extras", node=SrchVolEvalConfig)

class SrchVolEval(SemSegEval):
  def __init__(self, cfg):
    super().__init__(cfg)
    if cfg.load_external_pred:
      raise ValueError(
        "Search volume evaluation does not support external predictions")
    elif cfg.online_eval_period <= 0:
      raise ValueError("Search volume evaluation only runs in online eval mode")
    elif "SemanticRayFrontiersMap" not in cfg.mapping._target_:
      raise ValueError("Search volume evaluation only works with "
                       "SemanticRayFrontiersMap.")

    self.cache_validity_props["occ_gt"] = {
      "mapping.vox_size", "mapping.occ_thickness", "dataset", "seed"
    }
    self.cache_file_paths["occ_gt"] = "occupancy_gt.pt"

    # Used to keep track of search volumes we want to remove from vis.
    self.prev_ray_preds = set()

  def compute_occupancy_gt(self, bbox):
    logger.info("Computing ground truth occupancy map to define scene volume.")

    gt_occ_mapper = mapping.OccupancyVoxelMap(
      self.dataset.intrinsics_3x3, None, self.vis,
      max_pts_per_frame=-1,
      vox_size=self.cfg.mapping.vox_size,
      vox_accum_period=1,
      max_empty_pts_per_frame=-1,
      occ_observ_weight=100,
      max_occ_cnt=100,
      occ_thickness=self.cfg.mapping.occ_thickness,
      clip_bbox=bbox,
    )
    for i, batch in enumerate(self.dataloader):
      rgb_img = batch["rgb_img"].cuda()
      depth_img = batch["depth_img"].cuda()
      pose_4x4 = batch["pose_4x4"].cuda()
      if self.vis is not None:
        if i % self.cfg.vis.pose_period == 0:
          self.vis.log_pose(batch["pose_4x4"][-1])
        if i % self.cfg.vis.input_period == 0:
          self.vis.log_depth_img(batch["depth_img"][-1].squeeze())

      gt_occ_mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4)
      if self.vis is not None:
        if i % self.cfg.vis.map_period == 0:
          gt_occ_mapper.vis_map()

        self.vis.step()
    gt_occ_mapper.accum_occ_voxels()
    return gt_occ_mapper.global_vox_xyz, gt_occ_mapper.global_vox_occ

  def compute_srchvol_metrics(self, gt_xyz, gt,
                              obsrv_xyz, obsrv_occ_xyz, obsrv_occ_preds,
                              unobsrv_xyz, unobsrv_pred_masks, unobsrv_preds):
    """Compute search volume metrics for classes present in GT

    We divide space into observed/unobserved occupied/empty
    Each partition can have its own tp,fp,fn,tn.
    Observed partition includes both occupied and empty in the observed region
    Observed occupied partition includes only occupied in the observed region

    Args:
      gt_xyz: Nx3 float tensor describing xyz locations of ground truth.
        must be sorted.
      gt: N long tensor that has ground truth class indices
      obsrv_xyz: Mx3 float tensor describing xyz locations of observed volume
      obsrv_occ_xyz: Cx3 float tensor describing xyz locations of observed
        occupied vol.
      obsrv_occ_preds: C long tensor describing prediction class indices of the
        occupied observed volume.
      unobserv_xyz: Ux3 float tensor describing xyz locations of unobserved
        volume.
      unobserv_pred_masks: PxU bool tensor describing predictions about the 
        unobserved volume.
      unobserv_preds: P long tensor describing the class indices that correspond
        to each mask in unobserv_pred_masks
        We opt to split unobserved predictions to unobserv_pred_masks and 
        unobserv_preds since unobserved volume can have multi labels.
    """
    d = gt_xyz.device
    # Strong assumption gt_xyz is sorted ! TODO: Implement fail-safe
    union_xyz, flag = g3d.intersect_voxels(gt_xyz, obsrv_xyz,
                                           self.cfg.mapping.vox_size)
    obsrv_gt_xyz = union_xyz[flag==0]
    obsrv_gt = gt[flag[flag >= 0] == 0]
    unobsrv_gt_xyz = union_xyz[flag==1]
    unobsrv_gt = gt[flag[flag >= 0] == 1]

    assert obsrv_gt.shape[0] + unobsrv_gt.shape[0] == gt.shape[0]
    assert obsrv_gt_xyz.shape[0] + unobsrv_gt_xyz.shape[0] == gt_xyz.shape[0]

    obsrv_gt_non_ignore = torch.sum(obsrv_gt != 0)
    obsrv_occ_pred_non_ignore = torch.sum(obsrv_occ_preds != 0)

    if self.cfg.k > 0:
      obsrv_gt, obsrv_occ_preds = eval_utils.align_labels_with_knn(
        obsrv_gt_xyz, obsrv_gt, obsrv_occ_xyz, obsrv_occ_preds, k=self.cfg.k)
    else:
      obsrv_gt, obsrv_occ_preds = eval_utils.align_labels_with_vox_grid(
        obsrv_gt_xyz, obsrv_gt, obsrv_occ_xyz, obsrv_occ_preds,
        self.cfg.mapping.vox_size)

    assert obsrv_gt_non_ignore == torch.sum(obsrv_gt != 0)
    assert obsrv_occ_pred_non_ignore == torch.sum(obsrv_occ_preds != 0)

    m = dict()
    m["obsrv_occ_tp"], m["obsrv_occ_fp"], m["obsrv_occ_fn"], m["obsrv_occ_tn"] = \
      eval_utils.eval_gt_pred(
        obsrv_occ_preds, obsrv_gt, num_classes=self.num_classes)

    assert torch.all(
      m["obsrv_occ_tp"] + m["obsrv_occ_fn"] +
      m["obsrv_occ_fp"] + m["obsrv_occ_tn"] == obsrv_gt_non_ignore)

    # Eval for the unobserved volume is a bit trickier because we allow multilabel
    # Also the memory constrains us to evaluate each class separately.
    for k in ["unobsrv_tp", "unobsrv_fp", "unobsrv_fn", "unobsrv_tn",
              "unobsrv_occ_tp", "unobsrv_occ_fp", "unobsrv_occ_fn",
              "unobsrv_occ_tn"]:
      m[k] = torch.zeros_like(m["obsrv_occ_tp"])

    # First we iterate over predictions

    # This number is the union of predictions and GT which may not align exactly
    # so it may be bigger than unobsrv_xyz.shape[0]
    unobsrv_nvox = None

    for i, pred_class_id in enumerate(unobsrv_preds):
      pred_ids = torch.zeros_like(unobsrv_xyz[:, 0], dtype=torch.long)
      pred_ids[unobsrv_pred_masks[i]] = 2

      # We set ignore to -1, other classes to 1, and the class in question to 2.
      # we leave id=0 for voxels not present in the GT (Empty voxels) which
      # we will get after aligning the gt with the unobserved prediction.
      unobsrv_gt_single = torch.ones_like(unobsrv_gt)*-1
      unobsrv_gt_single[unobsrv_gt != 0] = 1
      unobsrv_gt_single[unobsrv_gt == pred_class_id] = 2

      if self.cfg.k > 0:
        aligned_preds, unobsrv_gt_single = eval_utils.align_labels_with_knn(
          unobsrv_xyz, pred_ids, unobsrv_gt_xyz, unobsrv_gt_single,
          k=self.cfg.k)
      else:
        aligned_preds, unobsrv_gt_single = eval_utils.align_labels_with_vox_grid(
          unobsrv_xyz, pred_ids, unobsrv_gt_xyz, unobsrv_gt_single,
          self.cfg.mapping.vox_size)

      assert unobsrv_nvox is None or unobsrv_nvox == aligned_preds.shape[0]
      unobsrv_nvox = aligned_preds.shape[0]

      # Now aligned_preds has 0 for no predictions and 2 for our class.
      # And unobsrv_gt_single has 0 for no GT/empty, -1 for explicitly ignore,
      # 1 for other classes, and 2 for our class.

      # For search volume in the unobserved region, we do not want to ignore
      # empty voxels and we want them to factor in as true negatives or
      # false positives. However we still ignore the voxels that have GT as -1;
      # unlike empty voxels that are guaranteed not to have the class in question,
      # voxels that have GT set to -1 are "unknown" so we cannot say if they are
      # or if they are not our class.
      # If we add 1 to the IDs then -1 will become the ignore index
      tp,fp,fn,tn = eval_utils.eval_gt_pred(
        aligned_preds+1, unobsrv_gt_single+1, 4)

      m["unobsrv_tp"][pred_class_id-1] += tp[-1]
      m["unobsrv_fp"][pred_class_id-1] += fp[-1]
      m["unobsrv_fn"][pred_class_id-1] += fn[-1]
      m["unobsrv_tn"][pred_class_id-1] += tn[-1]

      # For occupied unobserved voxels. We want to ignore empty voxels too so we
      # clamp -1 (empty) to 0 (ignore).
      unobsrv_gt_single = torch.clamp(unobsrv_gt_single, min=0)
      tp,fp,fn,tn = eval_utils.eval_gt_pred(aligned_preds, unobsrv_gt_single, 3)

      m["unobsrv_occ_tp"][pred_class_id-1] += tp[-1]
      m["unobsrv_occ_fp"][pred_class_id-1] += fp[-1]
      m["unobsrv_occ_fn"][pred_class_id-1] += fn[-1]
      m["unobsrv_occ_tn"][pred_class_id-1] += tn[-1]

    # Second we iterate over classes present in gt that had no predictions
    # These can only affect true negatives and false negatives.
    unobsrv_total = unobsrv_nvox - torch.sum(unobsrv_gt == 0)
    unobsrv_occ_total = torch.sum(unobsrv_gt != 0)
    gt_id_set = {x.item() for x in gt.unique()[1:]}
    pred_id_set = {x.item() for x in unobsrv_preds}
    gt_ids_not_in_pred = gt_id_set.difference(pred_id_set)
    for cls_id in gt_ids_not_in_pred:
      fn = torch.sum(unobsrv_gt == cls_id)
      m["unobsrv_fn"][cls_id-1] += fn
      m["unobsrv_tn"][cls_id-1] += unobsrv_total - fn

      m["unobsrv_occ_fn"][cls_id-1] += fn
      m["unobsrv_occ_tn"][cls_id-1] += unobsrv_occ_total - fn

    # Third we iterate over classes missing from both gt and predictions. These
    # are classes that have been used for prompting and may exist in the scene
    # just beyond the depth at which GT was generated. They should be excluded
    # from final metrics calculation.
    all_scene_ids_set = {x.item() for x in torch.arange(1, self.num_classes)}
    ids_not_in_gt_or_pred = \
      all_scene_ids_set.difference(gt_id_set).difference(pred_id_set)
    for cls_id in ids_not_in_gt_or_pred:
      m["unobsrv_tn"][cls_id-1] += unobsrv_total
      m["unobsrv_occ_tn"][cls_id-1] += unobsrv_occ_total

    assert torch.all(m["unobsrv_tp"] + m["unobsrv_fp"] + m["unobsrv_fn"] +
                    m["unobsrv_tn"] == unobsrv_total)

    assert torch.all(m["unobsrv_occ_tp"] + m["unobsrv_occ_fp"] +
                    m["unobsrv_occ_fn"] + m["unobsrv_occ_tn"] ==
                    unobsrv_occ_total)

    def calc_iou(prefix):
      p = prefix
      m[f"{p}_iou"] = m[f"{p}_tp"] / (m[f"{p}_tp"] + m[f"{p}_fn"] + m[f"{p}_fp"])
      freq = m[f"{p}_tp"] + m[f"{p}_fn"]
      m[f"{p}_iou"][freq==0] = torch.nan # Ignore classes not present in scene.
      m[f"{p}_fiou"] = (freq / torch.sum(freq)) * m[f"{p}_iou"]

    def f_beta(p, r, b):
      return (1+b**2) * (p*r) / (b**2 * p + r)
    
    def calc_fscore(prefix, betas=(0.5, 1, 2)):
      m[f"{prefix}_recall"] = m[f"{prefix}_tp"] / (m[f"{prefix}_tp"] + m[f"{prefix}_fn"])
      m[f"{prefix}_precision"] = m[f"{prefix}_tp"] / (m[f"{prefix}_tp"] + m[f"{prefix}_fp"])
      for b in betas:
        m[f"{prefix}_f{b}"] = f_beta(m[f"{prefix}_precision"], m[f"{prefix}_recall"], b)

    calc_iou("obsrv_occ")
    calc_iou("unobsrv")

    ## Calculate IoU for the whole scene
    # (Not including empty voxels in the unobserved region)
    m["scene_occ_tp"] = m["unobsrv_occ_tp"] + m["obsrv_occ_tp"]
    m["scene_occ_fp"] = m["unobsrv_occ_fp"] + m["obsrv_occ_fp"]
    m["scene_occ_fn"] = m["unobsrv_occ_fn"] + m["obsrv_occ_fn"]

    calc_iou("scene_occ")

    ## Calculate IoU for the whole scene
    # (Including empty voxels in the unobserved region)
    m["scene_tp"] = m["unobsrv_tp"] + m["obsrv_occ_tp"]
    m["scene_fp"] = m["unobsrv_fp"] + m["obsrv_occ_fp"]
    m["scene_fn"] = m["unobsrv_fn"] + m["obsrv_occ_fn"]
    calc_iou("scene")

    ## Calculate general volumes
    P = m["scene_fn"].shape[0]
    m["obsrv_vol"] = torch.tensor(obsrv_xyz.shape[0]).to(d).tile(P)
    m["unobsrv_vol"] = torch.tensor(unobsrv_total).to(d).tile(P)
    m["obsrv_occ_vol"] = torch.tensor(obsrv_occ_xyz.shape[0]).to(d).tile(P)
    m["scene_vol"] = m["obsrv_vol"] + m["unobsrv_vol"]

    calc_fscore("scene")
    calc_fscore("obsrv_occ")
    calc_fscore("unobsrv")
    calc_fscore("unobsrv_occ")
    calc_fscore("scene_occ")

    ## Calculate Search volume metrics
    m["unobsrv_srchvol"] = (m["unobsrv_tp"] + m["unobsrv_fp"]) / m["unobsrv_vol"]
    m["unobsrv_srchcutvol"] = 1 - (m["unobsrv_fp"] / m["unobsrv_vol"])
    m["unobsrv_srchcutvol_recall"] = m["unobsrv_recall"] * m["unobsrv_srchcutvol"]

    # This metric shows how much of the object has been observed using GT
    # Can be used to terminate a search.
    m["scene_gt_occ_vs_unobs_iou"] = (m["obsrv_occ_tp"] + m["obsrv_occ_fn"]) / \
      (m["scene_occ_tp"]+ m["scene_occ_fn"])

    # Calculate aggregate metrics
    m["mSCVR"] = torch.nanmean(m["unobsrv_srchcutvol_recall"])
    m["mSCV"] = torch.nanmean(m["unobsrv_srchcutvol"])
    m["mRecall"] = torch.nanmean(m["unobsrv_recall"])
    m["mIoU"] = torch.nanmean(m["obsrv_occ_iou"])
    m["fmIoU"] = torch.nansum(m["obsrv_occ_fiou"])

    return m

  def srchvol_eval(self, mapper, semseg_gt_xyz, semseg_gt_labels, occ_gt_xyz,
                   feat_queries, vis=True):
    observed_xyz = rayfronts_cpp.occ_vdb2sizedpc(mapper.occ_map_vdb)[:, :3]
    observed_xyz = observed_xyz.to(self.device)
    union_xyz, flag = g3d.intersect_voxels(occ_gt_xyz, observed_xyz,
                                           mapper.vox_size)
    unobserved_xyz = union_xyz[flag==1]
    # observed_xyz = union_xyz[flag==0]
    # assert observed_xyz.shape[0] + unobserved_xyz.shape[0] == occ_gt_xyz.shape[0]

    # Initialize unobserved volume predictions based on the default
    srchvol_mask_cls = semseg_gt_labels.unique()[1:] # skip ignore label
    f = torch.ones if self.cfg.default_prediction else torch.zeros
    srchvol_mask = f(srchvol_mask_cls.shape[0],
                     unobserved_xyz.shape[0],
                     dtype=torch.bool, device=self.device)

    is_default_pred = torch.ones(self.num_classes-1,
                                 dtype=torch.bool, device=self.device)

    ## We need to compute the predictions defined by all the rays
    if (mapper.global_rays_orig_angles is not None and
        mapper.global_rays_orig_angles.shape[0] > 0):
      feats = mapper.global_rays_feat
      if (self.feat_compressor is not None and 
          not self.cfg.querying.compressed):
        feats = self.feat_compressor.decompress(feats)
      feats = self.encoder.align_spatial_features_with_language(
        feats.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
      
      ray_preds = eval_utils.compute_semseg_preds(
        feats, feat_queries, self.cfg.prompt_denoising_thresh,
        self.cfg.prediction_thresh, self.cfg.chunk_size)

      if mapper.angle_bin_size >= 360 and not mapper.infer_direction:
        frontiers = mapper.global_rays_orig_angles[:, :3]
        ray_srchvol_masks, ray_srchvol_cls = eval_utils.frontiers_to_searchvol(
          frontiers, ray_preds, unobserved_xyz,
          self.cfg.srchvol_thresh, self.cfg.chunk_size,
          mapper.vox_size*mapper.fronti_subsampling
        )
      else:
        fovx, fovy = g3d.intrinsics_3x3_to_fov(self.dataset.intrinsics_3x3,
                                              (self.dataset.depth_h,
                                                self.dataset.depth_w))
        cone_angle = torch.rad2deg(torch.max(fovx, fovy) / 2)
        cone_start_radius = mapper.vox_size*mapper.fronti_subsampling
        ray_srchvol_masks, ray_srchvol_cls = eval_utils.rays_to_searchvol(
          mapper.global_rays_orig_angles, ray_preds, unobserved_xyz,
          mapper.vox_size, cone_angle, cone_start_radius,
          self.cfg.srchvol_thresh, self.cfg.chunk_size)

      # Update initial prediction about unobserved with new evidence
      masks_to_append = [srchvol_mask]
      cls_to_append = [srchvol_mask_cls]
      for j,cls in enumerate(ray_srchvol_cls):
        if cls == 0: # Ignore label
          continue
        a = torch.argwhere(srchvol_mask_cls == cls).squeeze(-1)
        if a.shape[0] == 0: # If prediction is not in GT
          masks_to_append.append(ray_srchvol_masks[j].unsqueeze(0))
          cls_to_append.append(cls.unsqueeze(0))
        else:
          assert srchvol_mask_cls[a[0]] == cls
          srchvol_mask[a[0]] = ray_srchvol_masks[j]

        is_default_pred[cls-1] = False

      srchvol_mask = torch.cat(masks_to_append, dim=0)
      srchvol_mask_cls = torch.cat(cls_to_append, dim=0)

    if mapper.global_vox_xyz is not None and mapper.global_vox_xyz.shape[0] > 0:
      feats = mapper.global_vox_feat
      if (self.feat_compressor is not None and 
          not self.cfg.querying.compressed):
        feats = self.feat_compressor.decompress(feats)
      feats = self.encoder.align_spatial_features_with_language(
        feats.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

      vox_preds = eval_utils.compute_semseg_preds(
        feats, feat_queries, self.cfg.prompt_denoising_thresh,
        self.cfg.prediction_thresh, self.cfg.chunk_size)
      vox_xyz = mapper.global_vox_xyz
    else:
      vox_xyz = torch.zeros_like(semseg_gt_xyz[1:1])
      vox_preds = torch.zeros_like(semseg_gt_labels[1:1])

    m = self.compute_srchvol_metrics(
      semseg_gt_xyz, semseg_gt_labels,
      observed_xyz,
      vox_xyz, vox_preds,
      unobserved_xyz, srchvol_mask, srchvol_mask_cls)
    m["is_def_pred"] = is_default_pred

    if (vis and self.vis is not None and
        mapper.global_rays_orig_angles is not None and
        mapper.global_rays_orig_angles.shape[0] > 0):

      self.vis.log_label_pc(mapper.global_vox_xyz, vox_preds,
                            layer="predictions/voxels")
      ray_orig = mapper.global_rays_orig_angles[:, :3]
      if mapper.angle_bin_size >= 360 and not mapper.infer_direction:
        self.vis.log_label_pc(
          ray_orig, ray_preds, layer="predictions/frontiers")
      else:
        angles = torch.deg2rad(mapper.global_rays_orig_angles[:, 3:])
        ray_dir = torch.stack(
          g3d.spherical_to_cartesian(1, angles[:, 0], angles[:, 1]), dim=-1)
        self.vis.log_label_arrows(
          ray_orig, ray_dir, ray_preds, layer="predictions/rays")

      for i in range(srchvol_mask.shape[0]):
        cat_name = self.dataset._cat_index_to_name[
          srchvol_mask_cls[i].item()].replace(" ", "_")
        self.vis.log_label_pc(
          unobserved_xyz[srchvol_mask[i]],
          pc_labels = srchvol_mask_cls[i],
          layer=f"srchvol/{cat_name}")

      s = {x.item() for x in srchvol_mask_cls}
      to_remove = self.prev_ray_preds.difference(s)
      for cls_id in to_remove:
        cat_name = self.dataset._cat_index_to_name[cls_id].replace(" ", "_")
        self.vis.log_label_pc(
          torch.empty([0,3]).float(),
          pc_labels = torch.empty([0,1]).long(),
          layer=f"srchvol/{cat_name}")
      self.prev_ray_preds = s

    return m

  @override
  def run(self):
    ## 1. Get Semantic Segmentation Ground Truth
    eval_utils.reset_seed(self.cfg.seed)
    if self.cfg.load_external_gt:
      semseg_gt_xyz, semseg_gt_label = self.load_externel_semseg_gt()
    else:
      if self.cache_is_valid("semseg_gt"):
        logger.info("Loading cached semseg ground truth 3D voxels...")
        d = torch.load(self.cache_file_paths["semseg_gt"], weights_only=True)
        semseg_gt_xyz = d["semseg_gt_xyz"].to(self.device)
        semseg_gt_label = d["semseg_gt_label"].to(self.device)
      else:
        semseg_gt_xyz, semseg_gt_label = self.compute_semseg_gt()
        logger.info("Ground truth voxel labels generated.")
        gt_file_name = self.cache_file_paths["semseg_gt"]
        if self.store_output and not self.cfg.load_external_gt:
          os.makedirs(os.path.dirname(gt_file_name), exist_ok=True)
          torch.save(dict(semseg_gt_xyz=semseg_gt_xyz,
                          semseg_gt_label=semseg_gt_label), gt_file_name)

    if self.vis is not None:
      self.vis.log_label_pc(semseg_gt_xyz, semseg_gt_label,
                            layer="ground_truth")

    ## 2. Get the volume of the full scene.
    eval_utils.reset_seed(self.cfg.seed)
    bbox_mn = torch.min(semseg_gt_xyz, dim=0).values
    bbox_mx = torch.max(semseg_gt_xyz, dim=0).values
    bbox = torch.stack((bbox_mn, bbox_mx), dim=0)
    if self.cache_is_valid("occ_gt"):
      logger.info("Loading cached ground truth occupancy voxels...")
      d = torch.load(self.cache_file_paths["occ_gt"], weights_only=True)
      occ_gt_xyz = d["occ_gt_xyz"]
      occ_gt_occ = d["occ_gt_occ"]
    else:
      occ_gt_xyz, occ_gt_occ = self.compute_occupancy_gt(bbox)

    # CLIP gt to what was observed
    union_xyz, flag = g3d.intersect_voxels(
      semseg_gt_xyz, occ_gt_xyz, vox_size=self.cfg.mapping.vox_size)
    semseg_gt_xyz = union_xyz[flag==0]
    # Strong assumption semseg_gt_xyz is sorted ! TODO: Implement fail-safe
    semseg_gt_label = semseg_gt_label[flag[flag >= 0] == 0]

    results_dict = OrderedDict()
    encoder_kwargs = dict()
    if "NARadioEncoder" in self.cfg.encoder._target_:
      encoder_kwargs["input_resolution"] = (self.dataset.rgb_h,
                                            self.dataset.rgb_w)
    if "classes" in self.cfg.encoder:
      encoder_kwargs["classes"] = self.dataset.cat_index_to_name[1:]
    self.encoder = hydra.utils.instantiate(self.cfg.encoder, **encoder_kwargs)

    self.feat_compressor = None
    if self.cfg.mapping.feat_compressor is not None:
      self.feat_compressor = hydra.utils.instantiate(
        self.cfg.mapping.feat_compressor)

    ## 3. Compute text embeddings.
    if self.cache_is_valid("text_embeds"):
      logger.info("Loading cached text embeddings...")
      d = torch.load(self.cache_file_paths["text_embeds"], weights_only=True)
      text_embeds = d["text_embeds"].to(self.device)
    else:
      text_embeds = self.compute_text_embeds()
      if self.store_output:
        torch.save(dict(text_embeds=text_embeds),
                   self.cache_file_paths["text_embeds"])

    ## 4. Compute Map features.
    eval_utils.reset_seed(self.cfg.seed)
    results_dict = dict()
    i = 0

    mapper = hydra.utils.instantiate(
      self.cfg.mapping, encoder=self.encoder,
      intrinsics_3x3=self.dataset.intrinsics_3x3, visualizer=self.vis,
      occ_pruning_period=-1, clip_bbox=bbox,
      feat_compressor=self.feat_compressor)
    for _, _ in self.mapping_loop(mapper, text_embeds):
      if (i != 0 and i % self.cfg.online_eval_period == 0):
        results_dict[i] = self.srchvol_eval(mapper,
          semseg_gt_xyz, semseg_gt_label, occ_gt_xyz, text_embeds)
        self.log_metrics(i, results_dict[i], prefix="srchvol")

      i += 1

    # 5. Predict and evaluate semantic segmentation
    results_dict[i] = self.srchvol_eval(mapper,
      semseg_gt_xyz, semseg_gt_label, occ_gt_xyz, text_embeds)
    self.log_metrics(i, results_dict[i], prefix="srchvol")

    # 6. Save the results
    self.save_metrics(results_dict, prefix="srchvol")

    OmegaConf.save(self.cfg, self.cache_cfg_fn)

@hydra.main(version_base="1.2",
            config_path="../rayfronts/configs",
            config_name="default")
@torch.inference_mode()
def main(cfg=None):
  srchvol_eval = SrchVolEval(cfg)
  srchvol_eval.run()

if __name__ == "__main__":
  main()


