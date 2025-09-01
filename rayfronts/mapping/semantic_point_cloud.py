"""Module defining a minimal semantic point cloud class.

Typical usage example:

  map = SemanticPointCloud(intrinsics_3x3, None, visualizer, encoder)
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
from rayfronts import geometry3d, image_encoders, visualizers, feat_compressors
from rayfronts.utils import compute_cos_sim

class SemanticPointCloud(SemanticRGBDMapping):
  """Minamilist semantic point cloud map with no filtering or aggregation.

  Mostly used for debugging. Largely inefficent since to maintain API
  consistency we requery and revisualize the whole map instead of just the 
  latest update. Even though in a point cloud with no filtering the new
  observations do not affect the previously visualized points.
  
  Given Posed RGBD input, an encoder is used to extract vision based semantics
  and the semantics are unprojected into the world as points. No filtering or
  aggregation is done and therefore can run out of memory quickly. Points can 
  then be queried with images or with text (If encoder produces language aligned
  features).

  Attributes:
    intrinsics_3x3: See base.
    device: See base.
    visualizer: See base.
    encoder: See base.
    feat_compressor: See base.
    interp_mode: See base.

    max_pts_per_frame: See __init__.

    global_xyz_pc: list of Nx3 Float tensors representing point coordinates.
    global_rgb_pc: list of Nx3 Float tensors representing RGB values in [0-1]
      corresponding to global_xyz_pc.
    global_feat_pc: list of NxC Float tensors representing feature values
      corresponding to global_xyz_pc.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: visualizers.Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               encoder: image_encoders.ImageSpatialEncoder = None,
               feat_compressor: feat_compressors.FeatCompressor = None,
               interp_mode: str = "bilinear",
               max_pts_per_frame: int = 1000):
    """
    Args:
      intrinsics_3x3: See base.
      device: See base.
      visualizer: See base.
      clip_bbox: See base.
      encoder: See base.
      feat_compressor: See base.
      interp_mode: See base.

      max_pts_per_frame: How many points to project per frame ? Set to -1 to
        project all valid points. Otherwise, the points will be uniformly 
        sampled.
    """
    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox, encoder,
                     feat_compressor, interp_mode)

    self.max_pts_per_frame = max_pts_per_frame

    # The point clouds are stored as a list of separate point clouds. to reduce
    # computation, we only aggregate into a single tensor when necessary.
    self.global_pc_xyz = list()
    self.global_pc_rgb = list()
    self.global_pc_feat = list()

  def _concat(self):
    self.global_pc_xyz = [torch.cat(self.global_pc_xyz, dim=0)]
    self.global_pc_rgb = [torch.cat(self.global_pc_rgb, dim=0)]
    self.global_pc_feat = [torch.cat(self.global_pc_feat, dim=0)]

  @override
  def save(self, file_path):
    torch.save(dict(global_xyz_pc=self.global_pc_xyz,
                    global_rgb_pc=self.global_pc_rgb,
                    global_feat_pc=self.global_pc_feat),
                    file_path)

  @override
  def load(self, file_path):
    d = torch.load(file_path, weights_only=True)
    self.global_pc_xyz = [pc.to(self.device) for pc in d["global_xyz_pc"]]
    self.global_pc_rgb = [pc.to(self.device) for pc in d["global_rgb_pc"]]
    self.global_pc_feat = [pc.to(self.device) for pc in d["global_feat_pc"]]

  @override
  def is_empty(self) -> bool:
    for x in self.global_pc_xyz:
      if x.shape[0] > 0:
        return False
    return True

  @override
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None,
                         feat_img: torch.FloatTensor = None) -> None:
    update_info = dict()

    pts_xyz, selected_indices = geometry3d.depth_to_pointcloud(
      depth_img, pose_4x4, self.intrinsics_3x3, conf_map,
      max_num_pts=self.max_pts_per_frame
    )

    pts_xyz, selected_indices = self._clip_pc(pts_xyz,
                                              selected_indices.unsqueeze(-1))
    selected_indices = selected_indices.squeeze(-1)

    B, _, rH, rW = rgb_img.shape
    B, _, dH, dW = depth_img.shape

    if rH != dH or rW != dW:
      rgb_pts = torch.nn.functional.interpolate(
        rgb_img,
        size=(dH, dW),
        mode=self.interp_mode,
        antialias=self.interp_mode in ["bilinear", "bicubic"])
    else:
      rgb_pts = rgb_img

    rgb_pts = rgb_pts.permute(0, 2, 3, 1).reshape(-1, 3)[selected_indices]

    self.global_pc_xyz.append(pts_xyz)
    self.global_pc_rgb.append(rgb_pts)

    # Encode and get features if needed
    if self.encoder is not None:
      if feat_img is None:
        feat_img = self.encoder.encode_image_to_feat_map(rgb_img)
      feat_img = self._proj_resize_feat_map(feat_img, dH, dW)
      update_info["feat_img"] = feat_img

      feat_pts = feat_img.permute(0, 2, 3, 1).reshape(-1, feat_img.shape[1])
      feat_pts = feat_pts[selected_indices]
      self.global_pc_feat.append(feat_pts)

    return update_info

  @override
  def vis_map(self) -> None:
    if self.visualizer is None:
      return
    self._concat()
    self.visualizer.log_pc(self.global_pc_xyz[0], self.global_pc_rgb[0])
    self.visualizer.log_feature_pc(self.global_pc_xyz[0],
                                   self.global_pc_feat[0])
    self.visualizer.step()

  @override
  def vis_update(self, **kwargs) -> None:
    if "feat_img" in kwargs:
      self.visualizer.log_feature_img(kwargs["feat_img"][-1].permute(1, 2, 0))

  @override
  def feature_query(self,
                    feat_query: torch.FloatTensor,
                    softmax: torch.FloatTensor = False,
                    compressed: bool = True) -> dict:
    if self.is_empty():
      return

    self._concat()
    pc_feat = self.global_pc_feat[-1]
    if self.feat_compressor is not None and not compressed:
      pc_feat = self.feat_compressor.decompress(pc_feat)

    pc_feat = self.encoder.align_spatial_features_with_language(
      pc_feat.unsqueeze(-1).unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)

    r = compute_cos_sim(feat_query, pc_feat, softmax=softmax).T
    return dict(pc_xyz=self.global_pc_xyz[-1], pc_sim=r)

  @override
  def vis_query_result(self,
                       query_results: dict,
                       vis_labels: List[str] = None,
                       vis_colors: Dict[str, str] = None,
                       vis_thresh: float = 0) -> None:

    pc_xyz = query_results["pc_xyz"]
    pc_sim = query_results["pc_sim"]
    for q in range(pc_sim.shape[0]):
      kwargs = dict()
      label = vis_labels[q]
      if vis_colors is not None and label in vis_colors.keys():
        kwargs["high_color"] = vis_colors[label]
        kwargs["low_color"] = (0, 0, 0)
      self.visualizer.log_heat_pc(pc_xyz, pc_sim[q, :],
        layer=f"queries/{label.replace(' ', '_').replace('/', '_')}",
        vis_thresh=vis_thresh,
        **kwargs)
