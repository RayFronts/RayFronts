"""Defines a dummy dataset that generates randomozied and preset data to test.

Requires open3d
"""

from typing import Union, Tuple
from typing_extensions import override

import torch
import open3d as o3d
import numpy as np

from rayfronts.datasets.base import PosedRgbdDataset

def create_room(size=4.0, height=3.0):
  walls = []
  def make_box(width, height_, depth, translate):
    box = o3d.geometry.TriangleMesh.create_box(
      width=width, height=height_, depth=depth)
    box.translate(translate)
    box.paint_uniform_color([0.5, 0.5, 0.5])
    box.compute_vertex_normals()
    return box

  walls.append(make_box(size, size, 0.05, [-size/2, -size/2, 0])) # Floor
  walls.append(make_box(size, size, 0.05, [-size/2, -size/2, height])) # Ceiling
  walls.append(make_box(size, 0.05, height, [-size/2, -size/2, 0]))
  walls.append(make_box(size, 0.05, height, [-size/2, size/2, 0]))
  walls.append(make_box(0.05, size, height, [-size/2, -size/2, 0]))
  walls.append(make_box(0.05, size, height, [size/2, -size/2, 0]))
  objects = []
  sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
  sphere.translate([0.5, 0, 0.4])
  sphere.paint_uniform_color([1.0, 0.0, 0.0])
  sphere.compute_vertex_normals()
  objects.append(sphere)

  cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=1.0)
  cylinder.translate([-0.5, -0.5, 0.5])
  cylinder.paint_uniform_color([0.0, 1.0, 0.0])
  cylinder.compute_vertex_normals()
  objects.append(cylinder)

  box = o3d.geometry.TriangleMesh.create_box(width=0.6, height=0.6, depth=0.6)
  box.translate([0.0, 0.8, 0.0])
  box.paint_uniform_color([0.0, 0.0, 1.0])
  box.compute_vertex_normals()
  objects.append(box)
  return objects, walls

def look_at_c2w_rdf(eye, center, up=(0, 0, 1)):
  """ Compute a 4x4 camera-to-world matrix in RDF convention.

  Args:
    eye:    camera position (3,)
    center: look-at target (3,)
    up:     up vector (3,)
  """
  eye = np.array(eye, dtype=np.float32)
  center = np.array(center, dtype=np.float32)
  up = np.array(up, dtype=np.float32)

  # Forward = viewing direction
  forward = center - eye
  forward /= np.linalg.norm(forward)

  # Right = forward × up   (ensures right-handed frame)
  right = np.cross(forward, up)
  right /= np.linalg.norm(right)

  # Recompute true up
  down = np.cross(forward, right)  # points "down" in RDF
  down /= np.linalg.norm(down)

  # Build rotation matrix in RDF convention
  # X=Right, Y=Down, Z=Forward
  R = np.stack([right, down, forward], axis=1)  # 3x3

  # Assemble 4x4 camera-to-world matrix
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, :3] = R
  c2w[:3, 3] = eye
  return c2w

class DummyDataset(PosedRgbdDataset):
  """A dummy dataset that just rotates around a room with some objects."""

  def __init__(self,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear",
               num_frames: int = -1,
               depth_noise_var: float = 0,
               fov: float = 90):
    """
    Args:
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
      num_frames: How many frames to output. Set to -1 to keep looping.
      depth_noise_var: Variance of gaussian noise to add to depth values.
      fov: Camera horizontal field of view in degrees.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    self.num_frames = num_frames
    self.depth_noise_var = depth_noise_var
    self.depth_renderer = o3d.visualization.rendering.OffscreenRenderer(
      self.depth_w, self.depth_h)
    self.rgb_renderer = o3d.visualization.rendering.OffscreenRenderer(
      self.rgb_w, self.rgb_h)

    aspect = self.depth_w / self.depth_h
    fov_rad = fov * np.pi / 180.0
    fx = fy = (self.depth_w / 2.0) / np.tan(fov_rad / 2.0)
    cx = self.depth_w / 2.0
    cy = self.depth_h / 2.0
    self.intrinsics_3x3 = torch.tensor([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]], dtype=torch.float)

    self._objects = list()
    self._walls = list()
    for j,ren in enumerate([self.rgb_renderer, self.depth_renderer]):
      objects, walls = create_room()
      self._objects.append(objects)
      self._walls.append(walls)
      mat = o3d.visualization.rendering.MaterialRecord()
      mat.shader = "defaultLit"
      scene = ren.scene
      scene.camera.set_projection(
        fov, aspect, 0.1, 10,
        o3d.visualization.rendering.Camera.FovType.Horizontal)

      for i, wall in enumerate(walls):
        scene.add_geometry(f"wall_{j}_{i}", wall, mat)
      for i, obj in enumerate(objects):
        scene.add_geometry(f"obj_{j}_{i}", obj, mat)


  @override
  def __iter__(self):
    f = 0
    # Camera orbit parameters
    center = [0, 0, 0.0]
    radius = 1.5
    elevation = 1.2

    while self.num_frames <= 0 or f < self.num_frames:
      if self.frame_skip > 0 and f % (self.frame_skip+1) != 0:
        continue
      theta = 2 * np.pi * (f / 180) # azimuth
      cam_pos = [
        radius * np.cos(theta),
        radius * np.sin(theta),
        elevation
      ]
      self.depth_renderer.scene.camera.look_at(center, cam_pos, [0, 0, 1])
      self.rgb_renderer.scene.camera.look_at(center, cam_pos, [0, 0, 1])

      rgb_img = torch.from_numpy(np.asarray(
        self.rgb_renderer.render_to_image())).permute(2, 0, 1).float()/255

      depth_img = torch.from_numpy(np.asarray(
        self.depth_renderer.render_to_depth_image(z_in_view_space=True))
        ).unsqueeze(0).float()

      # Apply Gaussian noise
      if self.depth_noise_var > 0:
        noise = torch.randn_like(depth_img) * np.sqrt(self.depth_noise_var)
        depth_img = torch.clamp(depth_img + noise, min=0.0)

      pose_4x4 = torch.from_numpy(look_at_c2w_rdf(cam_pos, center)).float()

      frame_data = dict(rgb_img=rgb_img, depth_img=depth_img, pose_4x4=pose_4x4)
      yield frame_data
      f += 1
