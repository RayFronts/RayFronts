#!/usr/bin/env bash
# Isaac Sim (ros2isaacsim): RGB + depth + Odometry + camera_info
# Run from anywhere; script cd's to RayFronts/ and sets PYTHONPATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYFRONTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RAYFRONTS_ROOT"
export PYTHONPATH="$RAYFRONTS_ROOT:${PYTHONPATH}"

exec python3 -m rayfronts.mapping_server \
  dataset=ros2isaacsim \
  mapping=semantic_ray_frontiers_map \
  mapping.vox_size=0.5 \
  mapping.vox_accum_period=2 \
  mapping.ray_accum_period=2 \
  mapping.ray_accum_phase=1 \
  mapping.max_pts_per_frame=3000 \
  mapping.max_empty_pts_per_frame=10000 \
  vis=rerun \
  vis.map_period=10 \
  vis.input_period=10 \
  vis.pose_period=10 \
  dataset.rgb_resolution=[240,240] \
  dataset.depth_resolution=[240,240] \
  dataset.frame_skip=10 \
  depth_limit=10 \
  querying.text_query_mode=labels \
  querying.query_file=labels.txt \
  messaging_service=ros \
  "$@"
