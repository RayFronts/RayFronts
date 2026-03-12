#!/usr/bin/env bash
# Isaac Sim (ros2isaacsim): RGB + depth + Odometry + camera_info
# Run from anywhere; script cd's to RayFronts/ and sets PYTHONPATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYFRONTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RAYFRONTS_ROOT"
export PYTHONPATH="$RAYFRONTS_ROOT:${PYTHONPATH}"

exec python3 -m rayfronts.mapping_server \
  dataset=ros2isaacsim \
  mapping=semantic_voxel_map \
  mapping.vox_size=0.2 \
  dataset.rgb_resolution=[224,224] \
  dataset.depth_resolution=[224,224] \
  dataset.frame_skip=10 \
  depth_limit=20 \
  "$@"
