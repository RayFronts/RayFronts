#!/usr/bin/env bash
# StarlingMax: ToF + RGB + LingBot + semantic ray frontiers
# Run from anywhere; script cd's to RayFronts/ and sets PYTHONPATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYFRONTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RAYFRONTS_ROOT"
export PYTHONPATH="$RAYFRONTS_ROOT:${PYTHONPATH}"

exec python3 -m rayfronts.mapping_server \
  mapping=semantic_ray_frontiers_map \
  dataset=starlingmax \
  vis=rerun \
  vis.map_period=1 \
  vis.input_period=1 \
  vis.pose_period=1 \
  mapping.vox_accum_period=1 \
  mapping.vox_size=0.1 \
  depth_estimator=lingbot \
  depth_limit=5 \
  dataset.rgb_resolution=[320,480] \
  mapping.ray_accum_period=1 \
  mapping.max_pts_per_frame=5000 \
  mapping.max_empty_pts_per_frame=20000 \
  querying.text_query_mode=labels \
  querying.query_file=labels.txt \
  messaging_service=ros \
  "$@"
