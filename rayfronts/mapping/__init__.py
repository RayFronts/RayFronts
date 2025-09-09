from rayfronts.mapping.base import RGBDMapping, SemanticRGBDMapping
from rayfronts.mapping.semantic_point_cloud import SemanticPointCloud
from rayfronts.mapping.semantic_voxel_map import SemanticVoxelMap
from rayfronts.mapping.occ_voxel_map import OccupancyVoxelMap

import logging
logger = logging.getLogger(__name__)

try:
  from rayfronts.mapping.occ_vdb_map import OccupancyVDBMap
  from rayfronts.mapping.frontier_vdb_map import FrontierVDBMap
  from rayfronts.mapping.semantic_occ_vdb_map import SemanticOccVDBMap
  from rayfronts.mapping.semantic_ray_frontiers_map import SemanticRayFrontiersMap

except ModuleNotFoundError as e:
  if e.name == "rayfronts_cpp" or e.name == "openvdb":
    logger.warning("Unable to import %s. Make sure you compiled "
                  "it. Will not import mappers that depend on it.", e.name)
  else:
    raise e
