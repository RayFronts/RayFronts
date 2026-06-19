# Mapping
This directory includes all mapping classes. We currently only support posed
RGBD mapping.

## Available Options
- [SemanticPointCloud](semantic_point_cloud.py): A minimal semantic point cloud with no filtering associating a semantic feature with every point. Very memory inefficient and usually used for debugging only.
- [SemanticVoxelMap](semantic_voxel_map.py): A minimal semantic voxel map storing semantic features with every voxel.
- [OccupancyVoxelMap](occ_voxel_map.py): A minimal occupancy voxel map where log odds occupancy is stored in each voxel. Both empty and occupied cells are kept. This map purely uses PyTorch tensors.
- [OccupancyVdbMap](occ_vdb_map.py): A minimal occupancy voxel map where log odds occupancy is stored in each voxel. Both empty and occupied cells are kept. The map is built on C++ library OpenVDB for efficiency in storing large empty areas.
- [FrontierVDBMap](frontier_vdb_map.py): Builds on top of OccupancyVdbMap but adds flexible frontier calculation. 
- [SemanticOccVdbMap](semantic_occ_vdb_map.py): Combines the logic of Semantic Voxel Map and Occupancy Vdb Map to maintain semantic voxels + occupancy vdb maps. The added synergy is that occupancy is used to filter the semantic voxels.
- [SemanticRayFrontiersMap](semantic_ray_frontiers_map.py) (RayFronts): This map combines occupancy + frontiers + semantic voxels + semantic ray frontiers to guide a robot with semantics within and beyond depth-range.

## Adding a mapper
0. Read the [CONTRIBUTING](../../CONTRIBUTING.md) file.
1. Create a new python file with the same name as your map.
2. Extend one of the base abstract classes found in [base.py](base.py).
3. Implement and override the inherited methods.
4. Add a config file with all your constructor arguments in configs/mapping. 
5. import your map in the mapping/__init__.py file.
6. Edit this README to include your new addition.
