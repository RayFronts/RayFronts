# Datasets / Data sources/streams

This directory hosts all available data sources.

## Available Options:
- [NiceReplica](replica.py): Replica dataset version processed by [Nice-Slam](https://github.com/cvg/nice-slam/)
- [SemanticNerfReplica](replica.py): Replica dataset version processed by [Semantic_nerf](https://github.com/Harry-Zhi/semantic_nerf). This version can provide 2D semantic segmentation images as well.
- [ScanNet](scannet.py): Reads from processed prerecorded indoor [ScanNet data](https://github.com/ScanNet/ScanNet/tree/master). 
- [TartanAirV2](tartanair.py) Reads from [TartanAirV2](https://tartanair.org/) and [TartanGround](https://tartanair.org/tartanground/).
- [Ros2Subscriber](ros.py): Subscribe to any ros2 topics that provide posed RGBD information.
- [Rosnpy](ros.py): (Deprecated).
- [ScanNet++](scannetpp.py): ScanNet++ dataset processed according to instructions mentioned in the official [ScanNet++ repo](https://github.com/scannetpp/scannetpp).

## Adding a Dataset
0. Read the [CONTRIBUTING](../../CONTRIBUTING.md) file.
1. Create a new python file with the same name as your dataset.
2. Extend one of the base abstract classes found in [base.py](base.py).
3. Implement and override the inherited methods.
4. Add a config file with all your constructor arguments in configs/dataset. 
5. import your encoder in the datasets/__init__.py file.
6. Edit this README to include your new addition.
