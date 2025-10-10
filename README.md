<p align="center"><img src="assets/logo.gif" width="70%"/></p>
<h1 align="center">Open-Set Semantic Ray Frontiers <br/>
  for Online Scene Understanding and Exploration</h1>

<p align="center">
  <a href="https://oasisartisan.github.io/"><strong>Omar Alama</strong></a>
  .
  <a href="https://www.linkedin.com/in/avigyan-bhattacharya"><strong>Avigyan Bhattacharya</strong></a>
  ·
  <a href="https://purenothingness24.github.io/"><strong>Haoyang He</strong></a>
  ·
  <a href="https://seungchan-kim.github.io/"><strong>Seungchan Kim</strong></a>
  <br>
  <a href="https://haleqiu.github.io/"><strong>Yuheng Qiu</strong></a>
  .
  <a href="https://theairlab.org/team/wenshan/"><strong>Wenshan Wang</strong></a>
  ·
  <a href="https://cherieho.com/"><strong>Cherie Ho</strong></a>
  ·
  <a href="https://nik-v9.github.io/"><strong>Nikhil Keetha</strong></a>
  ·
  <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
</p>

  <h3 align="center"><a href="https://arxiv.org/abs/2504.06994">Paper</a> | <a href="https://RayFronts.github.io/">Project Page</a> | <a href="https://www.youtube.com/watch?v=fFSKUBHx5gA">Video</a> | <a href="https://www.youtube.com/watch?v=_tIVlw1Wrh4">Podcast</a> | <a href="https://x.com/OmarAlama/status/1910102471587831997">Thread</a></h3>
  <div align="center"></div>


<img src="assets/method_teaser.gif">

- 🤖 Guide your robot with semantics within and beyond. RayFronts can be easily deployed as part of your robotics stack as it supports ROS2 inputs for mapping and querying and has robust visualizations.
- 🖼️ Stop using slow SAM crops + CLIP pipelines. Use our encoder to get dense language aligned features in one forward pass.
- 🚀 Bootstrap your semantic mapping project. Utilize the modular RayFronts mapping codebase with its supported datasets to build your project (Novel encoding, novel mapping, novel feature fusion...etc.) and get results fast.
- 💬 Reach out or raise an issue if you face any problems !
## News/Release
- [06.16.2025] 🔥🔥🔥 RayFronts has been accepted to [IROS25](https://www.iros25.org/). See you in Hangzhou !!
- [06.12.2025] 🔥🔥 RayFronts has been accepted to [RSS25](https://roboticsconference.org/) [SemRobs](https://semrob.github.io/) & [RoboReps](https://rss25-roboreps.github.io/) Workshops. See you in Los Angeles !!
- [06.11.2025] 🔥 RayFronts code is released !
- [8.4.2025] Initial public arxiv release.
---

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#environment-setup">Environment Setup</a>
    </li>
    <li>
      <a href="#running-rayfronts">Running RayFronts</a>
    </li>
    <li>
      <a href="#running-image-encoding">Running Image Encoding</a>
    </li>
    <li>
      <a href="#benchmarking">Benchmarking</a>
    </li>
    <li>
      <a href="#citing-rayfronts">Citation</a>
    </li>
  </ol>
</details>

## Abstract
<img src="assets/abstract_fig.jpg">
<b><i>RayFronts</i></b> is a real-time semantic mapping system that enables fine-grained scene understanding both within and beyond the depth perception range. Given an example mission through multi-modal queries to locate red buildings & a water tower, <b><i>RayFronts</i></b> enables: (1) Significant search volume reduction for online exploration (as shown by the red and blue cones at the top) and localization of far-away entities (e.g., the water & radio tower). (2) Online semantic mapping, where prior semantic ray frontiers evolve into semantic voxels as entities enter the depth perception range (e.g., the red buildings query on the right side). (3) Multi-objective fine-grained open-set querying supporting various open-set prompts such as "Road Cracks", "Metal Stairs", and "Green Dense Canopy".

## Environment Setup

### Conda/Mamba
For a minimal setup without ROS and without openvdb you can create a python environment with the [environment.yml](environment.yml) conda specification (Installing it one shot doesn't work usually and you may need to start with a pytorch enabled environment and install the rest of the dependencies with pip). This won't allow you to run the full RayFronts mapping however since it requires OpenVDB.

For a full local installation:
1. Install packages in environment.yml:
    ```bash
    conda env create --file environment.yml
    ```
2. (Optional) Install ros2-jazzy in a conda/mamba environment using [these instructions](https://robostack.github.io/GettingStarted.html):
    ```bash
    conda install -c robostack-jazzy ros-jazzy-desktop
    ```
3. Install pytorch 2.4 with cuda 12.1
    ```
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
    ```
4. Clone the [patched OpenVDB](https://github.com/OasisArtisan/openvdb), build and install in your conda environment.
    ```
    sudo apt install -y libboost-iostreams-dev libtbb-dev libblosc-dev

    git clone https://github.com/OasisArtisan/openvdb && cd openvdb

    cmake -B build \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DOPENVDB_BUILD_PYTHON_MODULE=ON \
    -DOPENVDB_BUILD_PYTHON_UNITTESTS=ON \
    -DOPENVDB_PYTHON_WRAP_ALL_GRID_TYPES=ON \
    -DUSE_NUMPY=ON \
    -Dnanobind_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nanobind/cmake

    cmake --build build --config Release -j16
    cmake --build build --target install
    ```
5. Build CPP extension by running `CMAKE_INSTALL_PREFIX=$CONDA_PREFIX ./compile.sh`

### Docker
Two docker build files are provided. [One for desktop](docker/desktop.Dockerfile), and [one for the NVIDIA Jetson platofrm](docker/jetson.Dockerfile) that give you a full installation of RayFronts with ROS2 and OpenVDB.

You can build the image by going to the docker directory then running:

    docker build . -f desktop.Dockerfile  -t rayfronts:desktop

To run the docker image, an example command is available at the top of each docker file.

## Running RayFronts

1. Setup the data source / dataset. Head to the [datasets](rayfronts/datasets) folder to learn more about the available options. Each dataset class documents how to download and structure your data. For now you can download [NiceSlam Replica](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) to test.
2. Configure RayFronts. RayFronts has many hyperparameters to choose from. Head over to [configs](rayfronts/configs) to learn more about the different configuration options. For now we will pass in configurations via the command line for simplicity.
3. There are many mapping systems to choose from from simple occupancy maps and semantic voxel maps to the full fledged Semantic Ray Frontiers (RayFronts). Head over to [mapping](rayfronts/mapping) to learn more about the different options. For now we will assume you want to run the full fledged RayFronts mapper. Run:
    ```
    python3 -m rayfronts.mapping_server dataset=niceslam_replica dataset.path="path_to_niceslam_replica" mapping=semantic_ray_frontiers_map mapping.vox_size=0.05 dataset.rgb_resolution=[640,480] dataset.depth_resolution=[640,480]
    ```
4. To add and visualize queries, setup a query file (named "prompts.txt" for e.g) and add a query at each line in the text file (You can add paths to images for image querying). Next, add the following command line options when running RayFronts `querying.text_query_mode=prompts querying.query_file=prompts.txt querying.compute_prob=True querying.period=100`. More information can be found about the querying options in the [default.yml](rayfronts/configs/default.yml) config file.

## Running Image Encoding
If you are interested in using the encoder on its own for zero-shot open-vocabulary semantic segmentation, follow the example at the top of the [NARADIO](rayfronts/image_encoders/naradio.py) module.
Or run the provided GRADIO app by installing gradio `pip install gradio` then running:

```
python scripts/encoder_semseg_app.py encoder=naradio encoder.model_version=radio_v2.5-b
```

## Benchmarking

For details on reproducing RayFronts tables, go to [experiments](experiments).

### Offline zero-shot 3D semantic segmentation evaluation.

Configure your evaluation parameters use [this](experiments/semseg_configs/replica_naradio.yaml) as an example.
Run:
```
python scripts/semseg_eval.py --config-dir <path_to_config_dir> --config-name <name_of_config_file_without_.yaml>
```
Results will populate in the eval_out directory set in the config
### Online semantic mapping & search volume evaluation
Configure your evaluation parameters use [this](experiments/srchvol_configs/rayfronts_10.yaml) as an example.
Run:
```
python scripts/srchvol_eval.py --config-dir <path_to_config_dir> --config-name <name_of_config_file_without_.yaml>
```
Results will populate in the eval_out directory set in the config.

Note that AUC values are computed after the initial results are computed. Use [summarize_srchvol_eval.py](scripts/summarize_srchvol_eval.py) to compute those and any additional derrivative metrics you are interested in.

## Citing RayFronts
If you find this repository useful, please consider giving a star and citation:

    @misc{alama2025rayfrontsopensetsemanticray,
          title={RayFronts: Open-Set Semantic Ray Frontiers for Online Scene Understanding and Exploration},
          author={Omar Alama and Avigyan Bhattacharya and Haoyang He and Seungchan Kim and Yuheng Qiu and Wenshan Wang and Cherie Ho and Nikhil Keetha and Sebastian Scherer},
          year={2025},
          eprint={2504.06994},
          archivePrefix={arXiv},
          primaryClass={cs.RO},
          url={https://arxiv.org/abs/2504.06994},
    }
