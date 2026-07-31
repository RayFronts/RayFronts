# Experiments
This directory will include instructions to reproduce the experiments in the 
RayFronts publication. Note that scores may differ slightly as the code base evolved and is evolving but the relative scores and insights stay the same. See [5efcfb7](/../../commit/5efcfb7e31d6858299012a0b95e3cc31154d9cae) and [0b7b4a7](/../../commit/0b7b4a718fd41804a86bc99ae42744a17a02161a) for normalization bugs fixed post publication that improve NARADIO and Trident segmentation. Revert if reproduction of paper numbers is your goal.

If anything is unclear or you face problems feel free to raise an issue.

## Online semantic mapping & search volume evaluation

### Setting up TartanGround

Disclaimer: RayFronts was released before the final shape of TartanGround took place and at the time it was unclear if it was going to be released under tartanairv2 or as its separate dataset. As such, you will find us using either name. However, we always used the ground vehicle forward motion trajectories for evaluation of RayFronts.

Make sure to download the [TartanGround](https://tartanair.org/tartanground/) scenes we use which are (Downtown,Factory,AbandonedCableDay,ConstructionSiteOvercast). 

TartanGround provide semantic labels in a single label segmentation setting, however many are ambiguous (e.g brick vs wall vs house) or inaccurate (e.g camera_actor). To mitigate the effect of ambiguity, we delete inaccurate classes and merge ambiguous classes. We provide the cleaned labels [here](tartanground_labels) which you should add to their respective directories. The expected dataset structure is:

```
root
--<scene_name>
----seg_label.json
----seg_label_clean.json
----Data_ground
------...
```

### Running evaluation
You can loop over the configs in [srchvol_configs](srchvol_configs) to run all experiments.

```
#!/bin/bash

CONFIGS=(
    rayfronts_0
    rayfronts_10
    rayfronts_20
    sempose_0
    spherical_semfronts_10
    spherical_semfronts_20
    unidir_semfronts_10
    unidir_semfronts_20
)

for config in "${CONFIGS[@]}"
do
    python scripts/srchvol_eval.py \
        --config-dir experiments/srchvol_configs/ \
        --config-name "$config" \
        dataset.path="<path_to_tartanground>" \
        --multirun
done
```
Note that the searchvolume evaluation requires more than 24GB of GPU memory at the moment for some scenarios. Scenarios where 20m depth range is needed, we run out of memory on AbandonedCableDay scene and downgrade the specs to `mapping.fronti_subsampling_min_fronti=10 mapping.max_rays_per_frame=1000`

Also note that for the numbers in the paper we use the slower raysampling algorithm instead of frustum culling as currently, frustum culling has issues with holes in the ground if the camera was too close which will result in lower mIoU, this will be fixed in the future. Change the default in rayfronts/geometry3d.py#L361

After successfully generating the results with no errors. We now need to parse them and compute AUC metrics using [summarize_srchvol_eval.py](../scripts/summarize_srchvol_eval.py). Head to that file and edit the top global variables to point to the correct paths and experiment names then run with `python scripts/summarize_srchvol_eval.py`. The generated file which will be at `{root_path}/srchvol_summary.csv` will contain the AUC average metrics for all scenes. The average row is what we report in the paper.

## Offline zero-shot 3D semantic segmentation evaluation.

### Setting up the datasets

The semantic segmentation experiments are evaluated on three datasets:
- [ScanNet](http://www.scan-net.org)
- [Replica](https://github.com/facebookresearch/Replica-Dataset)
- [TartanGround](https://tartanair.org/tartanground/)

Make sure you have downloaded and set up these datasets according to their respective instructions. The dataset paths should be configured in the config files.

### Running evaluation

We integrated the encoders for [NACLIP](https://github.com/sinahmr/NACLIP), [Trident](https://github.com/YuHengsss/Trident), and [ConceptFusion](https://github.com/concept-fusion/concept-fusion) into our RayFronts pipeline for the semantic segmentation evaluations (alpha=1 for conceptfusion). We also extracted the prediction results from [HOV-SG](https://github.com/hovsg/HOV-SG) and [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) from their respective repositories and evaluted ConceptGraphs with our pipeline. For replica, we use the NiceSlam version and we get the GT semantic labels from HOV-SG (Uploaded [here](https://cmu.box.com/s/x7si4h8y4sfk07dgmn9uwowaf2g74zjw) for convenience) since NiceSlam does not provide semantic labels without the original dataset.

You can run each of the experiments using the following command:

```
python scripts/semseg_eval.py \
    --config-dir experiments/semseg_configs/ \
    --config-name DATASET_ENCODER.yaml 
```

Add `--multirun` to run all scenes. You can set vis=null in the config to avoid visualization taking a lot of memory.
Once all scenes have been run for a dataset. A file `semseg_final_summary_results.csv` will be generated in `<eval_out>/<dataset>` showing the metrics for each scene. The average across all scenes is what we report in the paper.

The config file for each experiment is named after the dataset used and the encoder to evaluate (i.e. [replica_naclip.yaml](semseg_configs/replica_naclip.yaml) referring to running NACLIP encoder on the Replica Dataset). We record the configurations of our evaluation presented in our paper as shown in each of the config files. Please refer to each of the config files for more detailed instructions, including downloading respective checkpoints as required by each encoder. Note that:
* Our RayFronts encoder is referred to as naradio in the config files. 
* TartanGround is referred to as tartanair in the config files.
* ConceptGraphs is referred to as concpgr in the config files. We ran predictions for ConceptGraphs in their own pipeline, and only used our pipeline for evaluating against ground truth, hence we load external ground truths in the config files.

Since we ran the predictions of semantic segmentations for HOV-SG and ConceptGraphs in their respective pipelines, we also documented the exact modifications we made to each repository in the form of git patches. Please apply our [HOV-SG patch](patches/hovsg.patch) to this [HOV-SG commit](https://github.com/hovsg/HOV-SG/tree/6d9cd24d7d3b877896b00b7f1c97c2be86cdfa04) and our [ConceptGraphs patch](patches/concept_graphs.patch) to this [ConceptGraphs commit](https://github.com/concept-graphs/concept-graphs/tree/93277a02bd89171f8121e84203121cf7af9ebb5d). Please refer to our documentations within our modifications for details. 
