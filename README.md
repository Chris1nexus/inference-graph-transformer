# Inference ready graph-transformer for whole slide image classification

Adapted from the work published in *IEEE Transactions on Medical Imaging* (https://doi.org/10.1109/TMI.2022.3176598).
This project showcases the inference capabilities of transformer-based models for healthcare applications.
It has been adapted and developed during the Transformer hackaton @ LabLab.Ai<p align=“right”>
<img src="https://github.com/Chris1nexus/infercence-graph-transformer/blob/master/figures/lablab-logo.png" width=“80%” height=“80%”>
</p> 

# Introduction
This repository contains a PyTorch implementation of a deep learning based graph-transformer for whole slide image (WSI) classification. The original authors proposed a Graph-Transformer (GT) network that fuses a graph representation of a WSI and a transformer that can generate WSI-level predictions in a computationally efficient fashion.

<p align="center">
<img src="https://github.com/vkola-lab/graphCAM/blob/main/figures/framework.jpg" width="80%" height="80%">
</p>
This adaptation works on the freely available dataset TCGA. 
The model has been developed considering as primary site the Lungs, for which the available cancer subtypes are LUSC and LUAD.
An arbitrary dataset that contains cancer types associated with different primary sites can be built starting from the make_dataset.ipynb notebook.
The provided notebook allows creation of datasets of **equally sized WSI slide subsets for cancer subtype**, as well as setting the scale of the whole dataset in gigabytes or terabytes.



# Usage
## 1 Environment setup
Start by setting the environment variables by running:
```bash
source ./set_env.sh
```
Note, for each shell it is necessary to re-run the above command, as it sets the environment variables only for the current shell.

Then, download the required datasets, the gdc tool and the main pip dependencies with:
```bash
./download_dataset_manifest.sh
./run_setup.sh
```
## 2 Easy training
At this point, the whole training pipeline can be run by means of:
```bash
./run_train_pipeline.sh
```
The above is composed of four main steps:
1. extraction of patches from the original slide
2. training of the patch feature extractor
3. extraction of the graph of each patch obtained in step(1.) for a given slide, by means of the trained feature extractor
4. training and testing of the Graph Transformer
 

## 3 Easy testing
Finally, after all training steps are completed, inference can be tested with the streamlit.py demo:
```bash
./run_streamlit.sh
```

# Detailed pipeline: step by step
## 1. Graph Construction
<p align="center">
<img src="https://github.com/vkola-lab/graphCAM/blob/main/figures/fig_b.JPG" width="70%" height="70%">
</p>

### (a) Tiling Patch 
```
python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 20 -o <full_patch_to_output_folder> "full_path_to_input_slides/*/*.svs"
```
Mandatory parameters:
<li>-s is tile_size: 512 (512x512 pixel tiles)</li>
<li>-e is overlap, 0 (no overlap between adjacent tiles). Important: the overlap is defined as "the number of extra pixels to add to each interior edge of a tile". Which means that the final tile size is s + 2.e. So to get a 512px tile with a 50% overlap, you need to set s to 256 and e to 128. Also, tile from the edges of the slide will be smaller (since up to two sides have no "interior" edge)</li>
<li>-j is number of threads: 32</li>
<li>-B is Max Percentage of Background allowed: 50% (tiles removed if background percentage above this value)</li>
<li>-o is the path were the output images must be saved</li>
<li>-M set to -1 by default to tile the image at all magnifications. Set it to the value of the desired magnification to tile only at that magnification and save space</li>

### (b) Training Patch Feature Extractor
Go to './feature_extractor' and config 'config.yaml' before training. The trained feature extractor based on contrastive learning is saved in folder './feature_extractor/runs'. We train the model with patches cropped in single magnification (20X). Before training, put paths to all pathces in 'all_patches.csv' file.
```
python run.py
```

### (c) Constructing Graph
Go to './feature_extractor' and build graphs from patches:
```
python build_graphs.py --weights "path_to_pretrained_feature_extractor" --dataset "path_to_patches" --output "../graphs"
```

## 2. Training Graph-Transformer
Run the following script to train and store the model and logging files under "graph_transformer/saved_models" and "graph_transformer/runs".
```
bash scripts/train.sh
```
To evaluate the model. run
```bash scripts/test.sh```

Split training, validation, and testing dataset and store them in text files as:
```
sample1 \t label1
sample2 \t label2
LUAD/C3N-00293-23 \t luad
...
```

## 3. GraphCAM
To generate GraphCAM of the model on the WSI:
```
1. bash scripts/get_graphcam.sh
```
To visualize the GraphCAM:
```
2. bash scripts/vis_graphcam.sh
```
Note: Currently we only support generating GraphCAM for one WSI at each time.

More GraphCAM examples:
<p align="center">
<img src="https://github.com/vkola-lab/graphCAM/blob/main/figures/GraphCAM_example2.PNG" width="80%" height="80%">
</p>

GraphCAMs generated on WSIs across the runs performed via 5-fold cross validation are shown above. The same set of WSI regions are highlighted by our method across the various  cross-validation folds, thus indicating consistency of our technique in highlighting salient regions of interest. 

# Requirements
<li> WSI software: PixelView (deepPath, Inc.) </li>
Major dependencies are:
<li> python </li>
<li> pytorch </li>
<li> openslide-python </li>
<li> Weights & Biases </li>
