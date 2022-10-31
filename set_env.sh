
# environment variables for model training

export TMP_DIR_NAME=tmp/graph-transformer

export DATASET_NAME='dataset_large'
export DATASET_DIR=datasets/${DATASET_NAME}
export DATASET_SPLITS_DIR=datasets/${DATASET_NAME}/${DATASET_NAME}_splits
export MODEL_TMP_DIR=${DATASET_DIR}/${TMP_DIR_NAME}
export PATCHES_TRAIN_DIR=${MODEL_TMP_DIR}/patches
export GRAPHS_TRAIN_DIR=${MODEL_TMP_DIR}/graphs
mkdir -p $PATCHES_TRAIN_DIR
mkdir -p $GRAPHS_TRAIN_DIR


# environment variables for the inference api
export DATA_DIR=queries
export PATCHES_DIR=${DATA_DIR}/patches
export SLIDES_DIR=${DATA_DIR}/slides
export GRAPHCAM_DIR=${DATA_DIR}/graphcam_plots
mkdir $GRAPHCAM_DIR -p


# manually put the metadata in the metadata folder
export CLASS_METADATA='metadata/label_map.pkl'

# manually put the desired weights in the weights folder
export WEIGHTS_PATH='weights'
export FEATURE_EXTRACTOR_WEIGHT_PATH=${WEIGHTS_PATH}/feature_extractor/model.pth
export GT_WEIGHT_PATH=${WEIGHTS_PATH}/graph_transformer/GraphCAM.pth