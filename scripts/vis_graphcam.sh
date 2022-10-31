#python3 src/vis_graphcam.py \
#--path_file "cptac_lung_val2.txt" \
#--path_patches "/run/media/yizheng/YiZ/CPTAC_data/patches" \
#--path_WSI "/run/media/yizheng/YiZ/CPTAC/PKG - CPTAC-LUAD/LUAD" \
#--path_graph "/scratch2/zheng/cptac_data/CPTAC_LUNG_features/simclr_files" \


#python3 src/vis_graphcam.py \
#--path_file "../../dataset_splits/test_set.txt" \
#--path_patches "TCGA_LUNG_patches" \
#--path_WSI "../../dataset" \
#--path_graph "graphs/simclr_files" \
#--dataset_metadata_path '../../dataset_splits'

python3 src/vis_graphcam.py \
--path_file {DATASET_SPLITS_DIR}/test_set.txt \
--path_patches ${PATCHES_TRAIN_DIR} \
--path_WSI ${DATASET_DIR} \
--path_graph ${MODEL_TMP_DIR}/graphs/simclr_files \
--dataset_metadata_path ${DATASET_SPLITS_DIR}