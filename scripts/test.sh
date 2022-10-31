#export CUDA_VISIBLE_DEVICES=0
#python3 main.py \
#--data_path "graphs" \
#--val_set "../../dataset_splits/test_set.txt" \
#--model_path "graph_transformer/saved_models/" \
#--log_path "graph_transformer/runs/" \
#--task_name "GraphCAM" \
#--batch_size 1 \
#--test \
#--log_interval_local 6 \
#--resume "graph_transformer/saved_models/GraphCAM.pth" \
#--dataset_metadata_path '../../dataset_splits'

export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--data_path ${MODEL_TMP_DIR}/graphs \
--val_set ${DATASET_SPLITS_DIR}/test_set.txt \
--model_path ${MODEL_TMP_DIR}/graph_transformer/saved_models/ \
--log_path ${MODEL_TMP_DIR}/graph_transformer/runs/ \
--task_name "GraphCAM" \
--batch_size 1 \
--test \
--log_interval_local 6 \
--resume ${MODEL_TMP_DIR}/graph_transformer/saved_models/GraphCAM.pth \
--dataset_metadata_path ${DATASET_SPLITS_DIR}
