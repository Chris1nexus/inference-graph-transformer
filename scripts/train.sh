export CUDA_VISIBLE_DEVICES=0

#python3 main.py \
#--data_path 'graphs' \
#--train_set "../../dataset_splits/train_set.txt" \
#--val_set "../../dataset_splits/val_set.txt" \
#--model_path "graph_transformer/saved_models/" \
#--log_path "graph_transformer/runs/" \
#--task_name "GraphCAM" \
#--batch_size 8 \
#--train \
#--log_interval_local 6 \
#--dataset_metadata_path '../../dataset_splits'
#-- data_path "path_to_graph_data" \

python3 main.py \
--data_path ${MODEL_TMP_DIR}/graphs \
--train_set ${DATASET_SPLITS_DIR}/train_set.txt \
--val_set ${DATASET_SPLITS_DIR}/val_set.txt \
--model_path  ${MODEL_TMP_DIR}/graph_transformer/saved_models/ \
--log_path ${MODEL_TMP_DIR}/graph_transformer/runs/ \
--task_name "GraphCAM" \
--batch_size 8 \
--train \
--log_interval_local 6 \
--dataset_metadata_path ${DATASET_SPLITS_DIR}
