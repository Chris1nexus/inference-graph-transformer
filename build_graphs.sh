#python3 build_graphs.py --weights "runs/Oct29_16-15-55_xrh1/checkpoints/model.pth" --dataset "../TCGA_LUNG_patches/*/*" --output "../graphs"

(cd feature_extractor && python3 build_graphs.py --weights ../${FEATURE_EXTRACTOR_WEIGHT_PATH} \
 --dataset ../${PATCHES_TRAIN_DIR}/'*'/'*' \
 --output ../${GRAPHS_TRAIN_DIR} )