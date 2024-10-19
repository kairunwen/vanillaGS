#! /bin/bash

GPU_ID=3
DATA_ROOT_DIR="/ssd2/zhiwen/projects/w_workspace/datasets/InstantSplat/collated_instantsplat_data/eval"
DATASETS=(
    # mipnerf360
    #   Tanks
    # TT
    # MVimgNet
    Mipnerf
    )


SCENES=(
    garden
    # Barn
    #   Family
    # Francis
    # Horse
    # Ignatius
    )

N_VIEWS=(    
    0
    3
    6
    9
    12
    # 24
     #50
    #100
    #150
    )

# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=30000

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
            IMAGE_PATH=${SOURCE_PATH}images
            # IMAGE_PATH=${SOURCE_PATH}images_4
            MODEL_PATH=./output/${DATASET}/${SCENE}/${N_VIEW}_views/

            # ----- (1) Train: jointly optimize pose -----
            CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            -r 1 \
            --n_views ${N_VIEW}  \
            --iterations ${gs_train_iter} \
            --eval \
            "
            # --scene ${SCENE} \


            # ----- (2) Render -----
            CMD_R="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            -r 1 \
            --n_views ${N_VIEW}  \
            --eval \
            "            
            # --scene ${SCENE} \

            # ----- (3) Metrics -----
            CMD_M="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./metrics.py \
            -m ${MODEL_PATH}  \
            "


            echo "========= ${SCENE}: Train========="
            eval $CMD_T
            echo "========= ${SCENE}: Render ========="
            eval $CMD_R
            echo "========= ${SCENE}: Metric ========="
            eval $CMD_M
            done
        done
    done