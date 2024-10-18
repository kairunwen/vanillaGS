
# python train.py -s /home/workspace/datasets/usm3d_new_hr/Tanks_colmap/Barn/24_views -m ouput/Barn --eval
# python render.py  -m ouput/Barn
# python metrics.py  -m ouput/Barn

# kairun-241018

GPU_ID=0
NOW_TIME=$(date -u -d "+8 hours" +"%y%m%d-%H:%M:%S") 
echo $NOW_TIME

Resolution=(
    # images
    images_512
)

DATASETS=(
    # Tanks_dust3r
    Tanks_colmap
    # MVimgNet
    )

SCENES=(
    Barn
    Family
    Francis
    Horse
    Ignatius  
    )

N_VIEWS=(
    # 3
    # 6
    # 9
    # 12
    # 24
    # 50
    # 100
    # 150
    0
    )



for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            SOURCE_PATH=/home/workspace/datasets/usm3d_new_hr/${DATASET}/${SCENE}/24_views
            MODEL_PATH=./output/${DATASET}/${SCENE}/${N_VIEW}_views/

            ##### ----- (1) Train -----
            CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --eval \
            --images ${Resolution} \
            "

            ##### ----- (2) Render & Generate_high_resolution_video -----
            CMD_R="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./render.py \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --eval \
            --images ${Resolution} \
            "
            
            # --skip_test \
            # --skip_train \

            ##### ----- (3) Metrics -----
            CMD_M="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./metrics.py \
            -m ${MODEL_PATH}  \
            "
            
            
            echo "========= ${SCENE}: Train & Interpolate pose ========="
            eval $CMD_T
            echo "========= ${SCENE}: Render & Generate_high_resolution_video ========="
            eval $CMD_R
            echo "========= ${SCENE}: Metric ========="
            eval $CMD_M
        
            done
        done
    done