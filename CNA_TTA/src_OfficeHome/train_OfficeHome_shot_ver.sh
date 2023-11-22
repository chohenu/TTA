SEED=2021
SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/seed${SEED}/office-home/"

PORT=10019
MEMO="OfficeHome_LR"
SUB_MEMO="V2_SHOT"

for SRC_DOMAIN in Product RealWorld
do
    for TGT_DOMAIN in Art Clipart Product RealWorld
    do
        if [ "$TGT_DOMAIN" == "$SRC_DOMAIN" ]; then 
            echo "pass trainig"
        else
            CUDA_VISIBLE_DEVICES=6,7 python ../main_adacontrast.py \
            seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="office-home" \
            data.data_root="/mnt/data" data.workers=8 \
            data.dataset="OfficeHome" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
            model_src=shot.yaml \
            learn=targetv2.yaml \
            learn.epochs=100 \
            model_src.arch="resnet50" \
            model_tta.src_log_dir=${SRC_MODEL_DIR} \
            optim.lr=2e-4 
        fi
    done
done



(96.35+96.11+96.08)/3=96.18

(77.73+78.65+78.27)/3=78.21

(88.34+88.04+87.97)/3=88.12

(77.38+75.43+76.52)/3=76.44

(96.33+95.95+95.82)/3=96.03

(94.84+94.55+93.4)/3=94.26

(93.79+93.15+93.25)/3=93.40

(85.82+84.75+87.65)/3=86.07

(93.12+93.23+92.13)/3=92.82

(88.82+88.82+88.34)/3=88.66

(89.66+88.88+88.64)/3=89.06

(43.64+41.85+41.71)/3=42.4

(85.48+84.95+84.98)/3=85.14