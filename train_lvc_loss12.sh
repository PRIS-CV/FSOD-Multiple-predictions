python -m tools.train_net \
    --config-file configs/COCO-detection/loss12.yaml \
    --num-gpus 4 \
    --resume

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 checkpoints/coco/faster_rcnn/loss12/model_final.pth \
#     --save-dir checkpoints/coco/faster_rcnn/baseline_base

# python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_10shot.yaml \
#     --num-gpus 2 \
#     --resume

# python -m tools.ckpt_surgery \
#     --method combine \
#     --coco \
#     --src1 checkpoints/coco/faster_rcnn/baseline_base/model_final.pth \
#     --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_10shot/model_final.pth \
#     --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_10shot/

# python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_10shot_aug_ftmore_dropout.yaml \
#     --num-gpus 2
